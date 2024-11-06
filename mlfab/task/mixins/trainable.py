"""Defines a mixin for models that can be trained."""

import bisect
import itertools
import logging
import time
from dataclasses import dataclass, is_dataclass
from typing import Any, Generic, Iterator, Literal, Mapping, Sequence, TypeVar, cast, get_args

import numpy as np
import torch
from dpshdl.prefetcher import Prefetcher
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer

from mlfab.core.conf import field
from mlfab.core.state import Phase, State
from mlfab.nn.functions import recursive_chunk
from mlfab.task.base import BaseConfig, BaseTask
from mlfab.task.mixins.optimizer import OptimizerConfig, OptimizerMixin
from mlfab.task.mixins.parallel import ParallelConfig, ParallelMixin
from mlfab.task.mixins.profiler import ProfilerConfig, ProfilerMixin
from mlfab.utils.experiments import StateTimer, get_git_state, get_training_code

logger = logging.getLogger(__name__)

# Batch = TypeVar("Batch")
# Output = TypeVar("Output")

Batch = Any
Output = Any

Loss = Tensor | dict[str, Tensor]

StepKind = Literal["step", "sample", "second"]

PRINT_FINISH_TIME_EVERY_N_SECONDS = 60 * 2


def cast_step_kind(s: str) -> StepKind:
    assert s in get_args(StepKind), f"`step_kind` must be one of {get_args(StepKind)}, not {s}"
    return cast(StepKind, s)


@dataclass(kw_only=True)
class TrainableConfig(
    ProfilerConfig,
    OptimizerConfig,
    ParallelConfig,
    BaseConfig,
):
    valid_every_n_steps: int | None = field(None, help="Number of training steps to run per validation step")
    valid_on_first_step: bool = field(True, help="Run validation on the first step")
    disable_validation: bool = field(False, help="Disable validation")
    valid_every_n_seconds: float | None = field(60.0 * 10.0, help="Run validation every N seconds")
    valid_first_n_seconds: float | None = field(60.0, help="Run first validation after N seconds")
    batches_per_step: int = field(1, help="Batches to accumulate per training step, to simulate larger batch sizes")
    batches_per_step_schedule: list[int] | None = field(
        None,
        help=(
            "A schedule for increasing the effective batch size. The first segment will have batch size "
            "`batches_per_step`, the second will have `2 * batches_per_step`, the third will have "
            "`3 * batches_per_step`, and so on."
        ),
    )
    batch_chunks_per_step_schedule: list[int] | None = field(
        None,
        help=(
            "A schedule for splitting batches into chunks. The batches in the first segment will have "
            "`batch_size / (N + 1)` elements, the second will have `batch_size / N` elements, until "
            "the last segment has `batch_size` elements."
        ),
    )
    batch_dim: int = field(0, help="The batch dimension, for splitting batches into chunks")
    max_steps: int | None = field(None, help="Maximum number of steps to run")
    step_kind: str = field("step", help=f"How to measure a step; one of [{', '.join(get_args(StepKind))}]")


Config = TypeVar("Config", bound=TrainableConfig)


class TrainableModule(nn.Module):
    def __init__(self, mod: "TrainableMixin") -> None:
        super().__init__()

        self.base_mod = mod

    def forward(self, batch: Batch, state: State) -> Loss:
        return self.base_mod.get_loss(batch, state)


def get_step(step_kind: StepKind, state: State) -> int:
    match step_kind:
        case "step":
            return state.num_steps
        case "sample":
            return state.num_samples
        case "second":
            return int(state.elapsed_time_s)
        case _:
            raise ValueError(f"Invalid step kind {step_kind}")


class BatchIterator:
    def __init__(
        self,
        base_iter: Iterator[Batch],
        batch_dim: int,
        batch_chunks_schedule: list[int] | None,
        batches_per_step_schedule: list[int] | None,
        batches_per_step: int,
        step_kind: StepKind,
    ) -> None:
        self.batch_dim = batch_dim
        self.batch_chunks_schedule = batch_chunks_schedule
        self.batches_per_step_schedule = batches_per_step_schedule
        self.batches_per_step = batches_per_step
        self.step_kind = step_kind

        self._base_iter = base_iter

    def _get_batch_chunks(self, state: State) -> int:
        if (schedule := self.batch_chunks_schedule) is None:
            return 1
        step = get_step(self.step_kind, state)
        i = bisect.bisect_left(schedule, step + 1)
        return len(schedule) - i + 1

    def _get_batches_per_step(self, state: State) -> int:
        if (schedule := self.batches_per_step_schedule) is None:
            return self.batches_per_step
        step = get_step(self.step_kind, state)
        i = bisect.bisect_left(schedule, step)
        return self.batches_per_step * i

    def _chunked_batch_iterator(self, base_iter: Iterator[Batch], state: State) -> Iterator[Batch]:
        for batch in base_iter:
            num_chunks = self._get_batch_chunks(state)
            yield from recursive_chunk(batch, num_chunks, dim=self.batch_dim)

    def next(self, state: State) -> Iterator[tuple[Batch, bool]]:
        batches_per_step = self._get_batches_per_step(state)
        batch_iter = self._chunked_batch_iterator(self._base_iter, state)
        yield next(batch_iter), batches_per_step == 1
        for i in range(1, batches_per_step):
            try:
                yield next(batch_iter), i == batches_per_step - 1
            except StopIteration:
                pass


class TrainableMixin(
    ParallelMixin[Config],
    OptimizerMixin[Config],
    ProfilerMixin[Config],
    BaseTask[Config],
    Generic[Config],
):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        # This flag can be toggled to end training from anywhere in the task.
        self.__training_over_flag = False

        self.__last_printed_remaining_time = 0.0

        # The kind of step that was specified in the config.
        self.__step_kind = cast_step_kind(self.config.step_kind)

        self.last_valid_time: float | None = None
        self.last_valid_step: int | None = None
        self.first_valid_step_flag = True

        # Timers for iterations.
        self.state_timers: dict[Phase, StateTimer] = {phase: StateTimer() for phase in get_args(Phase)}

    def on_step_end(self, state: State, loss_dict: dict[str, Tensor]) -> None:
        super().on_step_end(state, loss_dict)
        state.elapsed_time_s = time.time() - state.start_time_s

    def set_training_over(self) -> None:
        self.__training_over_flag = True

    def is_training_over(self, state: State) -> bool:
        if self.__training_over_flag:
            return True
        remaining_percent = self.get_remaining_percent(state)
        if remaining_percent is None:
            return False
        self.log_scalar("percent", remaining_percent, namespace="⏰ remaining")
        self.maybe_log_termination_time(remaining_percent, state)
        return remaining_percent <= 0.0

    def batches_per_step_schedule(self) -> list[int] | None:
        schedule = self.config.batches_per_step_schedule
        if schedule is None:
            return None
        if any(s < 1 for s in schedule):
            raise ValueError("Batch chunk schedule must be positive")
        return list(itertools.accumulate([0] + schedule))

    def batch_chunks_schedule(self) -> list[int] | None:
        schedule = self.config.batch_chunks_per_step_schedule
        if schedule is None:
            return None
        if any(s < 1 for s in schedule):
            raise ValueError("Batch chunk schedule must be positive")
        return list(itertools.accumulate([0] + schedule))

    def is_valid_step(self, state: State) -> bool:
        if self.config.disable_validation:
            return False

        if self.last_valid_time is None or self.last_valid_step is None:
            self.last_valid_time = state.elapsed_time_s
            self.last_valid_step = state.num_steps
            return self.config.valid_on_first_step

        # Step-based validation.
        valid_every_n_steps = self.config.valid_every_n_steps
        if valid_every_n_steps is not None and state.num_steps > valid_every_n_steps + self.last_valid_step:
            self.last_valid_step = state.num_steps
            return True

        # Time-based validation.
        valid_every_n_seconds = self.config.valid_every_n_seconds
        if valid_every_n_seconds is not None and state.elapsed_time_s - self.last_valid_time >= valid_every_n_seconds:
            self.last_valid_time = state.elapsed_time_s
            return True

        # Time-based validation for first validation step.
        if self.first_valid_step_flag:
            valid_first_n_seconds = self.config.valid_first_n_seconds
            if valid_first_n_seconds is not None and state.elapsed_time_s >= valid_first_n_seconds:
                self.last_valid_time = state.elapsed_time_s
                self.first_valid_step_flag = False
                return True

        return False

    def log_state(self) -> None:
        self.logger.log_task_info(self.task_name, self.task_path)
        self.logger.log_git_state(get_git_state(self))
        self.logger.log_training_code(get_training_code(self))
        self.logger.log_config(cast(DictConfig, self.config))

    def get_trainable_module(self, mod: "TrainableMixin") -> TrainableModule:
        return TrainableModule(mod)

    def get_size_of_batch(self, batch: Batch) -> int | None:
        """Gets the batch size for the current batch.

        Args:
            batch: The current minibatch of samples.

        Returns:
            The parsed batch size, or None if the batch size could not be
            determined.
        """
        if isinstance(batch, (np.ndarray, Tensor)):
            return batch.shape[self.config.batch_dim]
        if is_dataclass(batch):
            for v in batch.__dict__.values():
                if bsz := self.get_size_of_batch(v):
                    return bsz
        if isinstance(batch, Mapping):
            for v in batch.values():
                if bsz := self.get_size_of_batch(v):
                    return bsz
        if isinstance(batch, Sequence):
            for i in batch:
                if bsz := self.get_size_of_batch(i):
                    return bsz
        return None

    def maybe_log_termination_time(self, remaining_percent: float, state: State) -> None:
        if self.__last_printed_remaining_time + PRINT_FINISH_TIME_EVERY_N_SECONDS > state.elapsed_time_s:
            return
        self.__last_printed_remaining_time = state.elapsed_time_s
        remaining_seconds = remaining_percent * state.elapsed_time_s / (1 - remaining_percent)
        termination_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remaining_seconds))
        logger.info("Estimated finish time: %s", termination_time)

    def log_train_step(self, batch: Batch, output: Output, state: State) -> None:
        """Override this function to do logging during the training phase.

        This function is called after the model forward pass and before the
        backward pass. It is called in the training phase.

        Args:
            batch: The batch from the dataloader.
            output: The model output.
            state: The current training state.
        """

    def log_valid_step(self, batch: Batch, output: Output, state: State) -> None:
        """Override this function to do logging during the validation phase.

        This function is called after the model forward pass. It is called in
        the validation phase.

        Args:
            batch: The batch from the dataloader.
            output: The model output.
            state: The current training state.
        """

    def log_step(self, batch: Batch, output: Output, state: State) -> None:
        with torch.no_grad():
            match state.phase:
                case "train":
                    self.log_train_step(batch, output, state)
                case "valid":
                    self.log_valid_step(batch, output, state)
                case _:
                    raise KeyError(f"Unknown phase: {state.phase}")

    def get_loss(self, batch: Batch, state: State) -> Loss:
        """Gets the loss for the current batch.

        By default, we assume the model's forward function takes the batch as
        input and returns the loss. We do some logging with the output, and
        return it as the loss. This function can be patched to do more complex
        operations instead.

        Args:
            batch: The current minibatch of samples.
            state: The current training state.

        Returns:
            The computed loss or losses, either a tensor or dictionary of
            tensors. The dictionary keys are used when logging the losses.
        """
        output = self(batch)
        self.log_step(batch, output, state)
        return output

    def get_remaining_percent(self, state: State) -> float | None:
        if self.config.max_steps is None:
            return None
        return (self.config.max_steps - get_step(self.__step_kind, state)) / self.config.max_steps

    def batch_iterator(self, pf: Prefetcher[Batch, Batch]) -> BatchIterator:
        return BatchIterator(
            pf,
            self.config.batch_dim,
            self.batch_chunks_schedule(),
            self.batches_per_step_schedule(),
            self.config.batches_per_step,
            self.__step_kind,
        )

    def get_single_loss(self, loss: Tensor | dict[str, Tensor]) -> tuple[Tensor, list[str]]:
        if isinstance(loss, Tensor):
            if loss.ndim == 0:
                return loss.unsqueeze(0), ["loss"]
            if loss.ndim == 1:
                return loss, ["loss"]
            return loss.sum().unsqueeze(0) / loss.size(0), ["loss"]
        assert isinstance(loss, dict), f"Single loss should be a scalar or dictionary, not {type(loss)}"
        keys, values = (list(i) for i in zip(*sorted(loss.items())))
        losses = [v.sum() / v.size(0) if v.ndim > 0 else v for v in values]
        single_loss = torch.stack(losses, dim=0)
        return single_loss, keys

    def log_loss_dict(self, loss: Mapping[str, int | float | Tensor], state: State) -> None:
        for k, v in loss.items():
            self.log_scalar(k, v, namespace="loss")
        timer = self.state_timers[state.phase]
        timer.step(state)
        for ns, d in timer.log_dict().items():
            for k, v in d.items():
                self.log_scalar(k, v, namespace=ns)

    def train_step(
        self,
        mod: nn.Module,
        opt: Optimizer,
        batches: Iterator[tuple[Batch, bool]],
        state: State,
    ) -> dict[str, Tensor]:
        with self.step_context("change_mode"):
            state.set_phase(self, "train")
        total_bsz: int | None = None
        losses: dict[str, tuple[Tensor, int]] = {}
        with self.step_context("zero_grads"):
            self.zero_optimizer(opt)
        num_steps = 0
        with self.autocast_context:
            for batch, is_last in batches:
                with self.get_grad_sync_context(mod, is_last):
                    bsz = self.get_size_of_batch(batch)
                    if bsz is not None:
                        total_bsz = bsz if total_bsz is None else total_bsz + bsz
                    with self.step_context("forward"):
                        loss = mod(batch, state)
                    with self.step_context("get_single_loss"):
                        single_loss, loss_names = self.get_single_loss(loss)
                    with self.step_context("backward"):
                        self.backward_grads(single_loss)
                    with self.step_context("log_losses"):
                        single_loss_detached = single_loss.detach()
                        for i, name in enumerate(loss_names):
                            new_loss = single_loss_detached[i]
                            if name in losses:
                                old_loss, count = losses[name]
                                losses[name] = (old_loss + new_loss, count + 1)
                            else:
                                losses[name] = (new_loss, 1)
                    num_steps += 1
        with self.step_context("log_losses"):
            self.log_mp_scale()
            loss_dict = {k: value / count for k, (value, count) in losses.items()}
            self.log_loss_dict(loss_dict, state)
        with self.step_context("step"):
            self.step_optimizer(mod, opt, num_steps)
        with self.step_context("write_logs"), self.autocast_context:
            self.write_logs(state)
        with self.step_context("update_state"):
            state.num_steps += 1
            if total_bsz is not None:
                state.num_samples += total_bsz
        return loss_dict

    @torch.no_grad()
    def val_step(self, mod: nn.Module, batches: Iterator[tuple[Batch, bool]], state: State) -> None:
        with self.step_context("change_mode"):
            state.set_phase(self, "valid")
        losses: dict[str, tuple[Tensor, int]] = {}
        with self.autocast_context:
            for batch, _ in batches:
                with self.step_context("forward"):
                    loss = mod(batch, state)
                with self.step_context("get_single_loss"):
                    single_loss, loss_names = self.get_single_loss(loss)
                with self.step_context("log_losses"):
                    single_loss_detached = single_loss.detach()
                    for i, name in enumerate(loss_names):
                        new_loss = single_loss_detached[i]
                        if name in losses:
                            old_loss, count = losses[name]
                            losses[name] = (old_loss + new_loss, count + 1)
                        else:
                            losses[name] = (new_loss, 1)
        with self.step_context("log_losses"):
            self.log_mp_scale()
            loss_dict = {k: value / count for k, (value, count) in losses.items()}
            self.log_loss_dict(loss_dict, state)
        with self.step_context("write_logs"), self.autocast_context:
            self.write_logs(state)
        with self.step_context("update_state"):
            state.num_valid_steps += 1
