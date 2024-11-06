"""Defines a mixin for running the training loop."""

import contextlib
import datetime
import logging
import signal
import sys
import textwrap
import time
import traceback
from dataclasses import dataclass
from threading import Thread
from typing import Generic, TypeVar

from mlfab.core.conf import field
from mlfab.nn.parallel import is_master
from mlfab.task.base import BaseConfig, BaseTask
from mlfab.task.mixins.checkpointing import CheckpointingConfig, CheckpointingMixin
from mlfab.task.mixins.compile import CompileConfig, CompileMixin
from mlfab.task.mixins.cpu_stats import CPUStatsConfig, CPUStatsMixin
from mlfab.task.mixins.data_loader import DataloadersConfig, DataloadersMixin
from mlfab.task.mixins.gpu_stats import GPUStatsConfig, GPUStatsMixin
from mlfab.task.mixins.meta import MetaConfig, MetaMixin
from mlfab.task.mixins.pretrained import PretrainedConfig, PretrainedMixin
from mlfab.task.mixins.runnable import RunnableConfig, RunnableMixin
from mlfab.task.mixins.trainable import TrainableConfig, TrainableMixin
from mlfab.utils.experiments import TrainingFinishedError
from mlfab.utils.text import format_timedelta, highlight_exception_message, show_info

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class TaskConfig(
    TrainableConfig,
    CheckpointingConfig,
    CompileConfig,
    MetaConfig,
    PretrainedConfig,
    DataloadersConfig,
    RunnableConfig,
    GPUStatsConfig,
    CPUStatsConfig,
    BaseConfig,
):
    init_state_strict: bool = field(True, help="Load the initial state strictly")


Config = TypeVar("Config", bound=TaskConfig)


class TaskMixin(
    TrainableMixin[Config],
    CheckpointingMixin[Config],
    CompileMixin[Config],
    MetaMixin[Config],
    PretrainedMixin[Config],
    DataloadersMixin[Config],
    RunnableMixin[Config],
    GPUStatsMixin[Config],
    CPUStatsMixin[Config],
    BaseTask[Config],
    Generic[Config],
):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def run(self) -> None:
        self.run_training_loop()

    def run_training_loop(self) -> None:
        """Runs the training loop.

        Args:
            model: The current model
            task: The current task
            optimizer: The current optimizer
            lr_scheduler: The current learning rate scheduler

        Raises:
            ValueError: If the task is not a supervised learning task
        """
        with contextlib.ExitStack() as ctx:
            self.set_loggers()

            with self.step_context("model_to_device"):
                mod = self.get_trainable_module(self)
                self.configure_model_(mod)
                mod = self.get_wrapped_model(mod)

            with self.step_context("create_optimizers"):
                opt = self.build_optimizer(mod)

            if is_master():
                Thread(target=self.log_state, daemon=True).start()

            with self.step_context("load_checkpoint"):
                state = self.load_ckpt_(model=mod, optimizer=opt, strict=self.config.init_state_strict)

            # Gets the datasets.
            with self.step_context("get_dataset"):
                valid_ds = self.get_dataset("valid")
                train_ds = self.get_dataset("train")

            # Gets the dataloaders.
            with self.step_context("get_dataloader"):
                valid_dl = self.get_dataloader(valid_ds, "valid")
                train_dl = self.get_dataloader(train_ds, "train")

            # Gets the prefetchers.
            with self.step_context("get_prefetcher"):
                valid_pf = self.device_manager.get_prefetcher(valid_dl)
                train_pf = self.device_manager.get_prefetcher(train_dl)

                # ctx.enter_context(self)
                ctx.enter_context(valid_pf)
                ctx.enter_context(train_pf)

                valid_batch_iterator = self.batch_iterator(valid_pf)
                train_batch_iterator = self.batch_iterator(train_pf)

            with self.step_context("training_start"):
                self.on_training_start(state)

            def on_exit() -> None:
                self.save_ckpt(
                    state=state,
                    model=mod,
                    optimizer=opt,
                )

            # Handle user-defined interrupts during the training loop.
            self.add_signal_handler(on_exit, signal.SIGUSR1)

            try:
                if (profile := self.get_profile()) is not None:
                    ctx.enter_context(profile)

                while True:
                    if self.is_training_over(state):
                        raise TrainingFinishedError

                    if self.is_valid_step(state):
                        with self.step_context("valid_step"):
                            self.val_step(mod, valid_batch_iterator.next(state), state)

                    with self.step_context("on_step_start"):
                        self.on_step_start(state)

                    with self.step_context("train_step"):
                        loss_dict = self.train_step(mod, opt, train_batch_iterator.next(state), state)

                    if self.should_save_ckpt(state):
                        with self.step_context("save_checkpoint"):
                            self.save_ckpt(
                                state=state,
                                model=mod,
                                optimizer=opt,
                            )

                    if profile is not None:
                        profile.step()

                    with self.step_context("on_step_end"):
                        self.on_step_end(state, loss_dict)

            except TrainingFinishedError:
                with self.step_context("save_checkpoint"):
                    self.save_ckpt(
                        state=state,
                        model=mod,
                        optimizer=opt,
                    )
                if is_master():
                    elapsed_time = format_timedelta(datetime.timedelta(seconds=time.time() - state.start_time_s))
                    show_info(
                        f"Finished training after {state.num_steps} steps, {state.num_samples} samples, {elapsed_time}",
                        important=True,
                    )

            except BaseException:
                exception_tb = textwrap.indent(highlight_exception_message(traceback.format_exc()), "  ")
                sys.stdout.write(f"Caught exception during training loop:\n\n{exception_tb}\n")
                sys.stdout.flush()

            finally:
                self.on_training_end(state)
