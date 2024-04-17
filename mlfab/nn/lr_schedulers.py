"""Defines the base learning rate scheduler.

Here's an example of building a custom learning rate scheduler using this API
and using it in your own workflow:

.. code-block:: python

    class ConstantLRScheduler(BaseLRScheduler):
        def get_lr_sclae(self, state: State) -> float:
            return 1.0

    opt = Adam(model.parameters(), lr=1e-3)
    sched = ConstantLRScheduler.get(opt)

    state = State.init_state()

    for sample in samples:
        opt.zero_grad()
        loss = model(sample)
        loss.backward()
        opt.step()
        sched.step(state)
        state.num_steps += 1

Note that each scheduler expects to recieve a State dataclass as input.
"""

import math
from abc import ABC, abstractmethod
from typing import Any, Sequence

from torch import nn
from torch.optim.optimizer import Optimizer

from mlfab.core.state import State


class SchedulerAdapter:
    """Defines a general-purpose learning rate scheduler adapter."""

    last_state: State | None

    def __init__(self, scheduler: "BaseLRScheduler", optimizer: Optimizer) -> None:
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.last_state = None

        for param_group in self.optimizer.param_groups:
            param_group["initial_lr"] = param_group["lr"]

        self.lr_scale = 0.0

    def state_dict(self) -> dict[str, Any]:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict)

    def step(self, state: State) -> None:
        self.last_state = state
        self.lr_scale = self.scheduler.get_lr_scale(state)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = param_group["initial_lr"] * self.lr_scale

    def parameters(self) -> Sequence[nn.Parameter]:
        return [p for param_group in self.optimizer.param_groups for p in param_group["params"]]


class BaseLRScheduler(ABC):
    """Defines the base learning rate scheduler."""

    @abstractmethod
    def get_lr_scale(self, state: State) -> float:
        """Given a state, returns the current learning rate.

        Args:
            state: The current trainer state

        Returns:
            The computed learning rate to use
        """

    def get(self, optimizer: Optimizer) -> SchedulerAdapter:
        return SchedulerAdapter(self, optimizer)


class ConstantLRScheduler(BaseLRScheduler):
    """Defines a constant learning rate scheduler.

    This scheduler is really only meant to be used as a quick stand-in for
    other schedulers when none is desired, or as an alternative method of
    sweeping the learning rate besides updating the optimizer.

    Parameters:
        factor: The constant learning rate factor.
    """

    def __init__(self, factor: float = 1.0) -> None:
        super().__init__()

        self.factor = factor

    def get_lr_scale(self, state: State) -> float:
        return self.factor


class LinearLRScheduler(BaseLRScheduler):
    """Ramps up to some value, then ramps back down.

    Parameters:
        total_steps: The total number of training steps.
        warmup_steps: The number of warmup steps.
        warmup_percent: The percent of total steps to use as warmup steps,
            if not specified.
        min_scale: The minimum learning rate scale.
        decay: Whether to decay the learning rate after warmup.
    """

    def __init__(
        self,
        total_steps: int | None = None,
        warmup_steps: int | None = None,
        warmup_percent: float = 0.01,
        min_scale: float = 1e-4,
        decay: bool = True,
    ) -> None:
        super().__init__()

        if warmup_steps is None:
            if total_steps is None:
                raise ValueError(
                    "If `total_steps` is not specified, then `warmup_steps` cannot be inferred from `warmup_percent`. "
                    "You should therefore specify the number of warmup steps explicitly."
                )
            warmup_steps = round(total_steps * warmup_percent)

        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_scale = min_scale
        self.decay = decay

    def get_lr_scale(self, state: State) -> float:
        warmup, total, min_scale = self.warmup_steps, self.total_steps, self.min_scale
        if state.num_steps < warmup:
            return state.num_steps / warmup
        if not self.decay:
            return 1.0
        if total is None:
            return 1.0
        if state.num_steps < total:
            return (1 - min_scale) * (total - state.num_steps) / (total - warmup) + min_scale
        return min_scale


class CosineLRScheduler(BaseLRScheduler):
    """Defines a cosine learning rate schedule.

    Parameters:
        total_steps: The total number of training steps.
        num_resets: The number of times to reset the learning rate.
        phase: The number of steps in a phase.
        ramp_up_percent: The percent of phase to spend ramping up.
        ramp_up_steps: The number of steps to spend ramping up.
        eta_min: The minimum learning rate scale.
        eta_max: The maximum learning rate scale.
    """

    def __init__(
        self,
        total_steps: int | None = None,
        num_resets: int = 0,
        phase: int | None = None,
        ramp_up_percent: float = 0.05,
        ramp_up_steps: int | None = None,
        eta_min: float = 0.01,
        eta_max: float = 1.0,
    ) -> None:
        super().__init__()

        if phase is None:
            if total_steps is None:
                raise ValueError("If `total_steps` is not specified, then `phase` must be specified.")
            phase = int(total_steps / (num_resets + 1))
        if ramp_up_steps is None:
            assert 0.0 <= ramp_up_percent < 1.0
            ramp_up_steps = round(phase * ramp_up_percent)
        else:
            assert ramp_up_steps < phase

        self.phase = phase
        self.ramp_up_steps = ramp_up_steps
        self.eta_min = eta_min
        self.eta_max = eta_max

    def get_lr_scale(self, state: State) -> float:
        phase, ramp_up = self.phase, self.ramp_up_steps
        eta_min, eta_max = self.eta_min, self.eta_max
        phase_steps = state.num_steps % (phase + ramp_up)
        if phase_steps < ramp_up:
            return (1.0 - eta_min) * (phase_steps / ramp_up) + eta_min
        sigma = (phase_steps - ramp_up) / phase
        return eta_min + (eta_max - eta_min) * (1 + math.cos(math.pi * sigma)) / 2


class CosineDecayLRScheduler(BaseLRScheduler):
    """Defines a cosine decay learning rate schedule.

    This is like a cosine learning rate scheduler, but it decays the learning
    rate over time. When using multiple resets, each reset will peak lower than
    the previous one.

    Parameters:
        total_steps: The total number of training steps.
        num_resets: The number of times to reset the learning rate.
        phase: The number of steps in a phase.
        ramp_up_percent: The percent of phase to spend ramping up.
        ramp_up_steps: The number of steps to spend ramping up.
        eta_min: The minimum learning rate scale.
        eta_max: The maximum learning rate scale.
        min_decay: The minimum learning rate decay.
    """

    def __init__(
        self,
        total_steps: int,
        num_resets: int = 0,
        phase: int | None = None,
        ramp_up_percent: float = 0.05,
        ramp_up_steps: int | None = None,
        eta_min: float = 0.01,
        eta_max: float = 1.0,
        min_decay: float = 1e-4,
    ) -> None:
        super().__init__()

        if phase is None:
            phase = int(total_steps / (num_resets + 1))
        if ramp_up_steps is None:
            assert 0.0 <= ramp_up_percent < 1.0
            ramp_up_steps = round(phase * ramp_up_percent)
        else:
            assert ramp_up_steps < phase

        self.phase = phase
        self.total_steps = total_steps
        self.ramp_up_steps = ramp_up_steps
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.min_decay = min_decay

    def get_lr_scale(self, state: State) -> float:
        phase, total, ramp_up = self.phase, self.total_steps, self.ramp_up_steps
        eta_min, eta_max, min_decay = self.eta_min, self.eta_max, self.min_decay
        decay = (1.0 - min_decay) * (total - state.num_steps) / total + min_decay
        phase_steps = state.num_steps % (phase + ramp_up)
        if phase_steps < ramp_up:
            return max(phase_steps / ramp_up, eta_min) * decay
        sigma = (phase_steps - ramp_up) / phase
        lr_scale_no_decay = eta_min + (eta_max - eta_min) * (1 + math.cos(math.pi * sigma)) / 2
        return lr_scale_no_decay * decay


class NoamLRScheduler(BaseLRScheduler):
    """Defines the Noam learning rate scheduler.

    This corresponds to increasing the learning rate linearly for the first
    ``warmup_steps`` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square
    root of the dimensionality of the model.

    Parameters:
        warmup_steps: The number of learning rate ramp up steps.
        model_size: The dimensionality of the model.
    """

    def __init__(self, warmup_steps: int, model_size: int = 1024) -> None:
        super().__init__()

        self.warmup_steps = warmup_steps
        self.model_size = model_size

    def get_lr_scale(self, state: State) -> float:
        return self.model_size**-0.5 * min(state.num_steps**-0.5, state.num_steps * self.warmup_steps**-1.5)
