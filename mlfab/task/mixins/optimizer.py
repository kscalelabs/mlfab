"""Defines a mixin which supports an optimizer and learning rate scheduler."""

from abc import ABC
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from omegaconf import MISSING
from torch import nn
from torch.optim.optimizer import Optimizer

from mlfab.core.conf import field
from mlfab.nn.optimizers import AdamWScheduleFree
from mlfab.task.base import BaseConfig, BaseTask

OptType = Callable[[nn.Module], Optimizer]


@dataclass(kw_only=True)
class OptimizerConfig(BaseConfig):
    set_grads_to_none: bool = field(True, help="If set, zero gradients by setting them to None")
    learning_rate: float = field(MISSING, help="Learning rate to use for optimizer")
    betas: tuple[float, float] = field(MISSING, help="Beta values for Adam optimizer")
    weight_decay: float = field(MISSING, help="Weight decay to use for the optimizer")
    warmup_steps: int = field(MISSING, help="Number of warmup steps to use for the optimizer")
    weight_lr_power: float = field(2.0, help="Power to raise the weight learning rate by")


Config = TypeVar("Config", bound=OptimizerConfig)


class OptimizerMixin(BaseTask[Config], Generic[Config], ABC):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self._optimizer: Optimizer | None = None

    @property
    def optimizer(self) -> Optimizer:
        assert self._optimizer is not None
        return self._optimizer

    def build_optimizer(self) -> OptType:
        """Gets the optimizer builder for the current model.

        If the return type is a single optimizer, then a constant learning rate
        will be used.

        Args:
            model: The model to optimize.

        Returns:
            The optimizer builder.
        """
        return AdamWScheduleFree.get(
            default_decay=True,
            separate_weight_decay_params=True,
            lr=self.config.learning_rate,
            betas=self.config.betas,
            warmup_steps=self.config.warmup_steps,
            weight_lr_power=self.config.weight_lr_power,
            weight_decay=self.config.weight_decay,
        )

    def set_optimizer(self, model: nn.Module) -> None:
        if self._optimizer is not None:
            raise RuntimeError("Optimizer has already been set!")
        optimizer_builder = self.build_optimizer()
        self._optimizer = optimizer_builder(model)

    def zero_optimizer(self) -> None:
        self.optimizer.zero_grad(set_to_none=self.config.set_grads_to_none)

    def load_task_state_dict_(
        self,
        state_dict: dict,
        strict: bool = True,
        assign: bool = False,
        weights_only: bool = False,
    ) -> None:
        if self._optimizer is None:
            return super().load_task_state_dict_(state_dict, strict, assign, weights_only)
        optimizer_state = state_dict.pop("optimizer", [])
        if not weights_only:
            self._optimizer.load_state_dict(optimizer_state)
        return super().load_task_state_dict_(state_dict, strict, assign, weights_only)

    def task_state_dict(self) -> dict:
        state_dict = super().task_state_dict()
        if self._optimizer is not None:
            state_dict.update({"optimizer": self._optimizer.state_dict()})
        return state_dict
