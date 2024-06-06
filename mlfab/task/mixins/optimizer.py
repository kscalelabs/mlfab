"""Defines a mixin which supports an optimizer and learning rate scheduler."""

from abc import ABC
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from torch import nn
from torch.optim.optimizer import Optimizer

from mlfab.core.conf import field
from mlfab.nn.optimizers import AdamWScheduleFree
from mlfab.task.base import BaseConfig, BaseTask

OptType = Callable[[nn.Module], Optimizer]


@dataclass
class OptimizerConfig(BaseConfig):
    set_grads_to_none: bool = field(True, help="If set, zero gradients by setting them to None")
    learning_rate: float = field(3e-4, help="Learning rate to use for optimizer")
    betas: tuple[float, float] = field((0.9, 0.999), help="Beta values for Adam optimizer")
    optimizer_warmup_steps: int = field(100, help="Number of warmup steps to use for the optimizer")
    optimizer_default_decay: bool = field(True, help="If set, decay any modules by default")
    optimizer_separate_weight_decay_params: bool = field(True, help="If set, avoid weight decaying certain modules")


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
            default_decay=self.config.optimizer_default_decay,
            separate_weight_decay_params=self.config.optimizer_separate_weight_decay_params,
            lr=self.config.learning_rate,
            betas=self.config.betas,
        )

    def set_optimizer(self, model: nn.Module) -> None:
        if self._optimizer is not None:
            raise RuntimeError("Optimizer has already been set!")
        optimizer_builder = self.build_optimizer()
        self._optimizer = optimizer_builder(model)

    def zero_optimizer(self) -> None:
        self.optimizer.zero_grad(set_to_none=self.config.set_grads_to_none)

    def load_task_state_dict(
        self,
        state_dict: dict,
        strict: bool = True,
        assign: bool = False,
        weights_only: bool = False,
    ) -> None:
        if self._optimizer is None:
            return super().load_task_state_dict(state_dict, strict, assign, weights_only)
        optimizer_state = state_dict.pop("optimizer", [])
        if not weights_only:
            self._optimizer.load_state_dict(optimizer_state)
        return super().load_task_state_dict(state_dict, strict, assign, weights_only)

    def task_state_dict(self) -> dict:
        state_dict = super().task_state_dict()
        if self._optimizer is not None:
            state_dict.update({"optimizer": self._optimizer.state_dict()})
        return state_dict
