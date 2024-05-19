# mypy: disable-error-code="no-untyped-def"
"""Defines a mixin for pre-trained models.

Usually, when adding pre-trained models to a task, you just create a PyTorch
module with the pre-trained model's weights. The problem with this approach is
that it treats the pre-trained model as a "submodule" of the task, meaning that
it will add the pre-trained model's parameters to the state dictionary, which
is usually a waste of space.

Instead, this interface lets you define a pre-trained model as a separate
module-like object that will be moved to the device, without storing the
model parameters in the task state checkpoint or doing train-eval mode changes.
"""

from abc import ABC
from dataclasses import dataclass
from typing import Any, Callable, Generic, Self, TypeVar

from torch import Tensor, nn
from torch.nn.modules.module import Module

from mlfab.task.base import BaseConfig, BaseTask


@dataclass
class PretrainedConfig(BaseConfig):
    pass


Config = TypeVar("Config", bound=PretrainedConfig)


class PretrainedModule:
    def __init__(self, module: nn.Module) -> None:
        super().__init__()

        self.module = module

    def __getattr__(self, name: str) -> Tensor | Module:
        return self.module.__getattr__(name)

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> Self:
        self.module._apply(fn, recurse)
        return self

    def forward(self, *args, **kwargs) -> Any:  # noqa: ANN401, ANN002, ANN003
        return self.module(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:  # noqa: ANN401, ANN002, ANN003
        return self.module.__call__(*args, **kwargs)


def pretrained(module: nn.Module) -> PretrainedModule:
    return PretrainedModule(module)


class PretrainedMixin(BaseTask[Config], Generic[Config], ABC):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self._pretrained_modules: list[PretrainedModule] = []

    def __setattr__(self, name: str, value: Tensor | Module) -> None:
        super().__setattr__(name, value)

        if isinstance(value, PretrainedModule):
            self._pretrained_modules.append(value)

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> Self:
        mod = super()._apply(fn, recurse)
        for pretrained_i in mod._pretrained_modules:
            pretrained_i._apply(fn, recurse)
        return mod
