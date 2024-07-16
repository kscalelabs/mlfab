# mypy: disable-error-code="no-untyped-def"
"""Defines a mixin for pre-trained models."""

from abc import ABC
from dataclasses import dataclass
from typing import Any, Callable, Generic, Self, TypeVar, cast

from torch import Tensor, nn
from torch.nn.modules.module import Module

from mlfab.task.base import BaseConfig, BaseTask


@dataclass(kw_only=True)
class PretrainedConfig(BaseConfig):
    pass


Config = TypeVar("Config", bound=PretrainedConfig)

Tmod = TypeVar("Tmod", bound=nn.Module)


class PretrainedModule:
    def __init__(
        self,
        module: nn.Module,
        load_fn: Callable[[nn.Module], None],
        use_device_dtype: bool = False,
    ) -> None:
        super().__init__()

        self.module = module
        self.module.eval()
        self.module.requires_grad_(False)

        self.load_fn = load_fn
        self.use_device_dtype = use_device_dtype

    def load(self) -> None:
        self.load_fn(self.module)

    def __getattribute__(self, name: str) -> Any:  # noqa: ANN401
        if name.startswith("__") or name in ("module", "forward", "load_fn", "load", "use_device_dtype"):
            return super().__getattribute__(name)
        return getattr(self.module, name)

    def forward(self, *args, **kwargs) -> Any:  # noqa: ANN401, ANN002, ANN003
        return self.module.forward(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:  # noqa: ANN401, ANN002, ANN003
        return self.module.__call__(*args, **kwargs)


def pretrained(
    module: Tmod,
    load_fn: Callable[[nn.Module], None] = lambda _: None,
    use_device_dtype: bool = False,
) -> Tmod:
    """Marks a module as pre-trained (i.e., don't store it's weights).

    Usually, when adding pre-trained models to a task, you just create a
    PyTorch module with the pre-trained model's weights. The problem with this
    approach is that it treats the pre-trained model as a "submodule" of the
    task, meaning that it will add the pre-trained model's parameters to the
    state dictionary, which is usually a waste of space.

    Instead, this interface lets you define a pre-trained model as a separate
    module-like object that will be moved to the device, without storing the
    model parameters in the task state checkpoint or doing train-eval mode
    changes.

    Args:
        module: The pre-trained module.
        load_fn: A function that loads the pre-trained model weights.
        use_device_dtype: This is a flag that downstream accessors can use to
            determine if the pretrained module should have it's weights
            converted to some lower-precision format.

    Returns:
        The pre-trained module.
    """
    return cast(Tmod, PretrainedModule(module, load_fn, use_device_dtype))


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
            pretrained_i.module._apply(fn, recurse)
        return mod
