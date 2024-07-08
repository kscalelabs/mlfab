"""Defines neural network utility functions and classes."""

from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Type, TypeVar

from torch import nn

T = TypeVar("T")


class ResetParametersMeta(ABCMeta):
    def __call__(cls: Type[T], *args: Any, **kwargs: Any) -> T:  # noqa: ANN401
        instance = super().__call__(*args, **kwargs)  # type: ignore[misc]
        if isinstance(instance, ResetParameters):
            instance.reset_parameters()
        return instance


class ResetParameters(nn.Module, ABC, metaclass=ResetParametersMeta):
    """Defines a module that can reset its parameters.

    This is useful when you want to initialize the parameters of a module
    after the module is created. For example, you might start by initializing
    the module with the meta device, moving the parameters to the GPU, then
    resetting the parameters on the GPU.
    """

    @abstractmethod
    def reset_parameters(self) -> None: ...
