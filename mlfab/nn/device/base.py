"""Utility functions for abstracting away the tranining device."""

import functools
from abc import ABC, abstractmethod
from typing import Callable

import torch

from mlfab.core.conf import load_user_config, parse_dtype


class base_device(ABC):  # noqa: N801
    """The base ."""

    def __str__(self) -> str:
        return f"device({self.device.type}, {self.device.index}, {self.dtype})"

    def __repr__(self) -> str:
        return str(self)

    @functools.cached_property
    def device(self) -> torch.device:
        return self._get_device()

    @functools.cached_property
    def dtype(self) -> torch.dtype:
        return self._get_floating_point_type_with_override()

    @classmethod
    @abstractmethod
    def has_device(cls) -> bool:
        """Detects whether or not the device is available.

        Returns:
            If the device is available
        """

    @abstractmethod
    def _get_device(self) -> torch.device:
        """Returns the device, for instantiating new tensors.

        Returns:
            The device
        """

    @abstractmethod
    def _get_floating_point_type(self) -> torch.dtype:
        """Returns the default floating point type to use.

        Returns:
            The dtype
        """

    @abstractmethod
    def get_torch_compile_backend(self) -> str | Callable:
        """Returns the backend to use for Torch compile.

        Returns:
            The backend
        """

    def _get_floating_point_type_with_override(self) -> torch.dtype:
        if (dtype := parse_dtype(load_user_config().device)) is not None:
            return dtype
        return self._get_floating_point_type()
