"""Utility functions for abstracting away the tranining device."""

import contextlib
import functools
from abc import ABC, abstractmethod
from typing import Callable, ContextManager, TypeVar

import numpy as np
import torch
from dpshdl.dataloader import Dataloader
from dpshdl.prefetcher import Prefetcher
from torch import Tensor, nn

from mlfab.core.conf import load_user_config, parse_dtype
from mlfab.nn.functions import recursive_apply, recursive_from_numpy

T = TypeVar("T")
Tc = TypeVar("Tc")


def allow_nonblocking_transfer(device_a: torch.device, device_b: torch.device) -> bool:
    return device_a.type in ("cpu", "cuda") and device_b.type in ("cpu", "cuda")


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


class DeviceManager:
    def __init__(
        self,
        bd: base_device,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.bd = bd
        self.device = bd.device if device is None else device
        self.dtype = bd.dtype if dtype is None else dtype

    def get_torch_compile_backend(self) -> str | Callable:
        return self.bd.get_torch_compile_backend()

    def sample_to_device(self, sample: Tc, pin_memory: bool = True) -> Tc:
        return recursive_apply(
            recursive_from_numpy(sample, pin_memory=pin_memory),
            lambda t: t.to(
                self.device,
                self.dtype if t.is_floating_point() else t.dtype,
                non_blocking=allow_nonblocking_transfer(t.device, self.device),
            ),
        )

    def get_prefetcher(self, dataloader: Dataloader[T, Tc]) -> Prefetcher[Tc, Tc]:
        return Prefetcher(self.sample_to_device, dataloader)

    def module_to(self, module: nn.Module, with_dtype: bool = False) -> None:
        if with_dtype:
            module.to(self.device, self.dtype)
        else:
            module.to(self.device)

    def tensor_to(self, tensor: np.ndarray | Tensor) -> Tensor:
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        if tensor.is_floating_point():
            return tensor.to(self.device, self.dtype)
        return tensor.to(self.device)

    def autocast_context(self, enabled: bool = True) -> ContextManager:
        device_type = self.device.type
        if device_type not in ("cpu", "cuda"):
            return contextlib.nullcontext()
        if device_type == "cpu" and self.dtype != torch.bfloat16:
            return contextlib.nullcontext()
        return torch.autocast(device_type=device_type, dtype=self.dtype, enabled=enabled)

    def __str__(self) -> str:
        return f"device_manager({self.device.type}, {self.device.index}, {self.dtype})"

    def __repr__(self) -> str:
        return str(self)
