"""Defines a utility function for detecting the training device."""

import contextlib
import logging
from typing import Callable, ContextManager, TypeVar

import numpy as np
import torch
from dpshdl.dataloader import Dataloader
from dpshdl.prefetcher import Prefetcher
from torch import Tensor, nn

from mlfab.nn.device.base import base_device
from mlfab.nn.device.cpu import cpu_device
from mlfab.nn.device.gpu import gpu_device
from mlfab.nn.device.metal import metal_device
from mlfab.nn.functions import recursive_apply, recursive_from_numpy

logger: logging.Logger = logging.getLogger(__name__)

# Earlier devices in list will take precedence.
ALL_DEVICE_TYPES: list[type[base_device]] = [
    metal_device,
    gpu_device,
    cpu_device,
]

T = TypeVar("T")
Tc = TypeVar("Tc")


def detect_device() -> base_device:
    for device_type in ALL_DEVICE_TYPES:
        if device_type.has_device():
            return device_type()
    raise RuntimeError("Could not automatically detect the device to use")


def allow_nonblocking_transfer(device_a: torch.device, device_b: torch.device) -> bool:
    return device_a.type in ("cpu", "cuda") and device_b.type in ("cpu", "cuda")


class DeviceManager:
    def __init__(
        self,
        bd: base_device | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if bd is None:
            bd = detect_device()

        self.bd = bd
        self.device = bd.device if device is None else device
        self.dtype = bd.dtype if dtype is None else dtype

    def get_torch_compile_backend(self) -> str | Callable:
        return self.bd.get_torch_compile_backend()

    def sample_to_device(self, sample: Tc, pin_memory: bool | None = None) -> Tc:
        if pin_memory is None:
            pin_memory = self.device.type == "cuda"
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
