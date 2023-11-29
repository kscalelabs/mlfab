"""CPU device type."""

from typing import Callable

import torch

from mlfab.nn.device.base import base_device


class cpu_device(base_device):  # noqa: N801
    """Mixin to support CPU training."""

    @classmethod
    def has_device(cls) -> bool:
        return True

    def _get_device(self) -> torch.device:
        return torch.device("cpu")

    def _get_floating_point_type(self) -> torch.dtype:
        return torch.float32

    def get_torch_compile_backend(self) -> str | Callable:
        return "aot_ts"
