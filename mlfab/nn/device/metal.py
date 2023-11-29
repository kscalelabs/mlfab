"""MPS device support for Metal GPUs (i.e., Apple Silicon)."""

import os
from typing import Callable

import torch

from mlfab.core.conf import load_user_config
from mlfab.nn.device.base import base_device


def get_env_bool(key: str) -> bool:
    val = int(os.environ.get(key, 0))
    assert val in (0, 1), f"Invalid value for {key}: {val}"
    return val == 1


class metal_device(base_device):  # noqa: N801
    """Mixin to support Metal training."""

    @classmethod
    def has_device(cls) -> bool:
        return load_user_config().device.metal and torch.backends.mps.is_available()

    def _get_device(self) -> torch.device:
        return torch.device("mps", 0)

    def _get_floating_point_type(self) -> torch.dtype:
        return torch.float32

    def get_torch_compile_backend(self) -> str | Callable:
        return "aot_ts"
