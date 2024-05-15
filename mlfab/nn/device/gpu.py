"""GPU device type."""

import functools
import logging
from typing import Callable

import torch

from mlfab.core.conf import load_user_config
from mlfab.nn.device.base import base_device
from mlfab.nn.parallel import get_local_rank

logger: logging.Logger = logging.getLogger(__name__)


class gpu_device(base_device):  # noqa: N801
    """Mixin to support single-GPU training."""

    @classmethod
    def has_device(cls) -> bool:
        return load_user_config().device.gpu and torch.cuda.is_available() and torch.cuda.device_count() > 0

    @functools.lru_cache(maxsize=None)
    def _get_device(self) -> torch.device:
        return torch.device("cuda", get_local_rank())

    @functools.lru_cache(maxsize=None)
    def _get_floating_point_type(self) -> torch.dtype:
        # BF16 is only supported for compute capability >= 8.0
        if torch.cuda.get_device_capability()[0] >= 8:
            return torch.bfloat16
        return torch.float16

    def get_torch_compile_backend(self) -> str | Callable:
        capability = torch.cuda.get_device_capability()
        if capability >= (7, 0):
            return "inductor"
        return "aot_ts_nvfuser"
