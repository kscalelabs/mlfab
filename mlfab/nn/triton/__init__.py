# mypy: disable-error-code="import"
"""Defines utility function for Triton."""

import functools

import torch

from mlfab.core.conf import load_user_config
from mlfab.nn.device.gpu import gpu_device
from mlfab.utils.text import show_warning


@functools.lru_cache(maxsize=None)
def supports_triton() -> bool:
    config = load_user_config().triton
    if not config.use_triton_if_available:
        return False

    if not gpu_device.has_device():
        return False

    try:
        import triton

        assert triton is not None
        return True
    except (ImportError, ModuleNotFoundError):
        if gpu_device.has_device():
            show_warning("Triton is not installed, but CUDA is available; install with `pip install triton`")
        return False
