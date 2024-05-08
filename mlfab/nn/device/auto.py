"""Defines a utility function for detecting the training device."""

import logging

from mlfab.nn.device.base import base_device
from mlfab.nn.device.cpu import cpu_device
from mlfab.nn.device.gpu import gpu_device
from mlfab.nn.device.metal import metal_device

logger: logging.Logger = logging.getLogger(__name__)

# Earlier devices in list will take precedence.
ALL_DEVICE_TYPES: list[type[base_device]] = [
    metal_device,
    gpu_device,
    cpu_device,
]


def detect_device() -> base_device:
    for device_type in ALL_DEVICE_TYPES:
        if device_type.has_device():
            return device_type()
    raise RuntimeError("Could not automatically detect the device to use")
