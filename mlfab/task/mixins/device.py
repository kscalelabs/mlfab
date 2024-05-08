"""Defines a mixin for abstracting the PyTorch tensor device."""

import functools
import logging
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch

from mlfab.core.conf import Device as BaseDeviceConfig, field, parse_dtype
from mlfab.nn.device.auto import detect_device
from mlfab.nn.device.base import DeviceManager
from mlfab.task.base import BaseConfig, BaseTask
from mlfab.utils.logging import LOG_PING

logger = logging.getLogger(__name__)


@dataclass
class DeviceConfig(BaseConfig):
    device: BaseDeviceConfig = field(BaseDeviceConfig(), help="Device configuration")


Config = TypeVar("Config", bound=DeviceConfig)


class DeviceMixin(BaseTask[Config], Generic[Config]):
    @functools.cached_property
    def device_manager(self) -> DeviceManager:
        dtype = parse_dtype(self.config.device)
        dm = DeviceManager(detect_device(), dtype=dtype)
        logger.log(LOG_PING, f"Using device: {dm}")
        return dm

    @functools.cached_property
    def torch_device(self) -> torch.device:
        return self.device_manager.device

    @functools.cached_property
    def torch_dtype(self) -> torch.dtype:
        return self.device_manager.dtype
