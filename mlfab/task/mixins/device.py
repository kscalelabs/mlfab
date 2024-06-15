"""Defines a mixin for abstracting the PyTorch tensor device."""

import functools
import logging
from dataclasses import dataclass
from typing import Generic, Self, TypeVar

import torch

from mlfab.core.conf import Device as BaseDeviceConfig, field, parse_dtype
from mlfab.nn.device.auto import DeviceManager, detect_device
from mlfab.task.base import BaseConfig, BaseTask, RawConfigType
from mlfab.utils.logging import LOG_INFO_ALL

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class DeviceConfig(BaseConfig):
    device: BaseDeviceConfig = field(BaseDeviceConfig(), help="Device configuration")


Config = TypeVar("Config", bound=DeviceConfig)


class DeviceMixin(BaseTask[Config], Generic[Config]):
    @classmethod
    def get_device_manager(cls, cfg: Config | None = None) -> DeviceManager:
        dtype = None if cfg is None else parse_dtype(cfg.device)
        device_manager = DeviceManager(detect_device(), dtype=dtype)
        logger.log(LOG_INFO_ALL, f"Using device: {device_manager}")
        return device_manager

    @functools.cached_property
    def device_manager(self) -> DeviceManager:
        return self.get_device_manager(self.config)

    @functools.cached_property
    def torch_device(self) -> torch.device:
        return self.device_manager.device

    @functools.cached_property
    def torch_dtype(self) -> torch.dtype:
        return self.device_manager.dtype

    @classmethod
    def get_task(cls, *cfgs: RawConfigType, use_cli: bool | list[str] = True) -> Self:
        cfg = cls.get_config(*cfgs, use_cli=use_cli)
        with cls.get_device_manager(cfg).device:
            return cls(cfg)
