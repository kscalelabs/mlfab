"""Defines a base trainer mixin for handling subprocess monitoring jobs."""

import logging
from dataclasses import dataclass
from multiprocessing.context import DefaultContext, ForkContext, ForkServerContext, SpawnContext
from multiprocessing.managers import SyncManager
from typing import Generic, TypeVar

import torch.multiprocessing as mp

from mlfab.core.conf import load_user_config
from mlfab.task.base import BaseConfig, BaseTask

logger: logging.Logger = logging.getLogger(__name__)

Context = DefaultContext | ForkServerContext | SpawnContext | ForkContext


@dataclass
class ProcessConfig(BaseConfig):
    pass


Config = TypeVar("Config", bound=ProcessConfig)


class ProcessMixin(BaseTask[Config], Generic[Config]):
    """Defines a base trainer mixin for handling monitoring processes."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        user_conf = load_user_config()
        mp_ctx = mp.get_context(user_conf.experiment.multiprocessing_start_method)
        if not isinstance(mp_ctx, (DefaultContext, ForkServerContext, SpawnContext, ForkContext)):
            raise RuntimeError(f"Unexpected context: {type(mp_ctx)}")
        self._mp_ctx = mp_ctx
        self._mp_manager = self._mp_ctx.Manager()

    @property
    def multiprocessing_context(self) -> Context:
        return self._mp_ctx

    @property
    def multiprocessing_manager(self) -> SyncManager:
        return self._mp_manager
