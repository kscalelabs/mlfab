"""Defines a base trainer mixin for handling subprocess monitoring jobs."""

import logging
import multiprocessing as mp
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from multiprocessing.managers import SyncManager
from typing import Generic, TypeVar

from mlfab.core.conf import load_user_config
from mlfab.core.state import State
from mlfab.task.base import BaseConfig, BaseTask

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class ProcessConfig(BaseConfig):
    pass


Config = TypeVar("Config", bound=ProcessConfig)


class ProcessMixin(BaseTask[Config], Generic[Config]):
    """Defines a base trainer mixin for handling monitoring processes."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        user_conf = load_user_config()
        self._mp_ctx = mp.get_context(user_conf.experiment.multiprocessing_start_method)
        self._mp_manager = self._mp_ctx.Manager()

    @property
    def multiprocessing_context(self) -> BaseContext:
        return self._mp_ctx

    @property
    def multiprocessing_manager(self) -> SyncManager:
        return self._mp_manager

    def on_training_start(self, state: State) -> None:
        super().on_training_start(state)

        self._mp_manager = mp.Manager()

    def on_training_end(self, state: State) -> None:
        super().on_training_end(state)

        self._mp_manager.shutdown()
        self._mp_manager.join()
