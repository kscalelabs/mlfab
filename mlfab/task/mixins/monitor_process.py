"""Defines a base trainer mixin for handling subprocess monitoring jobs."""

import logging
import multiprocessing as mp
from dataclasses import dataclass
from typing import Generic, TypeVar

from mlfab.core.state import State
from mlfab.task.base import BaseConfig, BaseTask

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class MonitorProcessConfig(BaseConfig):
    pass


Config = TypeVar("Config", bound=MonitorProcessConfig)


class MonitorProcessMixin(BaseTask[Config], Generic[Config]):
    """Defines a base trainer mixin for handling monitoring processes."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self._mp_manager = mp.Manager()

    def on_training_start(self, state: State) -> None:
        super().on_training_start(state)

        self._mp_manager = mp.Manager()

    def on_training_end(self, state: State) -> None:
        super().on_training_end(state)

        self._mp_manager.shutdown()
