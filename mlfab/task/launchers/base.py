"""Defines the base launcher class."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from mlfab.task.base import RawConfigType

if TYPE_CHECKING:
    from mlfab.task.mixins.train import TrainMixin


class BaseLauncher(ABC):
    """Defines the base launcher class."""

    @abstractmethod
    def launch(self, task: "type[TrainMixin]", *cfgs: RawConfigType) -> None:
        """Launches the training process.

        Args:
            task: The task class to train
            cfgs: The raw configuration to use for training
        """
