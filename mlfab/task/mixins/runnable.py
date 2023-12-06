"""Defines a mixin which provides a "run" method."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar

from mlfab.task.base import BaseConfig, BaseTask, RawConfigType
from mlfab.task.launchers.base import BaseLauncher
from mlfab.task.launchers.multi_process import MultiProcessLauncher


@dataclass
class RunnableConfig(BaseConfig):
    pass


Config = TypeVar("Config", bound=RunnableConfig)


class RunnableMixin(BaseTask[Config], ABC):
    @abstractmethod
    def run(self) -> None:
        """Runs the task."""

    @classmethod
    def launch(cls, *cfgs: RawConfigType, launcher: BaseLauncher | None = None, use_cli: bool = True) -> None:
        if launcher is None:
            launcher = MultiProcessLauncher()
        launcher.launch(cls, *cfgs, use_cli=use_cli)
