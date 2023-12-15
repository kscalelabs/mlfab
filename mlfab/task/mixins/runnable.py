"""Defines a mixin which provides a "run" method."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar

from mlfab.task.base import BaseConfig, BaseTask, RawConfigType
from mlfab.task.launchers.base import BaseLauncher


@dataclass
class RunnableConfig(BaseConfig):
    pass


Config = TypeVar("Config", bound=RunnableConfig)


class RunnableMixin(BaseTask[Config], ABC):
    @abstractmethod
    def run(self) -> None:
        """Runs the task."""

    @classmethod
    def launch(
        cls,
        *cfgs: RawConfigType,
        launcher: BaseLauncher | None = None,
        use_cli: bool | list[str] = True,
    ) -> None:
        if launcher is None:
            from mlfab.task.launchers.cli import CliLauncher

            launcher = CliLauncher()
        launcher.launch(cls, *cfgs, use_cli=use_cli)
