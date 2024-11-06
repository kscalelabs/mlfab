"""Composes the base task with all the mixins into a single task interface."""

from dataclasses import dataclass
from typing import Generic, TypeVar

from mlfab.task.mixins import (
    TaskConfig,
    TaskMixin,
)


@dataclass(kw_only=True)
class Config(TaskConfig):
    pass


ConfigT = TypeVar("ConfigT", bound=Config)


class Task(TaskMixin[ConfigT], Generic[ConfigT]):
    pass
