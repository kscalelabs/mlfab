"""Defines a launcher to train a model locally, in a single process."""

from typing import TYPE_CHECKING

from mlfab.task.base import RawConfigType
from mlfab.task.launchers.base import BaseLauncher
from mlfab.utils.logging import configure_logging

if TYPE_CHECKING:
    from mlfab.task.mixins.train import TrainMixin


def run_single_process_training(task: "type[TrainMixin]", *cfgs: RawConfigType) -> None:
    configure_logging()
    task_obj = task.get_task(*cfgs)
    task_obj.run_training_loop()


class SingleProcessLauncher(BaseLauncher):
    def launch(self, task: "type[TrainMixin]", *cfgs: RawConfigType) -> None:
        run_single_process_training(task, *cfgs)
