"""Defines a launcher to train a model locally, in a single process."""

from typing import TYPE_CHECKING

from mlfab.task.base import RawConfigType
from mlfab.task.launchers.base import BaseLauncher
from mlfab.utils.logging import configure_logging

if TYPE_CHECKING:
    from mlfab.task.mixins.runnable import Config, RunnableMixin


def run_single_process_training(
    task: "type[RunnableMixin[Config]]",
    *cfgs: RawConfigType,
    use_cli: bool = True,
) -> None:
    configure_logging()
    task_obj = task.get_task(*cfgs, use_cli=use_cli)
    task_obj.run()


class SingleProcessLauncher(BaseLauncher):
    def launch(self, task: "type[RunnableMixin[Config]]", *cfgs: RawConfigType, use_cli: bool = True) -> None:
        run_single_process_training(task, *cfgs, use_cli=use_cli)
