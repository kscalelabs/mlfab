"""Defines a launcher to train a model locally, in multiple processes."""

import functools
from typing import TYPE_CHECKING

import torch

from mlfab.nn.device.gpu import gpu_device
from mlfab.nn.parallel import MultiProcessConfig, get_rank, get_world_size, launch_subprocesses
from mlfab.task.base import RawConfigType
from mlfab.task.launchers.base import BaseLauncher
from mlfab.utils.logging import configure_logging

if TYPE_CHECKING:
    from mlfab.task.mixins.runnable import Config, RunnableMixin


def run_training_worker(
    task: "type[RunnableMixin[Config]]",
    *cfgs: RawConfigType,
    use_cli: bool | list[str] = True,
) -> None:
    configure_logging(rank=get_rank(), world_size=get_world_size())
    task_obj = task.get_task(*cfgs, use_cli=use_cli)
    task_obj.run()


def get_num_processes() -> int:
    if gpu_device.has_device():
        return torch.cuda.device_count()
    return 1


class MultiProcessLauncher(BaseLauncher):
    """Defines a launcher to train models locally, in multiple processes.

    Parameters:
        num_processes: The number of local training processes. If not specified,
            will use sensible defaults based on the hardware environment.
    """

    def __init__(self, num_processes: int | None = None) -> None:
        super().__init__()

        self.num_processes = get_num_processes() if num_processes is None else num_processes

    def launch(
        self,
        task: "type[RunnableMixin[Config]]",
        *cfgs: RawConfigType,
        use_cli: bool | list[str] = True,
    ) -> None:
        cfg = MultiProcessConfig(world_size=self.num_processes)
        train_fn = functools.partial(run_training_worker, task, *cfgs, use_cli=use_cli)
        launch_subprocesses(train_fn, cfg)
