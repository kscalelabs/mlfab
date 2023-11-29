"""Defines a launcher to train a model locally, in multiple processes."""

import functools
from typing import TYPE_CHECKING

import torch

from mlfab.nn.parallel import MultiprocessConfig, launch_subprocesses
from mlfab.task.base import RawConfigType
from mlfab.task.launchers.base import BaseLauncher
from mlfab.task.launchers.single_process import run_single_process_training

if TYPE_CHECKING:
    from mlfab.task.mixins.train import TrainMixin


def get_num_processes() -> int:
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1


class MultiProcessLauncher(BaseLauncher):
    """Defines a launcher to train models locally, in multiple processes.

    Parameters:
        num_processes: The number of local training processes. If not specified,
            will use sensible defaults based on the hardware environment.
    """

    def __init__(self, num_processes: int | None) -> None:
        super().__init__()

        self.num_processes = get_num_processes() if num_processes is None else num_processes

    def launch(self, task: "type[TrainMixin]", *cfgs: RawConfigType) -> None:
        cfg = MultiprocessConfig(world_size=self.num_processes)
        train_fn = functools.partial(run_single_process_training, task, *cfgs)
        launch_subprocesses(train_fn, cfg)
