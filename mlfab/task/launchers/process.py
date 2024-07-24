"""Defines a launcher to train a model locally, in multiple processes."""

import argparse
import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mlfab.nn.parallel import get_rank, get_world_size, init_and_run, launch_subprocesses
from mlfab.task.base import RawConfigType
from mlfab.task.launchers.base import BaseLauncher
from mlfab.utils.logging import configure_logging

if TYPE_CHECKING:
    from mlfab.task.mixins.runnable import Config, RunnableMixin


def run_training_worker(task: "type[RunnableMixin[Config]]", cfg: "Config") -> None:
    configure_logging(rank=get_rank(), world_size=get_world_size())
    with torch.device("meta"):
        task_obj = task(cfg)
    task_obj.run()


class SingleProcessLauncher(BaseLauncher):
    def launch(
        self,
        task: "type[RunnableMixin[Config]]",
        *cfgs: RawConfigType,
        use_cli: bool | list[str] = True,
    ) -> None:
        cfg = task.get_config(*cfgs, use_cli=use_cli)
        train_fn = functools.partial(run_training_worker, task, cfg)
        init_and_run(train_fn)


@dataclass(kw_only=True)
class MultiProcessArgs:
    num_processes: int


class MultiProcessLauncher(BaseLauncher):
    """Defines a launcher to train models locally, in multiple processes.

    Parameters:
        num_processes: The number of local training processes. If not specified,
            will use sensible defaults based on the hardware environment.
    """

    def __init__(self, num_processes: int | None = None) -> None:
        super().__init__()

        self.num_processes = num_processes

    @classmethod
    def parse_args_from_cli(cls, args: list[str] | None = None) -> tuple[MultiProcessArgs, list[str]]:
        parser = argparse.ArgumentParser(description="Launches a multi-process job.")
        parser.add_argument("--num-processes", type=int, default=None, help="The number of processes to use")
        args, remaining_args = parser.parse_known_intermixed_args(args=args)

        return (
            MultiProcessArgs(
                num_processes=args.num_processes,
            ),
            remaining_args,
        )

    def launch(
        self,
        task: "type[RunnableMixin[Config]]",
        *cfgs: RawConfigType,
        use_cli: bool | list[str] = True,
    ) -> None:
        cfg = task.get_config(*cfgs, use_cli=use_cli)
        if self.num_processes is not None:
            cfg.local_world_size = cfg.world_size = self.num_processes
        train_fn = functools.partial(run_training_worker, task, cfg)
        launch_subprocesses(train_fn, cfg)
