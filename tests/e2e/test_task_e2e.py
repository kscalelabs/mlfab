"""Runs end-to-end tests of supervised learning training.

This test is also useful for reasoning about and debugging the entire
training loop.
"""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from dpshdl.dataset import Dataset
from torch import Tensor, nn

import mlfab


@dataclass
class Config(mlfab.Config):
    num_layers: int = mlfab.field(2, help="Number of layers to use")


class DummyDataset(Dataset[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]):
    def next(self) -> tuple[Tensor, Tensor]:
        return torch.randn(3, 8), torch.randint(0, 9, (3,))

    def collate(self, items: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        return mlfab.collate_non_null(items)


class DummyTask(mlfab.Task[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.emb = nn.Embedding(10, 8)
        self.convs = nn.Sequential(*(nn.Conv1d(3, 3, 3, padding=1) for _ in range(config.num_layers)))
        self.lstm = nn.LSTM(8, 8, 2)

    def build_optimizer(self) -> mlfab.OptType:
        assert (max_steps := self.config.max_steps) is not None

        return (
            mlfab.Adam.get(self),
            mlfab.LinearLRScheduler(max_steps),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x, _ = self.lstm(x.float())
        z = x + self.emb(y)
        return self.convs(z)

    def get_loss(self, batch: tuple[Tensor, Tensor], state: mlfab.State) -> Tensor:
        o = self(*batch).sum()
        return o

    def get_dataset(self, phase: mlfab.Phase) -> DummyDataset:
        return DummyDataset()


@pytest.mark.slow
def test_e2e_training(tmpdir: Path) -> None:
    os.environ["RUN_DIR"] = str(tmpdir)
    os.environ["USE_METAL"] = "0"

    mlfab.configure_logging()

    DummyTask.launch(
        Config(
            num_layers=2,
            batch_size=2,
            train_dl=mlfab.DataloaderConfig(
                num_workers=0,
            ),
            max_steps=10,
        ),
        use_cli=False,
    )


@pytest.mark.slow
def test_e2e_training_mp(tmpdir: Path) -> None:
    os.environ["RUN_DIR"] = str(tmpdir)
    os.environ["TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    os.environ["USE_METAL"] = "0"

    mlfab.configure_logging()

    DummyTask.launch(
        Config(
            num_layers=2,
            batch_size=2,
            train_dl=mlfab.DataloaderConfig(
                num_workers=0,
            ),
            max_steps=10,
        ),
        launcher=mlfab.MultiProcessLauncher(num_processes=2),
        use_cli=False,
    )


@pytest.mark.slow
def test_staged_training(tmpdir: Path) -> None:
    os.environ["RUN_DIR"] = str(tmpdir)
    os.environ["USE_METAL"] = "0"

    mlfab.configure_logging()

    config = Config(batch_size=1)
    orig_task = DummyTask(config)
    task_key = orig_task.task_key
    task: mlfab.Task = mlfab.Task.from_task_key(task_key).get_task(config, use_cli=False)
    assert isinstance(task, DummyTask)


if __name__ == "__main__":
    # python -m tests.e2e.test_task_e2e
    # test_e2e_training(Path(tempfile.mkdtemp()))
    test_e2e_training_mp(Path(tempfile.mkdtemp()))
