"""Tests that the slurm launcher can write a config file.

This test doesn't actually launch a job because that wouldn't be possible on
CI, just runs through the process of staging a directory and writing an
sbatch file.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from dpshdl.dataset import Dataset
from torch import Tensor, nn

import mlfab


@dataclass
class Config(mlfab.Config):
    num_layers: int = mlfab.field(2, help="Number of layers to use")


class DummyDataset(Dataset[tuple[np.ndarray, np.ndarray], tuple[Tensor, Tensor]]):
    def next(self) -> tuple[np.ndarray, np.ndarray]:
        return np.random.randn(3, 8), np.random.randint(0, 9, (3,))

    def collate(self, items: list[tuple[np.ndarray, np.ndarray]]) -> tuple[Tensor, Tensor]:
        return mlfab.collate(items)


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
def test_slurm_launcher(tmpdir: Path) -> None:
    os.environ["RUN_DIR"] = str(tmpdir)
    os.environ["DEFAULT_SLURM_KEY"] = "test"

    (stage_dir := Path(tmpdir / "staging")).mkdir()
    os.environ["STAGE_DIR"] = str(stage_dir)

    launcher = mlfab.SlurmLauncher(
        partition="test",
        gpus_per_node=1,
        cpus_per_gpu=1,
    )

    task = DummyTask.get_task(Config(batch_size=16), use_cli=False)

    contents = launcher.sbatch_file_contents(task)
    match = re.search(r"python -m .+", contents)
    assert match is not None
