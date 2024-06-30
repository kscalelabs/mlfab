"""Tests model parallelism primitives."""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from dpshdl.dataset import Dataset
from torch import Tensor, nn

import mlfab


@dataclass(kw_only=True)
class Config(mlfab.Config):
    learning_rate: float = mlfab.field(1e-3)
    betas: tuple[float, float] = mlfab.field((0.9, 0.999))
    weight_decay: float = mlfab.field(1e-4)
    warmup_steps: int = mlfab.field(100)


class DummyDataset(Dataset[Tensor, Tensor]):
    def next(self) -> Tensor:
        return torch.randint(0, 9, (3,))

    def collate(self, items: list[Tensor]) -> Tensor:
        return mlfab.collate(items)


class DummyTask(mlfab.Task[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        # A simple embedding layer plus two-layer MLP.
        self.emb = mlfab.ParallelEmbedding(10, 12)
        self.l1 = mlfab.ColumnParallelLinear(12, 16, bias=False)
        self.l2 = mlfab.RowParallelLinear(16, 8, bias=False)
        self.l3 = nn.Linear(8, 8, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.l3(self.l2(self.l1(self.emb(x))))

    def get_loss(self, batch: Tensor, state: mlfab.State) -> Tensor:
        o = self(batch).sum()
        return o

    def get_dataset(self, phase: mlfab.Phase) -> DummyDataset:
        return DummyDataset()


@pytest.mark.slow
def test_e2e_parallel_training_mp(tmpdir: Path) -> None:
    os.environ["RUN_DIR"] = str(tmpdir)
    os.environ["TENSORBOARD_PORT"] = "-1"
    os.environ["TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    os.environ["USE_METAL"] = "0"

    mlfab.configure_logging()

    model_parallelism = 2

    config = Config(
        model_parallelism=model_parallelism,
        batch_size=2,
        num_train_dl_workers=0,
        max_steps=10,
    )

    # Launches the first task with multiple data parallel workers.
    DummyTask.launch(config, launcher=mlfab.MultiProcessLauncher(num_processes=model_parallelism * 2), use_cli=False)

    exp_dir = tmpdir / "dummy_task" / "run_0"
    assert exp_dir.exists()

    # Make sure a checkpoint was saved.
    for i in range(model_parallelism):
        assert Path(exp_dir / "checkpoints" / f"ckpt_{i}.pt").exists()

        # Checks that the model was saved correctly.
        # assert ckpt["model"]["mod.emb.weight"].shape == (10, 6)
        # assert ckpt["model"]["mod.l1.weight"].shape == (8, 12)
        # assert ckpt["model"]["mod.l2.weight"].shape == (8, 8)
        # assert ckpt["model"]["mod.l3.weight"].shape == (8, 8)

    # Run from the same experiment directory.
    config.exp_dir = str(exp_dir)
    config.max_steps = 20

    # Launches the second task with a single data parallel worker per model
    # parallel worker.
    DummyTask.launch(config, launcher=mlfab.MultiProcessLauncher(num_processes=model_parallelism), use_cli=False)


if __name__ == "__main__":
    # python -m tests.e2e.test_parallel_task_e2e
    test_e2e_parallel_training_mp(Path(tempfile.mkdtemp()))
