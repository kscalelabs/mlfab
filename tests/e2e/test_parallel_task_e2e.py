"""Tests model parallelism primitives."""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from dpshdl.dataset import Dataset
from omegaconf import MISSING
from torch import Tensor

import mlfab


@dataclass(kw_only=True)
class Config(mlfab.Config):
    use_ddp: bool = mlfab.field(True)
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

    def forward(self, x: Tensor) -> Tensor:
        return self.l2(self.l1(self.emb(x)))

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

    config = Config(
        pipeline_parallelism=1,
        model_parallelism=2,
        batch_size=2,
        num_train_dl_workers=0,
        max_steps=10,
    )

    DummyTask.launch(config, launcher=mlfab.MultiProcessLauncher(num_processes=4), use_cli=False)

    exp_dir = tmpdir / "dummy_task" / "run_0"
    assert exp_dir.exists()

    # Make sure a checkpoint was saved.
    for i in range(2):
        assert (ckpt_path := (exp_dir / "checkpoints" / f"ckpt_{i}.pt")).exists()
        ckpt = torch.load(ckpt_path)

        # Checks that the model was saved correctly.
        assert ckpt["weights"]["emb.weight"].shape == (10, 6)
        assert ckpt["weights"]["l1.weight"].shape == (8, 12)
        assert ckpt["weights"]["l2.weight"].shape == (8, 8)

    # Run from the same experiment directory.
    config.exp_dir = str(exp_dir)
    config.max_steps = 20

    DummyTask.launch(config, launcher=mlfab.MultiProcessLauncher(num_processes=2), use_cli=False)


if __name__ == "__main__":
    # python -m tests.e2e.test_parallel_task_e2e
    test_e2e_parallel_training_mp(Path(tempfile.mkdtemp()))
