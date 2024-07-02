"""Tests model parallelism primitives."""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from dpshdl.dataset import Dataset
from torch import Tensor, nn
from torch.distributed._tensor import (
    DeviceMesh,
    Replicate,
    Shard,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

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


class DummyParallelModule(mlfab.ParallelModule):
    def __init__(self) -> None:
        super().__init__()

        # A simple embedding layer plus two-layer MLP.
        self.emb = nn.Embedding(10, 12)
        self.l1 = nn.Linear(12, 16, bias=False)
        self.l2 = nn.Linear(16, 8, bias=False)

    def parallelize(self, mesh: DeviceMesh) -> None:
        return parallelize_module(
            module=self,
            device_mesh=mesh,
            parallelize_plan={
                "emb": RowwiseParallel(
                    input_layouts=Replicate(),
                    output_layouts=Shard(1),
                ),
                "l1": SequenceParallel(),
                "l2": ColwiseParallel(
                    input_layouts=Shard(1),
                ),
            },
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.l2(self.l1(self.emb(x)))


class DummyTask(mlfab.Task[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.parallel_module = DummyParallelModule()

    def forward(self, x: Tensor) -> Tensor:
        return self.parallel_module(x)

    def get_loss(self, batch: Tensor, state: mlfab.State) -> Tensor:
        o = self(batch).sum()
        return o

    def get_dataset(self, phase: mlfab.Phase) -> DummyDataset:
        return DummyDataset()


def _test_common(tmpdir: Path, use_ddp: bool, model_parallelism: int) -> None:
    os.environ["RUN_DIR"] = str(tmpdir)
    os.environ["TENSORBOARD_PORT"] = "-1"
    os.environ["TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    os.environ["USE_METAL"] = "0"

    mlfab.configure_logging()

    config = Config(
        use_ddp=use_ddp,
        model_parallelism=model_parallelism,
        batch_size=2,
        num_train_dl_workers=0,
        max_steps=10,
    )

    # Launches the first task with multiple data parallel workers.
    DummyTask.launch(config, launcher=mlfab.MultiProcessLauncher(num_processes=model_parallelism * 2), use_cli=False)

    raise NotImplementedError

    exp_dir = tmpdir / "dummy_task" / "run_0"
    assert exp_dir.exists()

    # Run from the same experiment directory.
    config.exp_dir = str(exp_dir)
    config.max_steps = 20

    # Launches the second task with a single data parallel worker per model
    # parallel worker.
    DummyTask.launch(config, launcher=mlfab.MultiProcessLauncher(num_processes=model_parallelism), use_cli=False)

    # Run from the same experiment directory.
    config.max_steps = 30


@pytest.mark.slow
def test_e2e_parallel_training_ddp(tmpdir: Path) -> None:
    _test_common(tmpdir, True, 1)


@pytest.mark.slow
def test_e2e_parallel_training_fsdp(tmpdir: Path) -> None:
    _test_common(tmpdir, False, 2)


if __name__ == "__main__":
    # python -m tests.e2e.test_parallel_task_e2e
    # test_e2e_parallel_training_ddp(Path(tempfile.mkdtemp()))
    test_e2e_parallel_training_fsdp(Path(tempfile.mkdtemp()))
