"""Runs end-to-end tests of supervised learning training.

This test is also useful for reasoning about and debugging the entire
training loop.
"""

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from dpshdl.dataset import Dataset
from torch import Tensor, nn

import mlfab

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class Config(mlfab.Config):
    num_layers: int = mlfab.field(2)
    learning_rate: float = mlfab.field(1e-2)
    betas: tuple[float, float] = mlfab.field((0.9, 0.999))
    weight_decay: float = mlfab.field(1e-4)
    warmup_steps: int = mlfab.field(100)
    min_warn_time: float = mlfab.field(0.0)


class DummyDataset(Dataset[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]):
    def next(self) -> tuple[Tensor, Tensor]:
        return torch.randn(3, 8), torch.randint(0, 9, (3,))

    def collate(self, items: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        return mlfab.collate(items)


class DummyTask(mlfab.Task[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.proj = nn.Linear(8, 8)

        with self.torch_device:
            custom_emb = nn.Embedding(10, 8)
            custom_emb.weight.data.fill_(3.14)
            self.emb = mlfab.pretrained(custom_emb)

        self.convs = nn.Sequential(*(nn.Conv1d(3, 3, 3, padding=1) for _ in range(config.num_layers)))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x, _ = self.proj(x)
        z = x + self.emb(y)
        return self.convs(z)

    def get_loss(self, batch: tuple[Tensor, Tensor], state: mlfab.State) -> Tensor:
        return self(*batch).mean()

    def get_dataset(self, phase: mlfab.Phase) -> DummyDataset:
        return DummyDataset()


@pytest.mark.timeout(120)
@pytest.mark.slow
@pytest.mark.parametrize("tensor_parallelism", (1, 2, 4))
def test_e2e_training_mp(tmpdir: Path, tensor_parallelism: int) -> None:
    os.environ["TENSORBOARD_PORT"] = "-1"
    if "TORCH_DISTRIBUTED_BACKEND" not in os.environ:
        os.environ["TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    os.environ["USE_METAL"] = "0"

    num_processes = 4

    mlfab.configure_logging()

    config = Config(
        num_layers=2,
        batch_size=2,
        num_train_dl_workers=0,
        max_steps=5,
        tensor_parallelism=tensor_parallelism,
        run_dir=str(tmpdir),
    )

    DummyTask.launch(config, launcher=mlfab.MultiProcessLauncher(num_processes=num_processes), use_cli=False)

    # Run from the same experiment directory.
    exp_dir = Path(tmpdir) / "dummy_task" / "run_0"
    assert exp_dir.exists(), f"Directory does not exist: {exp_dir}"
    config.exp_dir = str(exp_dir)
    config.max_steps = 10

    # Attempts to convert to a single torch checkpoint.
    ckpt_path = Path(tmpdir) / "ckpt.pt"
    mlfab.convert_dcp_to_torch(exp_dir / "ckpt", ckpt_path)
    assert ckpt_path.exists(), f"Checkpoint does not exist: {ckpt_path}"

    # Loads the checkpoint weights into a new model.
    state_dict = torch.load(ckpt_path, map_location="cpu")["model"]
    DummyTask(config).load_state_dict(state_dict)

    DummyTask.launch(config, launcher=mlfab.MultiProcessLauncher(num_processes=num_processes), use_cli=False)

    # Run from the same experiment directory, single-process.
    assert exp_dir.exists(), f"Experiment directory {tmpdir} contains files {list(Path(tmpdir).iterdir())}"
    config.max_steps = 15
    config.tensor_parallelism = 1

    DummyTask.launch(config, launcher=mlfab.MultiProcessLauncher(num_processes=1), use_cli=False)


@pytest.mark.slow
def test_staged_training(tmpdir: Path) -> None:
    os.environ["TENSORBOARD_PORT"] = "-1"
    os.environ["USE_METAL"] = "0"

    mlfab.configure_logging()

    config = Config(batch_size=1, run_dir=str(tmpdir))
    orig_task = DummyTask(config)
    task_key = orig_task.task_key
    task: mlfab.Task = mlfab.Task.from_task_key(task_key).get_task(config, use_cli=False)
    assert isinstance(task, DummyTask)


if __name__ == "__main__":
    # python -m tests.e2e.test_task_e2e
    test_e2e_training_mp(Path(tempfile.mkdtemp()), 4)
