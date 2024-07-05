"""Tests model checkpointing."""

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
    num_layers: int = mlfab.field(2)
    use_ddp: bool = mlfab.field(True)
    learning_rate: float = mlfab.field(1e-3)
    betas: tuple[float, float] = mlfab.field((0.9, 0.999))
    weight_decay: float = mlfab.field(1e-4)
    warmup_steps: int = mlfab.field(100)


class DummyDataset(Dataset[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]):
    def next(self) -> tuple[Tensor, Tensor]:
        return torch.randn(3, 8), torch.randint(0, 9, (3,))

    def collate(self, items: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        return mlfab.collate(items)


class DummyTask(mlfab.Task[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.emb = nn.Embedding(10, 8)
        self.convs = nn.Sequential(*(nn.Conv1d(3, 3, 3, padding=1) for _ in range(config.num_layers)))
        self.lstm = nn.LSTM(8, 8, 2)
        self.pretrained_linear = mlfab.pretrained(nn.Linear(8, 8))

    def get_dataset(self, phase: mlfab.Phase) -> DummyDataset:
        return DummyDataset()


@pytest.mark.slow
def test_model_serialization(tmpdir: Path) -> None:
    task = DummyTask(Config(batch_size=1))
    mod = task
    opt = task.build_optimizer(mod)
    ckpt_path = Path(tmpdir)
    assert not any(k.startswith("pretrained") for k in mod.state_dict().keys())
    task.save_ckpt(mlfab.State.init_state(), mod, opt, ckpt_path=ckpt_path)
    task.load_ckpt_(mod, opt, ckpt_path=ckpt_path)


if __name__ == "__main__":
    # python -m tests.task.test_checkpointing
    with tempfile.TemporaryDirectory() as tmpdir:
        test_model_serialization(Path(tmpdir))
