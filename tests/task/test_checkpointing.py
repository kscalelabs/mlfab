"""Tests model checkpointing."""

from dataclasses import dataclass
from pathlib import Path

import torch
from dpshdl.dataset import Dataset
from torch import Tensor, nn

import mlfab


@dataclass
class Config(mlfab.Config):
    num_layers: int = mlfab.field(2, help="Number of layers to use")
    use_ddp: bool = mlfab.field(True, help="Whether to use DDP instead of FSDP")
    learning_rate: float = mlfab.field(1e-3, help="Learning rate to use for optimizer")
    betas: tuple[float, float] = mlfab.field((0.9, 0.999), help="Beta values for Adam optimizer")
    weight_decay: float = mlfab.field(1e-4, help="Weight decay to use for the optimizer")
    warmup_steps: int = mlfab.field(100, help="Number of warmup steps to use for the optimizer")


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

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x, _ = self.lstm(x.float())
        z = x + self.emb(y)
        return self.convs(z)

    def get_loss(self, batch: tuple[Tensor, Tensor], state: mlfab.State) -> Tensor:
        o = self(*batch).sum()
        return o

    def get_dataset(self, phase: mlfab.Phase) -> DummyDataset:
        return DummyDataset()


def test_model_serialization(tmpdir: Path) -> None:
    task = DummyTask(Config(batch_size=1))
    ckpt_path = Path(tmpdir / "ckpt.pt")
    task.save_checkpoint(mlfab.State.init_state(), ckpt_path)
    task.load_checkpoint_(ckpt_path)
