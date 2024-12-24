"""Trains a simple convolutional neural network on the MNIST dataset.

Run this example with `python -m examples.mnist`.
"""

from dataclasses import dataclass

import torch.nn.functional as F
from dpshdl.impl.mnist import MNIST
from torch import Tensor, nn

import mlfab


@dataclass(kw_only=True)
class Config(mlfab.Config):
    in_dim: int = mlfab.field(1, help="Number of input dimensions")
    batches_per_step: int = mlfab.field(8, help="Number of batches to accumulate gradients over")
    learning_rate: float = mlfab.field(1e-3, help="Learning rate to use for optimizer")
    betas: tuple[float, float] = mlfab.field((0.9, 0.999), help="Beta values for Adam optimizer")
    weight_decay: float = mlfab.field(1e-4, help="Weight decay to use for the optimizer")
    warmup_steps: int = mlfab.field(100, help="Number of warmup steps to use for the optimizer")


class MnistClassification(mlfab.Task[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.model = nn.Sequential(
            nn.Conv2d(config.in_dim, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def get_dataset(self, phase: mlfab.Phase) -> MNIST:
        return MNIST(root_dir=mlfab.get_data_dir() / "mnist", train=phase == "train", dtype="float32")

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def get_loss(self, batch: tuple[Tensor, Tensor], state: mlfab.State) -> Tensor:
        x, y = batch
        yhat = self(x.unsqueeze(1))
        self.log_step(batch, yhat, state)
        loss = F.cross_entropy(yhat, y.long())
        return loss

    def log_train_step(self, batch: tuple[Tensor, Tensor], output: Tensor, state: mlfab.State) -> None:
        (_, y), yhat = batch, output
        self.log_scalar("acc", lambda: (y == yhat.argmax(dim=-1)).float().mean(), namespace="metrics")

    def log_valid_step(self, batch: tuple[Tensor, Tensor], output: Tensor, state: mlfab.State) -> None:
        max_images = 16
        (x, y), yhat = batch, output

        def get_label_strings() -> list[str]:
            ypred = yhat[:max_images].argmax(-1)
            return [f"ytrue={y[i]}, ypred={ypred[i]}" for i in range(len(y[:max_images]))]

        self.log_labeled_images("images", lambda: (x[:max_images].cpu(), get_label_strings()), namespace="samples")


if __name__ == "__main__":
    MnistClassification.launch(Config(batch_size=256, num_train_dl_workers=1))
