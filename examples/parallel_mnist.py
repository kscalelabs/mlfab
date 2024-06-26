"""Trains a simple convolutional neural network on the MNIST dataset.

Run this example with `python -m examples.parallel_mnist`.
"""

import warnings
from dataclasses import dataclass

import torch.nn.functional as F
from dpshdl.impl.mnist import MNIST
from torch import Tensor, nn

import mlfab


@dataclass(kw_only=True)
class Config(mlfab.Config):
    in_dim: int = mlfab.field(28 * 28, help="Number of input dimensions")
    learning_rate: float = mlfab.field(1e-3, help="Learning rate to use for optimizer")
    betas: tuple[float, float] = mlfab.field((0.9, 0.999), help="Beta values for Adam optimizer")
    weight_decay: float = mlfab.field(1e-4, help="Weight decay to use for the optimizer")
    warmup_steps: int = mlfab.field(100, help="Number of warmup steps to use for the optimizer")

    # Model parallelism.
    model_parallelism: int = mlfab.field(2, help="Model parallelism")


class ParallelMnistClassification(mlfab.Task[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        if config.model_parallelism == 1:
            warnings.warn(
                "Running with a single model parallel! This example will behave the same way as a vanilla MNIST model. "
                "Instead, set `model_parallelism` to some larger value."
            )

        self.model = nn.Sequential(
            mlfab.RowParallelLinear(config.in_dim, 32, input_is_parallel=False),
            nn.LayerNorm(32),
            nn.ReLU(),
            mlfab.ColumnParallelLinear(32, 32, gather_output=True),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def get_dataset(self, phase: mlfab.Phase) -> MNIST:
        return MNIST(root_dir=mlfab.get_data_dir() / "mnist", train=phase == "train", dtype="float32")

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def get_loss(self, batch: tuple[Tensor, Tensor], state: mlfab.State) -> Tensor:
        x, y = batch
        yhat = self(x.flatten(1))
        self.log_step(batch, yhat, state)
        loss = F.cross_entropy(yhat, y.long())
        return loss

    def log_valid_step(self, batch: tuple[Tensor, Tensor], output: Tensor, state: mlfab.State) -> None:
        (x, y), yhat = batch, output

        def get_label_strings() -> list[str]:
            ypred = yhat.argmax(-1)
            return [f"ytrue={y[i]}, ypred={ypred[i]}" for i in range(len(y))]

        self.log_labeled_images("images", lambda: (x.cpu(), get_label_strings()))


if __name__ == "__main__":
    ParallelMnistClassification.launch(Config(batch_size=16, num_train_dl_workers=1))
