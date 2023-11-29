"""Trains a GAN on the MNIST dataset."""

import logging
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as V
from torch import Tensor, nn
from torch.utils.data.dataset import Dataset, TensorDataset
from torchvision.datasets import MNIST

import mlfab
from mlfab.core.state import State

logger = logging.getLogger(__name__)


@dataclass
class Config(mlfab.Config):
    in_dim: int = mlfab.field(1, help="Number of input dimensions")
    embed_dim: int = mlfab.field(128, help="Embedding dimension")
    dim_scales: list[int] = mlfab.field([1, 2, 4, 8], help="List of dimension scales")


def cbr(in_c: int, out_c: int, kernel_size: int, stride: int, padding: int = 0) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.2),
    )


class MnistGan(mlfab.Task[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.generator = mlfab.UNet(
            in_dim=config.in_dim,
            embed_dim=config.embed_dim,
            dim_scales=config.dim_scales,
        )

        self.discriminator = nn.Sequential(
            cbr(config.in_dim, config.embed_dim, 2, 2),
            cbr(config.embed_dim, config.embed_dim, 2, 2),
            cbr(config.embed_dim, config.embed_dim, 3, 1),
            nn.Conv2d(config.embed_dim, 1, 1),
        )

    def set_loggers(self) -> None:
        self.add_logger(
            # mlfab.CursesLogger(),
            mlfab.StdoutLogger(),
            mlfab.TensorboardLogger(self.exp_dir),
        )

    def get_dataset(self, phase: mlfab.Phase) -> Dataset[tuple[Tensor]]:
        root_dir = mlfab.get_data_dir() / "mnist"
        mnist = MNIST(
            root=root_dir,
            train=phase == "train",
            download=not root_dir.exists(),
        )

        data = mnist.data.float()
        data = V.pad(data, [2, 2])
        data = (data - 127.5) / 127.5
        data = data.unsqueeze(1)
        return cast(Dataset[tuple[Tensor]], TensorDataset(data))

    def build_optimizer(self) -> mlfab.OptType:
        assert (max_steps := self.config.max_steps) is not None, "`max_steps` must be set"

        return [
            (
                mlfab.Adam.get(self.generator, lr=1e-3, weight_decay=0.0),
                mlfab.CosineDecayLRScheduler(max_steps),
            ),
            (
                mlfab.Adam.get(self.discriminator, lr=1e-3, weight_decay=1e-5),
                mlfab.CosineDecayLRScheduler(max_steps),
            ),
        ]

    def forward_generator(self, x: Tensor) -> Tensor:
        return torch.tanh(self.generator(x))

    def forward_discriminator(self, x: Tensor) -> Tensor:
        return self.discriminator(x)

    def get_loss(self, batch: tuple[Tensor], state: State) -> list[dict[str, Tensor]]:
        (y,) = batch
        yhat = self.forward_generator(torch.randn_like(y))
        real_pred, fake_pred = self.forward_discriminator(y), self.forward_discriminator(yhat)
        self.log_step(batch, yhat, state)

        # The order of the elements in this list must match the order of the
        # optimizers returned by `build_optimizer`.
        return [
            {
                "generator": F.binary_cross_entropy_with_logits(fake_pred, torch.ones_like(fake_pred)),
            },
            {
                "real": F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred)),
                "fake": F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred)),
            },
        ]

    def log_valid_step(self, batch: tuple[Tensor], output: Tensor, state: mlfab.State) -> None:
        (images,) = batch
        gen_images = output
        max_images = 9
        self.log_images("real", images, max_images=max_images, sep=2)
        self.log_images("generated", gen_images, max_images=max_images, sep=2)


if __name__ == "__main__":
    # python -m examples.gan
    MnistGan.launch(
        Config(
            batch_size=32,
            max_steps=2_000,
            valid_every_n_steps=500,
            valid_every_n_seconds=None,
            valid_first_n_seconds=None,
        ),
    )
