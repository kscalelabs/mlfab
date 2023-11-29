"""Trains a conditional consistency model on the MNIST dataset."""

import logging
from dataclasses import dataclass
from typing import cast

import torchvision.transforms.functional as V
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
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
    num_classes: int = mlfab.field(10, help="Number of unique classes")
    steps_to_evaluate: list[int] = mlfab.field([1, 2, 4, 8, 16], help="Step counts to evaluate")


class ConditionalConsistency(mlfab.Task[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.diff = mlfab.ConsistencyModel(
            total_steps=config.max_steps,
            loss_dim=1,
        )

        self.class_embs = nn.Embedding(config.num_classes, config.embed_dim)

        self.time_embs = nn.Sequential(
            mlfab.FourierEmbeddings(config.embed_dim),
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.SiLU(),
            nn.Linear(config.embed_dim, config.embed_dim),
        )

        self.model = mlfab.UNet(
            in_dim=config.in_dim,
            embed_dim=config.embed_dim,
            dim_scales=config.dim_scales,
            input_embedding_dim=config.embed_dim,
        )

    def set_loggers(self) -> None:
        self.add_logger(
            # mlfab.CursesLogger(),
            mlfab.StdoutLogger(),
            mlfab.TensorboardLogger(self.exp_dir),
        )

    def get_dataset(self, phase: mlfab.Phase) -> Dataset[tuple[Tensor, Tensor]]:
        root_dir = mlfab.get_data_dir() / "mnist"
        mnist = MNIST(
            root=root_dir,
            train=phase == "train",
            download=not root_dir.exists(),
        )

        # Loads the images into RAM.
        data = mnist.data.float()
        data = V.pad(data, [2, 2])
        data = (data - 127.5) / 127.5
        data = data.unsqueeze(1)

        # Loads the labels into RAM.
        labels = mnist.targets

        return cast(Dataset[tuple[Tensor, Tensor]], TensorDataset(data, labels))

    def build_optimizer(self) -> tuple[Optimizer, mlfab.CosineDecayLRScheduler]:
        assert (max_steps := self.config.max_steps) is not None, "`max_steps` must not be None"

        return (
            mlfab.Adam.get(self, lr=1e-3),
            mlfab.CosineDecayLRScheduler(max_steps),
        )

    def forward(self, x: Tensor, t: Tensor, class_id: Tensor) -> Tensor:
        c_emb = self.class_embs(class_id)
        t_emb = self.time_embs(t.to(x))
        emb = c_emb + t_emb
        return self.model(x, emb)

    def get_loss(self, batch: tuple[Tensor, Tensor], state: State) -> Tensor:
        x, class_id = batch
        loss = self.diff.loss(lambda x, t: self(x, t, class_id), x, state.num_steps)
        self.log_step(batch, loss, state)
        return loss

    def log_train_step(self, batch: tuple[Tensor, Tensor], output: Tensor, state: State) -> None:
        self.log_scalar("num_scales", lambda: self.diff._get_num_scales(state.num_steps))

    def log_valid_step(self, batch: tuple[Tensor, Tensor], output: Tensor, state: mlfab.State) -> None:
        max_images = 9
        images, class_id = batch
        images, class_id = images[:max_images], class_id[:max_images]
        self.log_images("real", images, sep=2)

        def vanilla_func(x: Tensor, t: Tensor) -> Tensor:
            return self(x, t, class_id)

        def cross_condition_func(x: Tensor, t: Tensor) -> Tensor:
            e_pos_cond = self(x, t, class_id.roll(1, 0))
            e_neg_cond = self(x, t, class_id)
            return e_neg_cond + 7.5 * (e_pos_cond - e_neg_cond)

        def per_sample_norm(x: Tensor) -> Tensor:
            # Need to normalize per sample, because consistency models
            # gradually decrease the noise from a large amount, so without
            # doing this only the maximum noise sample shows up.
            return x / x.std(dim=(1, 2, 3), keepdim=True)

        for num_steps in self.config.steps_to_evaluate:
            # Generates a sample from the model.
            gen = self.diff.sample(vanilla_func, images[:max_images].shape, images.device, num_steps)
            self.log_images(f"{num_steps}_gen", gen[0], max_images=max_images, sep=2)
            one_gen = per_sample_norm(gen[:, 0])
            self.log_images(f"single_{num_steps}", one_gen, max_images=max_images, sep=2)

            # Generates a cross-conditioned sample from the model.
            cross_gen = self.diff.partial_sample(cross_condition_func, images, 0.5, num_steps)
            self.log_images(f"{num_steps}_cross_gen", cross_gen[0], sep=2)
            one_cross_gen = per_sample_norm(cross_gen[:, 0])
            self.log_images(f"single_{num_steps}_cross_gen", one_cross_gen, max_images=max_images, sep=2)


if __name__ == "__main__":
    # python -m examples.conditional_consistency
    ConditionalConsistency.launch(
        Config(
            batch_size=256,
            max_steps=10_000,
            valid_every_n_steps=500,
            valid_every_n_seconds=None,
            valid_first_n_seconds=None,
            batches_per_step_schedule=[1000, 1000, 1000],
        ),
    )
