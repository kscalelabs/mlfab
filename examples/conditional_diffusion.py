"""Trains a conditional diffusion model on the MNIST dataset."""

import logging
from dataclasses import dataclass
from typing import get_args

import numpy as np
import torch
import torchvision.transforms.functional as V
from dpshdl.dataset import TensorDataset
from dpshdl.impl.mnist import MNIST
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer

import mlfab
from mlfab.core.state import State

logger = logging.getLogger(__name__)


@dataclass
class Config(mlfab.Config):
    in_dim: int = mlfab.field(1, help="Number of input dimensions")
    embed_dim: int = mlfab.field(128, help="Embedding dimension")
    dim_scales: list[int] = mlfab.field([1, 2, 4, 8], help="List of dimension scales")
    num_classes: int = mlfab.field(10, help="Number of unique classes")
    num_beta_steps: int = mlfab.field(500, help="Number of beta steps")
    num_sampling_steps: int | None = mlfab.field(50, help="Number of sampling steps")


class ConditionalDiffusion(mlfab.Task[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.diff = mlfab.GaussianDiffusion(
            beta_schedule="linear",
            num_beta_steps=config.num_beta_steps,
            pred_mode="pred_v",
            loss="mse",
            sigma_type="upper_bound",
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

    def get_dataset(self, phase: mlfab.Phase) -> TensorDataset[tuple[np.ndarray]]:
        root_dir = mlfab.get_data_dir() / "mnist"
        mnist = MNIST(root_dir=root_dir, train=phase == "train", dtype="float32")
        data = torch.from_numpy(mnist.images)
        data = V.pad(data, [2, 2])
        data = data - 0.5
        data = data.unsqueeze(1)
        return TensorDataset(data.numpy(), mnist.labels.astype(np.int64))

    def build_optimizer(self) -> Optimizer:
        return mlfab.Adam.get(self, lr=1e-3)

    def forward(self, x: Tensor, t: Tensor, class_id: Tensor) -> Tensor:
        c_emb = self.class_embs(class_id)
        t_emb = self.time_embs(t.to(x))
        emb = c_emb + t_emb
        return self.model(x, emb)

    def get_loss(self, batch: tuple[Tensor, Tensor], state: State) -> Tensor:
        x, class_id = batch
        loss = self.diff.loss(lambda x, t: self(x, t, class_id), x)
        self.log_step(batch, loss, state)
        return loss

    def log_valid_step(self, batch: tuple[Tensor, Tensor], output: Tensor, state: mlfab.State) -> None:
        num_sample = self.config.num_sampling_steps
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

        for solver_key in get_args(mlfab.ODESolverType):
            solver = mlfab.get_ode_solver(solver_key)

            # Generates a sample from the model.
            gen = self.diff.sample(vanilla_func, images.shape, images.device, num_sample, solver)
            self.log_images(f"generated_{solver_key}", gen[0], sep=2)
            one_gen = gen[:, 0]
            self.log_images(f"generated_single_{solver_key}", one_gen, max_images=max_images, sep=2)

            # Generates a cross-conditioned sample from the model.
            cross_gen = self.diff.partial_sample(cross_condition_func, images, 0.5, num_sample, solver)
            self.log_images(f"cross_generated_{solver_key}", cross_gen[0], sep=2)
            one_cross_gen = cross_gen[:, 0]
            self.log_images(f"cross_generated_single_{solver_key}", one_cross_gen, max_images=max_images, sep=2)


if __name__ == "__main__":
    # python -m examples.conditional_diffusion
    ConditionalDiffusion.launch(
        Config(
            batch_size=256,
            valid_every_n_steps=500,
        ),
    )
