"""Defines a general-purpose UNet model."""

import math
from typing import Sequence, cast

import torch
from torch import Tensor, nn

from mlfab.nn.activations import ActivationType, get_activation
from mlfab.nn.norms import NormType, get_norm_2d


class PositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_length: int = 10000) -> None:
        super().__init__()

        self.register_buffer("embedding", self.make_embedding(dim, max_length), persistent=False)

    embedding: Tensor

    def forward(self, x: Tensor) -> Tensor:
        return self.embedding[x]

    @staticmethod
    def make_embedding(dim: int, max_length: int = 10000) -> Tensor:
        embedding = torch.zeros(max_length, dim)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(max_length / 2 / math.pi) / dim))
        embedding[:, 0::2] = torch.sin(position * div_term)
        embedding[:, 1::2] = torch.cos(position * div_term)
        return embedding


class FFN(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int) -> None:
        super().__init__()

        self.init_embed = nn.Linear(in_dim, embed_dim)
        self.time_embed = PositionalEmbedding(embed_dim)
        self.model = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, in_dim),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = self.init_embed(x)
        t = self.time_embed(t)
        return self.model(x + t)


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        embed_c: int | None = None,
        act: ActivationType = "relu",
        norm: NormType = "batch_affine",
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = get_norm_2d(norm, dim=out_c)
        self.act1 = get_activation(act)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = get_norm_2d(norm, dim=out_c)
        self.act2 = get_activation(act)

        # Projects input embedding to embedding space.
        self.mlp_emb = (
            nn.Sequential(
                nn.Linear(embed_c, embed_c),
                get_activation(act),
                nn.Linear(embed_c, out_c),
            )
            if embed_c is not None
            else None
        )

        # Shortcut for residual connection.
        self.shortcut = (
            nn.Identity()
            if in_c == out_c
            else nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_c),
            )
        )

    def forward(self, x: Tensor, embedding: Tensor | None = None) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)

        if embedding is not None:
            assert self.mlp_emb is not None, "Embedding was provided but embedding projection is None"
            tx = self.mlp_emb(embedding)
            out = out + tx[..., None, None]
        else:
            assert self.mlp_emb is None, "Embedding projection is not None but no embedding was provided"

        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act2(out + self.shortcut(x))
        return out


class SelfAttention2d(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout_prob: float = 0.1) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.q_conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.k_conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.v_conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.o_conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: Tensor) -> Tensor:
        _, _, h, w = x.shape
        q = self.q_conv.forward(x)
        k = self.k_conv.forward(x)
        v = self.v_conv.forward(x)
        q = q.unflatten(1, (self.num_heads, self.dim // self.num_heads)).flatten(-2)
        k = k.unflatten(1, (self.num_heads, self.dim // self.num_heads)).flatten(-2)
        v = v.unflatten(1, (self.num_heads, self.dim // self.num_heads)).flatten(-2)
        a = torch.einsum("... c q, ... c k -> ... q k", q, k) / self.dim**0.5
        a = self.dropout(torch.softmax(a, dim=-1))
        o = torch.einsum("... s t, ... c t -> ... c s", a, v)
        o = o.flatten(1, 2).unflatten(-1, (h, w))
        return o


class UNet(nn.Module):
    """Defines a general-purpose UNet model.

    Parameters:
        in_dim: Number of input dimensions.
        embed_dim: Embedding dimension.
        dim_scales: List of dimension scales.
        input_embedding_dim: The input embedding dimension, if an input
            embedding is used (for example, when conditioning on time, or some
            class embedding).

    Inputs:
        x: Input tensor of shape ``(batch_size, in_dim, height, width)``.
        t: Time tensor of shape ``(batch_size)`` if ``use_time`` is ``True``
            and ``None`` otherwise.
        c: Class tensor of shape ``(batch_size, class_dim)`` if ``use_class``
            is ``True`` and ``None`` otherwise.

    Outputs:
        x: Output tensor of shape ``(batch_size, in_dim, height, width)``.
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        dim_scales: Sequence[int],
        input_embedding_dim: int | None = None,
    ) -> None:
        super().__init__()

        self.init_embed = nn.Conv2d(in_dim, embed_dim, 1)

        self.down_blocks = cast(list[BasicBlock | nn.Conv2d], nn.ModuleList())
        self.up_blocks = cast(list[BasicBlock | nn.ConvTranspose2d], nn.ModuleList())

        all_dims = (embed_dim, *[embed_dim * s for s in dim_scales])

        for idx, (in_c, out_c) in enumerate(zip(all_dims[:-1], all_dims[1:])):
            is_last = idx == len(all_dims) - 2
            self.down_blocks.extend(
                nn.ModuleList(
                    [
                        BasicBlock(in_c, in_c, input_embedding_dim),
                        BasicBlock(in_c, in_c, input_embedding_dim),
                        nn.Conv2d(in_c, out_c, 3, 2, 1) if not is_last else nn.Conv2d(in_c, out_c, 1),
                    ]
                )
            )

        for idx, (in_c, out_c, skip_c) in enumerate(zip(all_dims[::-1][:-1], all_dims[::-1][1:], all_dims[:-1][::-1])):
            is_last = idx == len(all_dims) - 2
            self.up_blocks.extend(
                nn.ModuleList(
                    [
                        BasicBlock(in_c + skip_c, in_c, input_embedding_dim),
                        BasicBlock(in_c + skip_c, in_c, input_embedding_dim),
                        nn.ConvTranspose2d(in_c, out_c, (2, 2), 2) if not is_last else nn.Conv2d(in_c, out_c, 1),
                    ]
                )
            )

        self.mid_blocks = cast(
            list[BasicBlock | SelfAttention2d],
            nn.ModuleList(
                [
                    BasicBlock(all_dims[-1], all_dims[-1], input_embedding_dim),
                    SelfAttention2d(all_dims[-1]),
                    BasicBlock(all_dims[-1], all_dims[-1], input_embedding_dim),
                ]
            ),
        )

        self.out_blocks = cast(
            list[BasicBlock | nn.Conv2d],
            nn.ModuleList(
                [
                    BasicBlock(embed_dim, embed_dim, input_embedding_dim),
                    nn.Conv2d(embed_dim, in_dim, 1, bias=True),
                ]
            ),
        )

    def forward(self, x: Tensor, embedding: Tensor | None = None) -> Tensor:
        x = self.init_embed(x)
        skip_conns = []
        residual = x.clone()

        for block in self.down_blocks:
            if isinstance(block, BasicBlock):
                x = block.forward(x, embedding)
                skip_conns.append(x)
            else:
                x = block.forward(x)

        for block in self.mid_blocks:
            if isinstance(block, BasicBlock):
                x = block.forward(x, embedding)
            else:
                x = block.forward(x)

        for block in self.up_blocks:
            if isinstance(block, BasicBlock):
                x = torch.cat((x, skip_conns.pop()), dim=1)
                x = block.forward(x, embedding)
            else:
                x = block.forward(x)

        x = x + residual
        for block in self.out_blocks:
            if isinstance(block, BasicBlock):
                x = block.forward(x, embedding)
            else:
                x = block.forward(x)

        return x
