"""Defines a general-purpose API for transformer embedding layers.

.. highlight:: python
.. code-block:: python

    from mlfab.nn.embeddings import get_positional_embeddings, cast_embedding_kind

    embeddings = get_positional_embeddings(
        max_tsz=1024,
        embed_dim=128,
        kind="sinusoidal",
        learnable=False,
    )

    x = torch.arange(3, 5, 8)

    # Time-based positional embeddings - the time tensor supplies the
    # times for each element in the input.
    times = torch.randint(0, 1024, (3, 5))
    y1 = embeddings(x, times=times)

    # Offset-based positional embeddings - the input is assumed to be in
    # temporal order, and the offset is the offset of the first element.
    y2 = embeddings(x, offset=1)

    assert y1.shape == y2.shape == x.shape

    # This lets you parametrize the embedding kind as a string.
    embeddings = get_positional_embeddings(..., kind=cast_embedding_kind(my_kind))

Choices for the embedding kind are:

- ``"identity"``: No positional embeddings are added.
- ``"learned"``: Positional embeddings are learned.
- ``"sinusoidal"``: Sinusoidal embeddings.
- ``"rotary"``: Rotary embeddings (popular for training transformers).
"""

import math
from typing import Literal, cast, get_args, overload

import torch
from torch import Tensor, nn

from mlfab.nn.init import InitializationType, init_

EmbeddingKind = Literal["identity", "learned", "sinusoidal", "rotary"]


def cast_embedding_kind(k: str) -> EmbeddingKind:
    args = get_args(EmbeddingKind)
    assert k in args, f"Invalid initialization type: '{k}' Valid options are {args}"
    return cast(EmbeddingKind, k)


class IdentityPositionalEmbeddings(nn.Module):
    def forward(self, x: Tensor, offset: int = 0, times_bt: Tensor | None = None) -> Tensor:
        return x


class LearnedPositionalEmbeddings(nn.Module):
    """Defines a learned embeddings module.

    Parameters:
        max_tsz: The maximum sequence length.
        embed_dim: The embedding dimension.
        weight_init: The initialization type for the embedding weight.
        learnable: Whether the embeddings are learnable.
    """

    def __init__(
        self,
        max_tsz: int,
        embed_dim: int,
        weight_init: InitializationType = "normal",
        learnable: bool = True,
    ) -> None:
        super().__init__()

        self.max_tsz = max_tsz
        self.embed_dim = embed_dim
        self.weight_init = weight_init

        self.embeddings_tc = nn.Parameter(torch.empty(max_tsz, embed_dim), requires_grad=learnable)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_(self.embeddings_tc.data, None, self.weight_init)

    def forward(self, x: Tensor, offset: int = 0, times_bt: Tensor | None = None) -> Tensor:
        emb_btc = (
            self.embeddings_tc[None, offset : offset + x.size(1)] if times_bt is None else self.embeddings_tc[times_bt]
        )
        return x + emb_btc


class SinusoidalEmbeddings(nn.Module):
    """Defines a sinusoidal embeddings module.

    Parameters:
        embed_dim: The embedding dimension.
        max_tsz: The maximum sequence length.
        learnable: Whether the embeddings are learnable.
        base: The base for the sinusoidal embeddings.
    """

    def __init__(
        self,
        embed_dim: int | None = None,
        max_tsz: int | None = None,
        learnable: bool = True,
        base: int = 10_000,
    ) -> None:
        super().__init__()

        self.max_tsz = max_tsz
        self.embed_dim = embed_dim
        self.base = base

        self.embeddings_tc: nn.Parameter | None = None
        if learnable:
            assert max_tsz is not None, "Learnable parameters require `max_tsz` to be set"
            assert embed_dim is not None, "Learnable parameters require `embed_dim` to be set"
            self.embeddings_tc = nn.Parameter(torch.empty(max_tsz, embed_dim), requires_grad=learnable)
            self.reset_parameters()

        self.embeddings_cached: Tensor | None = None

    def forward(self, x_btc: Tensor, offset: int = 0, times_bt: Tensor | None = None) -> Tensor:
        embeddings_tc: Tensor | None = self.embeddings_tc
        _, tsz, xdim = x_btc.shape
        if embeddings_tc is None:
            max_tsz = max(tsz, 0 if times_bt is None else int(times_bt.max().item()) + 1) + offset
            if self.embeddings_cached is None:
                self.embeddings_cached = self.get_embeddings(max_tsz, xdim, x_btc.device, x_btc.dtype)
            else:
                embed_tsz, embed_dim = self.embeddings_cached.shape
                embed_device, embed_dtype = self.embeddings_cached.device, self.embeddings_cached.dtype
                if (
                    embed_tsz < max_tsz
                    or embed_dim != xdim
                    or embed_device != x_btc.device
                    or embed_dtype != x_btc.dtype
                ):
                    self.embeddings_cached = self.get_embeddings(max_tsz, embed_dim, x_btc.device, x_btc.dtype)
            embeddings_tc = self.embeddings_cached
        return x_btc + (embeddings_tc[None, offset : offset + tsz] if times_bt is None else embeddings_tc[times_bt])

    def reset_parameters(self) -> None:
        if self.embeddings_tc is None:
            assert self.max_tsz is not None, "Learnable parameters require `max_tsz` to be set"
            assert self.embed_dim is not None, "Learnable parameters require `embed_dim` to be set"
            self.embeddings_tc.data.copy_(self.get_embeddings(self.max_tsz, self.embed_dim))

    def get_embeddings(
        self,
        tsz: int,
        embed_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        positions = torch.arange(tsz, device=device, dtype=torch.float32)
        dim = torch.arange(embed_dim, device=device, dtype=torch.float32)
        dim = self.base ** (2 * (dim // 2) / embed_dim)
        embeddings = positions[:, None] / dim[None, :]
        embeddings[:, 0::2] = torch.sin(embeddings[:, 0::2])
        embeddings[:, 1::2] = torch.cos(embeddings[:, 1::2])
        return embeddings.to(dtype)


def get_rotary_embeddings(
    tsz: int,
    embed_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    offset: int = 0,
    base: int = 10_000,
) -> Tensor:
    with torch.no_grad():
        assert embed_dim % 4 == 0, f"Embedding dimension must be divisible by 4, got {embed_dim}"
        half_d = embed_dim // 2
        theta = 1.0 / (base ** (torch.arange(0, half_d, 2, device=device, dtype=torch.float32) / half_d))
        seq_idx = torch.arange(offset, tsz + offset, device=device, dtype=torch.float32)
        idx_theta_tc = torch.einsum("t,c->tc", seq_idx, theta)
        idx_theta2_tc = torch.cat([idx_theta_tc, idx_theta_tc], dim=1)
        cos_tc, sin_tc = idx_theta2_tc.cos(), idx_theta2_tc.sin()
        emb_2tc = torch.stack((cos_tc, sin_tc), dim=0).to(dtype)
        return emb_2tc


def apply_rotary_embeddings(x_btc: Tensor, embs_2tc: Tensor, offset: int = 0, times_bt: Tensor | None = None) -> Tensor:
    cos_tc, sin_tc = embs_2tc.unbind(0)
    _, tsz, embed_dim = x_btc.shape
    half_d = embed_dim // 2
    quarter_d = embed_dim // 4
    x_rope_btc, x_pass_btc = x_btc[..., :half_d], x_btc[..., half_d:]
    neg_half_x_btc = torch.cat([-x_rope_btc[..., quarter_d:], x_rope_btc[..., :quarter_d]], dim=-1)
    cos_part_btc = cos_tc[None, offset : offset + tsz] if times_bt is None else cos_tc[times_bt]
    sin_part_btc = sin_tc[None, offset : offset + tsz] if times_bt is None else sin_tc[times_bt]
    x_rope_btc = x_rope_btc * cos_part_btc + neg_half_x_btc * sin_part_btc
    return torch.cat((x_rope_btc, x_pass_btc), dim=-1)


def rotary_embeddings(x_btc: Tensor, offset: int = 0, base: int = 10_000) -> Tensor:
    """Defines a single function for applying rotary embeddings.

    This is slower than using the module, but it doesn't require
    pre-initializing the embeddings, so it can be used when running online.

    Args:
        x_btc: The input tensor, with shape ``(batch, tsz, embed_dim)``.
        offset: The offset for the first element.
        base: The base for the sinusoidal embeddings.

    Returns:
        The input tensor with rotary embeddings applied.
    """
    (_, tsz, embed_dim), device, dtype = x_btc.shape, x_btc.device, x_btc.dtype
    emb_2tc = get_rotary_embeddings(tsz + offset, embed_dim, device, dtype, 0, base)
    return apply_rotary_embeddings(x_btc, emb_2tc, offset)


class RotaryEmbeddings(nn.Module):
    def __init__(self, base: int = 10_000) -> None:
        """Defines a rotary embeddings module.

        Args:
            base: The base for the sinusoidal embeddings.
        """
        super().__init__()

        self.base = base
        self.embeddings: Tensor | None = None

    def forward(self, x_btc: Tensor, offset: int = 0, times_bt: Tensor | None = None) -> Tensor:
        emb_2tc = self.embeddings
        _, tsz, embed_dim = x_btc.shape
        max_tsz = max(tsz, 0 if times_bt is None else int(times_bt.max().item()) + 1) + offset
        if emb_2tc is None or emb_2tc.shape[-2] < max_tsz:
            emb_2tc = get_rotary_embeddings(max_tsz, embed_dim, x_btc.device, x_btc.dtype, 0, self.base)
            self.embeddings = emb_2tc
        return apply_rotary_embeddings(x_btc, emb_2tc, offset, times_bt)


@overload
def get_positional_embeddings(kind: Literal["identity"]) -> IdentityPositionalEmbeddings: ...


@overload
def get_positional_embeddings(
    kind: Literal["learned"],
    *,
    max_tsz: int,
    embed_dim: int,
    weight_init: InitializationType = "normal",
    learnable: bool | None = None,
) -> LearnedPositionalEmbeddings: ...


@overload
def get_positional_embeddings(
    kind: Literal["sinusoidal"],
    *,
    max_tsz: int | None = None,
    embed_dim: int | None = None,
    learnable: bool | None = None,
    base: int = 10_000,
) -> SinusoidalEmbeddings: ...


@overload
def get_positional_embeddings(
    kind: Literal["rotary"],
    *,
    base: int = 10_000,
) -> RotaryEmbeddings: ...


@overload
def get_positional_embeddings(
    kind: EmbeddingKind,
    *,
    max_tsz: int | None = None,
    embed_dim: int | None = None,
    weight_init: InitializationType = "normal",
    learnable: bool | None = None,
    base: int = 10_000,
) -> IdentityPositionalEmbeddings | LearnedPositionalEmbeddings | SinusoidalEmbeddings | RotaryEmbeddings: ...


def get_positional_embeddings(
    kind: EmbeddingKind,
    *,
    max_tsz: int | None = None,
    embed_dim: int | None = None,
    weight_init: InitializationType = "normal",
    learnable: bool | None = None,
    base: int = 10_000,
) -> nn.Module:
    """Defines the common module for adding positional embeddings.

    Args:
        kind: The type of embedding to use.
        max_tsz: The maximum sequence length.
        embed_dim: The embedding dimension.
        weight_init: The weight initialization for learned embeddings.
        learnable: Whether the embeddings are learnable; if not provided,
            uses sensible defaults.
        base: The base for the sinusoidal embeddings.

    Returns:
        The positional embeddings module.

    Raises:
        ValueError: If an invalid embedding kind is supplied.
    """
    match kind:
        case "identity":
            return IdentityPositionalEmbeddings()

        case "learned":
            assert max_tsz is not None, "Learned embeddings require `max_tsz` to be set"
            assert embed_dim is not None, "Learned embeddings require `embed_dim` to be set"

            return LearnedPositionalEmbeddings(
                max_tsz=max_tsz,
                embed_dim=embed_dim,
                weight_init=weight_init,
                learnable=True if learnable is None else learnable,
            )

        case "sinusoidal":
            return SinusoidalEmbeddings(
                max_tsz=max_tsz,
                embed_dim=embed_dim,
                learnable=False if learnable is None else learnable,
                base=base,
            )

        case "rotary":
            return RotaryEmbeddings(base=base)

        case _:
            raise ValueError(f"Invalid embedding kind: {kind}")


def fourier_embeddings(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
    half = dim // 2
    idxs = torch.arange(start=0, end=half, device=t.device, dtype=torch.float32)
    freqs = torch.exp(-math.log(max_period) * idxs / half)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    # Adds an additional row of zeros to match the expected dimension.
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class FourierEmbeddings(nn.Module):
    """Defines a module for applying Fourier embeddings to timesteps.

    This module differs from the other positional embedding modules because it
    expects a continuous time input, rather than a discrete time input.

    Parameters:
        dim: The number of embedding dimensions. This value is used to determine
            how many different frequencies to use, and a higher value means
            higher frequencies.
        max_period: The maximum period for the embeddings. This should roughly
            be in line with the maximum number of timesteps; the default value
            of 10,000 is commonly used in NLP applications, and is derived from
            operating on sequence lengths of 100 to 1000 tokens.
    """

    __constants__ = ["dim", "max_period"]

    def __init__(self, dim: int, max_period: int = 10000) -> None:
        super().__init__()

        self.dim = dim
        self.max_period = max_period

    def forward(self, t: Tensor) -> Tensor:
        return fourier_embeddings(t, self.dim, self.max_period)
