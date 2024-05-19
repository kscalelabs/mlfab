"""Defines self-attention modules.

You can implement a self-attention model using the built-in PyTorch module:

.. code-block:: python

    from torch import nn

    self.attn = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=512,
            head_dims=64,
            feedforward_factor=4,
            dropout=0.1,
            activation='relu',
            batch_first=True,
        ),
        num_layers=6,
    )

However, when doing inference, you will end up recomputing a lot of previous
states. Instead, you can use the equivalent implementation in this file:

.. code-block:: python

    from ml.models.architectures.attention import TransformerEncoder, TransformerEncoderLayer

    self.attn = TransformerEncoder(
        TransformerEncoderLayer(
            d_model=512,
            head_dims=64,
            feedforward_factor=4,
            dropout=0.1,
            # activation='relu',  Always ReLU
            # batch_first=True,  Always batch first
            is_causal=is_causal,  # Additional argument to support causal attention
            use_rotary=use_rotary,  # Additional argument to support rotary embeddings
        ),
        num_layers=6,
    )

    x, state = self.attn(x, state)

This also eliminates the need to pass in an attention mask; instead, simply use
the ``is_causal`` argument to the ``forward`` method and it will automatically
apply the mask for you. This will default to the more performant PyTorch
attention implementation.
"""

import copy
import math
from typing import Literal, TypeVar, cast, overload

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mlfab.nn.architectures.next_token import SamplingStrategy, sample_from_logits
from mlfab.nn.embeddings import apply_rotary_embeddings, get_rotary_embeddings
from mlfab.nn.norms import get_norm_linear

MaskMode = Literal["causal", "lengths", "combine"]


def _bool_mask_as_dtype(mask: Tensor, dtype: torch.dtype | None) -> Tensor:
    if dtype == torch.bool:
        return mask
    return torch.zeros_like(mask, dtype=dtype).masked_fill(~mask, -float("inf"))


@overload
def get_attention_mask(
    mode: Literal["causal"],
    *,
    tsz_q: int,
    tsz_k: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor: ...


@overload
def get_attention_mask(
    mode: Literal["lengths"],
    *,
    lengths: Tensor,
    tsz_k: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor: ...


def get_attention_mask(
    mode: MaskMode,
    *,
    lengths: Tensor | None = None,
    tsz_q: int | None = None,
    tsz_k: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Returns a causal attention mask.

    Args:
        mode: Causal attention mode.
        lengths: The lengths tensor, of shape ``(bsz)``. Only required if
            ``mode="lengths"``.
        tsz_q: The number of queries.
        tsz_k: The number of keys.
        device: The output device.
        dtype: The output dtype.

    Returns:
        If in causal mode, returns a causal attention mask with shape
        ``(tsz_q, tsz_k)``. If in ``lengths`` mode, will return an attention
        mask with shape ``(bsz, tsz_k)``. If the dtype is boolean, will have
        True values for queries and keys that should attend to each other,
        False otherwise. If a float, will have have values of 0 for queries
        and keys that should attend to each other, and ``-inf`` otherwise, so
        that the mask can be applied by being added to the pre-softmax
        attention matrix.
    """
    with torch.no_grad():
        match mode:
            case "causal":
                assert tsz_q is not None, "`tsz_q` required for causal mask"
                assert tsz_k is not None, "`tsz_k` required for causal mask"
                attn_mask = torch.ones(tsz_q, tsz_k, device=device, dtype=torch.bool)
                attn_mask = attn_mask.tril(diagonal=0)
                return _bool_mask_as_dtype(attn_mask, dtype)

            case "lengths":
                assert lengths is not None, "`lengths` tensor required for lengths mask"
                if dtype is None:
                    dtype = lengths.dtype
                assert dtype in (torch.int32, torch.int64), f"Expected integer dtype, got {dtype}"
                assert lengths.dim() == 1, f"`lengths` tensor should have shape `(bsz)`, got {lengths.shape}"
                if device is None:
                    device = lengths.device
                if tsz_k is None:
                    tsz_k = int(lengths.max().item())
                idxs = torch.arange(tsz_k, device=device, dtype=dtype)
                attn_mask = idxs[None, :] < lengths[:, None]
                return _bool_mask_as_dtype(attn_mask, dtype)

            case _:
                raise ValueError(f"Invalid mask mode: {mode}")


Tq = TypeVar("Tq", Tensor, None)
Tk = TypeVar("Tk", Tensor, None)
Tv = TypeVar("Tv", Tensor, None)


class MultiheadAttention(nn.Module):
    """Defines a streamable multihead attention layer.

    This is a slightly modified implementation of ``nn.MultiheadAttention``
    that is built into PyTorch. The main difference is that this version
    supports streaming inference for causal attention, by passing in a
    state tuple that contains the previously projected key and value tensors.

    Parameters:
        embed_dim: The input and output embedding dimension.
        head_dim: The number of dimensions in each attention head.
        dropout: The dropout probability, applied to the attention matrix.
        bias: Whether to include a bias term in the projection layers.
        kdim: The dimension of the key projection. Defaults to ``embed_dim``.
        vdim: The dimension of the value projection. Defaults to ``embed_dim``.
        gqa_factor: The GQA factor to use, meaning the ratio of the number of
            queries to the number of keys. Higher values will result in more
            queries than keys, which can speed up inference.

    Inputs:
        query: The query tensor, of shape ``(B, T, C)``.
        key: The key tensor, of shape ``(B, T, C)``.
        value: The value tensor, of shape ``(B, T, C)``.
        state: The previous key and value tensors, of shape
            ``(B * H, T', C // H)``, where ``T'`` is the number of previous
            timesteps and ``H`` is the number of attention heads. This is
            only supported if ``is_causal=True``.
        is_causal: Whether to apply a causal mask to the attention matrix.
            Note that the "mask" is only applied implicitly and isn't actually
            instantiated as a tensor.

    Outputs:
        output: The output tensor, of shape ``(B, T, C)``, along with the
            key and value state for the next timestep.
    """

    __constants__ = [
        "num_heads",
        "gqa_factor",
        "kv_num_heads",
        "dropout",
        "head_dim",
        "embed_dim",
        "kv_embed_dim",
        "kdim",
        "vdim",
        "_qkv_same_embed_dim",
    ]

    def __init__(
        self,
        embed_dim: int,
        head_dim: int,
        dropout: float = 0.0,
        bias: bool = True,
        kdim: int | None = None,
        vdim: int | None = None,
        gqa_factor: int = 1,
    ) -> None:
        super().__init__()

        head_dim = min(head_dim, embed_dim)
        assert embed_dim % head_dim == 0, f"`{embed_dim=}` must be divisible by `{head_dim=}`"
        num_heads = embed_dim // head_dim
        assert num_heads % gqa_factor == 0, f"`{num_heads=}` must be divisible by `{gqa_factor=}`"

        # Stores some constant values.
        self.num_heads = num_heads
        self.gqa_factor = gqa_factor
        self.kv_num_heads = num_heads // gqa_factor
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        self.embed_dim = embed_dim
        self.kv_embed_dim = self.kv_num_heads * self.head_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = nn.Parameter(torch.empty((embed_dim, embed_dim)))
            self.k_proj_weight = nn.Parameter(torch.empty((self.kv_embed_dim, self.kdim)))
            self.v_proj_weight = nn.Parameter(torch.empty((self.kv_embed_dim, self.vdim)))
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty((embed_dim + 2 * self.kv_embed_dim, embed_dim)))
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(embed_dim + 2 * self.kv_embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if self._qkv_same_embed_dim:
            nn.init.normal_(self.in_proj_weight, std=math.sqrt(1 / self.embed_dim))
        else:
            nn.init.normal_(self.q_proj_weight, std=math.sqrt(1 / self.embed_dim))
            nn.init.normal_(self.k_proj_weight, std=math.sqrt(2 / (self.kv_embed_dim + self.embed_dim)))
            nn.init.normal_(self.v_proj_weight, std=math.sqrt(2 / (self.kv_embed_dim + self.embed_dim)))

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward_matmuls(
        self,
        query_bqc: Tq,
        key_bkc: Tk,
        value_bkc: Tv,
        rotary_q_2qc: Tensor | None = None,
        rotary_k_2kc: Tensor | None = None,
    ) -> tuple[Tq, Tk, Tv]:
        # Gets the query, key, and value weights and biases.
        splits = (self.embed_dim, self.kv_embed_dim, self.kv_embed_dim)
        if self._qkv_same_embed_dim:
            qw_cc, kw_cc, vw_cc = self.in_proj_weight.split(splits, dim=0)
        else:
            qw_cc, kw_cc, vw_cc = self.q_proj_weight, self.k_proj_weight, self.v_proj_weight
        qb_c, kb_c, vb_c = (None, None, None) if self.in_proj_bias is None else self.in_proj_bias.split(splits, dim=0)

        # Computes the query projection.
        if query_bqc is None:
            xq_bghqd = None
        else:
            assert query_bqc.dim() == 3
            xq_bqc = F.linear(query_bqc, qw_cc, qb_c)
            xq_bghqd = xq_bqc.unflatten(-1, (self.gqa_factor, self.kv_num_heads, self.head_dim)).permute(0, 2, 3, 1, 4)
            if rotary_q_2qc is not None:
                xq_bghqd = apply_rotary_embeddings(xq_bghqd.flatten(0, 2), rotary_q_2qc).view(xq_bghqd.shape)

        # Computes the key projection.
        if key_bkc is None:
            xk_bghkd = None
        else:
            assert key_bkc.dim() == 3
            xk_bkc = F.linear(key_bkc, kw_cc, kb_c)
            xk_bghkd = xk_bkc.unflatten(-1, (1, self.kv_num_heads, self.head_dim)).permute(0, 2, 3, 1, 4)
            if rotary_k_2kc is not None:
                xk_bghkd = apply_rotary_embeddings(xk_bghkd.flatten(0, 2), rotary_k_2kc).view(xk_bghkd.shape)

        # Computes the value projection.
        if value_bkc is None:
            xv_bghkd = None
        else:
            assert value_bkc.dim() == 3
            xv_bkc = F.linear(value_bkc, vw_cc, vb_c)
            xv_bghkd = xv_bkc.unflatten(-1, (1, self.kv_num_heads, self.head_dim)).permute(0, 2, 3, 1, 4)

        return xq_bghqd, xk_bghkd, xv_bghkd

    def forward_attn(
        self,
        xq_bghqd: Tensor,
        xk_bghkd: Tensor,
        xv_bghkd: Tensor,
        is_causal: bool = False,
        mask_bqk: Tensor | None = None,
    ) -> Tensor:
        # Computes attention
        dropout = self.dropout if self.training else 0.0
        if mask_bqk is None:
            xo_bghqc = F.scaled_dot_product_attention(
                xq_bghqd,
                xk_bghkd,
                xv_bghkd,
                dropout_p=dropout,
                is_causal=is_causal,
            )
        else:
            xo_bghqc = F.scaled_dot_product_attention(
                xq_bghqd,
                xk_bghkd,
                xv_bghkd,
                attn_mask=mask_bqk[:, None, None],
                dropout_p=dropout,
            )

        # Flattens (B, G, H, Tq, C) -> (B, Tq, G * H * C)
        xo_bqc = xo_bghqc.permute(0, 3, 1, 2, 4).flatten(2)

        # Applies output projection
        xo_bqc = self.out_proj(xo_bqc)

        return xo_bqc

    def forward(
        self,
        query_bqc: Tensor,
        key_bkc: Tensor,
        value_bkc: Tensor,
        is_causal: bool = False,
        rotary_q_2qc: Tensor | None = None,
        rotary_k_2kc: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        xq_bghqd, xk_bghkd, xv_bghkd = self.forward_matmuls(query_bqc, key_bkc, value_bkc, rotary_q_2qc, rotary_k_2kc)
        xo = self.forward_attn(xq_bghqd, xk_bghkd, xv_bghkd, is_causal, mask)
        return xo

    def get_attn_matrix(
        self,
        xq_bghqc: Tensor,
        xk_bghkc: Tensor,
        is_causal: bool = False,
        mask_bqk: Tensor | None = None,
    ) -> Tensor:
        """Computes the attention matrix for a given query and key.

        This function can be used for visualization purposes.

        Args:
            xq_bghqc: The query embeddings, with shape ``(B, G, H, Tq, C)``
            xk_bghkc: The key embeddings, with shape ``(B, G, H, Tk, C)``
            state: The previous state tensor.
            is_causal: Whether to apply a causal mask to the attention matrix.
                In this function, unlike in the forward pass, the mask is
                explicitly created if not provided.
            mask_bqk: The attention mask, of shape ``(B, Tq, Tk)``. If
                ``None``, don't apply an attention mask.

        Returns:
            The attention matrix, of shape ``(B, G, H, Tq, Tk)``.
        """
        # Computes the unnormalized attention scores.
        attn_bghqk = torch.einsum("bghqc,bghkc->bghqk", xq_bghqc, xk_bghkc)

        # Applies a causal mask.
        if is_causal:
            tsz_q, tsz_k, device, dtype = attn_bghqk.size(-2), attn_bghqk.size(-1), attn_bghqk.device, attn_bghqk.dtype
            causal_mask_bqk = get_attention_mask("causal", tsz_q=tsz_q, tsz_k=tsz_k, device=device, dtype=dtype)
            causal_mask_bqk = causal_mask_bqk.expand(attn_bghqk.size(0), *causal_mask_bqk.shape)
            attn_bghqk = attn_bghqk + causal_mask_bqk[:, None, None]

        # Applies the additional attention mask, if provided.
        if mask_bqk is not None:
            attn_bghqk = attn_bghqk + mask_bqk[:, None, None]

        # Normalizes.
        attn_bghqk = F.softmax(attn_bghqk, dim=-1)

        return attn_bghqk


class TransformerEncoderLayer(nn.Module):
    """Defines a transformer encoder layer.

    This layer is a drop-in replacement for ``nn.TransformerEncoderLayer``
    except that it returns the attention state for causal attention, which can
    be used to implement streaming inference.

    Parameters:
        d_model: The input and output embedding dimension.
        head_dims: The number of dimensions in each attention head.
        feedforward_factor: The factor by which the input number of dimensions
            is multiplied to get the feedforward hidden dimension.
        dropout: The dropout probability, applied to the attention matrix.
        norm_eps: The layer normalization epsilon value.
        norm_first: Whether to apply layer normalization before the attention
            layer.
        norm_type: The type of normalization to use.
        gqa_factor: The GQA factor to use, meaning the ratio of the number of
            queries to the number of keys. Higher values will result in more
            queries than keys, which can speed up inference.
        max_kv_cache_len: The maximum number of previous timesteps to cache
            for the key and value tensors. If ``None``, don't clip the maximum
            length.

    Inputs:
        src_btc: The input tensor, of shape ``(B, T, C)``.
        state: The previous state tensor, if applicable.
        is_causal: Whether to apply a causal mask to the attention matrix.
            Note that the "mask" is only applied implicitly and isn't actually
            instantiated as a tensor.
        rotary_q_2tc: The rotary embeddings for the query tensor, of shape
            ``(2, T, C // H)``. If ``None``, don't apply rotary embeddings.
        rotary_k_2tc: The rotary embeddings for the key tensor, of shape
            ``(2, T, C // H)``. If ``None``, don't apply rotary embeddings.
        mask_btt: The attention mask, of shape ``(B, T, T)``. If ``None``,
            don't apply an attention mask.

    Outputs:
        output_btc: The output tensor, of shape ``(B, T, C)``.
        state: The next state tensor.
    """

    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        head_dims: int = 64,
        feedforward_factor: int = 4,
        dropout: float = 0.1,
        norm_eps: float = 1e-5,
        norm_first: bool = True,
        norm_type: Literal["layer", "rms"] = "rms",
        gqa_factor: int = 1,
        max_kv_cache_len: int | None = None,
    ) -> None:
        super().__init__()

        # Stores some constant values.
        self.max_kv_cache_len = max_kv_cache_len

        # Self-attention layer.
        self.self_attn = MultiheadAttention(
            d_model,
            head_dims,
            dropout=dropout,
            gqa_factor=gqa_factor,
        )

        # Feed-forward layers.
        self.linear1 = nn.Linear(d_model, d_model * feedforward_factor)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model * feedforward_factor, d_model)

        # Extras (norms and dropout).
        self.norm_first = norm_first
        self.norm1 = get_norm_linear(norm_type, dim=d_model, eps=norm_eps)
        self.norm2 = get_norm_linear(norm_type, dim=d_model, eps=norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        src_btc: Tensor,
        state: Tensor | None = None,
        is_causal: bool = False,
        rotary_q_2tc: Tensor | None = None,
        rotary_k_2tc: Tensor | None = None,
        mask_btt: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        x_btc = src_btc
        if self.norm_first:
            xi_btc = self.norm1(x_btc)
            xi_btc, state = self._sa_block(xi_btc, state, is_causal, rotary_q_2tc, rotary_k_2tc, mask_btt)
            x_btc = x_btc + xi_btc
            x_btc = x_btc + self._ff_block(self.norm2(x_btc))
        else:
            xi_btc, state = self._sa_block(x_btc, state, is_causal, rotary_q_2tc, rotary_k_2tc, mask_btt)
            x_btc = self.norm1(x_btc + xi_btc)
            x_btc = self.norm2(x_btc + self._ff_block(x_btc))
        return x_btc, state

    def _get_qkv(
        self,
        x_btc: Tensor,
        state: Tensor | None,
        is_causal: bool,
        rotary_q_2tc: Tensor | None = None,
        rotary_k_2tc: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        xq_btc, xk_btc, xv_btc = self.self_attn.forward_matmuls(x_btc, x_btc, x_btc, rotary_q_2tc, rotary_k_2tc)

        # Concatenates previous states
        if state is not None:
            if is_causal:
                raise ValueError(
                    "Causal attention with state will lead to incorrect results. Instead, when unrolling the "
                    "attention component, set `is_causal=False` and pass samples one-at-a-time."
                )
            if x_btc.size(1) != 1:
                raise ValueError(
                    "Using a state implies that you are using causal attention, but you are passing multiple query "
                    "vectors. Instead, when unrolling the attention component, set `is_causal=False` and pass "
                    "samples one-at-a-time."
                )

            prev_k, prev_v = state.unbind(0)
            xk_btc = torch.cat((prev_k, xk_btc), dim=-2)
            xv_btc = torch.cat((prev_v, xv_btc), dim=-2)
            if self.max_kv_cache_len is not None:
                xk_btc = xk_btc[:, -self.max_kv_cache_len :]
                xv_btc = xv_btc[:, -self.max_kv_cache_len :]

        return xq_btc, xk_btc, xv_btc

    def _sa_block(
        self,
        x_btc: Tensor,
        state: Tensor | None,
        is_causal: bool,
        rotary_q_2tc: Tensor | None = None,
        rotary_k_2tc: Tensor | None = None,
        mask_btt: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        xq_btc, xk_btc, xv_btc = self._get_qkv(x_btc, state, is_causal, rotary_q_2tc, rotary_k_2tc)
        x_btc = self.self_attn.forward_attn(xq_btc, xk_btc, xv_btc, is_causal, mask_btt)
        return self.dropout1(x_btc), torch.stack((xk_btc, xv_btc), dim=0)

    def _ff_block(self, x_btc: Tensor) -> Tensor:
        x_btc = self.linear2(self.dropout(self.activation(self.linear1(x_btc))))
        return self.dropout2(x_btc)


class TransformerDecoderLayer(nn.Module):
    """Defines a transformer decoder layer.

    Unlike the PyTorch decoder layer, this layer only contains cross-attention.
    To mimic the original behavior, pair this layer with a self-attention
    layer.

    Parameters:
        d_model: The input and output embedding dimension.
        head_dims: The number of dimensions in each attention head.
        feedforward_factor: The factor by which the input number of dimensions
            is multiplied to get the feedforward hidden dimension.
        dropout: The dropout probability, applied to the attention matrix.
        norm_eps: The layer normalization epsilon value.
        norm_first: Whether to apply layer normalization before the attention
            layer.
        gqa_factor: The GQA factor to use, meaning the ratio of the number of
            queries to the number of keys. Higher values will result in more
            queries than keys, which can speed up inference.
        memory_dims: The number of dimensions in the memory tensor; if not
            provided, defaults to ``d_model``.

    Inputs:
        src_bqc: The input tensor, of shape ``(B, Tq, C)``.
        memory_bkc: The memory tensor, of shape ``(B, Tk, C)``
        state: The previous state tensor, if applicable.
        rotary_q_2qc: The rotary embeddings for the query tensor, of shape
            ``(2, Tq, C // H)``. If ``None``, don't apply rotary embeddings.
        rotary_k_2qc: The rotary embeddings for the key tensor, of shape
            ``(2, Tq, C // H)``. If ``None``, don't apply rotary embeddings.
        mask_bqk: The attention mask, of shape ``(B, Tq, Tk)``. If ``None``, don't
            apply an attention mask.

    Outputs:
        output_bqc: The output tensor, of shape ``(B, Tq, C)``.
        state: The next state tensor.
    """

    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        head_dims: int = 64,
        feedforward_factor: int = 4,
        dropout: float = 0.1,
        norm_eps: float = 1e-5,
        norm_type: Literal["layer", "rms"] = "rms",
        norm_first: bool = True,
        gqa_factor: int = 1,
        memory_dims: int | None = None,
    ) -> None:
        super().__init__()

        # Self-attention layer.
        self.cross_attn = MultiheadAttention(
            d_model,
            head_dims,
            dropout=dropout,
            gqa_factor=gqa_factor,
            kdim=memory_dims,
            vdim=memory_dims,
        )

        # Feed-forward layers.
        self.linear1 = nn.Linear(d_model, d_model * feedforward_factor)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model * feedforward_factor, d_model)

        # Extras (norms and dropout).
        self.norm_first = norm_first
        self.norm1 = get_norm_linear(norm_type, dim=d_model, eps=norm_eps)
        self.norm2 = get_norm_linear(norm_type, dim=d_model, eps=norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        src_bqc: Tensor,
        memory_bkc: Tensor,
        state: Tensor | None = None,
        mask_bqk: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        x_bqc = src_bqc
        if self.norm_first:
            xi_bqc = self.norm1(x_bqc)
            xi_bqc, state = self._sa_block(xi_bqc, memory_bkc, state, mask_bqk)
            x_bqc = x_bqc + xi_bqc
            x_bqc = x_bqc + self._ff_block(self.norm2(x_bqc))
        else:
            xi_bqc, state = self._sa_block(x_bqc, memory_bkc, state, mask_bqk)
            x_bqc = self.norm1(x_bqc + xi_bqc)
            x_bqc = self.norm2(x_bqc + self._ff_block(x_bqc))
        return x_bqc, state

    def _get_qkv(self, x_bqc: Tensor, memory_bkc: Tensor, state: Tensor | None) -> tuple[Tensor, Tensor, Tensor]:
        if state is None:
            xq_bqc, xk_bkc, xv_bkc = self.cross_attn.forward_matmuls(x_bqc, memory_bkc, memory_bkc)
            state = torch.stack((xk_bkc, xv_bkc))
        else:
            xq_bqc, _, _ = self.cross_attn.forward_matmuls(x_bqc, None, None)
            xk_bkc, xv_bkc = state.unbind(0)
        return xq_bqc, xk_bkc, xv_bkc

    def _sa_block(
        self,
        x_bqc: Tensor,
        memory_bkc: Tensor,
        state: Tensor | None,
        mask_bqk: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        xq_bqc, xk_bkc, xv_bkc = self._get_qkv(x_bqc, memory_bkc, state)
        x_bqc = self.cross_attn.forward_attn(xq_bqc, xk_bkc, xv_bkc, mask_bqk=mask_bqk)
        if state is None:
            state = torch.stack((xk_bkc, xv_bkc), dim=0)
        return self.dropout1(x_bqc), state

    def _ff_block(self, x_bqc: Tensor) -> Tensor:
        x_bqc = self.linear2(self.dropout(self.activation(self.linear1(x_bqc))))
        return self.dropout2(x_bqc)


class TransformerEncoder(nn.Module):
    """Defines a transformer encoder.

    This is a drop-in replacement for ``nn.TransformerEncoder`` except that it
    returns the attention state for causal attention, which can be used to
    implement streaming inference.

    This additionally supports using rotary embeddings for the key-query
    matrix multiplications. The rotary embedding tensors are computed at
    runtime and cached.

    Parameters:
        encoder_layer: The encoder layer to use.
        num_layers: The number of encoder layers.
        is_causal: Default value for ``is_causal`` in the ``forward`` method
            if not supplied. Controls causal verses bidirectional attention.
        use_rotary: Default value for ``use_rotary`` in the ``forward`` method
            if not supplied. Controls the use of rotary embeddings in the
            key-query matrix multiplication.
        rotary_base: The base value for rotary embeddings.

    Inputs:
        src: The input tensor, of shape ``(B, T, C)``.
        state: The previous state tensor, if applicable.
        is_causal: Whether to apply a causal mask to the attention matrix.
            Note that the "mask" is only applied implicitly and isn't actually
            instantiated as a tensor.
        use_rotary: If set, use rotary embeddings in the key-query matrix
            multiplication.
        mask: The attention mask, of shape ``(B, Tq, Tk)``. If ``None``, don't
            apply an attention mask.

    Outputs:
        output: The output tensor, of shape ``(B, T, C)``.
        state: The previous state tensor, if applicable.
    """

    __constants__ = ["num_heads", "head_dim", "kdim", "vdim", "embed_dim", "is_causal", "use_rotary", "rotary_base"]

    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        is_causal: bool | None = None,
        use_rotary: bool = False,
        rotary_base: int = 10_000,
    ) -> None:
        super().__init__()

        # Keeps some constant values in the top-level layer.
        self.num_heads = encoder_layer.self_attn.num_heads
        self.head_dim = encoder_layer.self_attn.head_dim
        self.embed_dim = encoder_layer.self_attn.embed_dim
        self.is_causal = is_causal
        self.use_rotary = use_rotary
        self.rotary_base = rotary_base

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.rotary_q: Tensor | None = None
        self.rotary_k: Tensor | None = None

    def _get_rotary_embeddings(
        self,
        q_tsz: int,
        k_tsz: int,
        state: Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        if state is None:
            if self.rotary_q is None or self.rotary_q.shape[-2] < q_tsz:
                self.rotary_q = get_rotary_embeddings(q_tsz, self.head_dim, device, dtype, 0, self.rotary_base)
            if self.rotary_k is None or self.rotary_k.shape[-2] < k_tsz:
                self.rotary_k = get_rotary_embeddings(k_tsz, self.head_dim, device, dtype, 0, self.rotary_base)
            return self.rotary_q[..., :q_tsz, :], self.rotary_k[..., :k_tsz, :]

        else:
            offset = state.shape[-2]
            rotary_q = get_rotary_embeddings(q_tsz, self.head_dim, device, dtype, offset, self.rotary_base)
            rotary_k = get_rotary_embeddings(k_tsz, self.head_dim, device, dtype, offset, self.rotary_base)
            return rotary_q, rotary_k

    def _default(self, *values: bool | None) -> bool:
        for value in values:
            if value is not None:
                return value
        return False  # Arbitrary.

    def forward(
        self,
        src_btc: Tensor,
        state: Tensor | None = None,
        is_causal: bool | None = None,
        use_rotary: bool | None = None,
        mask_btt: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        is_causal = self._default(is_causal, self.is_causal, state is None)
        use_rotary = self._default(use_rotary, self.use_rotary)

        output_btc = src_btc
        state_out = []
        _, tsz, _ = src_btc.shape
        rotary_q_2tc, rotary_k_2tc = (
            self._get_rotary_embeddings(
                q_tsz=tsz,
                k_tsz=tsz,
                state=state,
                device=src_btc.device,
                dtype=src_btc.dtype,
            )
            if use_rotary
            else (None, None)
        )
        for i, layer in enumerate(self.layers):
            state_i = None if state is None else state[i]
            output_btc, state_o_i = layer.forward(output_btc, state_i, is_causal, rotary_q_2tc, rotary_k_2tc, mask_btt)
            state_out.append(state_o_i)
        return output_btc, torch.stack(state_out, dim=0)


class TransformerDecoder(nn.Module):
    """Defines a transformer decoder.

    Parameters:
        encoder_layer: The encoder layer to use.
        num_layers: The number of encoder layers.
        is_causal: Default value for ``is_causal`` in the ``forward`` method
            if not supplied. Controls causal verses bidirectional attention.
        use_rotary: Default value for ``use_rotary`` in the ``forward`` method
            if not supplied. Controls the use of rotary embeddings in the
            key-query matrix multiplication.
        rotary_base: The base value for rotary embeddings.

    Inputs:
        src_bqc: The input tensor, of shape ``(B, Tq, C)``.
        memory_bkc: The memory tensor, of shape ``(B, Tk, C)``.
        state: The previous state tensor, if applicable.
        is_causal: Whether to apply a causal mask to the attention matrix.
            Note that the "mask" is only applied implicitly and isn't actually
            instantiated as a tensor.
        use_rotary: If set, use rotary embeddings in the key-query matrix
            multiplication.
        encoder_mask_bqq: The encoder attention mask, of shape ``(B, Tq, Tq)``.
            If ``None``, don't apply an attention mask to the encoder.
        decoder_mask_bqk: The decoder attention mask, of shape ``(B, Tq, Tk)``.
            If ``None``, don't apply an attention mask to the decoder.

    Outputs:
        output_bqc: The output tensor, of shape ``(B, Tq, C)``.
        state: The previous state tensor, if applicable.
    """

    __constants__ = ["num_heads", "head_dim", "kdim", "vdim", "embed_dim", "is_causal", "use_rotary", "rotary_base"]

    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        decoder_layer: TransformerDecoderLayer,
        num_layers: int,
        is_causal: bool | None = None,
        use_rotary: bool = False,
        rotary_base: int = 10_000,
    ) -> None:
        super().__init__()

        if encoder_layer.self_attn.embed_dim != decoder_layer.cross_attn.embed_dim:
            raise ValueError("Embedding dimensions for encoder and decoder layers do not match!")

        # Keeps some constant values in the top-level layer.
        self.enc_num_heads = encoder_layer.self_attn.num_heads
        self.enc_head_dim = encoder_layer.self_attn.head_dim
        self.dec_num_heads = encoder_layer.self_attn.num_heads
        self.dec_head_dim = encoder_layer.self_attn.head_dim
        self.embed_dim = encoder_layer.self_attn.embed_dim
        self.is_causal = is_causal
        self.use_rotary = use_rotary
        self.rotary_base = rotary_base

        self.encoder_layers = _get_clones(encoder_layer, num_layers)
        self.decoder_layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.rotary_q: Tensor | None = None
        self.rotary_k: Tensor | None = None

    def _get_rotary_embeddings(
        self,
        q_tsz: int,
        k_tsz: int,
        state: Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        if state is None:
            if self.rotary_q is None or self.rotary_q.shape[-2] < q_tsz:
                self.rotary_q = get_rotary_embeddings(q_tsz, self.enc_head_dim, device, dtype, 0, self.rotary_base)
            if self.rotary_k is None or self.rotary_k.shape[-2] < k_tsz:
                self.rotary_k = get_rotary_embeddings(k_tsz, self.enc_head_dim, device, dtype, 0, self.rotary_base)
            return self.rotary_q[..., :q_tsz, :], self.rotary_k[..., :k_tsz, :]

        else:
            offset = state.shape[-2]
            rotary_q = get_rotary_embeddings(q_tsz, self.enc_head_dim, device, dtype, offset, self.rotary_base)
            rotary_k = get_rotary_embeddings(k_tsz, self.enc_head_dim, device, dtype, offset, self.rotary_base)
            return rotary_q, rotary_k

    def _default(self, *values: bool | None) -> bool:
        for value in values:
            if value is not None:
                return value
        return False  # Arbitrary.

    def forward(
        self,
        src_bqc: Tensor,
        memory_bkc: Tensor,
        state: tuple[Tensor, Tensor] | None = None,
        is_causal: bool | None = None,
        use_rotary: bool | None = None,
        encoder_mask_bqq: Tensor | None = None,
        decoder_mask_bqk: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        is_causal = self._default(is_causal, self.is_causal, state is None)
        use_rotary = self._default(use_rotary, self.use_rotary)

        output_bqc = src_bqc
        e_state_out = []
        d_state_out = []
        _, tsz, _ = src_bqc.shape
        rotary_q_2qc, rotary_k_2kc = (
            self._get_rotary_embeddings(
                q_tsz=tsz,
                k_tsz=tsz,
                state=None if state is None else state[0],
                device=src_bqc.device,
                dtype=src_bqc.dtype,
            )
            if use_rotary
            else (None, None)
        )
        for i, (e_layer, d_layer) in enumerate(zip(self.encoder_layers, self.decoder_layers)):
            e_state_i, d_state_i = (None, None) if state is None else (state[0][i], state[1][i])
            output_bqc, e_state_out_i = e_layer.forward(
                output_bqc,
                e_state_i,
                is_causal,
                rotary_q_2qc,
                rotary_k_2kc,
                encoder_mask_bqq,
            )
            e_state_out.append(e_state_out_i)
            output_bqc, d_state_out_i = d_layer.forward(output_bqc, memory_bkc, d_state_i, decoder_mask_bqk)
            d_state_out.append(d_state_out_i)
        return output_bqc, (torch.stack(e_state_out, dim=0), torch.stack(d_state_out, dim=0))


class NextTokenTransformer(nn.Module):
    """Defines a next token prediction transformer module.

    This seems to be the most popular architecture for solving a large number
    of problems. This provides a tested implementation of the next token
    prediction transformer.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        vocab_size: int,
        head_dims: int = 64,
        feedforward_factor: int = 4,
        dropout: float = 0.1,
        norm_eps: float = 1e-5,
        norm_type: Literal["layer", "rms"] = "rms",
        norm_first: bool = True,
        final_norm_type: Literal["layer", "rms", "no_norm"] = "rms",
        gqa_factor: int = 1,
        max_kv_cache_len: int | None = None,
        use_rotary: bool = True,
        rotary_base: int = 10_000,
    ) -> None:
        super().__init__()

        self.init_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.attn = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=d_model,
                head_dims=head_dims,
                feedforward_factor=feedforward_factor,
                dropout=dropout,
                norm_eps=norm_eps,
                norm_type=norm_type,
                norm_first=norm_first,
                gqa_factor=gqa_factor,
                max_kv_cache_len=max_kv_cache_len,
            ),
            num_layers=num_layers,
            use_rotary=use_rotary,
            rotary_base=rotary_base,
        )
        self.proj = nn.Sequential(
            get_norm_linear(final_norm_type, dim=d_model, eps=norm_eps),
            nn.Linear(d_model, vocab_size),
        )

    def forward(self, tokens_bt: Tensor) -> Tensor:
        x_btc = self.embeddings(tokens_bt[:, :-1])
        x_btc = torch.cat((self.init_emb.expand(x_btc.size(0), 1, -1), x_btc), dim=1)
        x_btc, _ = self.attn(x_btc, is_causal=True)
        logits_btc = self.proj(x_btc)
        return logits_btc

    def infer(
        self,
        t: int,
        bsz: int = 1,
        sampling_strategy: SamplingStrategy = "top-p",
        k: int | None = None,
        p: float | None = 0.95,
        temperature: float = 1.0,
    ) -> Tensor:
        x_b1c: Tensor = self.init_emb.expand(bsz, 1, -1)
        x_b1c, state = self.attn(x_b1c)
        logits_b1l = self.proj(x_b1c)
        tokens_bt = sample_from_logits(logits_b1l, sampling_strategy, k=k, p=p, temperature=temperature)
        tokens_b1 = tokens_bt[:, :1]

        for _ in range(1, t):
            x_b1c = self.embeddings(tokens_b1)
            x_b1c, state = self.attn(x_b1c, state)
            logits_b1l = self.proj(x_b1c)
            tokens_b1 = sample_from_logits(logits_b1l, sampling_strategy, k=k, p=p, temperature=temperature)
            tokens_bt = torch.cat((tokens_bt, tokens_b1), dim=1)

        return tokens_bt


class NextTokenWithEmbeddingsTransformer(nn.Module):
    """Defines a next token prediction transformer module, over base embeddings.

    This is similar to the ``NextTokenTransformer`` except that each of the
    input timesteps also has an associated embedding, which is added to the
    input before the transformer.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        vocab_size: int,
        head_dims: int = 64,
        feedforward_factor: int = 4,
        dropout: float = 0.1,
        norm_eps: float = 1e-5,
        norm_type: Literal["layer", "rms"] = "rms",
        norm_first: bool = True,
        final_norm_type: Literal["layer", "rms", "no_norm"] = "rms",
        gqa_factor: int = 1,
        max_kv_cache_len: int | None = None,
        use_rotary: bool = True,
        rotary_base: int = 10_000,
    ) -> None:
        super().__init__()

        self.init_emb = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.attn = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=d_model,
                head_dims=head_dims,
                feedforward_factor=feedforward_factor,
                dropout=dropout,
                norm_eps=norm_eps,
                norm_type=norm_type,
                norm_first=norm_first,
                gqa_factor=gqa_factor,
                max_kv_cache_len=max_kv_cache_len,
            ),
            num_layers=num_layers,
            use_rotary=use_rotary,
            rotary_base=rotary_base,
        )
        self.proj = nn.Sequential(
            get_norm_linear(final_norm_type, dim=d_model, eps=norm_eps),
            nn.Linear(d_model, vocab_size),
        )

    def forward(self, tokens_bt: Tensor, emb_btc: Tensor) -> tuple[Tensor, Tensor]:
        x_btc = self.embeddings(tokens_bt[:, :-1])
        x_btc = torch.cat((self.init_emb.expand(x_btc.size(0), 1, -1), x_btc), dim=1)
        x_btc = x_btc + emb_btc
        x_btc, _ = self.attn(x_btc, is_causal=True)
        logits_btc = self.proj(x_btc)
        return logits_btc, x_btc

    def infer(
        self,
        emb_btc: Tensor,
        sampling_strategy: SamplingStrategy = "top-p",
        k: int | None = None,
        p: float | None = 0.95,
        temperature: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        x_b1c: Tensor = self.init_emb.expand(emb_btc.size(0), 1, -1)
        x_b1c = x_b1c + emb_btc[:, :1]
        x_b1c, state = self.attn(x_b1c)
        logits_b1l = self.proj(x_b1c)
        tokens_bt = sample_from_logits(logits_b1l, sampling_strategy, k=k, p=p, temperature=temperature)
        tokens_b1 = tokens_bt[:, :1]

        x_list_btc = [x_b1c]
        for t in range(1, emb_btc.size(1)):
            x_b1c = self.embeddings(tokens_b1) + emb_btc[:, t : t + 1]
            x_b1c, state = self.attn(x_b1c, state)
            x_list_btc.append(x_b1c)
            logits_b1l = self.proj(x_b1c)
            tokens_b1 = sample_from_logits(logits_b1l, sampling_strategy, k=k, p=p, temperature=temperature)
            tokens_bt = torch.cat((tokens_bt, tokens_b1), dim=1)

        return tokens_bt, torch.cat(x_list_btc, dim=1)


T = TypeVar("T", bound=nn.Module)


def _get_clones(module: T, num_layers: int) -> list[T]:
    return cast(list[T], nn.ModuleList([copy.deepcopy(module) for _ in range(num_layers)]))
