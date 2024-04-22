"""Defines some utility functions specific to RNNs."""

import math
import os
import warnings
from typing import Callable, Literal, cast, get_args

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd.function import Function, FunctionCtx, once_differentiable

from mlfab.nn.architectures.next_token import SamplingStrategy, sample_from_logits
from mlfab.nn.triton import supports_triton


class NextTokenLstm(nn.Module):
    """Defines a next token prediction LSTM module.

    This seems to be the most popular architecture for solving a large number
    of problems. This provides a tested implementation of the next token
    prediction LSTM.
    """

    def __init__(
        self,
        emb_dim: int,
        num_layers: int,
        vocab_size: int,
    ) -> None:
        super().__init__()

        self.init_emb = nn.Parameter(torch.randn(1, 1, emb_dim) * 0.02)
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.rwkv = nn.LSTM(
            emb_dim=emb_dim,
            num_layers=num_layers,
            wkv_key=wkv_key,
            feedforward_factor=feedforward_factor,
        )
        self.proj = nn.Linear(emb_dim, vocab_size)

    def forward(self, tokens_bt: Tensor) -> Tensor:
        x_btc = self.embeddings(tokens_bt[:, :-1])
        x_btc = torch.cat((self.init_emb.expand(x_btc.size(0), 1, -1), x_btc), dim=1)
        x_btc, _ = self.rwkv(x_btc)
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
        x_b1c, state = self.rwkv(x_b1c)
        logits_b1l = self.proj(x_b1c)
        tokens_bt = sample_from_logits(logits_b1l, sampling_strategy, k=k, p=p, temperature=temperature)
        tokens_b1 = tokens_bt[:, :1]

        for _ in range(1, t):
            x_b1c = self.embeddings(tokens_b1)
            x_b1c, state = self.rwkv(x_b1c, state)
            logits_b1l = self.proj(x_b1c)
            tokens_b1 = sample_from_logits(logits_b1l, sampling_strategy, k=k, p=p, temperature=temperature)
            tokens_bt = torch.cat((tokens_bt, tokens_b1), dim=1)

        return tokens_bt


class NextTokenWithEmbeddingsRwkv(nn.Module):
    """Defines a next token prediction RWKV module, over base embeddings.

    This is similar to the ``NextTokenTransformer`` except that each of the
    input timesteps also has an associated embedding, which is added to the
    input before the RWKV layers.
    """

    def __init__(
        self,
        emb_dim: int,
        num_layers: int,
        vocab_size: int,
        wkv_key: WkvFnKey | None = None,
        feedforward_factor: int = 4,
    ) -> None:
        super().__init__()

        self.init_emb = nn.Parameter(torch.randn(1, 1, emb_dim) * 0.02)
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.rwkv = RwkvStack(
            emb_dim=emb_dim,
            num_layers=num_layers,
            wkv_key=wkv_key,
            feedforward_factor=feedforward_factor,
        )
        self.proj = nn.Linear(emb_dim, vocab_size)

    def forward(self, tokens_bt: Tensor, emb_btc: Tensor) -> tuple[Tensor, Tensor]:
        x_btc = self.embeddings(tokens_bt[:, :-1])
        x_btc = torch.cat((self.init_emb.expand(x_btc.size(0), 1, -1), x_btc), dim=1)
        x_btc = x_btc + emb_btc
        x_btc, _ = self.rwkv(x_btc)
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
        x_b1c, state = self.rwkv(x_b1c)
        logits_b1l = self.proj(x_b1c)
        tokens_bt = sample_from_logits(logits_b1l, sampling_strategy, k=k, p=p, temperature=temperature)
        tokens_b1 = tokens_bt[:, :1]

        x_list_btc = [x_b1c]
        for t in range(1, emb_btc.size(1)):
            x_b1c = self.embeddings(tokens_b1) + emb_btc[:, t : t + 1]
            x_b1c, state = self.rwkv(x_b1c, state)
            x_list_btc.append(x_b1c)
            logits_b1l = self.proj(x_b1c)
            tokens_b1 = sample_from_logits(logits_b1l, sampling_strategy, k=k, p=p, temperature=temperature)
            tokens_bt = torch.cat((tokens_bt, tokens_b1), dim=1)

        return tokens_bt, torch.cat(x_list_btc, dim=1)
