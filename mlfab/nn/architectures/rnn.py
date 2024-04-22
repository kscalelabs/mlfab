"""Defines some utility functions specific to RNNs."""

import torch
from torch import Tensor, nn

from mlfab.nn.architectures.next_token import SamplingStrategy, sample_from_logits


class NextTokenGru(nn.Module):
    """Defines a next token prediction GRU module.

    This seems to be the most popular architecture for solving a large number
    of problems. This provides a tested implementation of the next token
    prediction GRU.
    """

    def __init__(
        self,
        input_size: int,
        num_layers: int,
        vocab_size: int,
        hidden_size: int | None = None,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_size is None:
            hidden_size = input_size

        self.init_emb = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.embeddings = nn.Embedding(vocab_size, input_size)

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            bidirectional=False,
            dropout=dropout,
        )
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens_bt: Tensor) -> Tensor:
        x_btc = self.embeddings(tokens_bt[:, :-1])
        x_btc = torch.cat((self.init_emb.expand(x_btc.size(0), 1, -1), x_btc), dim=1)
        x_btc, _ = self.rnn(x_btc)
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
        x_b1c, state = self.rnn(x_b1c)
        logits_b1l = self.proj(x_b1c)
        tokens_bt = sample_from_logits(logits_b1l, sampling_strategy, k=k, p=p, temperature=temperature)
        tokens_b1 = tokens_bt[:, :1]

        for _ in range(1, t):
            x_b1c = self.embeddings(tokens_b1)
            x_b1c, state = self.rnn(x_b1c, state)
            logits_b1l = self.proj(x_b1c)
            tokens_b1 = sample_from_logits(logits_b1l, sampling_strategy, k=k, p=p, temperature=temperature)
            tokens_bt = torch.cat((tokens_bt, tokens_b1), dim=1)

        return tokens_bt


class NextTokenWithEmbeddingsGru(nn.Module):
    """Defines a next token prediction GRU module, over base embeddings.

    This is similar to the ``NextTokenGru`` except that each of the
    input timesteps also has an associated embedding, which is added to the
    input before the RNN layers.
    """

    def __init__(
        self,
        input_size: int,
        num_layers: int,
        vocab_size: int,
        hidden_size: int | None = None,
        bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_size is None:
            hidden_size = input_size

        self.init_emb = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.embeddings = nn.Embedding(vocab_size, input_size)

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            bidirectional=False,
            dropout=dropout,
        )
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens_bt: Tensor, emb_btc: Tensor) -> tuple[Tensor, Tensor]:
        x_btc = self.embeddings(tokens_bt[:, :-1])
        x_btc = torch.cat((self.init_emb.expand(x_btc.size(0), 1, -1), x_btc), dim=1)
        x_btc = x_btc + emb_btc
        x_btc, _ = self.rnn(x_btc)
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
        x_b1c, state = self.rnn(x_b1c)
        logits_b1l = self.proj(x_b1c)
        tokens_bt = sample_from_logits(logits_b1l, sampling_strategy, k=k, p=p, temperature=temperature)
        tokens_b1 = tokens_bt[:, :1]

        x_list_btc = [x_b1c]
        for t in range(1, emb_btc.size(1)):
            x_b1c = self.embeddings(tokens_b1) + emb_btc[:, t : t + 1]
            x_b1c, state = self.rnn(x_b1c, state)
            x_list_btc.append(x_b1c)
            logits_b1l = self.proj(x_b1c)
            tokens_b1 = sample_from_logits(logits_b1l, sampling_strategy, k=k, p=p, temperature=temperature)
            tokens_bt = torch.cat((tokens_bt, tokens_b1), dim=1)

        return tokens_bt, torch.cat(x_list_btc, dim=1)
