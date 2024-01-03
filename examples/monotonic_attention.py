# mypy: disable-error-code="import-not-found"
"""Defines a dummy "letters" which is solvable by the monotonic attention model.

This model takes a random sequence of letters and outputs a new sequence
containing the unique letters repeated N times. For example, the input sequence
"abbbccdef" would be transformed into "aabbccddeeff".

Run this example using `python -m examples.monotonic_attention`.
"""

import random
from dataclasses import dataclass
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data.dataset import Dataset, IterableDataset

import mlfab
from mlfab.core.state import Phase, State
from mlfab.task.mixins.optimizer import OptType
from mlfab.task.mixins.train import Batch, Loss, Output
from mlfab.utils.data.collate import CollateMode

PADDING_IDX = 0


class Tokenizer:
    def __init__(self, num_letters: int) -> None:
        super().__init__()

        assert 2 <= num_letters <= 26, f"`{num_letters=}` must be between 2 and 26"

        self.num_letters = num_letters
        self.vocab = list("abcdefghijklmnopqrstuvwxyz"[:num_letters])

    def tokenize(self, s: str) -> Tensor:
        return Tensor([self.vocab.index(c) + 1 for c in s])

    def detokenize(self, t: Tensor) -> str:
        return "".join(self.vocab[int(i) - 1] for i in t.tolist())

    @property
    def vocab_size(self) -> int:
        return self.num_letters + 1


class LettersDataset(IterableDataset[tuple[Tensor, Tensor]]):
    def __init__(self, tokenizer: Tokenizer, seq_length: int) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        return self

    def __next__(self) -> tuple[Tensor, Tensor]:
        tokens_in: list[int] = []
        tokens_out: list[int] = []
        prev_letter: int | None = None
        while len(tokens_in) < self.seq_length:
            choices = [i for i in range(1, self.tokenizer.num_letters + 1) if i != prev_letter]
            letter = random.choice(choices)
            prev_letter = letter
            tokens_in.extend([letter] * min(self.seq_length - len(tokens_in), random.randint(2, 15)))
            tokens_out.extend([letter])

        tokens_in_t = torch.tensor(tokens_in)
        tokens_out_t = torch.tensor(tokens_out)
        return tokens_in_t, tokens_out_t

    def collate_fn(self, items: list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        tokens_in, tokens_out = zip(*items)

        # Pads the output tokens and creates a mask.
        max_out_len = max(len(t) for t in tokens_out)
        tokens_out_t = torch.full((len(tokens_out), max_out_len), fill_value=PADDING_IDX, dtype=torch.long)
        for i, token_out in enumerate(tokens_out):
            tokens_out_t[i, : len(token_out)] = token_out

        return torch.stack(tokens_in), tokens_out_t


class MonotonicSeq2Seq(nn.Module):
    """Defines a monotonic sequence-to-sequence model.

    Parameters:
        vocab_size: The vocabulary size
        dim: The number of embedding dimensions
    """

    def __init__(self, vocab_size: int, dim: int, use_rnn: bool) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.embs = nn.Embedding(vocab_size, dim, padding_idx=PADDING_IDX)
        self.init_emb = nn.Parameter(torch.zeros(1, 1, dim))
        self.rnn = nn.LSTM(dim, dim, batch_first=True) if use_rnn else None
        self.attn = mlfab.MonotonicAttention("many_keys_one_query", dim, num_heads=1)
        self.proj = nn.Linear(dim, vocab_size)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        bsz = tgt.size(0)
        src_emb = self.embs(src)
        tgt_emb = torch.cat((self.init_emb.expand(bsz, -1, -1), self.embs(tgt[..., :-1])), dim=1)
        x = self.attn(tgt_emb, src_emb, src_emb)
        if self.rnn is not None:
            x, _ = self.rnn(x)
        x = self.proj(x)
        return x

    def get_attention_matrix(self, src: Tensor, tgt: Tensor) -> Tensor:
        bsz = tgt.size(0)
        src_emb = self.embs(src)
        tgt_emb = torch.cat((self.init_emb.expand(bsz, -1, -1), self.embs(tgt[..., :-1])), dim=1)
        return self.attn.get_attn_matrix(tgt_emb, src_emb)


@dataclass
class Config(mlfab.Config):
    num_letters: int = mlfab.field(10, hlep="How many unique letters to use")
    seq_length: int = mlfab.field(64, help="Input sequence length")
    embedding_dims: int = mlfab.field(32, help="Number of embedding dimensions")
    use_rnn: bool = mlfab.field(False, help="Whether to use an RNN")


class MonotonicAttentionTask(mlfab.Task[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.tokenizer = Tokenizer(config.num_letters)
        self.model = MonotonicSeq2Seq(config.num_letters + 1, config.embedding_dims, config.use_rnn)

    def get_dataset(self, phase: Phase) -> Dataset:
        return LettersDataset(self.tokenizer, self.config.seq_length)

    def build_optimizer(self) -> OptType:
        return mlfab.Adam.get(self, lr=1e-3)

    def forward(self, tokens_in: Tensor, tokens_out: Tensor) -> Tensor:
        return self.model(tokens_in, tokens_out)

    def get_loss(self, batch: Batch, state: State) -> Loss:
        tokens_in, tokens_out = batch
        tokens_out_pred = self(tokens_in, tokens_out)
        self.log_step(batch, tokens_out_pred, state)
        return F.cross_entropy(
            tokens_out_pred.view(-1, self.config.num_letters + 1),
            tokens_out.view(-1).long(),
            ignore_index=PADDING_IDX,
        )

    def log_valid_step(self, batch: Batch, output: Output, state: State) -> None:
        tokens_in, tokens_out = batch
        letters_in = " ".join(self.tokenizer.detokenize(tokens_in[0]).split())
        letters_out = " ".join(self.tokenizer.detokenize(tokens_out[0]).split())
        attn_matrix = self.model.get_attention_matrix(tokens_in[:1], tokens_out[:1])[0, 0, 0].exp()
        self.log_labeled_image("attention", (attn_matrix, f"In: {letters_in}\nOut: {letters_out}"))

    @classmethod
    def collate_fn(cls, items: list[Any], *, mode: CollateMode = "stack") -> Any | None:  # noqa: ANN401
        tokens_in, tokens_out = zip(*items)

        # Pads the output tokens and creates a mask.
        max_out_len = max(len(t) for t in tokens_out)
        tokens_out_t = torch.full((len(tokens_out), max_out_len), fill_value=PADDING_IDX, dtype=torch.long)
        for i, token_out in enumerate(tokens_out):
            tokens_out_t[i, : len(token_out)] = token_out

        return torch.stack(tokens_in), tokens_out_t


if __name__ == "__main__":
    MonotonicAttentionTask.launch(Config(batch_size=16, num_dataloader_workers=0, valid_every_n_seconds=10))
