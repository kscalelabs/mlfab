"""Tests the embeddings API."""

from typing import get_args

import pytest
import torch

import mlfab


@pytest.mark.parametrize("kind", get_args(mlfab.EmbeddingKind))
def test_embeddings_api(kind: mlfab.EmbeddingKind) -> None:
    x = torch.randn(3, 5, 8)
    times = torch.arange(1, 6)[None, :].repeat(3, 1)
    emb = mlfab.get_positional_embeddings(max_tsz=12, embed_dim=8, kind=kind)
    y1 = emb(x, times=times)
    y2 = emb(x, offset=1)
    assert y1.shape == (3, 5, 8)
    assert y2.shape == (3, 5, 8)
    assert torch.allclose(y1, y2)


@pytest.mark.parametrize("offset", [0, 12])
def test_rotary_embeddings_inference(offset: int) -> None:
    x = torch.randn(3, 5, 8, dtype=torch.double)
    emb = mlfab.get_positional_embeddings(max_tsz=8 + offset, embed_dim=8, learnable=False, kind="rotary")
    emb = emb.double()
    y1 = emb(x, offset=offset)
    y2 = mlfab.rotary_embeddings(x, offset=offset)
    assert y1.shape == (3, 5, 8)
    assert y2.shape == (3, 5, 8)
    assert torch.allclose(y1, y2)
