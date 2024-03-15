"""Unit tests for transformer layers."""

import pytest
import torch
from torch import Tensor

import mlfab


def test_top_p_sampling() -> None:
    x = torch.randn(2, 10, 13, 32)

    assert mlfab.top_p_sampling(x, 0.5).shape == (2, 10, 13)
    assert mlfab.top_p_sampling(x, 0.5, dim=2).shape == (2, 10, 32)


def test_top_k_sampling() -> None:
    x = torch.randn(2, 10, 13, 32)

    assert mlfab.top_k_sampling(x, 5).shape == (2, 10, 13)
    assert mlfab.top_k_sampling(x, 5, dim=2).shape == (2, 10, 32)


@pytest.mark.parametrize("norm_first", [False, True])
@pytest.mark.parametrize("gqa_factor", (1, 2))
def test_transformer_encoder_layer(norm_first: bool, gqa_factor: int) -> None:
    bsz = 1

    model = mlfab.TransformerEncoderLayer(
        d_model=16,
        head_dims=8,
        dropout=0.0,
        norm_first=norm_first,
        gqa_factor=gqa_factor,
    )
    model.double()
    model.eval()

    # Runs batch inference.
    x = torch.randn(bsz, 10, 16, dtype=torch.float64)
    y1, _ = model.forward(x, is_causal=True)

    # Runs streaming inference.
    y2s = []
    state: Tensor | None = None
    for i in range(10):
        xi = x[:, i : i + 1]
        y2, state = model.forward(xi, state, is_causal=False)
        y2s.append(y2)
    with pytest.raises(ValueError):
        model.forward(x[:, -1:], state, is_causal=True)
    with pytest.raises(ValueError):
        model.forward(x, state, is_causal=False)
    y2 = torch.cat(y2s, dim=1)
    assert y2.shape == (bsz, 10, 16)
    assert torch.allclose(y1, y2)


@pytest.mark.parametrize("norm_first", [False, True])
@pytest.mark.parametrize("gqa_factor", (1, 2))
def test_transformer_decoder_layer(norm_first: bool, gqa_factor: int) -> None:
    bsz = 1

    model = mlfab.TransformerDecoderLayer(
        d_model=16,
        head_dims=8,
        dropout=0.0,
        norm_first=norm_first,
        gqa_factor=gqa_factor,
    )
    model.double()
    model.eval()

    # Runs batch inference.
    x = torch.randn(bsz, 10, 16, dtype=torch.float64)
    mem = torch.randn(bsz, 7, 16, dtype=torch.float64)
    y1, state = model.forward(x, mem)
    y2, _ = model.forward(x, None, state)  # type: ignore[arg-type]
    assert torch.allclose(y1, y2)


@pytest.mark.parametrize("norm_first", [True, False])
@pytest.mark.parametrize("use_rotary", [True, False])
@pytest.mark.parametrize("gqa_factor", (1, 2))
def test_transformer_encoder_module(norm_first: bool, use_rotary: bool, gqa_factor: int) -> None:
    bsz = 1

    model = mlfab.TransformerEncoder(
        mlfab.TransformerEncoderLayer(
            d_model=16,
            head_dims=8,
            dropout=0.0,
            norm_first=norm_first,
            gqa_factor=gqa_factor,
        ),
        num_layers=1,
        use_rotary=use_rotary,
    )
    model.double()
    model.eval()

    # Runs batch inference.
    x = torch.randn(bsz, 10, 16, dtype=torch.float64)
    y1, _ = model.forward(x)
    assert y1.shape == (bsz, 10, 16)

    # Runs streaming inference.
    y2s = []
    state: Tensor | None = None
    for i in range(10):
        xi = x[:, i : i + 1]
        y2, state = model.forward(xi, state)
        y2s.append(y2)
    y2 = torch.cat(y2s, dim=1)
    assert y2.shape == (bsz, 10, 16)
    assert torch.allclose(y1, y2)


@pytest.mark.parametrize("norm_first", [True, False])
@pytest.mark.parametrize("use_rotary", [True, False])
@pytest.mark.parametrize("gqa_factor", (1, 2))
def test_transformer_decoder_module(norm_first: bool, use_rotary: bool, gqa_factor: int) -> None:
    bsz = 1

    model = mlfab.TransformerDecoder(
        mlfab.TransformerEncoderLayer(
            d_model=16,
            head_dims=8,
            dropout=0.0,
            norm_first=norm_first,
            gqa_factor=gqa_factor,
        ),
        mlfab.TransformerDecoderLayer(
            d_model=16,
            head_dims=8,
            dropout=0.0,
            norm_first=norm_first,
            gqa_factor=gqa_factor,
        ),
        num_layers=1,
        use_rotary=use_rotary,
    )
    model.double()
    model.eval()

    # Runs batch inference.
    x = torch.randn(bsz, 10, 16, dtype=torch.float64)
    mem = torch.randn(bsz, 7, 16, dtype=torch.float64)
    y1, _ = model.forward(x, mem)
    assert y1.shape == (bsz, 10, 16)

    # Runs streaming inference.
    y2s = []
    state: tuple[Tensor, Tensor] | None = None
    for i in range(10):
        xi = x[:, i : i + 1]
        y2, state = model.forward(xi, mem, state)
        y2s.append(y2)
    y2 = torch.cat(y2s, dim=1)
    assert y2.shape == (bsz, 10, 16)
    assert torch.allclose(y1, y2)


def test_next_token_transformer() -> None:
    bsz, tsz = 2, 9

    model = mlfab.NextTokenTransformer(
        d_model=16,
        num_layers=2,
        vocab_size=32,
        head_dims=8,
        dropout=0.0,
    )
    model.double()
    model.eval()

    # Infers from the model.
    x_infer_bt = model.infer(tsz, bsz=bsz, sampling_strategy="greedy")
    assert x_infer_bt.shape == (bsz, tsz)

    # Gets the training logits.
    x_train_btl = model(x_infer_bt)
    x_train_bt = x_train_btl.argmax(-1)
    assert x_train_bt.shape == (bsz, tsz)

    # Compares the training and inference results.
    assert torch.allclose(x_infer_bt, x_train_bt)


def test_next_token_with_embeddings_transformer() -> None:
    bsz, tsz, d_model = 2, 9, 16

    model = mlfab.NextTokenWithEmbeddingsTransformer(
        d_model=d_model,
        num_layers=2,
        vocab_size=32,
        head_dims=8,
        dropout=0.0,
    )
    model.double()
    model.eval()

    emb_btc = torch.randn(bsz, tsz, d_model, dtype=torch.float64)

    # Infers from the model.
    x_infer_bt = model.infer(emb_btc, bsz=bsz, sampling_strategy="greedy")
    assert x_infer_bt.shape == (bsz, tsz)

    # Gets the training logits.
    x_train_btl = model(x_infer_bt, emb_btc)
    x_train_bt = x_train_btl.argmax(-1)
    assert x_train_bt.shape == (bsz, tsz)

    # Compares the training and inference results.
    assert torch.allclose(x_infer_bt, x_train_bt)
