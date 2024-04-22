"""Tests RNN operations."""

import torch

import mlfab


def test_next_token_gru() -> None:
    bsz, tsz = 2, 9

    model = mlfab.NextTokenGru(
        input_size=16,
        hidden_size=16,
        num_layers=2,
        vocab_size=32,
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


def test_next_token_with_embeddings_gru() -> None:
    bsz, tsz, emb_dim = 2, 9, 16

    model = mlfab.NextTokenWithEmbeddingsGru(
        input_size=emb_dim,
        hidden_size=emb_dim,
        num_layers=2,
        vocab_size=32,
    )
    model.double()
    model.eval()

    emb_btc = torch.randn(bsz, tsz, emb_dim, dtype=torch.float64)

    # Infers from the model.
    x_infer_bt, latent_infer_btc = model.infer(emb_btc, sampling_strategy="greedy")
    assert x_infer_bt.shape == (bsz, tsz)
    assert latent_infer_btc.shape == (bsz, tsz, emb_dim)

    # Gets the training logits.
    x_train_btl, latent_train_btc = model(x_infer_bt, emb_btc)
    x_train_bt = x_train_btl.argmax(-1)
    assert x_train_bt.shape == (bsz, tsz)
    assert latent_train_btc.shape == (bsz, tsz, emb_dim)

    # Compares the training and inference results.
    assert torch.allclose(x_infer_bt, x_train_bt)
