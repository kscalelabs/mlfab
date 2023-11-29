"""Tests the UNet model implementation."""

import pytest
import torch

import mlfab


@pytest.mark.parametrize("use_time", [False, True])
def test_unet_architecture(use_time: bool) -> None:
    model = mlfab.UNet(in_dim=3, embed_dim=8, dim_scales=[1, 2, 4], input_embedding_dim=8 if use_time else None)
    x = torch.randn(2, 3, 32, 32)
    t_emb = torch.randn(2, 8) if use_time else None
    out = model.forward(x, t_emb)
    assert out.shape == x.shape
