"""Tests the BiFPN model implementation."""

import torch

import mlfab


def test_bifpn_architecture() -> None:
    bsz = 2
    model = mlfab.BiFPN(input_size=[4, 8, 16], feature_size=8, num_layers=3)
    features = [torch.randn(bsz, 4, 4, 4), torch.randn(bsz, 8, 2, 2), torch.randn(bsz, 16, 1, 1)]
    out_features = model.forward(features)
    assert len(out_features) == 3
    assert out_features[0].shape == (bsz, 8, 4, 4)
    assert out_features[1].shape == (bsz, 8, 2, 2)
    assert out_features[2].shape == (bsz, 8, 1, 1)
