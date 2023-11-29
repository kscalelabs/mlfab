"""Tests the finite scalar quantization module."""

import torch

import mlfab


def test_fsq() -> None:
    dim = 3
    bsz = 7
    tsz = 5
    levels = [8, 6, 5]

    fsq = mlfab.FiniteScalarQuantization(levels=levels)

    x = torch.randn(bsz, tsz, dim)
    quantized = fsq.forward(x)
    assert quantized.shape == x.shape

    # Converts from indices to codes and back.
    indices = fsq.codes_to_indices(quantized)
    assert indices.shape == (bsz, tsz)
    codes = fsq.indices_to_codes(indices)
    assert torch.allclose(codes, quantized)
