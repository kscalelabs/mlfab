"""Tests the finite scalar quantization module."""

import pytest
import torch

import mlfab


@pytest.mark.parametrize("num_codebooks", [1, 2])
def test_fsq(num_codebooks: int) -> None:
    dim = 3
    bsz = 7
    tsz = 5

    lfq = mlfab.LookupFreeQuantization(dim=dim, num_codebooks=num_codebooks)

    x = torch.randn(bsz, tsz, dim)
    quantized, indices, _ = lfq.forward(x)
    assert quantized.shape == x.shape
    assert indices.shape == (bsz, tsz, num_codebooks)
