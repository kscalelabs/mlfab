"""Tests the codebook learning module.

This test checks that the codebook module is working as intended.
"""

import torch

import mlfab


def test_residual_vq() -> None:
    dim = 32
    codebook_size = 24
    codebook_dim = 16
    num_quantizers = 3
    bsz = 7
    tsz = 5

    vq = mlfab.VectorQuantization(
        dim=dim,
        codebook_size=codebook_size,
        codebook_dim=codebook_dim,
    )
    rvq = mlfab.ResidualVectorQuantization(vq, num_quantizers=num_quantizers)

    x = torch.randn(bsz, tsz, dim)
    quantized, indices, loss, _ = rvq.forward(x)

    assert quantized.shape == x.shape
    assert indices.shape == (num_quantizers, bsz, tsz)
    assert loss.numel() == num_quantizers

    enc = rvq.encode(x)
    assert enc.shape == (bsz, tsz, num_quantizers)

    dec = rvq.decode(enc)
    assert dec.shape == x.shape

    # Checks that all the parameters in the module recieve non-zero gradients.
    combined = loss.sum() + quantized.sum()
    combined.backward(torch.ones_like(combined))
    for n, p in rvq.named_parameters():
        assert p.grad is not None, f"{n} has no gradient"
        assert p.grad.sum().abs() > 1e-3, f"{n} has zero gradient"
