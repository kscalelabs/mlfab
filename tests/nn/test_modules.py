"""Tests shared modules."""

import pytest
import torch
from torch import Tensor

import mlfab


@pytest.mark.slow
@pytest.mark.parametrize("tsz", [32, 37, 51])
@pytest.mark.parametrize("ksize", [2, 3, 5, 7])
@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("padding", [0, 1, 2, 3])
@pytest.mark.parametrize("dilation", [1, 2, 3])
@pytest.mark.parametrize("chunk_size", [2, 3, 5, 33])
def test_streaming_conv_vanilla(
    tsz: int,
    ksize: int,
    stride: int,
    padding: int,
    dilation: int,
    chunk_size: int,
) -> None:
    bsz, cin, cout, groups = 2, 3, 6, 3

    # Deterministic values rather than random numbers can be easier to debug.
    # x = torch.arange(tsz, dtype=torch.double)[None, None, :].expand(bsz, cin, tsz, implicit=True)
    # weight = torch.arange(ksize, dtype=torch.double)[None, None, :].expand(cout, cin // groups, ksize, implicit=True)
    # bias = None

    x = torch.randn(bsz, cin, tsz, dtype=torch.double)
    weight = torch.randn(cout, cin // groups, ksize, dtype=torch.double)
    bias = torch.randn(cout, dtype=torch.double)

    batch_y = torch.conv1d(x, weight, bias, stride, padding, dilation, groups)
    infer_ys = []
    state: tuple[Tensor, int] | None = None
    s = 0
    for xi in x.split(chunk_size, dim=-1):
        infer_yi, state = mlfab.streaming_conv_1d(xi, state, weight, bias, stride, padding, dilation, groups)
        batch_yi = batch_y[..., s : s + infer_yi.shape[-1]]
        assert batch_yi.shape == infer_yi.shape
        assert torch.allclose(batch_yi, infer_yi)
        s += infer_yi.shape[-1]
        infer_ys.append(infer_yi)
    infer_y = torch.cat(infer_ys, dim=-1)
    assert batch_y.shape[-1] >= infer_y.shape[-1]
    assert torch.allclose(batch_y[..., : infer_y.shape[-1]], infer_y)


@pytest.mark.slow
@pytest.mark.parametrize("tsz", [5, 9])
@pytest.mark.parametrize("ksize", [2, 3, 5, 7])
@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("dilation", [1, 2, 3])
@pytest.mark.parametrize("chunk_size", [2, 3])
def test_streaming_conv_transpose(
    tsz: int,
    ksize: int,
    stride: int,
    dilation: int,
    chunk_size: int,
) -> None:
    bsz, cin, cout, groups = 2, 3, 6, 3

    # Deterministic values rather than random numbers can be easier to debug.
    # x = torch.arange(tsz, dtype=torch.double)[None, None, :].expand(bsz, cin, tsz, implicit=True)
    # weight = torch.arange(ksize, dtype=torch.double)[None, None, :].expand(cin, cout // groups, ksize, implicit=True)
    # bias = torch.arange(cout, dtype=torch.double)
    # bias = None

    x = torch.randn(bsz, cin, tsz, dtype=torch.double)
    weight = torch.randn(cin, cout // groups, ksize, dtype=torch.double)
    bias = torch.randn(cout, dtype=torch.double)

    batch_y = torch.conv_transpose1d(x, weight, bias, stride, 0, 0, groups, dilation)

    infer_ys = []
    state: tuple[Tensor, int] | None = None
    s = 0
    for xi in x.split(chunk_size, dim=-1):
        infer_yi, state = mlfab.streaming_conv_transpose_1d(xi, state, weight, bias, stride, dilation, groups)
        batch_yi = batch_y[..., s : s + infer_yi.shape[-1]]
        assert torch.allclose(batch_yi, infer_yi)
        s += infer_yi.shape[-1]
        infer_ys.append(infer_yi)
    infer_y = torch.cat(infer_ys, dim=-1)
    assert batch_y.shape[-1] >= infer_y.shape[-1]
    assert torch.allclose(batch_y[..., : infer_y.shape[-1]], infer_y)


def test_streaming_module_conv_vanilla() -> None:
    mod = mlfab.StreamingConv1d(3, 6, 3, 2, 1, 3)
    x = torch.randn(2, 3, 32)
    y, _ = mod(x)
    assert y.shape[:-1] == (2, 6)


def test_streaming_module_conv_transpose() -> None:
    mod = mlfab.StreamingConvTranspose1d(3, 6, 3, 2, 1, 3)
    x = torch.randn(2, 3, 7)
    y, _ = mod(x)
    assert y.shape[:-1] == (2, 6)
