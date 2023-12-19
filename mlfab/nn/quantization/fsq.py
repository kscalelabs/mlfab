"""Provides an implementation of Finite Scalar Quantization (FSQ).

FSQ is a quantization approach which has a relatively small number of parameters
compared with codebook learning. It was proposed in the paper
`Finite Scalar Quantization: VQ-VAE Made Simple
<https://arxiv.org/abs/2309.15505>`_.

This implementation is largely adapted from the `lucidrains implementation
<https://github.com/lucidrains/vector-quantize-pytorch>`_ which in turn tracks
very closely with the `original implementation
<https://github.com/google-research/google-research/tree/master/fsq>`_.
"""

import torch
from torch import Tensor, nn

from mlfab.nn.functions import device_fn


def round_ste(z: Tensor) -> Tensor:
    zhat = z.round()
    return z + (zhat - z).detach()


class FiniteScalarQuantization(nn.Module):
    """Defines a finite scalar quantization module.

    The original paper proposes the following number of levels, depending on
    the target codebook size:

    +------------------+------------------+
    | Codebook size    | Number of levels |
    +==================+==================+
    | 2^8              | 8, 6, 5          |
    +------------------+------------------+
    | 2^10             | 8, 5, 5, 5       |
    +------------------+------------------+
    | 2^12             | 7, 5, 5, 5, 5    |
    +------------------+------------------+
    | 2^14             | 8, 8, 8, 6, 5    |
    +------------------+------------------+
    | 2^16             | 8, 8, 8, 5, 5, 5 |
    +------------------+------------------+

    Parameters:
        levels: The number of levels. The product of the levels is the number
            of unique codes. The input to the module should be a tensor with
            shape ``(..., len(levels))``.

    Properties:
        dim: The number of dimensions of the quantized tensor, i.e. the length
            of the ``levels`` argument.
        n_codes: The number of unique codes.

    Inputs:
        z: A tensor of shape ``(..., len(levels))``.

    Outputs:
        quantized: A quantized tensor of shape ``(..., len(levels))``. The
            quantized values will be in the range ``[-1, 1]``.
    """

    @device_fn("cpu")
    def __init__(self, levels: list[int]) -> None:
        super().__init__()

        _levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32)
        self.register_buffer("_basis", _basis, persistent=False)

        self.dim = len(levels)
        self.n_codes = self._levels.prod().item()
        implicit_codebook = self.indices_to_codes(torch.arange(self.n_codes))
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

    _levels: Tensor
    _basis: Tensor
    implicit_codebook: Tensor

    def forward(self, z: Tensor) -> Tensor:
        return self.quantize(z)

    def quantize(self, z: Tensor) -> Tensor:
        if z.shape[-1] != self.dim:
            raise ValueError(f"Expected final dimension to be {self.dim}, but got input shape {z.shape}")
        quantized = round_ste(self._bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        half_l = (self._levels - 1) * (1 - eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).tan()
        return (z + shift).tanh() * half_l - offset

    def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Tensor) -> Tensor:
        assert zhat.shape[-1] == self.dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(torch.int32)

    def indices_to_codes(self, indices: Tensor) -> Tensor:
        indices = indices.unsqueeze(-1)
        codes_non_centered = (indices // self._basis) % self._levels
        return self._scale_and_shift_inverse(codes_non_centered)
