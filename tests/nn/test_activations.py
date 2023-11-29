"""Tests the activations API."""

from typing import get_args

import pytest
import torch

from mlfab.nn.activations import ActivationType, get_activation


@pytest.mark.parametrize("kind", get_args(ActivationType))
@pytest.mark.parametrize("inplace", [True, False])
def test_embeddings_api(kind: ActivationType, inplace: bool) -> None:
    x = torch.randn(3, 5, 8)
    act = get_activation(kind, inplace=inplace)
    y = act(x)
    assert y.shape == x.shape
