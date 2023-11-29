"""Tests container functions."""

from dataclasses import dataclass

import torch
from torch import Tensor

import mlfab


@dataclass
class DummyDataclass:
    x: list[Tensor]
    y: dict[str, Tensor]


def test_recursive_apply() -> None:
    container = {"x": [torch.randn(3), torch.randn(4)], "y": [torch.randn(3)]}
    conv_container = mlfab.recursive_apply(container, lambda t: t.to(torch.float64))
    assert conv_container["x"][0].dtype == torch.float64
    assert conv_container["y"][0].dtype == torch.float64


def test_recursive_chunk() -> None:
    list_container = [1, torch.randn(3), torch.randn(3, 4)]
    conv_list_container = list(mlfab.recursive_chunk(list_container, 3))
    assert len(conv_list_container) == 3
    assert all(cl[0] == 1 for cl in conv_list_container)
    assert all(cl[1].shape == (1,) for cl in conv_list_container)

    dict_container = {"x": [torch.randn(3), torch.randn(3, 4)], "y": [torch.randn(3)]}
    conv_dict_container = list(mlfab.recursive_chunk(dict_container, 3))
    assert len(conv_dict_container) == 3
    assert all(cd["x"][0].shape == (1,) for cd in conv_dict_container)

    dataclass_container = DummyDataclass(x=[torch.randn(3), torch.randn(3, 4)], y={"a": torch.randn(3)})
    conv_dataclass_container = list(mlfab.recursive_chunk(dataclass_container, 3))
    assert len(conv_dataclass_container) == 3
    assert all(cdc.x[0].shape == (1,) for cdc in conv_dataclass_container)
