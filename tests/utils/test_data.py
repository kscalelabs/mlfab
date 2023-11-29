"""Runs tests on data functions."""

import itertools
from typing import Iterator

from torch.utils.data.dataset import Dataset, IterableDataset

import mlfab


class DummyDataset(Dataset[int]):
    def __len__(self) -> int:
        return 5

    def __getitem__(self, index: int) -> int:
        return index + 1


class DummyIterableDataset(IterableDataset[int]):
    def __iter__(self) -> Iterator[int]:
        for i in range(5):
            yield i + 1


def test_small_dataset_sized() -> None:
    ds = mlfab.SmallDataset(3, DummyDataset())
    assert len(ds) == 3
    assert list(itertools.islice(iter(ds), 9)) == [1, 2, 3, 1, 2, 3, 1, 2, 3]


def test_small_dataset_iterable() -> None:
    ds = mlfab.SmallDataset(3, DummyIterableDataset())
    assert len(ds) == 3
    assert list(itertools.islice(iter(ds), 9)) == [1, 2, 3, 1, 2, 3, 1, 2, 3]
