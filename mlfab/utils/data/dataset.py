"""Defines custom dataset functions."""

from typing import Generic, Iterator, Sized, TypeVar

from torch.utils.data.dataset import Dataset, IterableDataset

T = TypeVar("T")


class SmallDataset(IterableDataset[T], Generic[T]):
    """Defines a dataset which caches a small number of items.

    This is useful for validating a model on a small dataset before running
    a larger training job. It automatically detects iterable verses
    non-iterable datasets. Note that all the elements of the dataset must fit
    into memory.

    After this dataset has filled the cache, it loops through the cached items
    until the training run is over.

    Parameters:
        size: The number of unique samples to yield from the dataset.
        dataset: The dataset to wrap.
    """

    def __init__(self, size: int, dataset: Dataset[T]) -> None:
        super().__init__()

        assert size > 0, "Size must be positive."

        self.size = self._get_size(size, dataset)
        self.counter = 0
        self.dataset = dataset
        self.cache: list[T] = []

        self._ds_iter: Iterator[T] | None = None

    def _get_size(self, max_size: int, dataset: Dataset[T]) -> int:
        if isinstance(dataset, IterableDataset):
            return max_size
        if isinstance(dataset, Dataset) and isinstance(dataset, Sized):
            return min(max_size, len(dataset))
        raise TypeError("Dataset must be iterable or indexable.")

    def _add_item(self, recursed: bool = False) -> None:
        if isinstance(self.dataset, IterableDataset):
            if self._ds_iter is None:
                self._ds_iter = iter(self.dataset)
            try:
                self.cache.append(next(self._ds_iter))
            except StopIteration:
                if recursed:
                    raise RuntimeError("Dataset doesn't yield any items!")
                self._ds_iter = None
                self._add_item(recursed=True)
        elif isinstance(self.dataset, Dataset) and isinstance(self.dataset, Sized):
            self.cache.append(self.dataset[len(self.cache)])
        else:
            raise TypeError("Dataset must be iterable or indexable.")

    def __iter__(self) -> Iterator[T]:
        self.counter = 0
        return self

    def __next__(self) -> T:
        if self.counter >= self.size:
            self.counter = 0
        while len(self.cache) <= self.counter:
            self._add_item()
        self.counter += 1
        return self.cache[self.counter - 1]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> T:
        assert 0 <= index < self.size, "Index out of bounds."
        while index >= len(self.cache):
            self._add_item()
        return self.cache[index]
