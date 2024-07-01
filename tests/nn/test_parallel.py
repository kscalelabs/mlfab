"""Tests model parallelism primitives."""

import pytest

import mlfab


@pytest.mark.parametrize("n", [13, 101])
@pytest.mark.parametrize("num_workers", [5, 7, 1])
def test_split_n_items_across_workers(n: int, num_workers: int) -> None:
    starts, ends = zip(*(mlfab.split_n_items_across_workers(n, i, num_workers) for i in range(num_workers)))
    lengths = [e - s for s, e in zip(starts, ends)]
    assert sum(lengths) == n
    assert starts[0] == 0
    assert ends[-1] == n
    assert all(starts[i] == ends[i - 1] for i in range(1, num_workers))
