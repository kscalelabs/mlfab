"""Tests that methods for reading and writing token datasets are correct."""

import random
from pathlib import Path

import pytest

import mlfab
from mlfab.utils.tokens import _arr_to_bytes, _bytes_to_arr


@pytest.mark.parametrize("offset", [0, 3, 6])
def test_arr_to_bytes_to_arr(offset: int) -> None:
    values = [1, 2, 3, 4, 5, 6, 7]
    assert _bytes_to_arr(_arr_to_bytes(values, 100, offset)[0], len(values), 100, offset) == values

    values += [253, 254]
    assert _bytes_to_arr(_arr_to_bytes(values, 255, offset)[0], len(values), 255, offset) == values

    values += [510, 511]
    assert _bytes_to_arr(_arr_to_bytes(values, 512, offset)[0], len(values), 512, offset) == values

    values += [50_000, 99_999]
    assert _bytes_to_arr(_arr_to_bytes(values, 100_000, offset)[0], len(values), 100_000, offset) == values

    values += [1_000_000, 99_999_999]
    assert _bytes_to_arr(_arr_to_bytes(values, 100_000_000, offset)[0], len(values), 100_000_000, offset) == values


@pytest.mark.parametrize("num_tokens", [255, 512, 100_000])
def test_read_write(num_tokens: int, tmpdir: Path) -> None:
    file_path = tmpdir / "dataset.bin"

    # Write the tokens to the dataset.
    all_tokens = []
    all_token_lengths = []
    with mlfab.token_file.open(file_path, "w", num_tokens=num_tokens) as writer:
        for _ in range(10):
            token_length = random.randint(10, 100)
            all_token_lengths.append(token_length)
            tokens = [random.randint(0, num_tokens - 1) for _ in range(token_length)]
            all_tokens.append(tokens)
            writer.write(tokens)

    # Read the tokens from the dataset.
    reader = mlfab.token_file.open(file_path, "r")
    num_samples = len(reader)
    assert num_samples == len(all_tokens)
    for i in range(num_samples):
        assert reader[i] == all_tokens[i]

    # Reads again, to test that the offsets file is used.
    reader = mlfab.token_file.open(file_path, "r")
    num_samples = len(reader)
    assert num_samples == len(all_tokens)
    for i in range(num_samples):
        assert reader[i] == all_tokens[i]

    # Checks reader properties.
    assert reader.lengths == all_token_lengths

    # Checks reading a subset of an index.
    assert reader[1, 3:101] == all_tokens[1][3:]
    assert reader[2, :5] == all_tokens[2][:5]
    assert reader[3, 5:] == all_tokens[3][5:]
    assert reader[4, :-5] == all_tokens[4][:-5]
    assert reader[5, -5:] == all_tokens[5][-5:]
    assert reader[1:3] == all_tokens[1:3]
    assert reader[1:] == all_tokens[1:]
    assert reader[:3] == all_tokens[:3]

    # Checks reading entirely into memory.
    reader = mlfab.token_file.open(file_path, "r")
    num_samples = len(reader)
    assert num_samples == len(all_tokens)
    for i in range(num_samples):
        assert reader[i] == all_tokens[i]
