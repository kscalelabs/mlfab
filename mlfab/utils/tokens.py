"""Defines utility functions for dealing with tokens and token datasets.

This file provides helper methods for reading and writing compressed datasets
of tokens. This compresses the tokens into ``ceil(log2(num_tokens))`` bits per
token, with padding at the end of each line to ensure that each line is a
multiple of 8 bits. This optimizes for making the file size as small as
possible while still being efficient to read from.

Here's an example of how to use the API:

.. highlight:: python
.. code-block:: python

    from mlfab.utils.tokens import TokenReader, TokenWriter

    num_tokens = 6
    file_path = "/path/to/dataset.bin"

    # Write the tokens to the dataset.
    with TokenWriter(file_path, num_tokens) as writer:
        for _ in range(10):
            writer.write([1, 2, 3, 4, 5])

    # Read the tokens from the dataset.
    reader = TokenReader(file_path)
    num_samples = len(reader)
    for i in range(num_samples):
        sample = reader[i]

You can also read some subset of the tokens in a line using slicing syntax.
This syntax will only read the required tokens from the file, rather than
reading the entire line and then slicing it. Here is an example:

.. highlight:: python
.. code-block:: python

    reader = TokenReader(file_path)
    first_line = reader[0]  # Gets the first line.
    first_line_part = reader[0, 1:3]  # Prints the first line, but only the second and third tokens.
"""

import functools
import logging
import math
import struct
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, ContextManager, Iterable, Literal, cast, overload

from smart_open import open

logger = logging.getLogger(__name__)

NumberFormat = Literal["Q", "I", "H", "B"]

MAGIC = b"MLTK"  # Magic number for the token file format.
OFFSET_MAGIC = b"MLTO"  # Magic number for the offsets file format.


def _arr_to_bytes(tokens: Iterable[int], num_tokens: int, offset: int = 0) -> tuple[bytes, int]:
    assert 0 <= offset < 8
    num_bits = (num_tokens - 1).bit_length()
    byte_arr = bytearray()
    cur_token = 0
    cur_bits = 0
    total_len = 0
    for token in tokens:
        total_len += 1
        assert 0 <= token <= num_tokens
        cur_token += token << cur_bits
        cur_bits += num_bits
        if offset > 0:
            cur_token <<= offset
            cur_bits += offset
            offset = 0
        while cur_bits >= 8:
            byte_arr.append(cur_token & 0xFF)
            cur_token >>= 8
            cur_bits -= 8
    if cur_bits:
        byte_arr.append(cur_token)
    return bytes(byte_arr), total_len


def _bytes_to_arr(data: bytes, seq_len: int, num_tokens: int, offset: int = 0) -> list[int]:
    assert 0 <= offset < 8
    num_bits = (num_tokens - 1).bit_length()
    arr: list[int] = []
    cur_token = 0
    cur_bits = 0
    mask = (1 << num_bits) - 1
    for byte in data:
        cur_token += byte << cur_bits
        cur_bits += 8
        if offset != 0:
            cur_token >>= offset
            cur_bits -= offset
            offset = 0
        while cur_bits >= num_bits:
            arr.append(cur_token & mask)
            if len(arr) == seq_len:
                return arr
            cur_token >>= num_bits
            cur_bits -= num_bits
    raise ValueError("Not enough bytes to fill sequence")


class TokenWriter(ContextManager):
    """Helper class for writing a dataset of tokens to a file.

    This class can be used in conjunction with :class:`TokenReader` to write
    and read datasets of tokens. The default numerical formats are chosen to
    work well with typical ranges of token datasets. At the upper end, this
    supports ``2 ^ 32`` tokens, ``2 ^ 32`` tokens per line, and ``2 ^ 64``
    tokens per file.

    Parameters:
        path: The path to the file to write to.
        num_tokens: The number of tokens in the dataset.
        overwrite_if_exists: Whether to overwrite the file if it already exists.
        num_tokens_fmt: The format string for the number of tokens.
        lengths_fmt: The format string for the lengths of each line.
        offset_fmt: The format string for the offsets of each line.
    """

    def __init__(
        self,
        path: str | Path,
        num_tokens: int,
        overwrite_if_exists: bool = False,
        *,
        num_tokens_fmt: NumberFormat = "I",
        lengths_fmt: NumberFormat = "I",
        offset_fmt: NumberFormat = "Q",
    ) -> None:
        self._path = Path(path)
        self._fp: BinaryIO | None = None
        self._offsets: list[int] = []
        self._offset_idx = -1
        self._num_tokens = num_tokens
        self._overwrite_if_exists = overwrite_if_exists
        self._num_tokens_fmt = num_tokens_fmt
        self._lengths_fmt = lengths_fmt
        self._offset_fmt = offset_fmt

    def __enter__(self) -> "TokenWriter":
        if self._path.exists():
            if self._overwrite_if_exists:
                logger.warning("Token file already exists and will be overwritten")
            else:
                raise FileExistsError(f"Token file already exists at {self._path}")
        self._fp = cast(BinaryIO, open(self._path, "wb"))

        self._offsets = []

        # Writes the file magic.
        self._fp.write(MAGIC)

        # Writes the number formats which were used in this file.
        self._fp.write((self._num_tokens_fmt + self._lengths_fmt + self._offset_fmt).encode("ascii"))

        # Writes the number of unique tokens.
        self._fp.write(struct.pack(self._num_tokens_fmt, self._num_tokens))

        # Writes a pointer to the start of the offsets table and the number
        # of offsets (i.e., the number of written rows).
        self._offset_idx = self._fp.tell()
        self._fp.write(struct.pack(f"2{self._offset_fmt}", 0, 0))

        return self

    def __exit__(self, _t: type[BaseException] | None, _e: BaseException | None, _tr: TracebackType | None) -> None:
        assert self._fp is not None
        assert self._offset_idx != -1

        # Writes the offsets table.
        offsets_start = self._fp.tell()
        self._fp.write(struct.pack(f"{len(self._offsets)}{self._offset_fmt}", *self._offsets))

        # Writes the pointer to the offsets table.
        self._fp.seek(self._offset_idx)
        self._fp.write(struct.pack(f"2{self._offset_fmt}", offsets_start, len(self._offsets)))

        self._fp.flush()
        self._fp.close()

    def write(self, tokens: Iterable[int]) -> None:
        assert self._fp is not None, "TokenWriter must be opened with a context manager"

        # Converts the tokens to a binary array.
        byte_data, num_tokens = _arr_to_bytes(tokens, self._num_tokens)

        # Writes the binary data
        self._offsets.append(self._fp.tell())
        self._fp.write(struct.pack(self._lengths_fmt, num_tokens))
        self._fp.write(byte_data)

    def writemany(self, tokens: Iterable[Iterable[int]]) -> None:
        assert self._fp is not None, "TokenWriter must be opened with a context manager"

        for line in tokens:
            self.write(line)

    def flush(self) -> None:
        assert self._fp is not None, "TokenWriter must be opened with a context manager"

        self._fp.flush()


class TokenReader:
    """Helper class for reading a dataset of tokens from a file.

    This class can be used in conjunction with :class:`TokenWriter` to write
    and read datasets of tokens.

    Parameters:
        path: The path to the file to read from.
        shard: Read a specific shard from the dataset.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

        with open(self._path, "rb") as f:
            magic = f.read(len(MAGIC))
            if magic != MAGIC:
                raise ValueError("Invalid token file")

            # Reads the number formats.
            fmt_strings = f.read(3).decode("ascii")
            self._num_tokens_fmt = fmt_strings[0]
            self._lengths_fmt = fmt_strings[1]
            self._offset_fmt = fmt_strings[2]

            # Reads the number of tokens.
            self._num_tokens = struct.unpack(self._num_tokens_fmt, f.read(struct.calcsize(self._num_tokens_fmt)))[0]

            # Reads the offset table start and length.
            offsets_vals = struct.unpack(f"2{self._offset_fmt}", f.read(struct.calcsize(self._offset_fmt) * 2))
            self._offset_start, self._num_rows = offsets_vals

            self._lengths_fmt_size = struct.calcsize(self._lengths_fmt)

            def read_offsets() -> list[int]:
                offsets: list[int] = []
                f.seek(self._offset_start)
                offset_bytes = f.read(struct.calcsize(self._offset_fmt) * self._num_rows)
                offsets.extend(struct.unpack(f"{self._num_rows}{self._offset_fmt}", offset_bytes))
                offsets.append(self._offset_start)
                return offsets

            self._offsets = read_offsets()

    @functools.cached_property
    def bits_per_token(self) -> int:
        return math.ceil(math.log2(self._num_tokens))

    def byte_length(self, index: int) -> int:
        start = self._offsets[index]
        end = self._offsets[index + 1]
        return end - start

    def length(self, index: int) -> int:
        return ((self.byte_length(index) - self._lengths_fmt_size) * 8) // self.bits_per_token

    @property
    def byte_lengths(self) -> list[int]:
        return [self.byte_length(i) for i in range(self._num_rows)]

    @property
    def lengths(self) -> list[int]:
        return [self.length(i) for i in range(self._num_rows)]

    @property
    def offsets(self) -> list[int]:
        return self._offsets

    def __len__(self) -> int:
        return self._num_rows

    @overload
    def __getitem__(self, index: int | tuple[int, slice]) -> list[int]: ...

    @overload
    def __getitem__(self, index: slice) -> list[list[int]]: ...

    def __getitem__(self, index: int | tuple[int, slice] | slice) -> list[int] | list[list[int]]:
        if isinstance(index, int):
            offset = self._offsets[index]
            seq_len = self.length(index)
            start, length = offset + self._lengths_fmt_size, (seq_len * self.bits_per_token + 7) // 8
            with open(self._path, "rb") as f:
                f.seek(start)
                byte_data = f.read(length)
            return _bytes_to_arr(byte_data, seq_len, self._num_tokens)

        if isinstance(index, tuple) and len(index) == 2 and isinstance(index[0], int) and isinstance(index[1], slice):
            index, seq_slice = index
            offset = self._offsets[index]
            seq_len = self.length(index)
            offset_start = offset + self._lengths_fmt_size

            def make_positive(n: int) -> int:
                return min(n if n >= 0 else n + seq_len, seq_len)

            # Breaks down the slice into start, stop, and step.
            start = 0 if seq_slice.start is None else make_positive(seq_slice.start)
            stop = seq_len if seq_slice.stop is None else make_positive(seq_slice.stop)
            if stop <= start:
                return cast(list[int], [])

            start_bit = start * self.bits_per_token
            start_byte, start_offset = start_bit // 8, start_bit % 8
            end_byte = (stop * self.bits_per_token + 7) // 8

            with open(self._path, "rb") as f:
                f.seek(offset_start)
                f.seek(start_byte, 1)
                byte_data = f.read(end_byte - start_byte)

            arr = _bytes_to_arr(byte_data, stop - start, self._num_tokens, offset=start_offset)
            if seq_slice.step is not None:
                arr = arr[:: seq_slice.step]
            return arr

        if isinstance(index, slice):

            def make_positive(n: int) -> int:
                return min(n if n >= 0 else n + len(self), len(self))

            start = 0 if index.start is None else make_positive(index.start)
            stop = len(self) if index.stop is None else make_positive(index.stop)
            if stop <= start:
                return cast(list[int], [])

            # Non-contiguous reads can just be done using existing logic.
            if index.step is not None and index.step != 1:
                return [self[i] for i in range(start, stop, index.step)]

            offsets = [self._offsets[i] for i in range(start, stop)]
            seq_lens = [self.length(i) for i in range(start, stop)]

            start = offsets[0] + self._lengths_fmt_size
            stop = offsets[-1] + self._lengths_fmt_size + (seq_lens[-1] * self.bits_per_token + 7) // 8
            with open(self._path, "rb") as f:
                f.seek(start)
                byte_data = f.read(stop - start)
            starts = [offset - offsets[0] for offset in offsets]
            return [
                _bytes_to_arr(byte_data[start:], seq_len, self._num_tokens) for start, seq_len in zip(starts, seq_lens)
            ]

        raise TypeError("Index must be an integer or a tuple of an integer and a slice")


class token_file:  # noqa: N801
    @classmethod
    def to_bytes(cls, tokens: Iterable[int], num_tokens: int) -> bytes:
        return _arr_to_bytes(tokens, num_tokens)[0]

    @classmethod
    def from_bytes(cls, tokens_enc: bytes, seq_len: int, num_tokens: int) -> list[int]:
        return _bytes_to_arr(tokens_enc, seq_len, num_tokens)

    @overload
    @classmethod
    def open(
        cls,
        path: str | Path,
        mode: Literal["w"],
        num_tokens: int,
        overwrite_if_exists: bool = False,
    ) -> TokenWriter: ...

    @overload
    @classmethod
    def open(cls, path: str | Path, mode: Literal["r"] = "r") -> TokenReader: ...

    @classmethod
    def open(
        cls,
        path: str | Path,
        mode: Literal["r", "w"] = "r",
        num_tokens: int | None = None,
        overwrite_if_exists: bool = False,
    ) -> TokenReader | TokenWriter:
        """Opens a token file for reading or writing.

        Args:
            path: The path to the token file.
            mode: The mode to open the file in. Can be either ``"r"`` for
                reading or ``"w"`` for writing.
            num_tokens: The number of tokens in the dataset. Required when
                opening in write mode.
            overwrite_if_exists: Whether to overwrite the file if it already
                exists. Only used when opening in write mode.

        Returns:
            A :class:`TokenReader` or :class:`TokenWriter` depending on mode.
        """
        match mode:
            case "r":
                return TokenReader(path)

            case "w":
                if num_tokens is None:
                    raise ValueError("`num_tokens` is required when opening in write mode")

                return TokenWriter(
                    path,
                    num_tokens=num_tokens,
                    overwrite_if_exists=overwrite_if_exists,
                )

            case _:
                raise ValueError(f"Unexpected mode: {mode}")
