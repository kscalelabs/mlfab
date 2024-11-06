"""Defines utility functions for handling checkpoints."""

import argparse
import warnings
from collections import deque
from pathlib import Path
from typing import Deque

import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.format_utils import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present


def convert_dcp_to_torch(input_path: Path, output_path: Path, key: str | None = None) -> None:
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path {input_path} is not a directory")
    if output_path.suffix != ".pt":
        warnings.warn(f"Expected output path to have a .pt extension, but got {output_path}")

    # Loads the checkpoint.
    sd: STATE_DICT_TYPE = {}
    _load_state_dict(
        sd,
        storage_reader=FileSystemReader(input_path),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )

    if key is not None:
        sd = sd[key]

    # Removes common prefixes.
    dicts: Deque[dict] = deque()
    if isinstance(sd, dict):
        dicts.append(sd)
    while dicts:
        d = dicts.popleft()
        for prefix in ("module.", "base_mod."):
            consume_prefix_in_state_dict_if_present(d, prefix)
        for v in d.values():
            if isinstance(v, dict):
                dicts.append(v)

    torch.save(sd, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert between distributed and single checkpoints.")
    parser.add_argument("input_path", type=Path, help="The path to the distributed checkpoint directory")
    parser.add_argument("output_path", type=Path, help="The path to the output checkpoint file")
    parser.add_argument("--key", type=str, help="The key to extract from the input artifact")
    args = parser.parse_args()

    convert_dcp_to_torch(args.input_path, args.output_path, args.key)


if __name__ == "__main__":
    # python -m mlfab.utils.checkpoint
    main()
