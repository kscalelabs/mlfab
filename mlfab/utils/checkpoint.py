"""Defines utility functions for handling checkpoints."""

import argparse
import pickle
from pathlib import Path
from typing import Any


class CustomPickler(pickle.Pickler):
    def persistent_id(self, obj: Any) -> Any:  # noqa: ANN401
        return None


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> type:
        try:
            return super().find_class(module, name)
        except AttributeError:
            return lambda *args, **kwargs: None  # type: ignore[return-value]


class CustomPickleModule:
    Pickler = CustomPickler
    Unpickler = CustomUnpickler


def run_checkpoint_conversion_cli() -> None:
    """Defines a CLI for converting from a distributed checkpoint to a single checkpoint, and vice versa."""
    parser = argparse.ArgumentParser(description="Convert between distributed and single checkpoints.")
    parser.add_argument("checkpoint", type=Path, help="The checkpoint to convert path")
    parser.add_argument("output", type=Path, help="The output checkpoint path")
    args = parser.parse_args()


if __name__ == "__main__":
    run_checkpoint_conversion_cli()
