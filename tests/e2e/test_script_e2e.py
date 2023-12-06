"""Runs end-to-end tests of running a general-purpose script."""

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

import mlfab


@dataclass
class Config(mlfab.ScriptConfig):
    text: str = mlfab.field("Hello, world!", help="Some text to print")


class DummyScript(mlfab.Script[Config]):
    def run(self) -> None:
        world_size, rank = mlfab.get_world_size(), mlfab.get_rank()
        sys.stdout.write(f"{self.config.text} {rank} / {world_size}\n")
        sys.stdout.flush()


@pytest.mark.slow
def test_e2e_script(tmpdir: Path) -> None:
    os.environ["RUN_DIR"] = str(tmpdir)
    os.environ["USE_METAL"] = "0"

    mlfab.configure_logging()

    DummyScript.launch(use_cli=False)


@pytest.mark.slow
def test_e2e_script_mp(tmpdir: Path) -> None:
    os.environ["RUN_DIR"] = str(tmpdir)
    os.environ["TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    os.environ["USE_METAL"] = "0"

    mlfab.configure_logging()

    DummyScript.launch(launcher=mlfab.MultiProcessLauncher(num_processes=2), use_cli=False)


if __name__ == "__main__":
    # python -m tests.e2e.test_script_e2e
    # test_e2e_script(Path(tempfile.mkdtemp()))
    test_e2e_script_mp(Path(tempfile.mkdtemp()))
