"""Defines a dataclass for keeping track of the current training state."""

import time
from dataclasses import dataclass
from typing import Literal, TypeVar, cast, get_args

from omegaconf import MISSING
from torch import nn

from mlfab.core.conf import field

Module = TypeVar("Module", bound=nn.Module)

Phase = Literal["train", "valid", "test"]


def cast_phase(raw_phase: str) -> Phase:
    args = get_args(Phase)
    assert raw_phase in args, f"Invalid phase: '{raw_phase}' Valid options are {args}"
    return cast(Phase, raw_phase)


@dataclass
class State:
    num_steps: int = field(MISSING, help="Number of steps so far")
    num_samples: int = field(MISSING, help="Number of sample so far")
    num_valid_steps: int = field(MISSING, help="Number of validation steps so far")
    num_test_steps: int = field(MISSING, help="Number of test steps so far")
    start_time_s: float = field(MISSING, help="Start time of training")
    elapsed_time_s: float = field(MISSING, help="Total elapsed time so far")
    raw_phase: str = field(MISSING, help="Current training phase")

    @property
    def phase(self) -> Phase:
        return cast_phase(self.raw_phase)

    @phase.setter
    def phase(self, new_phase: Phase) -> None:
        self.raw_phase = new_phase

    @classmethod
    def init_state(cls) -> "State":
        return cls(
            num_steps=0,
            num_samples=0,
            num_valid_steps=0,
            num_test_steps=0,
            start_time_s=time.time(),
            elapsed_time_s=0.0,
            raw_phase="train",
        )

    @property
    def training(self) -> bool:
        return self.phase == "train"

    def num_phase_steps(self, phase: Phase) -> int:
        match phase:
            case "train":
                return self.num_steps
            case "valid":
                return self.num_valid_steps
            case "test":
                return self.num_test_steps

    def set_phase(self, model: Module, new_phase: Phase) -> Module:
        self.phase = new_phase
        if new_phase == "train":
            if not model.training:
                model = model.train()
            return model
        else:
            if model.training:
                model = model.eval()
            return model
