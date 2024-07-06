"""Defines neural network utility functions and classes."""

from abc import ABC, abstractmethod

from torch import nn


class ResetParameters(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

        self.reset_parameters()

    @abstractmethod
    def reset_parameters(self) -> None: ...
