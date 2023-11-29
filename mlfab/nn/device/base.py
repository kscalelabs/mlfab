"""Utility functions for abstracting away the tranining device."""

import contextlib
import functools
from abc import ABC, abstractmethod
from typing import Any, Callable, ContextManager, Generic, Iterable, Iterator, TypeVar

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data.dataloader import DataLoader, _BaseDataLoaderIter

from mlfab.core.conf import load_user_config, parse_dtype
from mlfab.nn.functions import recursive_apply

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


def allow_nonblocking_transfer(device_a: torch.device, device_b: torch.device) -> bool:
    return device_a.type in ("cpu", "cuda") and device_b.type in ("cpu", "cuda")


class Prefetcher(Iterable[T_co], Generic[T_co]):
    """Helper class for pre-loading samples into device memory."""

    def __init__(
        self,
        to_device_func: Callable[[Any], Any],
        dataloader: DataLoader[T_co],
        raise_stop_iter: bool = False,
    ) -> None:
        super().__init__()

        self.to_device_func = to_device_func
        self.dataloader = dataloader
        self.raise_stop_iter = raise_stop_iter
        self.next_sample = None
        self._dataloader_iter: _BaseDataLoaderIter | None = None

    def infinite(self) -> "InfinitePrefetcher[T_co]":
        return InfinitePrefetcher(self)

    @property
    def dataloader_iter(self) -> _BaseDataLoaderIter:
        if self._dataloader_iter is None:
            self._dataloader_iter = iter(self.dataloader)
        return self._dataloader_iter

    def prefetch(self) -> None:
        try:
            next_sample = next(self.dataloader_iter)
            self.next_sample = self.to_device_func(next_sample)
        except StopIteration:
            self.next_sample = None

    def __iter__(self) -> Iterator[T_co]:
        # Yields one sample quickly.
        next_sample = next(self.dataloader_iter)
        yield self.to_device_func(next_sample)

        try:
            self.prefetch()
            while True:
                if self.next_sample is None:
                    raise StopIteration
                sample = self.next_sample
                self.prefetch()
                yield sample

        except StopIteration:
            # Resets the dataloader if the iteration has completed.
            self._dataloader_iter = iter(self.dataloader)
            if self.raise_stop_iter:
                raise


class InfinitePrefetcher(Iterable[T_co]):
    def __init__(self, prefetcher: Prefetcher[T_co]) -> None:
        self.prefetcher = prefetcher

    def __iter__(self) -> Iterator[T_co]:
        while True:
            for batch in self.prefetcher:
                yield batch


class base_device(ABC):  # noqa: N801
    """The base ."""

    def __str__(self) -> str:
        return f"device({self.device.type}, {self.device.index}, {self.dtype})"

    def __repr__(self) -> str:
        return str(self)

    @functools.cached_property
    def device(self) -> torch.device:
        return self._get_device()

    @functools.cached_property
    def dtype(self) -> torch.dtype:
        return self._get_floating_point_type_with_override()

    @classmethod
    @abstractmethod
    def has_device(cls) -> bool:
        """Detects whether or not the device is available.

        Returns:
            If the device is available
        """

    @abstractmethod
    def _get_device(self) -> torch.device:
        """Returns the device, for instantiating new tensors.

        Returns:
            The device
        """

    @abstractmethod
    def _get_floating_point_type(self) -> torch.dtype:
        """Returns the default floating point type to use.

        Returns:
            The dtype
        """

    @abstractmethod
    def get_torch_compile_backend(self) -> str | Callable:
        """Returns the backend to use for Torch compile.

        Returns:
            The backend
        """

    def _get_floating_point_type_with_override(self) -> torch.dtype:
        if (dtype := parse_dtype(load_user_config().device)) is not None:
            return dtype
        return self._get_floating_point_type()

    def sample_to_device(self, sample: T) -> T:
        return recursive_apply(
            sample,
            lambda t: t.to(
                self.device,
                self.dtype if t.is_floating_point() else t.dtype,
                non_blocking=allow_nonblocking_transfer(t.device, self.device),
            ),
        )

    def get_prefetcher(self, dataloader: DataLoader[T_co]) -> Prefetcher[T_co]:
        return Prefetcher(self.sample_to_device, dataloader)

    def module_to(self, module: nn.Module, with_dtype: bool = False) -> None:
        if with_dtype:
            module.to(self.device, self.dtype)
        else:
            module.to(self.device)

    def tensor_to(self, tensor: np.ndarray | Tensor) -> Tensor:
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        if tensor.is_floating_point():
            return tensor.to(self.device, self.dtype)
        return tensor.to(self.device)

    def autocast_context(self, enabled: bool = True) -> ContextManager:
        device_type = self.device.type
        if device_type not in ("cpu", "cuda"):
            return contextlib.nullcontext()
        if device_type == "cpu" and self.dtype != torch.bfloat16:
            return contextlib.nullcontext()
        return torch.autocast(device_type=device_type, dtype=self.dtype, enabled=enabled)

    def supports_grad_scaler(self) -> bool:
        return False
