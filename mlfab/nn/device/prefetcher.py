"""Defines a utility class for pre-loading samples into device memory.

When you are training a model, you usually get a sample from the dataloader,
then need to move it into device memory. This host-to-device transfer can be
slow, so it is beneficial to pre-load the next sample into device memory while
the current sample is being processed.
"""

from multiprocessing.synchronize import Event
from types import TracebackType
from typing import TYPE_CHECKING, Callable, ContextManager, Generic, Iterable, Iterator, TypeVar

import torch.multiprocessing as mp

if TYPE_CHECKING:
    from queue import Queue

T = TypeVar("T")
Tc_co = TypeVar("Tc_co", covariant=True)
Tp_co = TypeVar("Tp_co", covariant=True)


def enqueue_thread(
    dataloader: Iterator[Tc_co],
    to_device_func: Callable[[Tc_co], Tp_co],
    queue: "Queue[Tp_co]",
    stop_event: Event,
) -> None:
    for sample in dataloader:
        if stop_event.is_set():
            break
        queue.put(to_device_func(sample))


class Prefetcher(Iterable[Tp_co], Generic[Tc_co, Tp_co]):
    """Helper class for pre-loading samples into device memory."""

    def __init__(
        self,
        to_device_func: Callable[[Tc_co], Tp_co],
        dataloader: Iterator[Tc_co],
        prefetch_size: int = 2,
    ) -> None:
        super().__init__()

        self.to_device_func = to_device_func
        self.dataloader = dataloader
        self.sample_queue: "Queue[Tp_co]" = mp.Queue(maxsize=prefetch_size)
        self.stop_event = mp.Event()
        self.enqueue_thread: mp.Process | None = None

    def __iter__(self) -> Iterator[Tp_co]:
        if self.enqueue_thread is None:
            raise RuntimeError("Prefetcher is not running.")
        return self

    def __next__(self) -> Tp_co:
        if self.enqueue_thread is None:
            raise RuntimeError("Prefetcher is not running.")
        return self.sample_queue.get()

    def __enter__(self) -> "Prefetcher[Tc_co, Tp_co]":
        if isinstance(self.dataloader, ContextManager):
            self.dataloader = self.dataloader.__enter__()
        if self.enqueue_thread is not None:
            raise RuntimeError("Prefetcher is already running.")
        self.enqueue_thread = mp.Process(
            target=enqueue_thread,
            args=(self.dataloader, self.to_device_func, self.sample_queue, self.stop_event),
        )
        self.enqueue_thread.start()
        return self

    def __exit__(self, _t: type[BaseException] | None, _e: BaseException | None, _tr: TracebackType | None) -> None:
        if self.enqueue_thread is None:
            raise RuntimeError("Prefetcher is not running.")
        self.stop_event.set()
        self.enqueue_thread.join()
        self.enqueue_thread = None
        if isinstance(self.dataloader, ContextManager):
            self.dataloader.__exit__(_t, _e, _tr)
