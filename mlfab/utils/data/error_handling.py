"""Defines error handling wrappers for datasets."""

import bdb
import logging
import random
import sys
import time
from collections import Counter
from typing import Iterator, TypeVar, no_type_check

from torch.utils.data.dataset import Dataset, IterableDataset

from mlfab.core.conf import load_user_config
from mlfab.nn.parallel import get_data_worker_info
from mlfab.utils.text import TextBlock, render_text_blocks

logger: logging.Logger = logging.getLogger(__name__)

BatchT = TypeVar("BatchT")
DatasetT = TypeVar("DatasetT", Dataset, IterableDataset)


def get_loc(num_excs: int = 1) -> str:
    _, _, exc_tb = sys.exc_info()
    if exc_tb is None or (exc_tb := exc_tb.tb_next) is None:
        return "unknown"
    exc_strs: list[str] = []
    for _ in range(num_excs):
        exc_strs += [f"{exc_tb.tb_frame.f_code.co_filename}:{exc_tb.tb_lineno}"]
        if (exc_tb := exc_tb.tb_next) is None:
            break
    return "\n".join(exc_strs)


class ExceptionSummary:
    def __init__(self) -> None:
        self.config = load_user_config().error_handling
        self.steps = 0
        self.step_has_error = False
        self.total_exceptions = 0
        self.exceptions: Counter[str] = Counter()
        self.exception_classes: Counter[str] = Counter()
        self.exception_locs: Counter[str] = Counter()
        self.last_exception: Exception | None = None

    def add_exception(self, exc: Exception, loc: str) -> None:
        self.last_exception = exc
        self.exceptions[f"{exc.__class__.__name__}: {exc}"] += 1
        self.exception_classes[exc.__class__.__name__] += 1
        self.exception_locs[loc] += 1
        if not self.step_has_error:
            self.total_exceptions += 1
            self.step_has_error = True

    def step(self) -> None:
        if self.steps >= self.config.flush_exception_summary_every:
            self.flush()
        self.steps += 1
        self.step_has_error = False

    def summary(self) -> str:
        blocks: list[list[TextBlock]] = []
        blocks += [
            [
                TextBlock("Error Summary", color="red", bold=True, width=60, center=True),
                TextBlock("Count", color="yellow", bold=False, width=10, center=True),
                TextBlock("Percent", color="yellow", bold=False, width=10, center=True),
            ],
        ]

        def get_header(s: str) -> list[list[TextBlock]]:
            return [
                [
                    TextBlock(s, color="yellow", bold=True, width=60, no_sep=True),
                    TextBlock("", width=10, no_sep=True),
                    TextBlock("", width=10, no_sep=True),
                ],
            ]

        def get_line(ks: str, v: int) -> list[list[TextBlock]]:
            line = [
                TextBlock(ks, width=60, no_sep=True),
                TextBlock(f"{v}", width=10, no_sep=True),
                TextBlock(f"{int(v * 100 / self.steps)} %", width=10, no_sep=True),
            ]
            return [line]

        # Logs unique exception strings.
        blocks += get_header("Exceptions")
        for k, v in self.exceptions.most_common(self.config.report_top_n_exception_types):
            blocks += get_line(k, v)

        # Logs the individual exception classes.
        blocks += get_header("Types")
        for k, v in self.exception_classes.most_common(self.config.report_top_n_exception_types):
            blocks += get_line(k, v)

        # Logs by line number.
        blocks += get_header("Locations")
        for k, v in self.exception_locs.most_common(self.config.report_top_n_exception_types):
            blocks += get_line(k, v)

        # Logs the total number of exceptions.
        exception_prct = int(self.total_exceptions / self.steps * 100)
        blocks += [
            [
                TextBlock("", width=60, no_sep=True),
                TextBlock(f"{self.total_exceptions} / {self.steps}", color="red", bold=True, width=10, no_sep=True),
                TextBlock(f"{exception_prct} %", color="red", bold=True, width=10, no_sep=True),
            ],
        ]

        return render_text_blocks(blocks)

    def flush(self) -> None:
        worker_info = get_data_worker_info()
        if worker_info.worker_id == 0 and self.total_exceptions > 0:
            logger.info("Exception summary:\n\n%s\n", self.summary())
        self.exceptions.clear()
        self.exception_classes.clear()
        self.exception_locs.clear()
        self.steps = 0
        self.total_exceptions = 0


class ErrorHandlingDataset(Dataset[BatchT]):
    """Defines a wrapper for safely handling errors."""

    def __init__(self, dataset: Dataset[BatchT]) -> None:
        super().__init__()

        self.dataset = dataset
        self.exc_summary = ExceptionSummary()

    def __getitem__(self, index: int) -> BatchT:
        config = load_user_config().error_handling
        num_exceptions = 0
        backoff_time = config.sleep_backoff
        self.exc_summary.step()
        while num_exceptions < config.maximum_exceptions:
            try:
                return self.dataset[index]
            except bdb.BdbQuit as e:
                logger.info("User interrupted debugging session; aborting")
                raise e
            except Exception as e:
                if config.log_full_exception:
                    logger.exception("Caught exception on index %d", index)
                self.exc_summary.add_exception(e, get_loc(config.exception_location_traceback_depth))
                index = random.randint(0, len(self) - 1)
            num_exceptions += 1
            if num_exceptions > config.backoff_after:
                logger.error(
                    "Encountered %d exceptions for a single index, backing off for %f seconds",
                    num_exceptions,
                    backoff_time,
                )
                time.sleep(backoff_time)
                backoff_time *= config.sleep_backoff_power
        exc_message = f"Reached max exceptions {config.maximum_exceptions}\n{self.exc_summary.summary()}"
        if self.exc_summary.last_exception is None:
            raise RuntimeError(exc_message)
        raise RuntimeError(exc_message) from self.exc_summary.last_exception

    def __len__(self) -> int:
        if hasattr(self.dataset, "__len__"):
            return self.dataset.__len__()
        raise NotImplementedError("Base dataset doesn't implemenet `__len__`")


class ErrorHandlingIterableDataset(IterableDataset[BatchT]):
    """Defines a wrapper for safely handling errors in iterable datasets."""

    def __init__(self, dataset: IterableDataset[BatchT]) -> None:
        super().__init__()

        self.iteration = 0
        self.dataset = dataset
        self.exc_summary = ExceptionSummary()
        self.iter: Iterator[BatchT] | None = None

    def __iter__(self) -> Iterator[BatchT]:
        self.iter = self.dataset.__iter__()
        self.iteration = 0
        return self

    def __next__(self) -> BatchT:
        assert self.iter is not None, "Must call `__iter__` before `__next__`"
        config = load_user_config().error_handling
        num_exceptions = 0
        backoff_time = config.sleep_backoff
        self.exc_summary.step()
        self.iteration += 1
        while num_exceptions < config.maximum_exceptions:
            try:
                return self.iter.__next__()
            except bdb.BdbQuit as e:
                logger.info("User interrupted debugging session; aborting")
                raise e
            except StopIteration as e:
                raise e
            except Exception as e:
                if config.log_full_exception:
                    logger.exception("Caught exception on iteration %d", self.iteration)
                self.exc_summary.add_exception(e, get_loc(config.exception_location_traceback_depth))
            num_exceptions += 1
            if num_exceptions > config.backoff_after:
                logger.error(
                    "Encountered %d exceptions for a single index, backing off for %f seconds",
                    num_exceptions,
                    backoff_time,
                )
                time.sleep(backoff_time)
                backoff_time *= config.sleep_backoff_power
        raise RuntimeError(f"Reached max exceptions {config.maximum_exceptions}\n{self.exc_summary.summary()}")


@no_type_check
def error_handling_dataset(dataset: DatasetT) -> DatasetT:
    """Returns a dataset which wraps the base dataset and handles errors.

    Args:
        dataset: The dataset to handle errors for

    Returns:
        The wrapped dataset, which catches some errors

    Raises:
        NotImplementedError: If the dataset type is not supported
    """
    if isinstance(dataset, IterableDataset):
        return ErrorHandlingIterableDataset(dataset)
    elif isinstance(dataset, Dataset):
        return ErrorHandlingDataset(dataset)
    raise NotImplementedError(f"Unexpected type: {dataset}")


def test_exception_summary() -> None:
    summary = ExceptionSummary()
    for i in range(10):
        try:
            if i < 7:
                raise RuntimeError("test")
            else:
                raise ValueError("test 2")
        except Exception as e:
            summary.add_exception(e, get_loc())
        summary.step()
    sys.stdout.write(summary.summary())
    sys.stdout.flush()


if __name__ == "__main__":
    # python -m mlfab.utils.data.error_handling
    test_exception_summary()
