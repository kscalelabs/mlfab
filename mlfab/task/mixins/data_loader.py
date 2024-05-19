"""Defines a mixin for instantiating dataloaders."""

import logging
from dataclasses import dataclass
from typing import Generic, TypeVar

from dpshdl.dataloader import Dataloader
from dpshdl.dataset import Dataset, ErrorHandlingDataset
from omegaconf import II, MISSING

from mlfab.core.conf import field, is_missing, load_user_config
from mlfab.core.state import Phase
from mlfab.nn.functions import set_random_seed
from mlfab.nn.parallel import get_rank
from mlfab.task.base import BaseConfig, BaseTask
from mlfab.task.mixins.process import ProcessConfig, ProcessMixin

logger = logging.getLogger(__name__)

Sample = TypeVar("Sample")
Batch = TypeVar("Batch")

T = TypeVar("T")
Tc = TypeVar("Tc")


@dataclass
class DataloadersConfig(ProcessConfig, BaseConfig):
    batch_size: int = field(MISSING, help="Size of each batch")
    num_train_dl_workers: int = field(II("mlfab.num_workers:-1"), help="Number of workers for loading samples")
    num_test_dl_workers: int = field(1, help="Number of workers for loading samples")
    prefetch_factor: int = field(2, help="Number of items to pre-fetch on the host")
    debug_dataloader: bool = field(False, help="Debug dataloaders")
    use_pytorch_dataloader: bool = field(False, help="Use PyTorch dataloaders")


Config = TypeVar("Config", bound=DataloadersConfig)


class DataloadersMixin(ProcessMixin[Config], BaseTask[Config], Generic[Config]):
    def __init__(self, config: Config) -> None:
        if is_missing(config, "batch_size"):
            config.batch_size = self.get_batch_size()

        super().__init__(config)

    def get_batch_size(self) -> int:
        raise NotImplementedError(
            "When `batch_size` is not specified in your training config, you should override the `get_batch_size` "
            "method to return the desired training batch size."
        )

    def get_dataset(self, phase: Phase) -> Dataset:
        """Returns the dataset for the given phase.

        Args:
            phase: The phase for the dataset to return.

        Raises:
            NotImplementedError: If this method is not overridden
        """
        raise NotImplementedError("The task should implement `get_dataset`")

    def get_dataloader(
        self,
        dataset: Dataset[Sample, Batch],
        phase: Phase,
    ) -> Dataloader[Sample, Batch]:
        debugging = self.config.debug_dataloader
        if debugging:
            logger.warning("Parallel dataloaders disabled in debugging mode")

        conf = load_user_config()
        if conf.error_handling.enabled:
            dataset = ErrorHandlingDataset(
                dataset,
                sleep_backoff=conf.error_handling.sleep_backoff,
                sleep_backoff_power=conf.error_handling.sleep_backoff_power,
                maximum_exceptions=conf.error_handling.maximum_exceptions,
                backoff_after=conf.error_handling.backoff_after,
                traceback_depth=conf.error_handling.exception_location_traceback_depth,
                flush_every_n_seconds=conf.error_handling.flush_exception_summary_every,
                flush_every_n_steps=conf.error_handling.flush_exception_summary_every,
            )

        num_workers = self.config.num_train_dl_workers if phase == "train" else self.config.num_test_dl_workers

        # Creates a globally unique name for each dataloader.
        rank = get_rank()
        name = f"{phase}-{rank}"

        return Dataloader(
            dataset=dataset,
            num_workers=0 if debugging else num_workers,
            batch_size=self.config.batch_size,
            prefetch_factor=self.config.prefetch_factor,
            mp_context=None if debugging or num_workers < 1 else self.multiprocessing_context,
            mp_manager=None if debugging or num_workers < 1 else self.multiprocessing_manager,
            collate_worker_init_fn=self.collate_worker_init_fn,
            dataloader_worker_init_fn=self.data_worker_init_fn,
            name=name,
        )

    @classmethod
    def data_worker_init_fn(cls, worker_id: int, num_workers: int) -> None:
        set_random_seed(offset=worker_id + 1)

    @classmethod
    def collate_worker_init_fn(cls) -> None:
        set_random_seed(offset=0)
