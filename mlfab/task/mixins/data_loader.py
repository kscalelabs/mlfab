"""Defines a mixin for instantiating dataloaders."""

import logging
from dataclasses import dataclass
from typing import Generic, TypeVar

from dpshdl.dataloader import Dataloader
from dpshdl.dataset import Dataset
from omegaconf import II, MISSING

from mlfab.core.conf import field, is_missing
from mlfab.core.state import Phase
from mlfab.nn.functions import set_random_seed
from mlfab.task.base import BaseConfig, BaseTask
from mlfab.task.mixins.process import ProcessConfig, ProcessMixin

logger = logging.getLogger(__name__)

Sample = TypeVar("Sample")
Batch = TypeVar("Batch")


@dataclass
class DataloaderConfig:
    num_workers: int = field(MISSING, help="Number of workers for loading samples")
    prefetch_factor: int = field(2, help="Number of items to pre-fetch on each worker")


@dataclass
class DataloadersConfig(ProcessConfig, BaseConfig):
    batch_size: int = field(MISSING, help="Size of each batch")
    train_dl: DataloaderConfig = field(
        DataloaderConfig(num_workers=II("mlfab.num_workers:-1")),
        help="Train dataloader config",
    )
    test_dl: DataloaderConfig = field(
        DataloaderConfig(num_workers=1),
        help="Valid dataloader config",
    )
    debug_dataloader: bool = field(False, help="Debug dataloaders")


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

    def dataloader_config(self, phase: Phase) -> DataloaderConfig:
        match phase:
            case "train":
                return self.config.train_dl
            case "valid":
                return self.config.test_dl
            case "test":
                return self.config.test_dl
            case _:
                raise KeyError(f"Unknown phase: {phase}")

    def get_dataset(self, phase: Phase) -> Dataset:
        """Returns the dataset for the given phase.

        Args:
            phase: The phase for the dataset to return.

        Raises:
            NotImplementedError: If this method is not overridden
        """
        raise NotImplementedError("The task should implement `get_dataset`")

    def get_dataloader(self, dataset: Dataset[Sample, Batch], phase: Phase) -> Dataloader[Sample, Batch]:
        debugging = self.config.debug_dataloader
        if debugging:
            logger.warning("Parallel dataloaders disabled in debugging mode")

        cfg = self.dataloader_config(phase)

        return Dataloader(
            dataset=dataset,
            num_workers=0 if debugging else cfg.num_workers,
            batch_size=self.config.batch_size,
            prefetch_factor=cfg.prefetch_factor,
            ctx=self.multiprocessing_context,
            dataloader_worker_init_fn=self.dataloader_worker_init_fn,
            collate_worker_init_fn=self.collate_worker_init_fn,
        )

    @classmethod
    def dataloader_worker_init_fn(cls, worker_id: int, num_workers: int) -> None:
        set_random_seed(offset=worker_id)

    @classmethod
    def collate_worker_init_fn(cls) -> None:
        set_random_seed(offset=-1)
