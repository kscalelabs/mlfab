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
    batch_size: int = field(MISSING, help="Size of each batch")
    batch_size_multiplier: float = field(MISSING, help="Batch size multiplier")
    shuffle: bool = field(MISSING, help="Should the batches be shuffled on each iteration")
    num_workers: int = field(MISSING, help="Number of workers for loading samples")
    pin_memory: bool = field(MISSING, help="Should memory be pinned to it's GPU location")
    drop_last: bool = field(MISSING, help="Should the last batch be dropped if not full")
    timeout: float = field(0, help="How long to wait for a sample to be ready")
    prefetch_factor: int = field(2, help="Number of items to pre-fetch on each worker")
    persistent_workers: bool = field(False, help="Persist worker processes between epochs")
    seed: int = field(1337, help="Dataloader random seed")


@dataclass
class DataloadersConfig(ProcessConfig, BaseConfig):
    batch_size: int = field(MISSING, help="Size of each batch")
    num_dataloader_workers: int = field(II("mlfab.num_workers:-1"), help="Default number of dataloader workers")
    train_dl: DataloaderConfig = field(
        DataloaderConfig(
            batch_size=II("batch_size"),
            batch_size_multiplier=1.0,
            shuffle=True,
            num_workers=II("num_dataloader_workers"),
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        ),
        help="Train dataloader config",
    )
    test_dl: DataloaderConfig = field(
        DataloaderConfig(
            batch_size=II("batch_size"),
            batch_size_multiplier=II("train_dl.batch_size_multiplier"),
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
        ),
        help="Valid dataloader config",
    )
    debug_dataloader: bool = field(False, help="Debug dataloaders")


Config = TypeVar("Config", bound=DataloadersConfig)


class DataloadersMixin(ProcessMixin[Config], BaseTask[Config], Generic[Config]):
    def __init__(self, config: Config) -> None:
        if is_missing(config, "batch_size") and (
            is_missing(config.train_dl, "batch_size") or is_missing(config.test_dl, "batch_size")
        ):
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
            batch_size=round(cfg.batch_size * cfg.batch_size_multiplier),
            prefetch_factor=cfg.prefetch_factor,
            ctx=self.multiprocessing_context,
        )

    @classmethod
    def worker_init_fn(cls, worker_id: int) -> None:
        set_random_seed(offset=worker_id)
