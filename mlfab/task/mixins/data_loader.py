"""Defines a mixin for instantiating dataloaders."""

import logging
from dataclasses import dataclass
from typing import Generic, Self, TypeVar, cast

from dpshdl.dataloader import Dataloader
from dpshdl.dataset import Dataset, ErrorHandlingDataset
from omegaconf import II, MISSING
from torch.utils.data.dataloader import DataLoader as PytorchDataloader
from torch.utils.data.dataset import IterableDataset as PytorchIterableDataset

from mlfab.core.conf import field, is_missing, load_user_config
from mlfab.core.state import Phase
from mlfab.nn.functions import set_random_seed
from mlfab.nn.parallel import get_data_worker_info
from mlfab.task.base import BaseConfig, BaseTask
from mlfab.task.mixins.process import ProcessConfig, ProcessMixin

logger = logging.getLogger(__name__)

Sample = TypeVar("Sample")
Batch = TypeVar("Batch")

T = TypeVar("T")
Tc = TypeVar("Tc")


class DatasetWrapper(PytorchIterableDataset[T], Generic[T, Tc]):
    def __init__(self, dataset: Dataset[T, Tc]) -> None:
        super().__init__()

        self.dataset = dataset

    def __iter__(self) -> Self:
        self.dataset.__iter__()
        return self

    def __next__(self) -> T:
        return self.dataset.next()


@dataclass
class DataloaderConfig:
    num_workers: int = field(MISSING, help="Number of workers for loading samples")
    host_prefetch_factor: int = field(2, help="Number of items to pre-fetch on the host")
    device_prefetch_factor: int = field(2, help="Number of items to pre-fetch to the device")


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
    use_pytorch_dataloader: bool = field(False, help="If set, use PyTorch dataloaders")


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

    def get_dataloader(
        self,
        dataset: Dataset[Sample, Batch],
        phase: Phase,
    ) -> Dataloader[Sample, Batch] | PytorchDataloader[Batch]:
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

        cfg = self.dataloader_config(phase)

        if self.config.use_pytorch_dataloader:
            return PytorchDataloader(
                dataset=cast(DatasetWrapper[Batch, Batch], DatasetWrapper(dataset)),
                num_workers=0 if debugging else cfg.num_workers,
                collate_fn=dataset.collate,
                batch_size=self.config.batch_size,
                prefetch_factor=cfg.host_prefetch_factor,
                multiprocessing_context=self.multiprocessing_context,
                worker_init_fn=self.pytorch_worker_init_fn,
            )

        else:
            return Dataloader(
                dataset=dataset,
                num_workers=0 if debugging else cfg.num_workers,
                batch_size=self.config.batch_size,
                prefetch_factor=cfg.host_prefetch_factor,
                mp_manager=self.multiprocessing_manager,
                dataloader_worker_init_fn=self.dataloader_worker_init_fn,
                collate_worker_init_fn=self.collate_worker_init_fn,
            )

    @classmethod
    def pytorch_worker_init_fn(cls, worker_id: int) -> None:
        info = get_data_worker_info()
        cls.dataloader_worker_init_fn(info.worker_id, info.num_workers)

    @classmethod
    def dataloader_worker_init_fn(cls, worker_id: int, num_workers: int) -> None:
        set_random_seed(offset=worker_id + 1)

    @classmethod
    def collate_worker_init_fn(cls) -> None:
        set_random_seed(offset=0)
