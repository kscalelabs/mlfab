"""Composes the base task with all the mixins into a single task interface."""

from dataclasses import dataclass
from typing import Generic, TypeVar

from mlfab.task.base import BaseConfig, BaseTask
from mlfab.task.mixins import (
    ArtifactsConfig,
    ArtifactsMixin,
    CheckpointingConfig,
    CheckpointingMixin,
    CompileConfig,
    CompileMixin,
    CPUStatsConfig,
    CPUStatsMixin,
    DataLoadersConfig,
    DataLoadersMixin,
    DeviceConfig,
    DeviceMixin,
    GPUStatsConfig,
    GPUStatsMixin,
    LoggerConfig,
    LoggerMixin,
    MixedPrecisionConfig,
    MixedPrecisionMixin,
    OptimizerConfig,
    OptimizerMixin,
    ProcessConfig,
    ProcessMixin,
    ProfilerConfig,
    ProfilerMixin,
    StepContextConfig,
    StepContextMixin,
    TrainConfig,
    TrainMixin,
)


@dataclass
class Config(
    TrainConfig,
    CheckpointingConfig,
    OptimizerConfig,
    CompileConfig,
    MixedPrecisionConfig,
    DataLoadersConfig,
    CPUStatsConfig,
    DeviceConfig,
    GPUStatsConfig,
    ProcessConfig,
    ProfilerConfig,
    LoggerConfig,
    StepContextConfig,
    ArtifactsConfig,
    BaseConfig,
):
    pass


ConfigT = TypeVar("ConfigT", bound=Config)


class Task(
    TrainMixin[ConfigT],
    CheckpointingMixin[ConfigT],
    OptimizerMixin[ConfigT],
    CompileMixin[ConfigT],
    MixedPrecisionMixin[ConfigT],
    DataLoadersMixin[Config],
    CPUStatsMixin[ConfigT],
    DeviceMixin[ConfigT],
    GPUStatsMixin[ConfigT],
    ProcessMixin[ConfigT],
    ProfilerMixin[ConfigT],
    LoggerMixin[ConfigT],
    StepContextMixin[ConfigT],
    ArtifactsMixin[ConfigT],
    BaseTask[ConfigT],
    Generic[ConfigT],
):
    pass
