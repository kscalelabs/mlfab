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
    HeartbeatConfig,
    HeartbeatMonitorMixin,
    LoggerConfig,
    LoggerMixin,
    MixedPrecisionConfig,
    MixedPrecisionMixin,
    MonitorProcessConfig,
    MonitorProcessMixin,
    OptimizerConfig,
    OptimizerMixin,
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
    DataLoadersConfig,
    CheckpointingConfig,
    OptimizerConfig,
    CompileConfig,
    MixedPrecisionConfig,
    CPUStatsConfig,
    DeviceConfig,
    GPUStatsConfig,
    HeartbeatConfig,
    MonitorProcessConfig,
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
    DataLoadersMixin[Config],
    CheckpointingMixin[ConfigT],
    OptimizerMixin[ConfigT],
    CompileMixin[ConfigT],
    MixedPrecisionMixin[ConfigT],
    CPUStatsMixin[ConfigT],
    DeviceMixin[ConfigT],
    GPUStatsMixin[ConfigT],
    HeartbeatMonitorMixin[ConfigT],
    MonitorProcessMixin[ConfigT],
    ProfilerMixin[ConfigT],
    LoggerMixin[ConfigT],
    StepContextMixin[ConfigT],
    ArtifactsMixin[ConfigT],
    BaseTask[ConfigT],
    Generic[ConfigT],
):
    pass
