"""Defines a mixin to support parallel model training."""

import contextlib
import functools
import logging
from dataclasses import dataclass
from typing import Any, ContextManager, Generic, Sequence, TypeVar

import torch
from torch import Tensor, nn
from torch.distributed import ProcessGroup
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.distributed.fsdp.api import ShardingStrategy
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import Optimizer

from mlfab.core.conf import field
from mlfab.nn.parallel import all_params_are_cuda, get_world_size, parallel_group_info
from mlfab.task.mixins.device import DeviceConfig, DeviceMixin
from mlfab.task.mixins.logger import LoggerConfig, LoggerMixin
from mlfab.utils.experiments import clip_grad_norm_, get_weight_norm

logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig(DeviceConfig, LoggerConfig):
    fsdp_cpu_offload: bool = field(False, help="CPU offloading for FSDP")
    fsdp_backward_prefetch: str | None = field(None, help="Backward prefetch for FSDP")
    fsdp_use_orig_params: bool = field(True, help="Use original parameters for FSDP")
    fsdp_sharding_strategy: ShardingStrategy = field(ShardingStrategy.HYBRID_SHARD, help="Sharding strategy")
    fsdp_sync_module_states: bool = field(True, help="Whether to sync module states on initialization")
    clip_grad_norm: float = field(10.0, help="What to clip the gradient norm to")
    clip_grad_norm_type: Any = field(2, help="Type of norm to use")


Config = TypeVar("Config", bound=ParallelConfig)


def ddp(model: nn.Module) -> DDP:
    group_info = parallel_group_info()
    return DDP(model, process_group=group_info.dp.group)


def fsdp(model: nn.Module, cfg: ParallelConfig, mixed_precision: MixedPrecision | None = None) -> FSDP:
    group_info = parallel_group_info()

    process_group: tuple[ProcessGroup, ProcessGroup] | ProcessGroup
    if cfg.fsdp_sharding_strategy in (ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2):
        process_group = group_info.mp.group, group_info.dp.group
    else:
        process_group = group_info.mp.group

    if cfg.fsdp_cpu_offload:
        logger.warning("CPU offloading doesn't support gradient accumulation")

    if mixed_precision is None:
        mixed_precision = MixedPrecision(
            param_dtype=None,
            reduce_dtype=None,
            buffer_dtype=None,
            keep_low_precision_grads=False,
            cast_forward_inputs=False,
            cast_root_forward_inputs=True,
        )

    return FSDP(
        model,
        process_group=process_group,
        sharding_strategy=cfg.fsdp_sharding_strategy,
        sync_module_states=cfg.fsdp_sync_module_states and all_params_are_cuda(model),
        cpu_offload=CPUOffload(cfg.fsdp_cpu_offload),
        backward_prefetch=None if cfg.fsdp_backward_prefetch is None else BackwardPrefetch[cfg.fsdp_backward_prefetch],
        mixed_precision=mixed_precision,
        use_orig_params=cfg.fsdp_use_orig_params,
    )


class ParallelMixin(DeviceMixin[Config], LoggerMixin[Config], Generic[Config]):
    """Defines a trainer mixin for doing FP16 scaling."""

    def maybe_fsdp(self, model: nn.Module) -> nn.Module | FSDP:
        if get_world_size() > 1 and torch.cuda.is_available() and not isinstance(model, FSDP):
            return fsdp(model, self.config)
        return model

    def get_grad_sync_context(self, mod: nn.Module, is_last: bool) -> ContextManager:
        if isinstance(mod, FSDP) and not is_last:
            return mod.no_sync()
        return contextlib.nullcontext()

    def backward_grads(
        self,
        loss: Tensor,
        retain_graph: bool | None = None,
        inputs: Sequence[Tensor] | None = None,
    ) -> None:
        if loss.numel() > 1:
            loss = loss.sum()
        isnan = not bool(torch.isfinite(loss))
        if isnan:
            loss.backward(torch.zeros_like(loss), retain_graph=retain_graph, inputs=inputs)
        else:
            loss.backward(retain_graph=retain_graph, inputs=inputs)

    @torch.no_grad()
    def step_optimizer(self, mod: nn.Module, optim: Optimizer, num_steps: int = 1) -> None:
        clip_norm = self.config.clip_grad_norm
        norm_type = self.config.clip_grad_norm_type

        # When accumulating multiple steps of gradients per backward pass, we
        # need to divide the gradients by the number of steps.
        if num_steps > 1:
            for p in mod.parameters():
                if p.grad is not None:
                    p.grad /= num_steps

        # Clips gradients.
        if isinstance(mod, FSDP):
            total_norm = mod.clip_grad_norm_(clip_norm, norm_type)
            was_clipped = bool(torch.isfinite(total_norm))
        else:
            total_norm, was_clipped = clip_grad_norm_(
                mod.parameters(),
                max_norm=clip_norm,
                norm_type=norm_type,
                foreach=None,
            )

        # Logs weight and gradient norms.
        self.log_scalar("weight_norm", lambda: get_weight_norm(mod.parameters()), namespace="📉 optim")
        self.log_scalar("grad_norm", total_norm, namespace="📉 optim")

        # Steps the optimizer.
        if was_clipped:
            optim.step()

    @functools.cached_property
    def autocast_context(self) -> ContextManager:
        return self.device_manager.autocast_context()
