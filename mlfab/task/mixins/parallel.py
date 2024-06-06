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
    fsdp_use_orig_params: bool = field(True, help="Use original parameters for FSDP")
    fsdp_backward_prefetch: BackwardPrefetch | None = field(None, help="Backward prefetch for FSDP")
    fsdp_sharding_strategy: ShardingStrategy | None = field(None, help="Sharding strategy")
    fsdp_sync_module_states: bool = field(True, help="Whether to sync module states on initialization")
    fsdp_keep_low_precision_grads: bool = field(False, help="Whether to keep low precision grads")
    fsdp_cast_forward_inputs: bool = field(False, help="Whether to cast forward inputs")
    fsdp_cast_root_forward_inputs: bool = field(True, help="Whether to cast root forward inputs")
    use_ddp: bool = field(False, help="Whether to use DDP instead of FSDP")
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

    return FSDP(
        model,
        process_group=process_group,
        sharding_strategy=cfg.fsdp_sharding_strategy,
        sync_module_states=cfg.fsdp_sync_module_states and all_params_are_cuda(model),
        cpu_offload=CPUOffload(cfg.fsdp_cpu_offload),
        backward_prefetch=cfg.fsdp_backward_prefetch,
        mixed_precision=mixed_precision,
        use_orig_params=cfg.fsdp_use_orig_params,
    )


class ParallelMixin(DeviceMixin[Config], LoggerMixin[Config], Generic[Config]):
    """Defines a trainer mixin for converting models to FSDP."""

    def get_fsdp_mixed_precision(self) -> MixedPrecision | None:
        dtype = self.device_manager.dtype

        return MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=dtype,
            buffer_dtype=dtype,
            keep_low_precision_grads=self.config.fsdp_keep_low_precision_grads,
            cast_forward_inputs=self.config.fsdp_cast_forward_inputs,
            cast_root_forward_inputs=self.config.fsdp_cast_root_forward_inputs,
        )

    def get_wrapped_model(self, model: nn.Module) -> nn.Module | FSDP | DDP:
        if get_world_size() <= 1:
            return model
        if isinstance(model, (FSDP, DDP)):
            return model
        if self.config.use_ddp:
            return ddp(model)
        if not torch.cuda.is_available():
            raise RuntimeError("FSDP requires CUDA")
        return fsdp(model, self.config, self.get_fsdp_mixed_precision())

    def get_grad_sync_context(self, mod: nn.Module, is_last: bool) -> ContextManager:
        if isinstance(mod, (FSDP, DDP)) and not is_last:
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
