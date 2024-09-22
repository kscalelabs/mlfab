"""Defines a mixin to support parallel model training."""

import contextlib
import functools
import json
import logging
from dataclasses import dataclass
from typing import Any, ContextManager, Generic, Sequence, TypeVar

import torch
from torch import Tensor, nn
from torch.distributed._tensor import DeviceMesh
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import CustomPolicy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.optimizer import Optimizer

from mlfab.core.conf import field
from mlfab.nn.parallel import all_params_are_cuda, device_mesh, get_world_size, parallel_group_info
from mlfab.task.mixins.device import DeviceConfig, DeviceMixin
from mlfab.task.mixins.logger import LoggerConfig, LoggerMixin
from mlfab.utils.experiments import MinGradScaleError, NaNError, clip_grad_norm_, get_weight_norm

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class GradScalerConfig:
    init_scale: float = field(2.0**16, help="Initial scaling factor")
    growth_factor: float = field(2.0, help="Factor by which the scale is multiplied if no gradient NaNs occur")
    backoff_factor: float = field(0.5, help="Factor by which the scale is multiplied if gradient NaNs occur")
    growth_interval: int = field(2000, help="How often to grow the scale")
    min_grad_scale: float = field(1e-4, help="Minimum allowable gradient scale")
    foreach: bool | None = field(None, help="If set, use foreach implementation")


@dataclass(kw_only=True)
class ParallelConfig(DeviceConfig, LoggerConfig):
    fsdp_cpu_offload: bool = field(False, help="CPU offloading for FSDP")
    fsdp_use_orig_params: bool = field(False, help="Use original parameters for FSDP")
    fsdp_wrap: bool = field(False, help="If set, use FSDP wrapping")
    fsdp_forward_prefetch: bool = field(True, help="Prefetch forward pass of FSDP")
    fsdp_backward_prefetch: BackwardPrefetch | None = field(BackwardPrefetch.BACKWARD_PRE, help="Backward prefetching")
    fsdp_sharding_strategy: ShardingStrategy | None = field(None, help="Sharding strategy")
    fsdp_limit_all_gathers: bool = field(True, help="Limit all gathers in FSDP computation")
    fsdp_sync_module_states: bool = field(True, help="Whether to sync module states on initialization")
    fsdp_keep_low_precision_grads: bool = field(False, help="Whether to keep low precision grads")
    fsdp_cast_forward_inputs: bool = field(False, help="Whether to cast forward inputs")
    fsdp_cast_root_forward_inputs: bool = field(True, help="Whether to cast root forward inputs")
    ddp_static_graph: bool = field(True, help="Whether to use a static graph for DDP")
    ddp_find_unused_parameters: bool = field(False, help="Whether to find unused parameters for DDP")
    use_ddp: bool = field(False, help="Whether to use DDP")
    grad_scaler: GradScalerConfig = field(GradScalerConfig(), help="Gradient scaler configuration")
    grad_scaler_enabled: bool = field(True, help="If set, should FP16 training be enabled")
    clip_grad_norm: float = field(10.0, help="What to clip the gradient norm to")
    clip_grad_norm_type: Any = field(2, help="Type of norm to use")
    debug_nan_grads: bool = field(False, help="If set, should NaN gradients be debugged")


Config = TypeVar("Config", bound=ParallelConfig)


def fsdp(
    model: nn.Module,
    cfg: ParallelConfig,
    device: torch.device,
    mixed_precision: MixedPrecision | None = None,
) -> FSDP:
    group_info = parallel_group_info()

    if (sharding_strategy := cfg.fsdp_sharding_strategy) is None:
        if group_info.tp.world_size == 1:
            logger.info("Using NO_SHARD FSDP strategy")
            sharding_strategy = ShardingStrategy.NO_SHARD
        elif group_info.dp.world_size == 1:
            logger.info("Using FULL_SHARD FSDP strategy")
            sharding_strategy = ShardingStrategy.FULL_SHARD
        else:
            logger.info("Using HYBRID_SHARD FSDP strategy")
            sharding_strategy = ShardingStrategy.HYBRID_SHARD

    if cfg.fsdp_cpu_offload:
        logger.warning("CPU offloading doesn't support gradient accumulation")

    def should_wrap(mod: nn.Module) -> bool:
        return bool(getattr(mod, "__wrap_fsdp__", False))

    kwargs = {
        "sharding_strategy": sharding_strategy,
        "auto_wrap_policy": CustomPolicy(should_wrap) if cfg.fsdp_wrap else None,
        "cpu_offload": CPUOffload(cfg.fsdp_cpu_offload),
        "backward_prefetch": cfg.fsdp_backward_prefetch,
        "mixed_precision": mixed_precision,
        "device_id": device,
        "sync_module_states": cfg.fsdp_sync_module_states and all_params_are_cuda(model),
        "forward_prefetch": cfg.fsdp_forward_prefetch,
        "limit_all_gathers": cfg.fsdp_limit_all_gathers,
        "use_orig_params": cfg.fsdp_use_orig_params,
    }

    if sharding_strategy in (ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2):
        mesh = device_mesh(device.type)
    else:
        mesh = device_mesh(device.type)["dp"]

    model = FSDP(model, device_mesh=mesh, **kwargs)  # type: ignore[arg-type]

    return model


def ddp(model: nn.Module, cfg: ParallelConfig) -> nn.Module:
    return DDP(
        model,
        find_unused_parameters=cfg.ddp_find_unused_parameters,
        static_graph=cfg.ddp_static_graph,
    )


class ParallelMixin(DeviceMixin[Config], LoggerMixin[Config], Generic[Config]):
    """Defines a task mixin for converting models to FSDP."""

    @functools.cached_property
    def grad_scaler(self) -> ShardedGradScaler | None:
        if not self.config.grad_scaler_enabled:
            return None
        if self.device_manager.device.type != "cuda":
            return None
        if self.device_manager.dtype not in (torch.float16, torch.bfloat16):
            return None
        return ShardedGradScaler(
            init_scale=self.config.grad_scaler.init_scale,
            growth_factor=self.config.grad_scaler.growth_factor,
            backoff_factor=self.config.grad_scaler.backoff_factor,
            growth_interval=self.config.grad_scaler.growth_interval,
            enabled=True,
        )

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

    def get_wrapped_model(self, model: nn.Module) -> FSDP | DDP | nn.Module:
        if get_world_size() <= 1:
            return model
        if self.config.use_ddp:
            return ddp(model, self.config)
        return fsdp(model, self.config, self.torch_device, self.get_fsdp_mixed_precision())

    def get_grad_sync_context(self, mod: nn.Module, is_last: bool) -> ContextManager:
        if (isinstance(mod, DDP | FSDP)) and not is_last:
            return mod.no_sync()
        return contextlib.nullcontext()

    def backward_grads(
        self,
        loss: Tensor,
        retain_graph: bool | None = None,
        inputs: Sequence[Tensor] | None = None,
    ) -> None:
        if self.grad_scaler is not None:
            loss = self.grad_scaler.scale(loss)
        if loss.numel() > 1:
            loss = loss.sum()
        isnan = not bool(torch.isfinite(loss))
        if isnan:
            loss.backward(torch.zeros_like(loss), retain_graph=retain_graph, inputs=inputs)
        else:
            loss.backward(retain_graph=retain_graph, inputs=inputs)

        if isnan:
            if any(not torch.isfinite(p).all() for p in self.parameters()):
                raise NaNError("One or more model parameters are NaN")
            if self.grad_scaler is not None:
                with torch.no_grad():
                    new_scale = self.grad_scaler.get_scale() * self.grad_scaler.get_backoff_factor()
                    if new_scale < self.config.grad_scaler.min_grad_scale:
                        raise MinGradScaleError("Minimum gradient scale reached; your loss is probably exploding")
                    logger.warning("Loss NaNs detected; reducing scale to %.2g", new_scale)
                    self.grad_scaler.update(new_scale)

    @functools.cached_property
    def device_mesh(self) -> DeviceMesh:
        return parallel_group_info().device_mesh(self.torch_device.type)

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
        else:
            if self.config.debug_nan_grads:
                logger.warning(
                    "Found NaN gradients for parameters %s",
                    [p for p in mod.parameters() if p.grad is not None and not torch.isfinite(p.grad).all()],
                )
            if self.grad_scaler is not None:
                with torch.no_grad():
                    new_scale = self.grad_scaler.get_scale() * self.grad_scaler.get_backoff_factor()
                    if new_scale < self.config.grad_scaler.min_grad_scale:
                        raise MinGradScaleError("Minimum gradient scale reached; your loss is probably exploding")
                    logger.warning("Loss NaNs detected; reducing scale to %.2g", new_scale)
                    self.grad_scaler.update(new_scale)

    @functools.cached_property
    def autocast_context(self) -> ContextManager:
        return self.device_manager.autocast_context()

    def scale_mixed_precision(self, tensor: Tensor) -> Tensor:
        if self.grad_scaler is not None:
            return self.grad_scaler.scale(tensor)
        return tensor

    def log_mp_scale(self) -> None:
        if (scaler := self.grad_scaler) is not None and scaler._enabled:
            self.log_scalar("scale", scaler.get_scale, namespace="⚖️ fp16")
            self.log_scalar("growth", scaler._get_growth_tracker, namespace="⚖️ fp16")

    def load_task_state_dict_(
        self,
        state_dict: dict,
        strict: bool = True,
        assign: bool = False,
        weights_only: bool = False,
    ) -> None:
        if self.grad_scaler is not None and "grad_scaler" in state_dict:
            self.grad_scaler.load_state_dict(json.loads(state_dict["grad_scaler"]))
        super().load_task_state_dict_(state_dict, strict, assign, weights_only)

    def task_state_dict(self) -> dict:
        state_dict = super().task_state_dict()
        if self.grad_scaler is not None:
            assert "grad_scaler" not in state_dict, "Duplicate keys!"
            state_dict["grad_scaler"] = json.dumps(self.grad_scaler.state_dict())
        return state_dict
