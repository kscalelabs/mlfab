"""Defines a mixin to support initializing models with the meta device."""

import logging
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.rnn import RNNBase, RNNCellBase

from mlfab.core.conf import field
from mlfab.task.mixins.device import DeviceConfig, DeviceMixin
from mlfab.utils.nn import ResetParameters

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class MetaConfig(DeviceConfig):
    throw_error_on_failed_parameter_reset: bool = field(False, help="If set, all modules must have a weight init")


Config = TypeVar("Config", bound=MetaConfig)


class MetaMixin(DeviceMixin[Config], Generic[Config]):
    """Defines a task mixin for initializing models to the meta device."""

    def configure_model_(self, model: nn.Module) -> None:
        self.meta_to_empty_(model)
        self.reset_parameters_(model)

    def meta_to_empty_(self, module: nn.Module) -> None:
        def to_empty(t: Tensor) -> Tensor:
            if t.is_floating_point():
                return torch.empty_like(t, device=self.torch_device, dtype=self.torch_dtype)
            return torch.empty_like(t, device=self.torch_device)

        module._apply(to_empty, recurse=True)

    def reset_parameters_(self, model: nn.Module) -> None:
        model.apply(self.init_weights_)

    def init_weights_(self, module: nn.Module) -> None:
        if isinstance(
            module,
            (
                _ConvNd,
                _BatchNorm,
                nn.Linear,
                nn.LazyLinear,
                nn.Bilinear,
                nn.Embedding,
                nn.EmbeddingBag,
                nn.LSTM,
                RNNBase,
                nn.PReLU,
                RNNCellBase,
                nn.LayerNorm,
                nn.GroupNorm,
                nn.AdaptiveLogSoftmaxWithLoss,
            ),
        ):
            module.reset_parameters()
        elif isinstance(module, (nn.MultiheadAttention, nn.Transformer)):
            module._reset_parameters()
        elif isinstance(module, ResetParameters):
            module.reset_parameters()
        elif hasattr(module, "reset_parameters"):
            logger.warning(
                "Module %s has a `reset_parameters` method but is not a known module type; "
                "assuming duck-typed `reset_parameters` method. You should subclass `mlfab.ResetParameters` instead.",
                type(module),
            )
            module.reset_parameters()
        elif any(True for _ in module.parameters(recurse=False)) or any(True for _ in module.buffers(recurse=False)):
            raise RuntimeError(f"Encountered a module without a weight initialization: {module}")
