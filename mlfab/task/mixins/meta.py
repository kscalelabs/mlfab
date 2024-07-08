"""Defines a mixin to support initializing models with the meta device."""

import itertools
import logging
from dataclasses import dataclass
from queue import Queue
from typing import Generic, TypeVar

import torch
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.rnn import RNNBase, RNNCellBase

from mlfab.core.conf import field
from mlfab.task.mixins.device import DeviceConfig, DeviceMixin
from mlfab.task.mixins.pretrained import PretrainedModule
from mlfab.utils.nn import ResetParameters

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class MetaConfig(DeviceConfig):
    throw_error_on_failed_parameter_reset: bool = field(False, help="If set, all modules must have a weight init")


Config = TypeVar("Config", bound=MetaConfig)


def has_meta(module: nn.Module, recurse: bool = True) -> bool:
    return any(
        itertools.chain(
            (param.is_meta for param in module.parameters(recurse=recurse)),
            (buffer.is_meta for buffer in module.buffers(recurse=recurse)),
        )
    )


class MetaMixin(DeviceMixin[Config], Generic[Config]):
    """Defines a task mixin for initializing models to the meta device."""

    def configure_model_(self, model: nn.Module) -> None:
        self.reset_parameters_(model)

    def to_empty_if_meta(self, t: Tensor) -> Tensor:
        if t.is_meta:
            if t.is_floating_point():
                return torch.empty_like(t, device=self.torch_device, dtype=self.torch_dtype)
            return torch.empty_like(t, device=self.torch_device)

        if t.is_floating_point():
            return t.to(self.torch_device, self.torch_dtype)
        return t.to(self.torch_device)

    def reset_parameters_(self, model: nn.Module) -> None:
        """Recursively resets model parameters.

        Since modules are initialized to the empty device by default, we need
        to move them to the torch device. For pre-trained modules, we call the
        load function to load the pre-trained weights, and check that there
        aren't any meta tensors left after loading. For other modules, we call
        the `init_weights_` method to reset the parameters.
        """
        module_queue: Queue[nn.Module] = Queue()
        module_queue.put(model)
        while not module_queue.empty():
            module = module_queue.get()
            module._apply(self.to_empty_if_meta, recurse=False)
            if isinstance(module, PretrainedModule):
                module.load()
                if has_meta(module):
                    raise RuntimeError("Pretrained module has meta tensors after loading!")
            else:
                self.init_weights_(module)
                for child in module.children():
                    module_queue.put(child)

    def init_weights_(self, module: nn.Module) -> None:
        if isinstance(
            module,
            (
                _BatchNorm,
                _ConvNd,
                nn.AdaptiveLogSoftmaxWithLoss,
                nn.Bilinear,
                nn.Embedding,
                nn.EmbeddingBag,
                nn.GroupNorm,
                nn.LayerNorm,
                nn.LazyLinear,
                nn.Linear,
                nn.LSTM,
                nn.PReLU,
                RNNBase,
                RNNCellBase,
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
