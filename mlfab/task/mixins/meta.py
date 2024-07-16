"""Defines a mixin to support initializing models with the meta device."""

import functools
import itertools
import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Generic, TypeVar

import torch
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.rnn import RNNBase, RNNCellBase

from mlfab.core.conf import field
from mlfab.task.mixins.device import DeviceConfig, DeviceMixin
from mlfab.task.mixins.pretrained import PretrainedMixin, PretrainedModule
from mlfab.utils.nn import ResetParameters

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class MetaConfig(DeviceConfig):
    throw_error_on_failed_parameter_reset: bool = field(False, help="If set, all modules must have a weight init")


Config = TypeVar("Config", bound=MetaConfig)


class MetaMixin(DeviceMixin[Config], Generic[Config]):
    """Defines a task mixin for initializing models to the meta device."""

    def configure_model_(self, model: nn.Module) -> None:
        """Configures the model parameters.

        Since modules are initialized to the empty device by default, we need
        to move them to the torch device. For pre-trained modules, we call the
        load function to load the pre-trained weights, and check that there
        aren't any meta tensors left after loading. For other modules, we call
        the `init_weights_` method to reset the parameters.
        """

        def iter_tensors(module: nn.Module, recurse: bool) -> itertools.chain[Tensor]:
            return itertools.chain(module.parameters(recurse=recurse), module.buffers(recurse=recurse))

        def has_meta(module: nn.Module, recurse: bool) -> bool:
            return any(p.is_meta for p in iter_tensors(module, recurse))

        def to_empty(t: Tensor, use_device_dtype: bool) -> Tensor:
            if t.is_meta:
                if t.is_floating_point() and use_device_dtype:
                    return torch.empty_like(t, device=self.torch_device, dtype=self.torch_dtype)
                return torch.empty_like(t, device=self.torch_device)

            if t.is_floating_point() and use_device_dtype:
                return t.to(self.torch_device, self.torch_dtype)
            return t.to(self.torch_device)

        def init_weights_(module: nn.Module) -> None:
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
                    "Module %s has a `reset_parameters` method but is not a known module type; assuming duck-typed "
                    "`reset_parameters` method. You should subclass `mlfab.ResetParameters` instead.",
                    type(module),
                )
                module.reset_parameters()
            elif any(True for _ in iter_tensors(module, recurse=False)):
                raise RuntimeError(f"Encountered a module without a weight initialization: {module}")

        module_queue: Deque[nn.Module | PretrainedModule] = deque()
        module_queue.append(model)
        weight_reset_queue: Deque[nn.Module] = deque()

        # Converts meta tensors to empty tensors on the target device.
        while len(module_queue) > 0:
            module = module_queue.popleft()
            if isinstance(module, PretrainedModule):
                module.module._apply(
                    functools.partial(to_empty, use_device_dtype=module.use_device_dtype),
                    recurse=True,
                )
                module.load()
                if has_meta(module.module, recurse=True):
                    raise RuntimeError("Pretrained module has meta tensors after loading!")
            else:
                module._apply(functools.partial(to_empty, use_device_dtype=False), recurse=False)
                weight_reset_queue.append(module)
                for child in module.children():
                    module_queue.append(child)
                if isinstance(module, PretrainedMixin):
                    for pretrained_child in module._pretrained_modules:
                        module_queue.append(pretrained_child)

        # Resets the module weights, starting at the leaves of the model.
        while len(weight_reset_queue) > 0:
            init_weights_(weight_reset_queue.pop())
