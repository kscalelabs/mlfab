"""Helper utilities for using LoRA layers.

LoRA layers are drop-in replacements for certain modules, which can be used
for fine-tuning pre-trained models. It is described in the paper
`LoRA: Low-Rank Adaptation of Large Language Models <https://arxiv.org/abs/2106.09685>`_.

.. highlight:: python
.. code-block:: python

    from mlfab.nn.lora import lora

    # The pre-trained model weights can be loaded into the LoRA model.
    model = nn.Sequential(nn.Linear(5, 7), nn.Linear(7, 5))
    lora_model = nn.Sequential(lora(nn.Linear(5, 7)), lora(nn.Linear(7, 5)))
    lora_model.load_state_dict(model.state_dict())  # No errors

    from mlfab.nn.lora import LoRALinear

    # Alternatively, you can just substitute the module name.
    model = nn.Sequential(LoRALinear(5, 7), LoRALinear(7, 5))

The modules which can be wrapped with LoRA modules are:

- ``nn.Embedding``
- ``nn.Linear``
- ``nn.Conv1d``
- ``nn.ConvTranspose1d``
- ``nn.Conv2d``
- ``nn.ConvTranspose2d``
- ``nn.LSTM``
- ``nn.GRU``
- ``ColumnParallelLinear``
- ``RowParallelLinear``
- ``ParallelEmbedding``

In the paper, the authors typically use values of 1, 2, 4, or 8 for the
``r`` parameter. The ``lora_alpha`` parameter is typically set to 1.0, but
can be tuned to improve performance.
"""

import math
import warnings
import weakref
from abc import abstractmethod
from typing import Any, TypeVar, Union, cast, overload

import torch
import torch.nn.functional as F
from torch import _VF, Tensor, nn
from torch.nn.modules.module import _IncompatibleKeys

from mlfab.nn.init import InitializationType
from mlfab.nn.parallel import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
    mp_copy,
    mp_gather,
    mp_reduce,
    mp_scatter,
)

T = TypeVar("T")

SupportedModuleNonParallel = Union[
    nn.Embedding,
    nn.Linear,
    nn.Conv1d,
    nn.ConvTranspose1d,
    nn.Conv2d,
    nn.ConvTranspose2d,
    nn.LSTM,
    nn.GRU,
    nn.LSTMCell,
    nn.GRUCell,
]

SupportedModule = Union[
    SupportedModuleNonParallel,
    ColumnParallelLinear,
    RowParallelLinear,
    ParallelEmbedding,
]


def _lora_post_hook(module: "_Lora", incompatible_keys: _IncompatibleKeys) -> None:
    lora_keys = [k for k in incompatible_keys.missing_keys if k.split(".")[-1].startswith("lora_")]
    for lora_key in lora_keys:
        incompatible_keys.missing_keys.remove(lora_key)


class _Lora(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(*args, **kwargs)

        # This allows modules to use LoRA layers as drop-in replacements for
        # non-LoRA pretrained models without throwing annoying errors for
        # state dict incompatibility.
        self.register_load_state_dict_post_hook(_lora_post_hook)

    @abstractmethod
    def reset_lora_parameters(self) -> None:
        """Resets LoRA parameters in-place."""


class LoraEmbedding(nn.Embedding, _Lora):
    __constants__ = nn.Embedding.__constants__ + ["r", "lora_alpha", "scaling", "merge", "merged"]

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        merge: bool = False,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
    ) -> None:
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
        )

        assert r > 0

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        self.merge = merge

        self.dropout = nn.Identity() if lora_dropout == 0.0 else nn.Dropout(p=lora_dropout)
        self.merged = False

        self.lora_a = nn.Parameter(self.weight.new_empty((r, num_embeddings)))
        self.lora_b = nn.Parameter(self.weight.new_empty((embedding_dim, r)))
        self.weight.requires_grad_(False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()

        if hasattr(self, "lora_a") and hasattr(self, "lora_b"):
            self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_normal_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def train(self, mode: bool = True) -> "LoraEmbedding":
        super().train(mode)

        if mode:
            if self.merge and self.merged:
                # Make sure that the weights are not merged
                if self.lora_a is not None and self.lora_b is not None:
                    self.weight.data -= (self.lora_b @ self.lora_a).transpose(0, 1) * self.scaling
                self.merged = False
        elif self.merge and not self.merged:
            # Merge the weights and mark it
            if self.lora_a is not None and self.lora_b is not None:
                self.weight.data += (self.lora_b @ self.lora_a).transpose(0, 1) * self.scaling
            self.merged = True

        return self

    def forward(self, x: Tensor) -> Tensor:
        if self.lora_a is not None and self.lora_b is not None and not self.merged:
            result = super().forward(x)
            after_a = F.embedding(
                x,
                self.lora_a.transpose(0, 1),
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
            return result + (after_a @ self.lora_b.transpose(0, 1)) * self.scaling

        return super().forward(x)


class LoraLinear(nn.Linear, _Lora):
    __constants__ = nn.Linear.__constants__ + ["r", "lora_alpha", "scaling", "merge", "fan_in_fan_out", "merged"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
        )

        assert r > 0

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        self.merge = merge
        self.fan_in_fan_out = fan_in_fan_out

        self.dropout = nn.Identity() if lora_dropout == 0.0 else nn.Dropout(p=lora_dropout)
        self.merged = False

        self.lora_a = nn.Parameter(self.weight.new_empty((r, in_features)))
        self.lora_b = nn.Parameter(self.weight.new_empty((out_features, r)))
        self.weight.requires_grad_(False)

        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self) -> None:
        super().reset_parameters()

        if hasattr(self, "lora_a") and hasattr(self, "lora_b"):
            self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_normal_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def _t(self, w: Tensor) -> Tensor:
        return w.transpose(0, 1) if self.fan_in_fan_out else w

    def train(self, mode: bool = True) -> "LoraLinear":
        super().train(mode)

        if mode:
            if self.merge and self.merged:
                # Make sure that the weights are not merged
                if self.lora_a is not None and self.lora_b is not None:
                    self.weight.data -= self._t(self.lora_b @ self.lora_a) * self.scaling
                self.merged = False

        elif self.merge and not self.merged:
            # Merge the weights and mark it
            if self.lora_a is not None and self.lora_b is not None:
                self.weight.data += self._t(self.lora_b @ self.lora_a) * self.scaling
            self.merged = True

        return self

    def forward(self, x: Tensor) -> Tensor:
        if self.lora_a is not None and self.lora_b is not None and not self.merged:
            result = F.linear(x, self._t(self.weight), bias=self.bias)
            mm = self.dropout(x) @ self.lora_a.transpose(0, 1) @ self.lora_b.transpose(0, 1)
            return result + mm * self.scaling

        return F.linear(x, self._t(self.weight), bias=self.bias)


class LoraConv1d(nn.Conv1d, _Lora):
    __constants__ = nn.Conv1d.__constants__ + ["r", "lora_alpha", "scaling", "merge", "merged"]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        r: int,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        merge: bool = False,
        stride: int | tuple[int] = 1,
        padding: str | int | tuple[int] = 0,
        dilation: int | tuple[int] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        assert r > 0

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        self.merge = merge

        self.dropout = nn.Identity() if lora_dropout == 0.0 else nn.Dropout(p=lora_dropout)
        self.merged = False

        self.lora_a = nn.Parameter(self.weight.new_empty((r, in_channels, *self.kernel_size)))
        self.lora_b = nn.Parameter(self.weight.new_empty((out_channels, r, 1)))
        self.weight.requires_grad_(False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()

        if hasattr(self, "lora_a") and hasattr(self, "lora_b"):
            self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_normal_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def train(self, mode: bool = True) -> "LoraConv1d":
        super().train(mode)

        if mode:
            if self.merge and self.merged:
                # Make sure that the weights are not merged
                if self.lora_a is not None and self.lora_b is not None:
                    self.weight.data -= self.lora_b @ self.lora_a * self.scaling
                self.merged = False

        elif self.merge and not self.merged:
            # Merge the weights and mark it
            if self.lora_a is not None and self.lora_b is not None:
                self.weight.data += self.lora_b @ self.lora_a * self.scaling
            self.merged = True

        return self

    def forward(self, x: Tensor) -> Tensor:
        if self.lora_a is not None and self.lora_b is not None and not self.merged:
            result = F.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            mm_a = F.conv1d(self.dropout(x), self.lora_a, None, self.stride, self.padding, self.dilation, self.groups)
            mm = F.conv1d(mm_a, self.lora_b)
            return result + mm * self.scaling

        return F.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LoraConvTranspose1d(nn.ConvTranspose1d, _Lora):
    __constants__ = nn.ConvTranspose1d.__constants__ + ["r", "lora_alpha", "scaling", "merge", "merged"]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        r: int,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        merge: bool = False,
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] = 0,
        output_padding: int | tuple[int] = 0,
        dilation: int | tuple[int] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        assert r > 0

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        self.merge = merge

        self.dropout = nn.Identity() if lora_dropout == 0.0 else nn.Dropout(p=lora_dropout)
        self.merged = False

        self.lora_a = nn.Parameter(self.weight.new_empty((in_channels, r, *self.kernel_size)))
        self.lora_b = nn.Parameter(self.weight.new_empty((r, out_channels, 1)))
        self.weight.requires_grad_(False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()

        if hasattr(self, "lora_a") and hasattr(self, "lora_b"):
            self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_normal_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def train(self, mode: bool = True) -> "LoraConvTranspose1d":
        super().train(mode)

        if mode:
            if self.merge and self.merged:
                # Make sure that the weights are not merged
                if self.lora_a is not None and self.lora_b is not None:
                    self.weight.data -= self.lora_b @ self.lora_a * self.scaling
                self.merged = False

        elif self.merge and not self.merged:
            # Merge the weights and mark it
            if self.lora_a is not None and self.lora_b is not None:
                self.weight.data += self.lora_b @ self.lora_a * self.scaling
            self.merged = True

        return self

    def forward(self, x: Tensor, output_size: list[int] | None = None) -> Tensor:
        assert isinstance(self.padding, tuple)

        if self.lora_a is not None and self.lora_b is not None and not self.merged:
            result = F.conv_transpose1d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.output_padding,
                self.groups,
                self.dilation,
            )
            mm_a = F.conv_transpose1d(
                self.dropout(x),
                self.lora_a,
                None,
                self.stride,
                self.padding,
                self.output_padding,
                self.groups,
                self.dilation,
            )
            mm = F.conv_transpose1d(mm_a, self.lora_b)
            return result + mm * self.scaling

        return F.conv_transpose1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LoraConv2d(nn.Conv2d, _Lora):
    __constants__ = nn.Conv2d.__constants__ + ["r", "lora_alpha", "scaling", "merge", "merged"]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        r: int,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        merge: bool = False,
        stride: int | tuple[int, int] = (1, 1),
        padding: str | int | tuple[int, int] = (0, 0),
        dilation: int | tuple[int, int] = (1, 1),
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        assert r > 0

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        self.merge = merge

        self.dropout = nn.Identity() if lora_dropout == 0.0 else nn.Dropout(p=lora_dropout)
        self.merged = False

        self.lora_a = nn.Parameter(self.weight.new_empty((r, in_channels, *self.kernel_size)))
        self.lora_b = nn.Parameter(self.weight.new_empty((out_channels, r, 1, 1)))
        self.weight.requires_grad_(False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()

        if hasattr(self, "lora_a") and hasattr(self, "lora_b"):
            self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_normal_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def train(self, mode: bool = True) -> "LoraConv2d":
        super().train(mode)

        if mode:
            if self.merge and self.merged:
                # Make sure that the weights are not merged
                if self.lora_a is not None and self.lora_b is not None:
                    self.weight.data -= self.lora_b @ self.lora_a * self.scaling
                self.merged = False

        elif self.merge and not self.merged:
            # Merge the weights and mark it
            if self.lora_a is not None and self.lora_b is not None:
                self.weight.data += self.lora_b @ self.lora_a * self.scaling
            self.merged = True

        return self

    def forward(self, x: Tensor) -> Tensor:
        if self.lora_a is not None and self.lora_b is not None and not self.merged:
            result = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            mm_a = F.conv2d(self.dropout(x), self.lora_a, None, self.stride, self.padding, self.dilation, self.groups)
            mm = F.conv2d(mm_a, self.lora_b)
            return result + mm * self.scaling

        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LoraConvTranspose2d(nn.ConvTranspose2d, _Lora):
    __constants__ = nn.ConvTranspose2d.__constants__ + ["r", "lora_alpha", "scaling", "merge", "merged"]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        r: int,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        merge: bool = False,
        stride: int | tuple[int, int] = (1, 1),
        padding: int | tuple[int, int] = (0, 0),
        output_padding: int | tuple[int, int] = (0, 0),
        dilation: int | tuple[int, int] = (1, 1),
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        assert r > 0

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        self.merge = merge

        self.dropout = nn.Identity() if lora_dropout == 0.0 else nn.Dropout(p=lora_dropout)
        self.merged = False

        self.lora_a = nn.Parameter(self.weight.new_empty((in_channels, r, *self.kernel_size)))
        self.lora_b = nn.Parameter(self.weight.new_empty((r, out_channels, 1, 1)))
        self.weight.requires_grad_(False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()

        if hasattr(self, "lora_a") and hasattr(self, "lora_b"):
            self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_normal_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def train(self, mode: bool = True) -> "LoraConvTranspose2d":
        super().train(mode)

        if mode:
            if self.merge and self.merged:
                # Make sure that the weights are not merged
                if self.lora_a is not None and self.lora_b is not None:
                    self.weight.data -= self.lora_b @ self.lora_a * self.scaling
                self.merged = False

        elif self.merge and not self.merged:
            # Merge the weights and mark it
            if self.lora_a is not None and self.lora_b is not None:
                self.weight.data += self.lora_b @ self.lora_a * self.scaling
            self.merged = True

        return self

    def forward(self, x: Tensor, output_size: list[int] | None = None) -> Tensor:
        assert isinstance(self.padding, tuple)

        if self.lora_a is not None and self.lora_b is not None and not self.merged:
            result = F.conv_transpose2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.output_padding,
                self.groups,
                self.dilation,
            )
            mm_a = F.conv_transpose2d(
                self.dropout(x),
                self.lora_a,
                None,
                self.stride,
                self.padding,
                self.output_padding,
                self.groups,
                self.dilation,
            )
            mm = F.conv_transpose2d(mm_a, self.lora_b)
            return result + mm * self.scaling

        return F.conv_transpose2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )


class _LoraRNN(nn.RNNBase, _Lora):
    __constants__ = nn.RNNBase.__constants__ + ["r", "lora_alpha", "scaling"]

    def __init__(
        self,
        mode: str,
        input_size: int,
        hidden_size: int,
        gate_mul: int,
        r: int,
        lora_alpha: float = 1.0,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
    ) -> None:
        super().__init__(
            mode=mode,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            proj_size=proj_size,
        )

        assert r > 0

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r

        num_directions = 2 if bidirectional else 1
        gate_size = gate_mul * hidden_size

        for layer in range(num_layers):
            for direction in range(num_directions):
                real_hidden_size = proj_size if proj_size > 0 else hidden_size
                layer_input_size = input_size if layer == 0 else real_hidden_size * num_directions

                suffix = "_reverse" if direction == 1 else ""
                w_ih: Tensor = getattr(self, f"weight_ih_l{layer}{suffix}")
                w_hh: Tensor = getattr(self, f"weight_hh_l{layer}{suffix}")
                w_ih.requires_grad_(False)
                w_hh.requires_grad_(False)
                lora_a_ih = nn.Parameter(w_ih.new_empty((r, gate_size)))
                lora_b_ih = nn.Parameter(w_ih.new_empty((layer_input_size, r)))
                lora_a_hh = nn.Parameter(w_hh.new_empty((r, gate_size)))
                lora_b_hh = nn.Parameter(w_hh.new_empty((real_hidden_size, r)))
                setattr(self, f"lora_a_ih_l{layer}{suffix}", lora_a_ih)
                setattr(self, f"lora_b_ih_l{layer}{suffix}", lora_b_ih)
                setattr(self, f"lora_a_hh_l{layer}{suffix}", lora_a_hh)
                setattr(self, f"lora_b_hh_l{layer}{suffix}", lora_b_hh)

                if self.proj_size != 0:
                    w_hr: Tensor = getattr(self, f"weight_hr_l{layer}{suffix}")
                    w_hr.requires_grad_(False)
                    lora_a_hr = nn.Parameter(w_hr.new_empty((r, proj_size)))
                    lora_b_hr = nn.Parameter(w_hr.new_empty((hidden_size, r)))
                    setattr(self, f"lora_a_hr_l{layer}{suffix}", lora_a_hr)
                    setattr(self, f"lora_b_hr_l{layer}{suffix}", lora_b_hr)

        self._init_flat_weights()

        self.reset_parameters()

    def _lora_names(self, weight_name: str) -> tuple[str, str]:
        weight_name = weight_name[len("weight_") :]
        lora_a_name, lora_b_name = f"lora_a_{weight_name}", f"lora_b_{weight_name}"
        return lora_a_name, lora_b_name

    def _get_weight(self, weight_name: str) -> Tensor:
        weight = getattr(self, weight_name)
        if weight_name.startswith("bias_"):
            return weight
        lora_a_name, lora_b_name = self._lora_names(weight_name)
        if not hasattr(self, lora_a_name) or not hasattr(self, lora_b_name):
            return weight
        lora_a, lora_b = getattr(self, lora_a_name), getattr(self, lora_b_name)
        return weight + (lora_a.transpose(0, 1) @ lora_b.transpose(0, 1)) * self.scaling

    def _init_flat_weights(self) -> None:
        self._flat_weights = [self._get_weight(wn) if hasattr(self, wn) else None for wn in self._flat_weights_names]
        self._flat_weight_refs = [weakref.ref(w) if w is not None else None for w in self._flat_weights]
        self.flatten_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()

        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        for wn in self._flat_weights_names:
            lora_a_name, lora_b_name = self._lora_names(wn)
            if hasattr(self, lora_a_name) and hasattr(self, lora_b_name):
                lora_a, lora_b = getattr(self, lora_a_name), getattr(self, lora_b_name)
                nn.init.kaiming_normal_(lora_a, a=math.sqrt(5))
                nn.init.zeros_(lora_b)


class LoraLSTM(nn.LSTM, _LoraRNN):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        r: int,
        lora_alpha: float = 1.0,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
    ) -> None:
        _LoraRNN.__init__(
            self,
            mode="LSTM",
            input_size=input_size,
            hidden_size=hidden_size,
            gate_mul=4,
            r=r,
            lora_alpha=lora_alpha,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            proj_size=proj_size,
        )


class LoraGRU(nn.GRU, _LoraRNN):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        r: int,
        lora_alpha: float = 1.0,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
    ) -> None:
        _LoraRNN.__init__(
            self,
            mode="GRU",
            input_size=input_size,
            hidden_size=hidden_size,
            gate_mul=3,
            r=r,
            lora_alpha=lora_alpha,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            proj_size=proj_size,
        )


class _LoraRNNCellBase(nn.RNNCellBase, _Lora):
    __constants__ = nn.RNNCell.__constants__ + ["r", "lora_alpha", "scaling"]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool,
        num_chunks: int,
        r: int,
        lora_alpha: float = 1.0,
    ) -> None:
        super().__init__(input_size=input_size, hidden_size=hidden_size, bias=bias, num_chunks=num_chunks)

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r

        self.lora_a_ih = nn.Parameter(self.weight_ih.new_empty((r, input_size)))
        self.lora_b_ih = nn.Parameter(self.weight_ih.new_empty((hidden_size * num_chunks, r)))
        self.lora_a_hh = nn.Parameter(self.weight_hh.new_empty((r, hidden_size)))
        self.lora_b_hh = nn.Parameter(self.weight_hh.new_empty((hidden_size * num_chunks, r)))
        self.weight_ih.requires_grad_(False)
        self.weight_hh.requires_grad_(False)

    def reset_parameters(self) -> None:
        super().reset_parameters()

        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        if hasattr(self, "lora_a_ih") and hasattr(self, "lora_b_ih"):
            nn.init.kaiming_normal_(self.lora_a_ih, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b_ih)
        if hasattr(self, "lora_a_hh") and hasattr(self, "lora_b_hh"):
            nn.init.kaiming_normal_(self.lora_a_hh, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b_hh)


class LoraLSTMCell(nn.LSTMCell, _LoraRNNCellBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        r: int,
        bias: bool = True,
        lora_alpha: float = 1.0,
    ) -> None:
        _LoraRNNCellBase.__init__(
            self,
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            num_chunks=4,
            r=r,
            lora_alpha=lora_alpha,
        )

    def forward(self, input: Tensor, hx: tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, Tensor]:
        assert input.dim() in (1, 2), f"LSTMCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)
        if hx is None:
            zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        else:
            hx = (hx[0].unsqueeze(0), hx[1].unsqueeze(0)) if not is_batched else hx
        lora_ih = (self.lora_b_ih @ self.lora_a_ih) * self.scaling
        lora_hh = (self.lora_b_hh @ self.lora_a_hh) * self.scaling
        ret = _VF.lstm_cell(input, hx, self.weight_ih + lora_ih, self.weight_hh + lora_hh, self.bias_ih, self.bias_hh)
        if not is_batched:
            ret = (ret[0].squeeze(0), ret[1].squeeze(0))
        return ret


class LoraGRUCell(nn.GRUCell, _LoraRNNCellBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        r: int,
        bias: bool = True,
        lora_alpha: float = 1.0,
    ) -> None:
        _LoraRNNCellBase.__init__(
            self,
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            num_chunks=3,
            r=r,
            lora_alpha=lora_alpha,
        )

    def forward(self, input: Tensor, hx: Tensor | None = None) -> Tensor:
        assert input.dim() in (1, 2), f"GRUCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        else:
            hx = hx.unsqueeze(0) if not is_batched else hx
        lora_ih = (self.lora_b_ih @ self.lora_a_ih) * self.scaling
        lora_hh = (self.lora_b_hh @ self.lora_a_hh) * self.scaling
        ret = _VF.gru_cell(input, hx, self.weight_ih + lora_ih, self.weight_hh + lora_hh, self.bias_ih, self.bias_hh)
        if not is_batched:
            ret = ret.squeeze(0)
        return ret


class LoraParallelEmbedding(ParallelEmbedding, _Lora):
    __constants__ = ParallelEmbedding.__constants__ + ["r", "lora_alpha", "merge", "scaling", "merged"]

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        merge: bool = False,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        init_type: InitializationType = "xavier_normal",
    ) -> None:
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            init_type=init_type,
        )

        assert r > 0

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        self.merge = merge

        self.dropout = nn.Identity() if lora_dropout == 0.0 else nn.Dropout(p=lora_dropout)
        self.merged = False

        self.lora_a = nn.Parameter(self.weight.new_empty((r, num_embeddings)))
        self.lora_b = nn.Parameter(self.weight.new_empty((self.embedding_dim_per_rank, r)))
        self.weight.requires_grad_(False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()

        if hasattr(self, "lora_a") and hasattr(self, "lora_b"):
            self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_normal_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def train(self, mode: bool = True) -> "LoraParallelEmbedding":
        super().train(mode)

        if mode:
            if self.merge and self.merged:
                # Make sure that the weights are not merged
                if self.lora_a is not None and self.lora_b is not None:
                    self.weight.data -= (self.lora_b @ self.lora_a).transpose(0, 1) * self.scaling
                self.merged = False
        elif self.merge and not self.merged:
            # Merge the weights and mark it
            if self.lora_a is not None and self.lora_b is not None:
                self.weight.data += (self.lora_b @ self.lora_a).transpose(0, 1) * self.scaling
            self.merged = True

        return self

    def forward(self, x: Tensor) -> Tensor:
        x = mp_copy(x)

        if self.lora_a is not None and self.lora_b is not None and not self.merged:
            output_parallel = F.embedding(
                x,
                self.weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )

            after_a_parallel = F.embedding(
                x,
                self.lora_a.transpose(0, 1),
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )

            return mp_gather(output_parallel + (after_a_parallel @ self.lora_b.transpose(0, 1)) * self.scaling)

        return mp_gather(output_parallel)


class LoraColumnParallelLinear(ColumnParallelLinear, _Lora):
    __constants__ = ColumnParallelLinear.__constants__ + [
        "r",
        "lora_alpha",
        "scaling",
        "merge",
        "fan_in_fan_out",
        "merged",
    ]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge: bool = False,
        bias: bool = True,
        gather_output: bool = True,
        init_type: InitializationType = "xavier_normal",
        stride: int = 1,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            gather_output=gather_output,
            init_type=init_type,
            stride=stride,
        )

        assert r > 0

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        self.merge = merge
        self.fan_in_fan_out = fan_in_fan_out

        self.dropout = nn.Identity() if lora_dropout == 0.0 else nn.Dropout(p=lora_dropout)
        self.merged = False

        self.lora_a = nn.Parameter(self.weight.new_empty((r, in_features)))
        self.lora_b = nn.Parameter(self.weight.new_empty((self.output_size_per_partition, r)))
        self.weight.requires_grad_(False)

        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self) -> None:
        super().reset_parameters()

        if hasattr(self, "lora_a") and hasattr(self, "lora_b"):
            self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_normal_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def _t(self, w: Tensor) -> Tensor:
        return w.transpose(0, 1) if self.fan_in_fan_out else w

    def train(self, mode: bool = True) -> "LoraColumnParallelLinear":
        super().train(mode)

        if mode:
            if self.merge and self.merged:
                # Make sure that the weights are not merged
                if self.lora_a is not None and self.lora_b is not None:
                    self.weight.data -= self._t(self.lora_b @ self.lora_a) * self.scaling
                self.merged = False

        elif self.merge and not self.merged:
            # Merge the weights and mark it
            if self.lora_a is not None and self.lora_b is not None:
                self.weight.data += self._t(self.lora_b @ self.lora_a) * self.scaling
            self.merged = True

        return self

    def forward(self, x: Tensor) -> Tensor:
        input_parallel = mp_copy(x)

        if self.lora_a is not None and self.lora_b is not None and not self.merged:
            output_parallel = F.linear(input_parallel, self._t(self.weight), bias=self.bias)
            mm = self.dropout(input_parallel) @ self.lora_a.transpose(0, 1) @ self.lora_b.transpose(0, 1)
            output_parallel = output_parallel + mm * self.scaling
            return mp_gather(output_parallel) if self.gather_output else output_parallel

        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        return mp_gather(output_parallel) if self.gather_output else output_parallel


class LoraRowParallelLinear(RowParallelLinear, _Lora):
    __constants__ = RowParallelLinear.__constants__ + [
        "r",
        "lora_alpha",
        "scaling",
        "merge",
        "fan_in_fan_out",
        "merged",
    ]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge: bool = False,
        bias: bool = True,
        input_is_parallel: bool = False,
        init_type: InitializationType = "xavier_normal",
        stride: int = 1,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            input_is_parallel=input_is_parallel,
            init_type=init_type,
            stride=stride,
        )

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        self.merge = merge
        self.fan_in_fan_out = fan_in_fan_out

        self.dropout = nn.Identity() if lora_dropout == 0.0 else nn.Dropout(p=lora_dropout)
        self.merged = False

        self.lora_a = nn.Parameter(self.weight.new_empty((r, self.input_size_per_partition)))
        self.lora_b = nn.Parameter(self.weight.new_empty((out_features, r)))
        self.weight.requires_grad_(False)

        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self) -> None:
        super().reset_parameters()

        if hasattr(self, "lora_a") and hasattr(self, "lora_b"):
            self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_normal_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def _t(self, w: Tensor) -> Tensor:
        return w.transpose(0, 1) if self.fan_in_fan_out else w

    def train(self, mode: bool = True) -> "LoraRowParallelLinear":
        super().train(mode)

        if mode:
            if self.merge and self.merged:
                # Make sure that the weights are not merged
                if self.lora_a is not None and self.lora_b is not None:
                    self.weight.data -= self._t(self.lora_b @ self.lora_a) * self.scaling
                self.merged = False

        elif self.merge and not self.merged:
            # Merge the weights and mark it
            if self.lora_a is not None and self.lora_b is not None:
                self.weight.data += self._t(self.lora_b @ self.lora_a) * self.scaling
            self.merged = True

        return self

    def forward(self, x: Tensor) -> Tensor:
        input_parallel = x if self.input_is_parallel else mp_scatter(x)

        if self.lora_a is not None and self.lora_b is not None and not self.merged:
            output_parallel = F.linear(input_parallel, self._t(self.weight), bias=self.bias)
            mm = self.dropout(input_parallel) @ self.lora_a.transpose(0, 1) @ self.lora_b.transpose(0, 1)
            output_parallel = output_parallel + mm * self.scaling
            output = mp_reduce(output_parallel)
            return output if self.bias is None else output + self.bias

        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        output = mp_reduce(output_parallel)
        return output if self.bias is None else output + self.bias


@overload
def lora(
    module: nn.Embedding,
    r: int,
    alpha: float = 1.0,
    dropout: float = 0.0,
    merge: bool = False,
) -> LoraEmbedding: ...


@overload
def lora(
    module: nn.Linear,
    r: int,
    alpha: float = 1.0,
    dropout: float = 0.0,
    merge: bool = False,
) -> LoraLinear: ...


@overload
def lora(
    module: nn.Conv1d,
    r: int,
    alpha: float = 1.0,
    dropout: float = 0.0,
    merge: bool = False,
) -> LoraConv1d: ...


@overload
def lora(
    module: nn.ConvTranspose1d,
    r: int,
    alpha: float = 1.0,
    dropout: float = 0.0,
    merge: bool = False,
) -> LoraConv1d: ...


@overload
def lora(
    module: nn.Conv2d,
    r: int,
    alpha: float = 1.0,
    dropout: float = 0.0,
    merge: bool = False,
) -> LoraConv2d: ...


@overload
def lora(
    module: nn.ConvTranspose2d,
    r: int,
    alpha: float = 1.0,
    dropout: float = 0.0,
    merge: bool = False,
) -> LoraConv2d: ...


@overload
def lora(
    module: nn.LSTM,
    r: int,
    alpha: float = 1.0,
    dropout: float = 0.0,
    merge: bool = False,
) -> LoraLSTM: ...


@overload
def lora(
    module: nn.GRU,
    r: int,
    alpha: float = 1.0,
    dropout: float = 0.0,
    merge: bool = False,
) -> LoraGRU: ...


@overload
def lora(
    module: nn.LSTMCell,
    r: int,
    alpha: float = 1.0,
    dropout: float = 0.0,
    merge: bool = False,
) -> LoraLSTMCell: ...


@overload
def lora(
    module: nn.GRUCell,
    r: int,
    alpha: float = 1.0,
    dropout: float = 0.0,
    merge: bool = False,
) -> LoraGRUCell: ...


@overload
def lora(
    module: ParallelEmbedding,
    r: int,
    alpha: float = 1.0,
    dropout: float = 0.0,
    merge: bool = False,
) -> LoraParallelEmbedding: ...


@overload
def lora(
    module: ColumnParallelLinear,
    r: int,
    alpha: float = 1.0,
    dropout: float = 0.0,
    merge: bool = False,
) -> LoraColumnParallelLinear: ...


@overload
def lora(
    module: RowParallelLinear,
    r: int,
    alpha: float = 1.0,
    dropout: float = 0.0,
    merge: bool = False,
) -> LoraRowParallelLinear: ...


@overload
def lora(
    module: SupportedModule,
    r: int,
    alpha: float = 1.0,
    dropout: float = 0.0,
    merge: bool = False,
) -> nn.Module: ...


def lora(
    module: SupportedModule,
    r: int,
    alpha: float = 1.0,
    dropout: float = 0.0,
    merge: bool = False,
) -> nn.Module:
    """Wraps a module with LoRA.

    This function takes a base module and returns the LoRA version of that
    module. The new module is effectively a drop-in replacement for the
    original module; for example, it can load the same state dict, and it has
    the same input and output shapes.

    Args:
        module: The module to wrap.
        r: The number of LoRA components to use. If 0, then LoRA is not used.
        alpha: The scaling factor for the LoRA components. A higher value
            means that more weight is given to the LoRA components.
        dropout: The dropout probability applied to the input value before
            computing the LoRA components. This parameter is not supported
            for RNNs (because it would require modifying the underyling kernel).
        merge: Whether to merge the LoRA components into the original
            weights. If True, then the LoRA components are merged into the
            weights during training, and the original weights are used during
            evaluation. If False, then the LoRA components are used during
            both training and evaluation.

    Returns:
        The LoRA version of the module.

    Raises:
        ValueError: If the module is not supported.
    """
    if isinstance(module, nn.Embedding):
        embedding = LoraEmbedding(
            module.num_embeddings,
            module.embedding_dim,
            padding_idx=module.padding_idx,
            max_norm=module.max_norm,
            norm_type=module.norm_type,
            scale_grad_by_freq=module.scale_grad_by_freq,
            sparse=module.sparse,
            r=r,
            lora_alpha=alpha,
            merge=merge,
        )
        embedding.weight.data.copy_(module.weight.data)
        return embedding

    if isinstance(module, nn.Linear):
        linear = LoraLinear(
            module.in_features,
            module.out_features,
            r=r,
            lora_alpha=alpha,
            merge=merge,
            bias=module.bias is not None,
        )
        linear.weight.data.copy_(module.weight.data)
        if module.bias is not None and linear.bias is not None:
            linear.bias.data.copy_(module.bias.data)
        return linear

    if isinstance(module, nn.Conv1d):
        conv_1d = LoraConv1d(
            module.in_channels,
            module.out_channels,
            cast(tuple[int], module.kernel_size),
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            merge=merge,
            stride=cast(tuple[int], module.stride),
            padding=cast(str | tuple[int], module.padding),
            dilation=cast(tuple[int], module.dilation),
            groups=module.groups,
            bias=module.bias is not None,
        )
        conv_1d.weight.data.copy_(module.weight.data)
        if module.bias is not None and conv_1d.bias is not None:
            conv_1d.bias.data.copy_(module.bias.data)
        return conv_1d

    if isinstance(module, nn.ConvTranspose1d):
        conv_transpose_1d = LoraConvTranspose1d(
            module.in_channels,
            module.out_channels,
            cast(tuple[int], module.kernel_size),
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            merge=merge,
            stride=cast(tuple[int], module.stride),
            padding=cast(tuple[int], module.padding),
            output_padding=cast(tuple[int], module.output_padding),
            dilation=cast(tuple[int], module.dilation),
            groups=module.groups,
            bias=module.bias is not None,
        )
        conv_transpose_1d.weight.data.copy_(module.weight.data)
        if module.bias is not None and conv_transpose_1d.bias is not None:
            conv_transpose_1d.bias.data.copy_(module.bias.data)
        return conv_transpose_1d

    if isinstance(module, nn.Conv2d):
        conv_2d = LoraConv2d(
            module.in_channels,
            module.out_channels,
            cast(tuple[int, int], module.kernel_size),
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            merge=merge,
            stride=cast(tuple[int, int], module.stride),
            padding=cast(str | tuple[int, int], module.padding),
            dilation=cast(tuple[int, int], module.dilation),
            groups=module.groups,
            bias=module.bias is not None,
        )
        conv_2d.weight.data.copy_(module.weight.data)
        if module.bias is not None and conv_2d.bias is not None:
            conv_2d.bias.data.copy_(module.bias.data)
        return conv_2d

    if isinstance(module, nn.ConvTranspose2d):
        conv_transpose_2d = LoraConvTranspose2d(
            module.in_channels,
            module.out_channels,
            cast(tuple[int, int], module.kernel_size),
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            merge=merge,
            stride=cast(tuple[int, int], module.stride),
            padding=cast(tuple[int, int], module.padding),
            output_padding=cast(tuple[int, int], module.output_padding),
            dilation=cast(tuple[int, int], module.dilation),
            groups=module.groups,
            bias=module.bias is not None,
        )
        conv_transpose_2d.weight.data.copy_(module.weight.data)
        if module.bias is not None and conv_transpose_2d.bias is not None:
            conv_transpose_2d.bias.data.copy_(module.bias.data)
        return conv_transpose_2d

    if isinstance(module, nn.LSTM):
        if dropout > 0.0:
            warnings.warn("LoRA dropout is not supported for LSTMs")

        lstm = LoraLSTM(
            module.input_size,
            module.hidden_size,
            r=r,
            lora_alpha=alpha,
            num_layers=module.num_layers,
            batch_first=module.batch_first,
            dropout=module.dropout,
            bidirectional=module.bidirectional,
            proj_size=module.proj_size,
            bias=module.bias,
        )
        for param_name, param_value in module.named_parameters():
            getattr(lstm, param_name).data.copy_(param_value.data)
        return lstm

    if isinstance(module, nn.GRU):
        if dropout > 0.0:
            warnings.warn("LoRA dropout is not supported for GRUs")

        gru = LoraGRU(
            module.input_size,
            module.hidden_size,
            r=r,
            lora_alpha=alpha,
            num_layers=module.num_layers,
            bias=module.bias,
            batch_first=module.batch_first,
            dropout=module.dropout,
            bidirectional=module.bidirectional,
            proj_size=module.proj_size,
        )
        for param_name, param_value in module.named_parameters():
            getattr(gru, param_name).data.copy_(param_value.data)
        return gru

    if isinstance(module, nn.LSTMCell):
        if dropout > 0.0:
            warnings.warn("LoRA dropout is not supported for LSTMCells")

        lstm_cell = LoraLSTMCell(
            module.input_size,
            module.hidden_size,
            r=r,
            lora_alpha=alpha,
            bias=module.bias,
        )
        lstm_cell.weight_hh.data.copy_(module.weight_hh.data)
        lstm_cell.weight_ih.data.copy_(module.weight_ih.data)
        if module.bias:
            lstm_cell.bias_hh.data.copy_(module.bias_hh.data)
            lstm_cell.bias_ih.data.copy_(module.bias_ih.data)
        return lstm_cell

    if isinstance(module, nn.GRUCell):
        if dropout > 0.0:
            warnings.warn("LoRA dropout is not supported for GRUCells")

        gru_cell = LoraGRUCell(
            module.input_size,
            module.hidden_size,
            r=r,
            lora_alpha=alpha,
            bias=module.bias,
        )
        gru_cell.weight_hh.data.copy_(module.weight_hh.data)
        gru_cell.weight_ih.data.copy_(module.weight_ih.data)
        if module.bias:
            gru_cell.bias_hh.data.copy_(module.bias_hh.data)
            gru_cell.bias_ih.data.copy_(module.bias_ih.data)
        return gru_cell

    if isinstance(module, ParallelEmbedding):
        parallel_embedding = LoraParallelEmbedding(
            module.num_embeddings,
            module.embedding_dim,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            merge=merge,
            padding_idx=module.padding_idx,
            max_norm=module.max_norm,
            norm_type=module.norm_type,
            scale_grad_by_freq=module.scale_grad_by_freq,
            sparse=module.sparse,
        )
        parallel_embedding.weight.data.copy_(module.weight.data)
        return parallel_embedding

    if isinstance(module, RowParallelLinear):
        row_parallel_linear = LoraRowParallelLinear(
            module.in_features,
            module.out_features,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            merge=merge,
            bias=module.bias is not None,
        )
        row_parallel_linear.weight.data.copy_(module.weight.data)
        if module.bias is not None and row_parallel_linear.bias is not None:
            row_parallel_linear.bias.data.copy_(module.bias.data)
        return row_parallel_linear

    if isinstance(module, ColumnParallelLinear):
        column_parallel_linear = LoraColumnParallelLinear(
            module.in_features,
            module.out_features,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            merge=merge,
            bias=module.bias is not None,
        )
        column_parallel_linear.weight.data.copy_(module.weight.data)
        if module.bias is not None and column_parallel_linear.bias is not None:
            column_parallel_linear.bias.data.copy_(module.bias.data)
        return column_parallel_linear

    raise ValueError(f"Unsupported module type {type(module)}")


T_module = TypeVar("T_module", bound=SupportedModule)


def maybe_lora(
    module: T_module,
    r: int | None,
    alpha: float = 1.0,
    dropout: float = 0.0,
    merge: bool = False,
    freeze: bool = True,
) -> T_module:
    """Apply LoRA to a supported module, if a LoRA rank is provided.

    Args:
        module: A supported module.
        r: The LoRA rank.
        alpha: The LoRA alpha parameter.
        dropout: The LoRA dropout rate.
        merge: Whether to merge the LoRA rank into the input dimension.
        freeze: Whether to freeze the module's parameters if a LoRA rank is
            not provided. This argument has no effect if a LoRA rank is
            provided, since downstream users can always freeze just the module
            themselves. Typically, when trying out LoRA fine-tuning, downstream
            users will want to freeze most of the module parameters and apply
            LoRA only to a subset of the module's layers, so this is the
            default behavior.

    Returns:
        The module with LoRA applied, if a LoRA rank is provided.
    """
    if freeze and r is None:
        module = cast(T_module, module.requires_grad_(False))
    return module if r is None else lora(module, r, alpha, dropout, merge)


def maybe_lora_weight_norm(
    module: T_module,
    r: int | None,
    alpha: float = 1.0,
    dropout: float = 0.0,
    merge: bool = False,
    freeze: bool = True,
) -> T_module:
    module = maybe_lora(module, r=r, alpha=alpha, dropout=dropout, merge=merge, freeze=freeze)
    return nn.utils.weight_norm(module)


def reset_lora_weights_(module: nn.Module) -> None:
    """Resets any LoRA weights in the module.

    All of the LoRA modules have a ``reset_lora_parameters`` method that will
    reset the LoRA weights in-place. This function looks for any modules with
    this method and calls it.

    Args:
        module: The module to reset, in-place.
    """
    for _, submodule in module.named_modules():
        if isinstance(submodule, _Lora):
            submodule.reset_lora_parameters()


def freeze_non_lora_(module: nn.Module) -> None:
    """Freezes any non-LoRA parameters in the module.

    Args:
        module: The module to freeze, in-place.
    """
    for _, submodule in module.named_modules():
        if isinstance(submodule, _Lora):
            continue
        for param in submodule.parameters(recurse=False):
            param.requires_grad_(False)
