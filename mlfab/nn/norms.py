"""Defines general-purpose helper functions for initializing norm layers.

.. highlight:: python
.. code-block:: python

    from mlfab.nn.norms import get_norm_linear, get_norm_1d, get_norm_2d, get_norm_3d, cast_norm_type

    linear = nn.Sequential(nn.Linear(32, 32), get_norm_linear("layer", dim=32))
    conv_1d = nn.Sequential(nn.Conv1d(32, 32, 3), get_norm_1d("layer", dim=32, groups=4))
    conv_2d = nn.Sequential(nn.Conv2d(32, 32, 3), get_norm_2d("layer", dim=32, groups=4))
    conv_3d = nn.Sequential(nn.Conv3d(32, 32, 3), get_norm_3d("layer", dim=32, groups=4))

    # This lets you parametrize the norm type as a string.
    linear = nn.Sequential(nn.Linear(32, 32), get_norm_linear(cast_norm_type(my_norm), dim=32))

Choices for the norm type are:

- ``"no_norm"``: No normalization
- ``"batch"`` or ``"batch_affine"``: Batch normalization
- ``"instance"`` or ``"instance_affine"``: Instance normalization
- ``"group"`` or ``"group_affine"``: Group normalization
- ``"layer"`` or ``"layer_affine"``: Layer normalization

Note that instance norm and group norm are not available for linear layers.
"""

from typing import Literal, TypeVar, cast, get_args

import torch
from torch import Tensor, nn

T_module = TypeVar("T_module", bound=nn.Module)

NormType = Literal[
    "no_norm",
    "batch",
    "batch_affine",
    "instance",
    "instance_affine",
    "group",
    "group_affine",
    "layer",
    "layer_affine",
    "rms",
]

ParametrizationNormType = Literal[
    "no_norm",
    "weight",
    "spectral",
]


def cast_norm_type(s: str) -> NormType:
    args = get_args(NormType)
    assert s in args, f"Invalid norm type: '{s}' Valid options are {args}"
    return cast(NormType, s)


def cast_parametrize_norm_type(s: str) -> ParametrizationNormType:
    args = get_args(ParametrizationNormType)
    assert s in args, f"Invalid parametrization norm type: '{s}' Valid options are {args}"
    return cast(ParametrizationNormType, s)


class RMSNorm(torch.nn.Module):
    """Defines root-mean-square normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class LastBatchNorm(nn.Module):
    """Applies batch norm along final dimension without transposing the tensor.

    The normalization is pretty simple, it basically just tracks the running
    mean and variance for each channel, then normalizes each channel to have
    a unit normal distribution.

    Input:
        x: Tensor with shape (..., N)

    Output:
        The tensor, normalized by the running mean and variance
    """

    __constants__ = ["channels", "momentum", "affine", "eps"]

    mean: Tensor
    var: Tensor

    def __init__(
        self,
        channels: int,
        momentum: float = 0.99,
        affine: bool = True,
        eps: float = 1e-4,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.momentum = momentum
        self.affine = affine
        self.eps = eps

        if dtype is None:
            mean_tensor = torch.zeros(channels, device=device)
            var_tensor = torch.ones(channels, device=device)
        else:
            mean_tensor = torch.zeros(channels, device=device, dtype=dtype)
            var_tensor = torch.ones(channels, device=device, dtype=dtype)
        self.register_buffer("mean", mean_tensor)
        self.register_buffer("var", var_tensor)

        if self.affine:
            self.affine_transform = nn.Linear(channels, channels, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        if self.affine:
            x = self.affine_transform(x)
        if self.training:
            x_flat = x.flatten(0, -2)
            mean, var = x_flat.mean(dim=0).detach(), x_flat.var(dim=0).detach()
            new_mean = mean * (1 - self.momentum) + self.mean * self.momentum
            new_var = var * (1 - self.momentum) + self.var * self.momentum
            x_out = (x - new_mean.expand_as(x)) / (new_var.expand_as(x) + self.eps)
            self.mean.copy_(new_mean, non_blocking=True)
            self.var.copy_(new_var, non_blocking=True)
        else:
            x_out = (x - self.mean.expand_as(x)) / (self.var.expand_as(x) + self.eps)
        return x_out


class ConvLayerNorm(nn.Module):
    __constants__ = ["channels", "eps", "elementwise_affine", "static_shape"]

    def __init__(
        self,
        channels: int,
        *,
        dims: int | None = None,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.channels = channels
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if dtype is None:
                self.weight = nn.Parameter(torch.empty(self.channels, device=device))
                self.bias = nn.Parameter(torch.empty(self.channels, device=device))
            else:
                self.weight = nn.Parameter(torch.empty(self.channels, device=device, dtype=dtype))
                self.bias = nn.Parameter(torch.empty(self.channels, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.static_shape = None if dims is None else (1, -1) + (1,) * dims

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        mean = inputs.mean(dim=1, keepdim=True)
        var = torch.square(inputs - mean).mean(dim=1, keepdim=True)
        normalized_inputs = (inputs - mean) / (var + self.eps).sqrt()
        if self.elementwise_affine:
            if self.static_shape is None:
                weight = self.weight.unflatten(0, (-1,) + (1,) * (len(inputs.shape) - 2))
                bias = self.bias.unflatten(0, (-1,) + (1,) * (len(inputs.shape) - 2))
            else:
                weight = self.weight.view(self.static_shape)
                bias = self.bias.view(self.static_shape)
            normalized_inputs = normalized_inputs * weight + bias
        return normalized_inputs


def get_norm_1d(
    norm: NormType,
    *,
    dim: int | None = None,
    groups: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    eps: float = 1e-5,
) -> nn.Module:
    """Returns a normalization layer for tensors with shape (B, C, T).

    Args:
        norm: The norm type to use
        dim: The number of dimensions in the input tensor
        groups: The number of groups to use for group normalization
        device: The device to use for the layer
        dtype: The dtype to use for the layer
        eps: The epsilon value to use for normalization

    Returns:
        A normalization layer

    Raises:
        NotImplementedError: If `norm` is not a valid 1D norm type
    """
    match norm:
        case "no_norm":
            return nn.Identity()
        case "batch" | "batch_affine":
            if dim is None:
                return nn.LazyBatchNorm1d(eps=eps, affine=norm == "batch_affine", device=device, dtype=dtype)
            return nn.BatchNorm1d(dim, eps=eps, affine=norm == "batch_affine", device=device, dtype=dtype)
        case "instance" | "instance_affine":
            if dim is None:
                return nn.LazyInstanceNorm1d(eps=eps, affine=norm == "instance_affine", device=device, dtype=dtype)
            return nn.InstanceNorm1d(dim, eps=eps, affine=norm == "instance_affine", device=device, dtype=dtype)
        case "group" | "group_affine":
            assert dim is not None, "`dim` is required for group norm"
            assert groups is not None, "`groups` is required for group norm"
            return nn.GroupNorm(groups, dim, eps=eps, affine=norm == "group_affine", device=device, dtype=dtype)
        case "layer" | "layer_affine":
            assert dim is not None, "`dim` is required for layer norm"
            return ConvLayerNorm(
                dim,
                dims=1,
                eps=eps,
                elementwise_affine=norm == "layer_affine",
                device=device,
                dtype=dtype,
            )
        case _:
            raise NotImplementedError(f"Invalid 1D norm type: {norm}")


def get_norm_2d(
    norm: NormType,
    *,
    dim: int | None = None,
    groups: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    eps: float = 1e-5,
) -> nn.Module:
    """Returns a normalization layer for tensors with shape (B, C, H, W).

    Args:
        norm: The norm type to use
        dim: The number of dimensions in the input tensor
        groups: The number of groups to use for group normalization
        device: The device to use for the layer
        dtype: The dtype to use for the layer
        eps: The epsilon value to use for normalization

    Returns:
        A normalization layer

    Raises:
        NotImplementedError: If `norm` is not a valid 2D norm type
    """
    match norm:
        case "no_norm":
            return nn.Identity()
        case "batch" | "batch_affine":
            if dim is None:
                return nn.LazyBatchNorm2d(eps=eps, affine=norm == "batch_affine", device=device, dtype=dtype)
            return nn.BatchNorm2d(dim, eps=eps, affine=norm == "batch_affine", device=device, dtype=dtype)
        case "instance" | "instance_affine":
            if dim is None:
                return nn.LazyInstanceNorm2d(eps=eps, affine=norm == "instance_affine", device=device, dtype=dtype)
            return nn.InstanceNorm2d(dim, eps=eps, affine=norm == "instance_affine", device=device, dtype=dtype)
        case "group" | "group_affine":
            assert dim is not None, "`dim` is required for group norm"
            assert groups is not None, "`groups` is required for group norm"
            return nn.GroupNorm(groups, dim, eps=eps, affine=norm == "group_affine", device=device, dtype=dtype)
        case "layer" | "layer_affine":
            assert dim is not None, "`dim` is required for layer norm"
            return ConvLayerNorm(
                dim,
                dims=2,
                eps=eps,
                elementwise_affine=norm == "layer_affine",
                device=device,
                dtype=dtype,
            )
        case _:
            raise NotImplementedError(f"Invalid 2D norm type: {norm}")


def get_norm_3d(
    norm: NormType,
    *,
    dim: int | None = None,
    groups: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    eps: float = 1e-5,
) -> nn.Module:
    """Returns a normalization layer for tensors with shape (B, C, D, H, W).

    Args:
        norm: The norm type to use
        dim: The number of dimensions in the input tensor
        groups: The number of groups to use for group normalization
        device: The device to use for the layer
        dtype: The dtype to use for the layer
        eps: The epsilon value to use for normalization

    Returns:
        A normalization layer

    Raises:
        NotImplementedError: If `norm` is not a valid 3D norm type
    """
    match norm:
        case "no_norm":
            return nn.Identity()
        case "batch" | "batch_affine":
            if dim is None:
                return nn.LazyBatchNorm3d(eps=eps, affine=norm == "batch_affine", device=device, dtype=dtype)
            return nn.BatchNorm3d(dim, eps=eps, affine=norm == "batch_affine", device=device, dtype=dtype)
        case "instance" | "instance_affine":
            if dim is None:
                return nn.LazyInstanceNorm3d(eps=eps, affine=norm == "instance_affine", device=device, dtype=dtype)
            return nn.InstanceNorm3d(dim, eps=eps, affine=norm == "instance_affine", device=device, dtype=dtype)
        case "group" | "group_affine":
            assert dim is not None, "`dim` is required for group norm"
            assert groups is not None, "`groups` is required for group norm"
            return nn.GroupNorm(groups, dim, eps=eps, affine=norm == "group_affine", device=device, dtype=dtype)
        case "layer" | "layer_affine":
            assert dim is not None, "`dim` is required for layer norm"
            return ConvLayerNorm(
                dim,
                dims=3,
                eps=eps,
                elementwise_affine=norm == "layer_affine",
                device=device,
                dtype=dtype,
            )
        case _:
            raise NotImplementedError(f"Invalid 3D norm type: {norm}")


def get_norm_linear(
    norm: NormType,
    *,
    dim: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    eps: float = 1e-5,
) -> nn.Module:
    """Returns a normalization layer for tensors with shape (B, ..., C).

    Args:
        norm: The norm type to use
        dim: The number of dimensions in the input tensor
        device: The device to use for the layer
        dtype: The dtype to use for the layer
        eps: The epsilon value to use for normalization

    Returns:
        A normalization layer

    Raises:
        NotImplementedError: If `norm` is not a valid linear norm type
    """
    match norm:
        case "no_norm":
            return nn.Identity()
        case "batch" | "batch_affine":
            assert dim is not None, "`dim` is required for batch norm"
            return LastBatchNorm(dim, affine=norm == "batch_affine", eps=eps, device=device, dtype=dtype)
        case "layer" | "layer_affine":
            assert dim is not None, "`dim` is required for layer norm"
            return nn.LayerNorm(dim, elementwise_affine=norm == "layer_affine", eps=eps, device=device, dtype=dtype)
        case "rms":
            assert dim is not None, "`dim` is required for RMS norm"
            return RMSNorm(dim, eps=eps)
        case _:
            raise NotImplementedError(f"Invalid linear norm type: {norm}")


def get_parametrization_norm(
    module: T_module,
    norm: ParametrizationNormType,
    *,
    name: str = "weight",
    n_power_iterations: int = 1,
    eps: float = 1e-12,
    weight_dim: int = 0,
    spectral_dim: int | None = None,
) -> T_module:
    """Returns a parametrized version of the module.

    Args:
        module: The module to parametrize
        norm: The parametrization norm type to use
        name: The name of the parameter to use for the parametrization; this
            should reference the name on the module (for instance, ``weight``
            for a ``nn.Linear`` module)
        n_power_iterations: The number of power iterations to use for spectral
            normalization
        eps: The epsilon value to use for spectral normalization
        weight_dim: The dimension of the weight parameter to normalize when
            using weight normalization
        spectral_dim: The dimension of the weight parameter to normalize when
            using spectral normalization

    Returns:
        The parametrized module
    """
    if all(p.device.type == "meta" for p in module.parameters()):
        return module
    match norm:
        case "no_norm":
            return module
        case "weight":
            return nn.utils.weight_norm(
                module,
                name=name,
                dim=weight_dim,
            )
        case "spectral":
            return nn.utils.spectral_norm(
                module,
                name=name,
                n_power_iterations=n_power_iterations,
                eps=eps,
                dim=spectral_dim,
            )
        case _:
            raise NotImplementedError(f"Invalid parametrization norm type: {norm}")
