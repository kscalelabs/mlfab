"""Defines a general-purpose API for activation functions.

.. highlight:: python
.. code-block:: python

    from ml.models.activations import get_activation, cast_activation_type

    model = nn.Sequential(nn.Linear(4, 5), get_activation("relu"))

    # This lets you parametrize the activation function as a string.
    model = nn.Sequential(nn.Linear(4, 5), get_activation(cast_activation_type(my_activation)))

Choices for the activation functions are:

- ``"no_act"``
- ``"relu"``
- ``"relu6"``
- ``"relu2"``
- ``"clamp6"``
- ``"leaky_relu"``
- ``"elu"``
- ``"celu"``
- ``"selu"``
- ``"gelu"``
- ``"gelu_fast"``
- ``"sigmoid"``
- ``"log_sigmoid"``
- ``"hard_sigomid"``
- ``"tanh"``
- ``"softsign"``
- ``"softplus"``
- ``"silu"``
- ``"mish"``
- ``"swish"``
- ``"hard_swish"``
- ``"soft_shrink"``
- ``"hard_shrink"``
- ``"tanh_shrink"``
- ``"soft_sign"``
- ``"relu_squared"``
- ``"laplace"``
"""

import math
from typing import Literal, cast, get_args

import torch
from torch import Tensor, nn

ActivationType = Literal[
    "no_act",
    "relu",
    "relu6",
    "relu2",
    "clamp6",
    "leaky_relu",
    "elu",
    "celu",
    "selu",
    "gelu",
    "gelu_fast",
    "sigmoid",
    "log_sigmoid",
    "hard_sigomid",
    "tanh",
    "softsign",
    "softplus",
    "silu",
    "mish",
    "swish",
    "hard_swish",
    "soft_shrink",
    "hard_shrink",
    "tanh_shrink",
    "soft_sign",
    "relu_squared",
    "laplace",
]


def cast_activation_type(s: str) -> ActivationType:
    args = get_args(ActivationType)
    assert s in args, f"Invalid activation type: '{s}' Valid options are {args}"
    return cast(ActivationType, s)


class Clamp(nn.Module):
    __constants__ = ["min_value", "max_value", "inplace"]

    def __init__(
        self,
        *,
        value: float | None = None,
        value_range: tuple[float, float] | None = None,
        inplace: bool = False,
    ) -> None:
        super().__init__()

        assert (value is None) != (value_range is None), "Exactly one of `value` or `value_range` must be specified."

        if value is not None:
            value_range = (-value, value)
        else:
            assert value_range is not None

        self.min_value, self.max_value = value_range
        self.inplace = inplace

        assert self.min_value < self.max_value, f"{self.min_value=} >= {self.max_value=}"

    def forward(self, x: Tensor) -> Tensor:
        return x.clamp_(self.min_value, self.max_value) if self.inplace else x.clamp(self.min_value, self.max_value)


class Clamp6(Clamp):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__(value=6.0, inplace=inplace)


class ReLUSquared(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        relu_applied = nn.functional.relu(x)
        squared = torch.square(relu_applied)
        return squared


class FastGELU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


class QuickGELU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(1.702 * x)


class LaplaceActivation(nn.Module):
    __constants__ = ["mu", "sigma"]

    def __init__(self, mu: float = 0.707107, sigma: float = 0.282095) -> None:
        super().__init__()

        self.mu = mu
        self.sigma = sigma

    def forward(self, x: Tensor) -> Tensor:
        x = (x - self.mu).div(self.sigma * math.sqrt(2.0))
        return 0.5 * (1.0 + torch.erf(x))


def get_activation(act: ActivationType, *, inplace: bool = True) -> nn.Module:
    """Returns an activation function from a keyword string.

    Args:
        act: The keyword for the activation function (None for identity)
        inplace: If set, use the inplace version of the activation function

    Returns:
        The activation function as a module

    Raises:
        NotImplementedError: If the activation function is invalid
    """
    match act:
        case "no_act":
            return nn.Identity()
        case "relu":
            return nn.ReLU(inplace=inplace)
        case "relu2":
            return nn.ReLU(inplace=inplace)
        case "relu6":
            return nn.ReLU6(inplace=inplace)
        case "clamp6":
            return Clamp6(inplace=inplace)
        case "leaky_relu":
            return nn.LeakyReLU(inplace=inplace)
        case "elu":
            return nn.ELU(inplace=inplace)
        case "celu":
            return nn.CELU(inplace=inplace)
        case "selu":
            return nn.SELU(inplace=inplace)
        case "gelu":
            return nn.GELU()
        case "gelu_fast":
            return FastGELU()
        case "gelu_quick":
            return QuickGELU()
        case "sigmoid":
            return nn.Sigmoid()
        case "log_sigmoid":
            return nn.LogSigmoid()
        case "hard_sigomid":
            return nn.Hardsigmoid(inplace=inplace)
        case "tanh":
            return nn.Tanh()
        case "softsign":
            return nn.Softsign()
        case "softplus":
            return nn.Softplus()
        case "silu" | "swish":
            return nn.SiLU()
        case "mish":
            return nn.Mish(inplace=inplace)
        case "hard_swish":
            return nn.Hardswish(inplace=inplace)
        case "soft_shrink":
            return nn.Softshrink()
        case "hard_shrink":
            return nn.Hardshrink()
        case "tanh_shrink":
            return nn.Tanhshrink()
        case "soft_sign":
            return nn.Softsign()
        case "relu_squared":
            return ReLUSquared()
        case "laplace":
            return LaplaceActivation()
        case _:
            raise NotImplementedError(f"Activation function '{act}' is not implemented.")
