"""Defines a general-purpose API for weight initialization.

.. highlight:: python
.. code-block:: python

    from mlfab.nn.init import init_, cast_init_type

    linear = nn.Linear(32, 32)
    init_(linear.weight, linear.bias, "orthogonal")

    # This lets you parametrize the initialization type as a string.
    init_(linear.weight, linear.bias, cast_init_type(my_init_type))

Choices for the initialization type are:

- ``"orthogonal"``: Orthogonal initialization, meaning that the weights are initialized to an orthogonal matrix.
- ``"normal"``: Initializes weights with a normal distribution
- ``"biased_normal"``: Initializes both weights and biases with a normal distribution
- ``"uniform"``: Initializes weights with a uniform distribution
- ``"kaiming_uniform"`` or ``"kaiming_normal"``: Initializes weights with a Kaiming normal or uniform distribution
- ``"xavier_uniform"`` or ``"xavier_normal"``: Initializes weights with a Xavier normal or uniform distribution
- ``"zeros"``: Initializes weights to all zeros
- ``"ones"``: Initializes weights to all ones
"""

import math
from typing import Literal, cast, get_args

import torch
from torch import Tensor, nn

InitializationType = Literal[
    "orthogonal",
    "normal",
    "biased_normal",
    "uniform",
    "kaiming_uniform",
    "kaiming_normal",
    "xavier_uniform",
    "xavier_normal",
    "trunc_normal",
    "dirac",
    "constant",
    "zeros",
    "ones",
]


def cast_init_type(s: str) -> InitializationType:
    args = get_args(InitializationType)
    assert s in args, f"Invalid initialization type: '{s}' Valid options are {args}"
    return cast(InitializationType, s)


def _uniform_bias(weight: Tensor, bias: Tensor | None) -> Tensor | None:
    if bias is None:
        return None
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    if fan_in == 0:
        nn.init.zeros_(bias)
    else:
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)
    return bias


def _zeros(t: Tensor | None) -> Tensor | None:
    return None if t is None else nn.init.zeros_(t)


def init_(
    weight: Tensor,
    bias: Tensor | None,
    init: InitializationType,
    *,
    mean: float = 0.0,
    std: float = 0.01,
    scale: float = 0.02,
    groups: int = 1,
    trunc_clip: tuple[float, float] = (-2.0, 2.0),
) -> tuple[Tensor, Tensor | None]:
    """Initializes the weight and bias in-place, using an initialization key.

    The weight and bias are from a convolution or linear layer.

    Args:
        weight: The weight tensor
        bias: The bias tensor
        init: The initialization type to use
        mean: The mean for normal initialization
        std: The standard deviation for normal initialization
        scale: The scale amount for uniform or constant initialization
        groups: The number of groups, if argument is necessary
        trunc_clip: The min and max values for trunc_normal initialization

    Returns:
        The initialized weight and bias (which can be discarded, since the
        initialization happens in-place).

    Raises:
        NotImplementedError: If the initialization mode isn't implemented
    """
    # Don't do anything for meta tensors.
    if weight.is_meta:
        return weight, bias
    if isinstance(weight, nn.Parameter):
        weight = weight.data
    if isinstance(bias, nn.Parameter):
        bias = bias.data
    match init:
        case "orthogonal":
            if weight.dtype in (torch.float16, torch.bfloat16):
                return (
                    weight.copy_(nn.init.orthogonal_(weight.float(), gain=0.01).to(weight)),
                    _zeros(bias),
                )
            return nn.init.orthogonal_(weight), _zeros(bias)
        case "normal":
            return nn.init.normal_(weight, mean=mean, std=std), _zeros(bias)
        case "biased_normal":
            return (
                nn.init.normal_(weight, mean=mean, std=std),
                None if bias is None else nn.init.normal_(bias, mean=mean, std=std),
            )
        case "uniform":
            return nn.init.uniform_(weight, b=scale), _zeros(bias)
        case "kaiming_uniform":
            return nn.init.kaiming_uniform_(weight), _uniform_bias(weight, bias)
        case "kaiming_normal":
            return nn.init.kaiming_normal_(weight), _uniform_bias(weight, bias)
        case "xavier_uniform":
            return nn.init.xavier_uniform_(weight), _uniform_bias(weight, bias)
        case "xavier_normal":
            return nn.init.xavier_normal_(weight), _uniform_bias(weight, bias)
        case "trunc_normal":
            a, b = trunc_clip
            return nn.init.trunc_normal_(weight, mean=mean, std=std, a=a, b=b), _zeros(bias)
        case "dirac":
            return nn.init.dirac_(weight, groups=groups), _zeros(bias)
        case "constant":
            return nn.init.constant_(weight, scale), _zeros(bias)
        case "zeros":
            return nn.init.zeros_(weight), _zeros(bias)
        case "ones":
            return nn.init.ones_(weight), _zeros(bias)
        case _:
            raise NotImplementedError(f"Unexpected initialization: {init}")
