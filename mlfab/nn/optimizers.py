"""Defines utility functions for optimizers.

Some of these functions can be used to improve the performance of some part of
the optimizer.
"""

import logging
from typing import Callable, Iterable, Literal, NotRequired, Self, TypedDict, Unpack, cast

import torch
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim.adam import Adam as AdamBase
from torch.optim.adamw import AdamW as AdamWBase
from torch.optim.optimizer import Optimizer

from mlfab.nn.triton import supports_triton

logger = logging.getLogger(__name__)


class ParamGroup(TypedDict):
    params: list[nn.Parameter]
    weight_decay: float


Params = Iterable[Tensor] | Iterable[ParamGroup]


def separate_decayable_params(model: nn.Module, default_decay: bool, weight_decay: float) -> Iterable[ParamGroup]:
    """Separates weight-decayable parameters from other parameters.

    In practice, it is a good idea to not apply weight decay to norm parameters,
    and instead apply it only to weights and biases. This function separates
    weight-decayable parameters from other parameters, and returns two
    parameter groups: one with weight-decayable parameters, and one without.

    Args:
        model: Model to separate parameters for.
        default_decay: Whether to apply weight decay to parameters by default,
            if they are not explicitly specified. This controls how custom
            parameters will be handled, such as the initial embedding for an
            autoregressive model.
        weight_decay: Weight decay to use for weight-decayable parameters.

    Returns:
        A list of parameter groups, with the first group containing
        weight-decayable parameters, and the second group containing other
        parameters.
    """
    wd_params: set[str] = set()
    no_wd_params: set[str] = set()
    seen: set[str] = set()

    always_decay = (
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
        nn.MultiheadAttention,
    )

    never_decay = (
        _BatchNorm,
        nn.LocalResponseNorm,
        nn.GroupNorm,
        nn.LayerNorm,
        nn.Embedding,
        nn.EmbeddingBag,
    )

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            if not p.requires_grad:
                continue
            fpn = f"{mn}.{pn}" if mn else pn
            if fpn in seen:
                continue
            seen.add(fpn)
            if p.ndim < 2:
                no_wd_params.add(fpn)
            elif isinstance(m, never_decay):
                no_wd_params.add(fpn)
            elif isinstance(m, always_decay):
                wd_params.add(fpn)
            else:
                (wd_params if default_decay else no_wd_params).add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    inter_params = wd_params & no_wd_params
    union_params = wd_params | no_wd_params
    assert len(inter_params) == 0, "Parameters made it into both decay and no-decay sets!"
    assert len(param_dict.keys() - union_params) == 0, "Parameters were not separated into decay or no-decay set!"

    groups: list[ParamGroup] = []
    if wd_params:
        groups.append({"params": [param_dict[pn] for pn in sorted(list(wd_params))], "weight_decay": weight_decay})
    if no_wd_params:
        groups.append({"params": [param_dict[pn] for pn in sorted(list(no_wd_params))], "weight_decay": 0.0})
    return groups


def can_use_fused(model: nn.Module) -> bool:
    return all(p.is_cuda and p.is_floating_point() for p in model.parameters())


def can_use_foreach(model: nn.Module) -> bool:
    return all(p.device.type in ("cpu", "cuda") and p.is_floating_point() for p in model.parameters())


def _lion_update_fn_vanilla(
    p: nn.Parameter,
    grad: Tensor,
    exp_avg: Tensor,
    lr: float,
    wd: float,
    beta1: float,
    beta2: float,
) -> None:
    """Runs the update function for a given parameter.

    This can be made slightly faster using the Triton backend.

    Args:
        p: Parameter to update.
        grad: Gradient for the parameter.
        exp_avg: Exponential average of the gradient.
        lr: Learning rate.
        wd: Weight decay.
        beta1: First momentum coefficient.
        beta2: Second momentum coefficient.
    """
    update = exp_avg.clone().mul_(beta1).add(grad, alpha=1 - beta1).sign_()
    p.data.mul_(1 - lr * wd).add_(update, alpha=-lr)
    exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)


def get_lion_update_fn(cpu: bool) -> Callable[[nn.Parameter, Tensor, Tensor, float, float, float, float], None]:
    if cpu or not supports_triton():
        return _lion_update_fn_vanilla

    from mlfab.nn.triton.lion import update_fn as triton_update_fn

    return triton_update_fn


class LionKwargs(TypedDict):
    lr: NotRequired[float]
    betas: NotRequired[tuple[float, float]]
    weight_decay: NotRequired[float]


class Lion(Optimizer):
    """Defines the Lion optimizer.

    This optimizer was proposed in `Symbolic Discovery of Optimization
    Algorithms <https://arxiv.org/abs/2302.06675>`_.

    Lion stands for "Evolved Sign Momentum" (yes, actually). It is more
    memory-efficient than Adam since it only keeps track of momentum.

    In the original paper, the authors suggest using a larger batch size and a
    smaller learning rate compared to Adam.

    This optimizer shines for tasks like contrasitve learning and diffusion
    which optimize proxy objectives rather than doing something like
    cross-entropy classification, although in the paper the authors show that
    it performs comparably to Adam on language modeling.

    This implementation is based on the ``lucidrain's`` implementation
    `here <https://github.com/lucidrains/lion-pytorch/>`_ and on the
    pseudo-code from the paper, which is reproduced below:

    .. code-block:: python

        def train(weight, gradient, momentum, lr):
            update = interp(gradient, momentum, beta1)
            update = sign(update)
            momentum = interp(gradient, momentum, beta2)
            update = update + weight * weight_decay
            update = update * lr
            return update, momentum

    The default values for ``betas`` are (0.9, 0.99), which are roughly the
    same as default Adam. However, the authors suggest using (0.95, 0.98) for
    better stability.
    """

    def __init__(self, params: Params, use_triton: bool = True, **kwargs: Unpack[LionKwargs]) -> None:
        lr = kwargs.pop("lr", 1e-4)
        betas = kwargs.pop("betas", (0.9, 0.99))
        weight_decay = kwargs.pop("weight_decay", 0.0)

        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not all([0.0 <= beta <= 1.0 for beta in betas]):
            raise ValueError(f"Invalid beta: {betas}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)  # type: ignore[arg-type]

        self.update_fn = get_lion_update_fn(True)
        self.update_fn_cuda = get_lion_update_fn(use_triton)

    @classmethod
    def stable(cls, model: nn.Module, **kwargs: Unpack[LionKwargs]) -> Self:
        kwargs.setdefault("lr", 1e-4)
        kwargs.setdefault("betas", (0.95, 0.98))
        kwargs.setdefault("weight_decay", 0.0)
        return cls(separate_decayable_params(model, True, kwargs["weight_decay"]), **kwargs)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                p = cast(Tensor, p)
                if p.grad is None:
                    continue

                grad = p.grad.data
                lr: float = group["lr"]
                wd: float = group["weight_decay"]
                beta1, beta2 = group["betas"]
                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                update_fn = self.update_fn_cuda if grad.is_cuda else self.update_fn
                update_fn(p, grad, state["exp_avg"], lr, wd, beta1, beta2)

        return loss


class AdanKwargs(TypedDict):
    lr: NotRequired[float]
    betas: NotRequired[tuple[float, float, float]]
    eps: NotRequired[float]
    weight_decay: NotRequired[float]


class Adan(Optimizer):
    def __init__(self, params: Params, **kwargs: Unpack[AdanKwargs]) -> None:
        lr = kwargs.pop("lr", 1e-3)
        betas = kwargs.pop("betas", (0.1, 0.1, 0.001))
        eps = kwargs.pop("eps", 1e-8)
        weight_decay = kwargs.pop("weight_decay", 0.0)

        assert len(betas) == 3

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}

        super().__init__(params, defaults)  # type: ignore[arg-type]

    @classmethod
    def get(cls, model: nn.Module, **kwargs: Unpack[AdanKwargs]) -> Self:
        kwargs.setdefault("lr", 1e-3)
        kwargs.setdefault("betas", (0.1, 0.1, 0.001))
        kwargs.setdefault("eps", 1e-8)
        kwargs.setdefault("weight_decay", 0.0)
        return cls(separate_decayable_params(model, True, kwargs["weight_decay"]), **kwargs)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
        loss: float | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2, beta3 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                data, grad = p.data, p.grad.data
                assert not grad.is_sparse

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = grad.clone()
                    state["v"] = torch.zeros_like(grad)
                    state["n"] = grad**2

                step, m, v, n = state["step"], state["m"], state["v"], state["n"]

                zeroth_step = step == 0
                first_step = step == 1

                if not zeroth_step:
                    prev_grad = state["prev_grad"]
                    m.mul_(1 - beta1).add_(grad, alpha=beta1)
                    grad_diff = grad - prev_grad
                    if not first_step:
                        v.mul_(1 - beta2).add_(grad_diff, alpha=beta2)
                    else:
                        v.add_(grad_diff)
                    next_n = (grad + (1 - beta2) * grad_diff) ** 2
                    n.mul_(1 - beta3).add_(next_n, alpha=beta3)

                weighted_step_size = lr / (n + eps).sqrt()
                denom = 1 + weight_decay * lr

                data.addcmul_(weighted_step_size, (m + (1 - beta2) * v), value=-1.0).div_(denom)
                state["prev_grad"] = grad.clone()
                state["step"] += 1

        return loss


class AdamKwargs(TypedDict):
    lr: NotRequired[float]
    betas: NotRequired[tuple[float, float]]
    weight_decay: NotRequired[float]
    eps: NotRequired[float]
    amsgrad: NotRequired[bool]
    maximize: NotRequired[bool]
    foreach: NotRequired[bool | None]
    capturable: NotRequired[bool]
    differentiable: NotRequired[bool]
    fused: NotRequired[bool | None]


AdamGpt3Size = Literal["small", "medium", "large"]
AdamRobertaSize = Literal["base", "large"]


class Adam:
    @classmethod
    def gpt3(
        cls,
        model: nn.Module,
        size: AdamGpt3Size = "small",
        default_decay: bool = True,
        **kwargs: Unpack[AdamKwargs],
    ) -> AdamBase | AdamWBase:
        match size:
            case "small":
                kwargs.setdefault("lr", 6e-4)
                kwargs.setdefault("betas", (0.9, 0.95))
                kwargs.setdefault("weight_decay", 0.1)
            case "medium":
                kwargs.setdefault("lr", 3e-4)
                kwargs.setdefault("betas", (0.9, 0.95))
                kwargs.setdefault("weight_decay", 0.1)
            case "large":
                kwargs.setdefault("lr", 2.5e-4)
                kwargs.setdefault("betas", (0.9, 0.95))
                kwargs.setdefault("weight_decay", 0.1)
            case _:
                raise ValueError(f"Invalid GPT-3 size: {size}")
        return cls.get(model, default_decay=default_decay, **kwargs)

    @classmethod
    def roberta(
        cls,
        model: nn.Module,
        size: AdamRobertaSize = "base",
        default_decay: bool = True,
        **kwargs: Unpack[AdamKwargs],
    ) -> AdamBase | AdamWBase:
        match size:
            case "base":
                kwargs.setdefault("lr", 6e-4)
                kwargs.setdefault("betas", (0.9, 0.98))
                kwargs.setdefault("weight_decay", 0.01)
            case "large":
                kwargs.setdefault("lr", 4e-4)
                kwargs.setdefault("betas", (0.9, 0.98))
                kwargs.setdefault("weight_decay", 0.01)
            case _:
                raise ValueError(f"Invalid RoBERTa size: {size}")
        return cls.get(model, default_decay=default_decay, **kwargs)

    @classmethod
    def get(cls, model: nn.Module, default_decay: bool = True, **kwargs: Unpack[AdamKwargs]) -> AdamBase | AdamWBase:
        kwargs.setdefault("lr", 3e-4)
        kwargs.setdefault("betas", (0.9, 0.95))
        kwargs.setdefault("eps", 1e-8)
        kwargs.setdefault("weight_decay", 1e-2)
        kwargs.setdefault("amsgrad", False)
        kwargs.setdefault("maximize", False)
        kwargs.setdefault("foreach", None)
        kwargs.setdefault("capturable", False)
        kwargs.setdefault("differentiable", False)
        kwargs.setdefault("fused", None)

        # Sets default values for foreach and fused variants.
        fused, foreach = kwargs.pop("fused"), kwargs.pop("foreach")
        if fused is None and foreach is None:
            if can_use_fused(model):
                fused = True
            elif can_use_foreach(model):
                foreach = True
        if fused is None:
            fused = False
        if foreach is None:
            foreach = False
        kwargs["fused"] = fused
        kwargs["foreach"] = foreach

        weight_decay = kwargs.pop("weight_decay")
        if weight_decay == 0.0:
            return AdamBase(separate_decayable_params(model, default_decay, weight_decay), **kwargs)  # type: ignore[arg-type]
        return AdamWBase(separate_decayable_params(model, default_decay, weight_decay), **kwargs)  # type: ignore[arg-type]
