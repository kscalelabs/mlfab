# mypy: disable-error-code="import, override"
"""Defines modules for RWKV blocks.

RWKV blocks are similar to Transformer blocks, but use a different attention
mechanism that doesn't require a linearly growing KV cache.

Training requires CUDA kernel requires installing ``triton``:

.. code-block:: bash

    pip install triton
"""

import math
import os
import warnings
from typing import Callable, Literal, cast, get_args

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd.function import Function, FunctionCtx, once_differentiable

from mlfab.nn.architectures.next_token import SamplingStrategy, sample_from_logits
from mlfab.nn.triton import supports_triton

WkvFnKey = Literal["eps", "log"]

RwkvAttentionState = tuple[Tensor, Tensor]
RwkvFeedForwardState = Tensor
RwkvState = tuple[RwkvAttentionState, RwkvFeedForwardState]

EPS = 1e-4


@torch.jit.script
def wkv_with_eps_forward(w: Tensor, u: Tensor, k: Tensor, v: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
    bsz, tsz, chans = k.shape

    assert w.shape == u.shape == (chans,)
    assert v.shape == (bsz, tsz, chans)
    assert state.shape == (bsz, 3, 1, chans)

    alpha, beta, eps = state[:, :, -1].chunk(3, dim=1)  # (B, 1, D), (B, 1, D), (B, 1, D)

    _, tsz, _ = k.shape

    wkvs = []
    alphas = [alpha]
    betas = [beta]
    epss = [eps]

    for t in range(tsz):
        kt, vt = k[:, t : t + 1], v[:, t : t + 1]
        ukt = u + kt
        tau = torch.maximum(ukt, eps)
        e1 = torch.exp(eps - tau)
        e2 = torch.exp(ukt - tau)
        wkv = (e1 * alpha + e2 * vt) / (e1 * beta + e2)
        wkvs.append(wkv)

        w_eps = eps - w
        eps = torch.maximum(w_eps, kt)
        e1 = torch.exp(w_eps - eps)
        e2 = torch.exp(kt - eps)
        alpha = e1 * alpha + e2 * vt
        beta = e1 * beta + e2

        alphas.append(alpha)
        betas.append(beta)
        epss.append(eps)

    alpha = torch.stack(alphas, dim=2)
    beta = torch.stack(betas, dim=2)
    eps = torch.stack(epss, dim=2)

    return torch.cat(wkvs, 1), torch.cat((alpha, beta, eps), dim=1)


@torch.jit.script
def wkv_with_eps_backward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor,
    grad_wkv: Tensor,
    grad_state: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    bsz, tsz, chans = k.shape

    assert w.shape == u.shape == (chans,)
    assert v.shape == (bsz, tsz, chans)
    assert state.shape == (bsz, 3, tsz + 1, chans)
    assert grad_wkv.shape == (bsz, tsz, chans)
    assert grad_state.shape == (bsz, 3, 1, chans)

    alpha, beta, eps = state.chunk(3, dim=1)  # (B, 1, T + 1, D), (B, 1, T + 1, D), (B, 1, T + 1, D)
    grad_alpha, grad_beta, grad_eps = grad_state[:, :, 0].chunk(3, dim=1)  # (B, 1, D), (B, 1, D), (B, 1, D)
    grad_eps = grad_eps.clone()

    grad_w = torch.zeros_like(w)
    grad_u = torch.zeros_like(u)
    grad_k = torch.zeros_like(k)
    grad_v = torch.zeros_like(v)

    for t in range(tsz - 1, -1, -1):
        kt, vt = k[:, t : t + 1], v[:, t : t + 1]
        alpha_prev, beta_prev, eps_prev = alpha[:, :, t], beta[:, :, t], eps[:, :, t]
        alpha_curr, beta_curr, eps_curr = alpha[:, :, t + 1], beta[:, :, t + 1], eps[:, :, t + 1]
        ukt = u + kt
        tau = torch.maximum(ukt, eps_prev)
        e1 = torch.exp(eps_prev - tau)
        e2 = torch.exp(ukt - tau)

        euke = torch.exp(ukt + eps_prev - 2 * tau)

        denom = e1 * beta_prev + e2
        denom_sq = denom * denom

        grad_wkvt = grad_wkv[:, t : t + 1]

        # Backpropagates wkv gradients.
        grad_uk = grad_wkvt * e2 * (e1 * beta_prev * vt - e1 * alpha_prev) / denom_sq
        grad_u += grad_uk.flatten(0, -2).sum(0)
        grad_k[:, t : t + 1] += grad_uk
        grad_v[:, t : t + 1] += grad_wkvt * e2 / denom

        grad_alpha_wkv = grad_wkvt * e1 / denom
        grad_beta_wkv = -grad_wkvt * e1 * (e2 * vt + e1 * alpha_prev) / denom_sq
        grad_eps_wkv = grad_wkvt * euke * (alpha_prev - vt * beta_prev) / (e1 * beta_prev + e2) ** 2

        e1 = torch.exp(eps_prev - eps_curr - w)
        e2 = torch.exp(kt - eps_curr)

        # Backpropagates alpha gradients.
        grad_alpha_we = grad_alpha * e1 * alpha_prev
        grad_w -= grad_alpha_we.flatten(0, -2).sum(0)
        grad_k[:, t : t + 1] += grad_alpha * e2 * vt
        grad_v[:, t : t + 1] += grad_alpha * e2
        grad_eps += grad_alpha * -alpha_curr

        # Backpropagates beta gradients.
        grad_beta_we = grad_beta * e1 * beta_prev
        grad_w -= grad_beta_we.flatten(0, -2).sum(0)
        grad_k[:, t : t + 1] += grad_beta * e2
        grad_eps += grad_beta * -beta_curr

        # Backpropagates epsilon gradients.
        eps_grad_mask = eps_prev - w > kt
        grad_eps_we = torch.where(eps_grad_mask, grad_eps, torch.zeros_like(grad_eps))
        grad_w -= grad_eps_we.flatten(0, -2).sum(0)
        grad_k[:, t : t + 1] += torch.where(eps_grad_mask, torch.zeros_like(grad_eps), grad_eps)

        # Computes gradients for alpha, beta and epsilon.
        grad_alpha = grad_alpha * e1 + grad_alpha_wkv
        grad_beta = grad_beta * e1 + grad_beta_wkv
        grad_eps = grad_alpha_we + grad_beta_we + grad_eps_we + grad_eps_wkv

    return grad_w, grad_u, grad_k, grad_v, torch.stack((grad_alpha, grad_beta, grad_eps), dim=1)


class WkvWithEps(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        w: Tensor,
        u: Tensor,
        k: Tensor,
        v: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        wkv, state_out = wkv_with_eps_forward(w, u, k, v, state)
        ctx.save_for_backward(w, u, k, v, state_out)
        return wkv, state_out[:, :, -1:]

    @staticmethod
    @once_differentiable
    def backward(
        ctx: FunctionCtx,
        grad_wkv: Tensor,
        grad_state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        w, u, k, v, state = cast(tuple[Tensor, ...], ctx.saved_tensors)
        return wkv_with_eps_backward(w, u, k, v, state, grad_wkv, grad_state)


def initial_state_with_eps(emb_dim: int) -> Tensor:
    return torch.zeros(1, 3, 1, emb_dim)


def wkv_with_eps(w: Tensor, u: Tensor, k: Tensor, v: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
    """Runs the core WKV computation.

    Args:
        w: The decay tensor, with shape (D)
        u: The output multiplier tensor, with shape (D)
        k: The K tensor, with shape (B, T, D)
        v: The V tensor, with shape (B, T, D)
        state: The state tensor, with shape (B, 3, T, D), consisting of the
            alpha, beta and eps tensors, each with shape (B, 1, T, D)

    Returns:
        The WKV tensor, with shape (B, T, D), and the next state, with shape
        (B, 3, 1, D), consisting of the next alpha, beta and eps tensors, each
        with shape (B, 1, 1, D)
    """
    return WkvWithEps.apply(w, u, k, v, state)


@torch.jit.script
def logaddexp(a: Tensor, b: Tensor) -> Tensor:
    max_ab = torch.maximum(a, b)
    return max_ab + torch.log(torch.exp(a - max_ab) + torch.exp(b - max_ab))


@torch.jit.script
def logsubexp(a: Tensor, b: Tensor, log_eps: float) -> Tensor:
    max_ab = torch.clamp_min(torch.maximum(a, b), log_eps)
    return max_ab + torch.log(torch.exp(a - max_ab) - torch.exp(b - max_ab))


@torch.jit.script
def wkv_log_space_forward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor,
    eps: float = EPS,
    normalize: bool = False,
) -> tuple[Tensor, Tensor]:
    bsz, tsz, chans = k.shape

    assert w.shape == u.shape == (chans,)
    assert v.shape == (bsz, tsz, chans)
    assert state.shape == (bsz, 3, 1, chans)

    ln_alpha_p, ln_alpha_m, ln_beta = state[:, :, -1].chunk(3, dim=1)

    log_eps = math.log(eps)

    wkvs = []
    ln_alpha_ps = [ln_alpha_p]
    ln_alpha_ms = [ln_alpha_m]
    ln_betas = [ln_beta]

    for t in range(tsz):
        kt, vt = k[:, t : t + 1], v[:, t : t + 1]
        vt_p, vt_m = torch.clamp_min(vt, 0) + eps, torch.clamp_min(-vt, 0) + eps
        ln_v_p, ln_v_m = torch.log(vt_p), torch.log(vt_m)

        if normalize:
            ln_alpha_pm = torch.minimum(ln_alpha_p, ln_alpha_m) - eps
            ln_alpha_p = logsubexp(ln_alpha_p, ln_alpha_pm, log_eps)
            ln_alpha_m = logsubexp(ln_alpha_m, ln_alpha_pm, log_eps)

        ln_wkv_p = logaddexp(u + kt + ln_v_p, ln_alpha_p) - logaddexp(u + kt, ln_beta)
        ln_wkv_m = logaddexp(u + kt + ln_v_m, ln_alpha_m) - logaddexp(u + kt, ln_beta)

        wkv = torch.exp(ln_wkv_p) - torch.exp(ln_wkv_m)
        wkvs.append(wkv)

        ln_alpha_p = logaddexp(ln_alpha_p - w, kt + ln_v_p)
        ln_alpha_m = logaddexp(ln_alpha_m - w, kt + ln_v_m)
        ln_beta = logaddexp(ln_beta - w, kt)

        ln_alpha_ps.append(ln_alpha_p)
        ln_alpha_ms.append(ln_alpha_m)
        ln_betas.append(ln_beta)

    ln_alpha_p = torch.stack(ln_alpha_ps, dim=2)
    ln_alpha_m = torch.stack(ln_alpha_ms, dim=2)
    ln_beta = torch.stack(ln_betas, dim=2)

    return torch.cat(wkvs, 1), torch.cat((ln_alpha_p, ln_alpha_m, ln_beta), dim=1)


@torch.jit.script
def wkv_log_space_backward(
    w: Tensor,
    u: Tensor,
    k: Tensor,
    v: Tensor,
    state: Tensor,
    grad_wkv: Tensor,
    grad_state: Tensor,
    eps: float = EPS,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    bsz, tsz, chans = k.shape

    assert w.shape == u.shape == (chans,)
    assert v.shape == (bsz, tsz, chans)
    assert state.shape == (bsz, 3, tsz, chans)
    assert grad_wkv.shape == (bsz, tsz, chans)
    assert grad_state.shape == (bsz, 3, 1, chans)

    grad_ln_alpha_p, grad_ln_alpha_m, grad_ln_beta = grad_state[:, :, 0].chunk(3, dim=1)

    grad_w = torch.zeros_like(w)
    grad_u = torch.zeros_like(u)
    grad_k = torch.zeros_like(k)
    grad_v = torch.zeros_like(v)

    for t in range(tsz - 1, -1, -1):
        kt, vt = k[:, t : t + 1], v[:, t : t + 1]
        vt_p, vt_m = torch.clamp_min(vt, 0) + eps, torch.clamp_min(-vt, 0) + eps
        ln_v_p, ln_v_m = torch.log(vt_p), torch.log(vt_m)

        ln_alpha_p_prev, ln_alpha_m_prev, ln_beta_prev = state[:, :, t].chunk(3, dim=1)

        uk = u + kt
        ukv_p, ukv_m = uk + ln_v_p, uk + ln_v_m

        ukb = logaddexp(uk, ln_beta_prev)
        wkv_p = torch.exp(logaddexp(ukv_p, ln_alpha_p_prev) - ukb)
        wkv_m = torch.exp(logaddexp(ukv_m, ln_alpha_m_prev) - ukb)

        grad_wkvt = grad_wkv[:, t : t + 1]
        grad_ln_wkv_p, grad_ln_wkv_m = grad_wkvt * wkv_p, grad_wkvt * -wkv_m

        # Backpropagates wkv gradients.
        e_num_p = torch.exp(ln_alpha_p_prev - ukv_p)
        e_num_m = torch.exp(ln_alpha_m_prev - ukv_m)
        e_den = torch.exp(ln_beta_prev - uk)
        grad_wkv_den_p = grad_ln_wkv_p / (1 + e_den)
        grad_wkv_den_m = grad_ln_wkv_m / (1 + e_den)
        grad_kv_p = grad_ln_wkv_p / (1 + e_num_p)
        grad_kv_m = grad_ln_wkv_m / (1 + e_num_m)
        grad_uk = grad_kv_p + grad_kv_m - grad_wkv_den_p - grad_wkv_den_m
        grad_u += grad_uk.flatten(0, -2).sum(0)
        grad_k[:, t : t + 1] += grad_uk
        grad_v[:, t : t + 1] += torch.where(vt > 0, grad_kv_p / vt_p, grad_kv_m / -vt_m)

        grad_ln_alpha_wkv_p = grad_ln_wkv_p / (1 + (1 / e_num_p))
        grad_ln_alpha_wkv_m = grad_ln_wkv_m / (1 + (1 / e_num_m))
        grad_ln_beta_wkv = -grad_ln_wkv_p / (1 + (1 / e_den)) - grad_ln_wkv_m / (1 + (1 / e_den))

        # Backpropagates alpha gradients.
        e_alpha_p = torch.exp(kt + ln_v_p + w - ln_alpha_p_prev)
        e_alpha_m = torch.exp(kt + ln_v_m + w - ln_alpha_m_prev)
        grad_wa_p = grad_ln_alpha_p / (1 + e_alpha_p)
        grad_wa_m = grad_ln_alpha_m / (1 + e_alpha_m)
        grad_w -= (grad_wa_p + grad_wa_m).flatten(0, -2).sum(0)
        grad_kv_p = grad_ln_alpha_p / (1 + (1 / e_alpha_p))
        grad_kv_m = grad_ln_alpha_m / (1 + (1 / e_alpha_m))
        grad_k[:, t : t + 1] += grad_kv_p + grad_kv_m
        grad_v[:, t : t + 1] += torch.where(vt > 0, grad_kv_p / vt_p, -grad_kv_m / vt_m)

        # Backpropagates beta gradients.
        e_beta = torch.exp(kt + w - ln_beta_prev)
        grad_wb = grad_ln_beta / (1 + e_beta)
        grad_w -= grad_wb.flatten(0, -2).sum(0)
        grad_k[:, t : t + 1] += grad_ln_beta / (1 + (1 / e_beta))

        # Compute gradients for log alpha and log beta.
        grad_ln_alpha_p = grad_wa_p + grad_ln_alpha_wkv_p
        grad_ln_alpha_m = grad_wa_m + grad_ln_alpha_wkv_m
        grad_ln_beta = grad_wb + grad_ln_beta_wkv

    return grad_w, grad_u, grad_k, grad_v, torch.stack((grad_ln_alpha_p, grad_ln_alpha_m, grad_ln_beta), dim=1)


class WkvLogSpace(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        w: Tensor,
        u: Tensor,
        k: Tensor,
        v: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        wkv, state_out = wkv_log_space_forward(w, u, k, v, state)
        ctx.save_for_backward(w, u, k, v, state_out[:, :, :-1])
        return wkv, state_out[:, :, -1:]

    @staticmethod
    @once_differentiable
    def backward(
        ctx: FunctionCtx,
        grad_wkv: Tensor,
        grad_state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        w, u, k, v, state = cast(tuple[Tensor, ...], ctx.saved_tensors)
        return wkv_log_space_backward(w, u, k, v, state, grad_wkv, grad_state)


def initial_state_log_space(emb_dim: int) -> Tensor:
    return torch.full((1, 3, 1, emb_dim), float("-inf"))


def wkv_log_space(w: Tensor, u: Tensor, k: Tensor, v: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
    """Runs the core WKV computation.

    Args:
        w: The decay tensor, with shape (D)
        u: The output multiplier tensor, with shape (D)
        k: The K tensor, with shape (B, T, D)
        v: The V tensor, with shape (B, T, D)
        state: The state tensor, with shape (B, 3, D), consisting of the
            alpha plus, alpha minus and beta tensors, each with shape (B, 1, D)

    Returns:
        The WKV tensor, with shape (B, T, D), and the next state, with shape
        (B, 2, D), consisting of the next alpha plus, alpha minus and beta
        tensors, each with shape (B, 1, D)
    """
    return WkvLogSpace.apply(w, u, k, v, state)


def get_wkv_fn(key: WkvFnKey) -> Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor]]:
    match key:
        case "eps":
            return wkv_with_eps
        case "log":
            return wkv_log_space
        case _:
            raise ValueError(f"Unsupported key: {key}")


def get_wkv_fn_cuda(key: WkvFnKey) -> Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], tuple[Tensor, Tensor]]:
    if not supports_triton():
        return get_wkv_fn(key)

    from mlfab.nn.triton.rwkv import wkv_triton_log_space, wkv_triton_with_eps

    match key:
        case "eps":
            return wkv_triton_with_eps
        case "log":
            return wkv_triton_log_space
        case _:
            raise ValueError(f"Unsupported key: {key}")


def get_default_wkv_fn_key() -> WkvFnKey:
    if "WKV_FN" in os.environ:
        assert (wkv_fn_str := os.environ["WKV_FN"]) in get_args(WkvFnKey), f"Unsupported WKV_FN: {wkv_fn_str}"
        return cast(WkvFnKey, wkv_fn_str)

    warnings.warn("Using default WKV_FN: eps")
    return "eps"


class RwkvAttention(nn.Module):
    init_x: Tensor
    init_state: Tensor

    def __init__(self, dim: int, wkv_key: WkvFnKey | None = None) -> None:
        super().__init__()

        self.time_decay = nn.Parameter(torch.ones(dim))
        self.time_first = nn.Parameter(torch.ones(dim))

        self.time_mix_k = nn.Parameter(torch.ones(1, 1, dim))
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, dim))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, dim))

        self.key = nn.Linear(dim, dim, False)
        self.value = nn.Linear(dim, dim, False)
        self.receptance = nn.Linear(dim, dim, False)
        self.output = nn.Linear(dim, dim, False)

        if wkv_key is None:
            wkv_key = get_default_wkv_fn_key()

        self.wkv_fn = get_wkv_fn(wkv_key)
        self.wkv_fn_cuda = get_wkv_fn_cuda(wkv_key)

        self.register_buffer("init_x", torch.zeros(1, 1, dim), persistent=False)
        self.register_buffer("init_state", initial_state_with_eps(dim), persistent=False)

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.key.weight)
        nn.init.orthogonal_(self.receptance.weight)
        nn.init.orthogonal_(self.value.weight)
        nn.init.orthogonal_(self.output.weight)

    def time_shift(self, last_x: Tensor, x: Tensor) -> Tensor:
        _, tsz, _ = x.shape
        if tsz > 1:
            last_x = torch.cat((last_x, x[..., :-1, :]), dim=-2)
        return last_x

    def forward(self, x: Tensor, state: RwkvAttentionState | None) -> tuple[Tensor, RwkvAttentionState]:
        bsz, _, _ = x.shape

        if state is None:
            last_x = self.init_x.repeat_interleave(bsz, dim=0)
            last_state = self.init_state.repeat_interleave(bsz, dim=0)
        else:
            last_x, last_state = state
        last_x = self.time_shift(last_x, x)

        k = self.key(x * self.time_mix_k + last_x * (1 - self.time_mix_k))
        v = self.value(x * self.time_mix_v + last_x * (1 - self.time_mix_v))
        r = self.receptance(x * self.time_mix_r + last_x * (1 - self.time_mix_r))
        sr = torch.sigmoid(r)

        w, u = self.time_decay, self.time_first
        w = torch.exp(w)
        wkv_fn = self.wkv_fn_cuda if x.is_cuda else self.wkv_fn
        wkv, next_state = wkv_fn(w, u, k, v, last_state)
        rwkv = wkv * sr

        return self.output(rwkv), (x[..., -1:, :], next_state)


class RwkvFeedForward(nn.Module):
    init_state: Tensor

    def __init__(self, dim: int, ffn_dim: int) -> None:
        super().__init__()

        self.time_mix_k = nn.Parameter(torch.ones(1, 1, dim))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, dim))

        self.key = nn.Linear(dim, ffn_dim, False)
        self.receptance = nn.Linear(dim, dim, False)
        self.value = nn.Linear(ffn_dim, dim, False)

        self.register_buffer("init_state", torch.zeros(1, 1, dim), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.key.weight)
        nn.init.orthogonal_(self.receptance.weight)
        nn.init.orthogonal_(self.value.weight)

    def time_shift(self, last_x: Tensor, x: Tensor) -> Tensor:
        _, tsz, _ = x.shape
        if tsz > 1:
            last_x = torch.cat((last_x, x[..., :-1, :]), dim=-2)
        return last_x

    def forward(self, x: Tensor, state: RwkvFeedForwardState | None = None) -> tuple[Tensor, RwkvFeedForwardState]:
        bsz = x.shape[0]

        last_x = self.time_shift(self.init_state.repeat(bsz, 1, 1) if state is None else state, x)

        k = self.key(x * self.time_mix_k + last_x * (1 - self.time_mix_k))
        r = self.receptance(x * self.time_mix_r + last_x * (1 - self.time_mix_r))
        vk = self.value(F.relu(k) ** 2)

        return torch.sigmoid(r) * vk, x[..., -1:, :]


class RwkvBlock(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        pre_norm: bool = False,
        wkv_key: WkvFnKey | None = None,
        feedforward_factor: int = 4,
    ) -> None:
        super().__init__()

        self.ln0 = nn.LayerNorm(emb_dim) if pre_norm else None
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.att = RwkvAttention(emb_dim, wkv_key=wkv_key)
        self.ffn = RwkvFeedForward(emb_dim, emb_dim * feedforward_factor)

    def run_attn(self, x: Tensor, state: RwkvState | None = None) -> tuple[Tensor, RwkvAttentionState]:
        return self.att.forward(self.ln1(x), None if state is None else state[0])

    def run_ffn(self, x: Tensor, state: RwkvState | None = None) -> tuple[Tensor, RwkvFeedForwardState]:
        return self.ffn.forward(self.ln2(x), None if state is None else state[1])

    def forward(self, x: Tensor, state: RwkvState | None = None) -> tuple[Tensor, RwkvState]:
        if self.ln0 is not None:
            x = self.ln0(x)
        dx, att_state_out = self.run_attn(x, state)
        x = x + dx
        dx, ffn_state_out = self.run_ffn(x, state)
        x = x + dx
        return x, (att_state_out, ffn_state_out)


class RwkvStack(nn.Module):
    """Defines a stack of RWKV modules.

    Parameters:
        emb_dim: The number of embedding dimensions in each block
        num_layers: The number of layers in the stack
        wkv_key: The WKV algorithm to use
        feedforward_factor: The factor by which the input number of dimensions
            is multiplied to get the feedforward hidden dimension.

    Inputs:
        x: The input tensor, with shape ``(B, T, D)``
        state: The previous state

    Outputs:
        The output tensor, with shape ``(B, T, D)``, and the next state
    """

    def __init__(
        self,
        emb_dim: int,
        num_layers: int,
        wkv_key: WkvFnKey | None = None,
        feedforward_factor: int = 4,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                RwkvBlock(
                    emb_dim,
                    pre_norm=i == 0,
                    wkv_key=wkv_key,
                    feedforward_factor=feedforward_factor,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x: Tensor, state: list[RwkvState] | None = None) -> tuple[Tensor, list[RwkvState]]:
        state_out: list[RwkvState] = []
        for i, block in enumerate(self.blocks):
            x, state_out_i = block(x, None if state is None else state[i])
            state_out.append(state_out_i)
        return x, state_out


class NextTokenRwkv(nn.Module):
    """Defines a next token prediction RWKV module.

    This seems to be the most popular architecture for solving a large number
    of problems. This provides a tested implementation of the next token
    prediction RWKV module.
    """

    def __init__(
        self,
        emb_dim: int,
        num_layers: int,
        vocab_size: int,
        wkv_key: WkvFnKey | None = None,
        feedforward_factor: int = 4,
    ) -> None:
        super().__init__()

        self.init_emb = nn.Parameter(torch.randn(1, 1, emb_dim) * 0.02)
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.rwkv = RwkvStack(
            emb_dim=emb_dim,
            num_layers=num_layers,
            wkv_key=wkv_key,
            feedforward_factor=feedforward_factor,
        )
        self.proj = nn.Linear(emb_dim, vocab_size)

    def forward(self, tokens_bt: Tensor) -> Tensor:
        x_btc = self.embeddings(tokens_bt[:, :-1])
        x_btc = torch.cat((self.init_emb.expand(x_btc.size(0), 1, -1), x_btc), dim=1)
        x_btc, _ = self.rwkv(x_btc)
        logits_btc = self.proj(x_btc)
        return logits_btc

    def infer(
        self,
        t: int,
        bsz: int = 1,
        sampling_strategy: SamplingStrategy = "top-p",
        k: int | None = None,
        p: float | None = 0.95,
        temperature: float = 1.0,
    ) -> Tensor:
        x_b1c: Tensor = self.init_emb.expand(bsz, 1, -1)
        x_b1c, state = self.rwkv(x_b1c)
        logits_b1l = self.proj(x_b1c)
        tokens_bt = sample_from_logits(logits_b1l, sampling_strategy, k=k, p=p, temperature=temperature)
        tokens_b1 = tokens_bt[:, :1]

        for _ in range(1, t):
            x_b1c = self.embeddings(tokens_b1)
            x_b1c, state = self.rwkv(x_b1c, state)
            logits_b1l = self.proj(x_b1c)
            tokens_b1 = sample_from_logits(logits_b1l, sampling_strategy, k=k, p=p, temperature=temperature)
            tokens_bt = torch.cat((tokens_bt, tokens_b1), dim=1)

        return tokens_bt


class NextTokenWithEmbeddingsRwkv(nn.Module):
    """Defines a next token prediction RWKV module, over base embeddings.

    This is similar to the ``NextTokenRwkv`` except that each of the
    input timesteps also has an associated embedding, which is added to the
    input before the RWKV layers.
    """

    def __init__(
        self,
        emb_dim: int,
        num_layers: int,
        vocab_size: int,
        wkv_key: WkvFnKey | None = None,
        feedforward_factor: int = 4,
    ) -> None:
        super().__init__()

        self.init_emb = nn.Parameter(torch.randn(1, 1, emb_dim) * 0.02)
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.rwkv = RwkvStack(
            emb_dim=emb_dim,
            num_layers=num_layers,
            wkv_key=wkv_key,
            feedforward_factor=feedforward_factor,
        )
        self.proj = nn.Linear(emb_dim, vocab_size)

    def forward(self, tokens_bt: Tensor, emb_btc: Tensor) -> tuple[Tensor, Tensor]:
        x_btc = self.embeddings(tokens_bt[:, :-1])
        x_btc = torch.cat((self.init_emb.expand(x_btc.size(0), 1, -1), x_btc), dim=1)
        x_btc = x_btc + emb_btc
        x_btc, _ = self.rwkv(x_btc)
        logits_btc = self.proj(x_btc)
        return logits_btc, x_btc

    def infer(
        self,
        emb_btc: Tensor,
        sampling_strategy: SamplingStrategy = "top-p",
        k: int | None = None,
        p: float | None = 0.95,
        temperature: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        x_b1c: Tensor = self.init_emb.expand(emb_btc.size(0), 1, -1)
        x_b1c = x_b1c + emb_btc[:, :1]
        x_b1c, state = self.rwkv(x_b1c)
        logits_b1l = self.proj(x_b1c)
        tokens_bt = sample_from_logits(logits_b1l, sampling_strategy, k=k, p=p, temperature=temperature)
        tokens_b1 = tokens_bt[:, :1]

        x_list_btc = [x_b1c]
        for t in range(1, emb_btc.size(1)):
            x_b1c = self.embeddings(tokens_b1) + emb_btc[:, t : t + 1]
            x_b1c, state = self.rwkv(x_b1c, state)
            x_list_btc.append(x_b1c)
            logits_b1l = self.proj(x_b1c)
            tokens_b1 = sample_from_logits(logits_b1l, sampling_strategy, k=k, p=p, temperature=temperature)
            tokens_bt = torch.cat((tokens_bt, tokens_b1), dim=1)

        return tokens_bt, torch.cat(x_list_btc, dim=1)
