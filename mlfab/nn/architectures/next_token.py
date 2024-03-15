"""Defines some common functions for next-token prediction."""

from typing import Literal, overload

import torch
import torch.nn.functional as F
from torch import Tensor

MaskMode = Literal["causal", "lengths", "combine"]

SamplingStrategy = Literal["top-p", "top-k", "greedy"]


def top_p_sampling(logits_btc: Tensor, p: float, temperature: float = 1.0, dim: int = -1) -> Tensor:
    """Samples from a distribution using top-P sampling.

    This is a modified version of ``torch.multinomial`` that uses top-p
    sampling instead of top-k sampling. The difference is that top-k sampling
    sets the probability of all values outside the top-k to zero, whereas
    top-p sampling sets the probability of all values outside the top-p
    to zero.

    Parameters:
        logits_btc: The input tensor, of shape ``(B, T, C)``.
        p: The probability threshold.
        temperature: The temperature to apply to the logits.
        dim: The dimension to sample from. Defaults to ``-1``.

    Returns:
        The sampled indices, of shape ``(B, T)``.
    """
    with torch.no_grad():
        assert 0.0 <= p <= 1.0, f"`{p=}` must be between 0 and 1"
        if dim != -1:
            logits_btc = logits_btc.transpose(dim, -1)
        orig_shape = logits_btc.shape[:-1]
        logits_nl = logits_btc.flatten(0, -2)
        probs_nl = F.softmax(logits_nl / temperature, dim=-1)
        sorted_probs_nl, indices_nl = torch.sort(probs_nl, descending=True, dim=-1)
        cum_sum_probs_nl = torch.cumsum(sorted_probs_nl, dim=-1)
        top_p_nl = cum_sum_probs_nl < p
        top_p_nl[:, 1:] = top_p_nl[:, :-1].clone()
        top_p_nl[:, 0] = 1
        sorted_probs_nl[~top_p_nl] = 0.0
        sampled_sorted_indexes_n1 = torch.multinomial(sorted_probs_nl, 1)
        sample_b1 = torch.gather(indices_nl, -1, sampled_sorted_indexes_n1)
        sample_btc1 = sample_b1.view(*orig_shape, 1)
        if dim != -1:
            sample_btc1 = sample_btc1.transpose(dim, -1)
        return sample_btc1.squeeze(dim)


def top_k_sampling(logits_btc: Tensor, k: int, temperature: float = 1.0, dim: int = -1) -> Tensor:
    """Samples from a distribution using top-k sampling.

    This function is a modified version of ``torch.multinomial`` that uses
    top-k sampling instead of top-p sampling. The difference is that top-k
    sampling sets the probability of all values outside the top-k to zero,
    whereas top-p sampling sets the probability of all values outside the
    top-p to zero.

    Parameters:
        logits_btc: The input tensor, of shape ``(B, T, C)``.
        k: The number of top values to consider.
        temperature: The temperature to apply to the logits.
        dim: The dimension to sample from. Defaults to ``-1``.

    Returns:
        The sampled indices, of shape ``(B, T)``.
    """
    with torch.no_grad():
        if dim != -1:
            logits_btc = logits_btc.transpose(dim, -1)
        orig_shape = logits_btc.shape[:-1]
        logits_nl = logits_btc.flatten(0, -2)
        probs_nl = F.softmax(logits_nl / temperature, dim=-1)
        sorted_probs_nl, indices_nl = torch.sort(probs_nl, descending=True, dim=-1)
        sorted_probs_nl[:, k:] = 0.0
        sorted_probs_nl = sorted_probs_nl / sorted_probs_nl.sum(dim=-1, keepdim=True)
        sampled_sorted_indexes_n1 = torch.multinomial(sorted_probs_nl, 1)
        sample_b1 = torch.gather(indices_nl, -1, sampled_sorted_indexes_n1)
        sample_btc1 = sample_b1.view(*orig_shape, 1)
        if dim != -1:
            sample_btc1 = sample_btc1.transpose(dim, -1)
        return sample_btc1.squeeze(dim)


@overload
def sample_from_logits(
    logits_btc: Tensor,
    strategy: Literal["top-p"],
    *,
    p: float,
    temperature: float = 1.0,
) -> Tensor: ...


@overload
def sample_from_logits(
    logits_btc: Tensor,
    strategy: Literal["top-k"],
    *,
    k: int,
    temperature: float = 1.0,
) -> Tensor: ...


@overload
def sample_from_logits(logits_btc: Tensor, strategy: Literal["greedy"]) -> Tensor: ...


@overload
def sample_from_logits(
    logits_btc: Tensor,
    strategy: SamplingStrategy,
    *,
    k: int | None = None,
    p: float | None = None,
    temperature: float = 1.0,
) -> Tensor: ...


def sample_from_logits(
    logits_btc: Tensor,
    strategy: SamplingStrategy,
    *,
    k: int | None = None,
    p: float | None = None,
    temperature: float = 1.0,
) -> Tensor:
    """Samples from a distribution using a given strategy.

    This function is a wrapper around the various sampling strategies, such as
    top-p sampling, top-k sampling, and greedy sampling.

    Parameters:
        logits_btc: The input tensor, of shape ``(B, T, C)``.
        strategy: The sampling strategy to use.
        k: The number of top values to consider, for top-k sampling.
        p: The probability threshold, for top-p sampling.
        temperature: The temperature to apply to the logits.

    Returns:
        The sampled indices, of shape ``(B, T)``.
    """
    match strategy:
        case "top-p":
            if p is None:
                raise ValueError("Top-P sampling requires a probability threshold.")
            return top_p_sampling(logits_btc, p, temperature)
        case "top-k":
            if k is None:
                raise ValueError("Top-K sampling requires a number of top values to consider.")
            return top_k_sampling(logits_btc, k, temperature)
        case "greedy":
            return torch.argmax(logits_btc, dim=-1)
        case _:
            raise ValueError(f"Invalid sampling strategy: {strategy}")
