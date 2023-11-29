# mypy: disable-error-code="import, no-untyped-def"
# ruff: noqa: ANN001, ANN201, ANN202, N803, N806
"""Implements a Triton kernel for the Lion optimizer."""

from typing import Any

import triton
import triton.language as tl
from torch import Tensor, nn


@triton.jit
def update_fn_kernel(
    p_ptr,
    grad_ptr,
    exp_avg_ptr,
    lr,
    wd,
    beta1,
    beta2,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    b_idx = tl.program_id(0)

    block_start = b_idx * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    offset_p_ptr = p_ptr + offsets
    offset_grad_ptr = grad_ptr + offsets
    offset_exp_avg_ptr = exp_avg_ptr + offsets

    p = tl.load(offset_p_ptr, mask=mask)
    grad = tl.load(offset_grad_ptr, mask=mask)
    exp_avg = tl.load(offset_exp_avg_ptr, mask=mask)

    update = tl.where((exp_avg * beta1 + grad * (1 - beta1)) < 0, lr, -lr)
    p = p * (1 - lr * wd) + update
    exp_avg = exp_avg * beta2 + grad * (1 - beta2)

    tl.store(offset_p_ptr, p, mask=mask)
    tl.store(offset_exp_avg_ptr, exp_avg, mask=mask)


def update_fn(
    p: nn.Parameter,
    grad: Tensor,
    exp_avg: Tensor,
    lr: float,
    wd: float,
    beta1: float,
    beta2: float,
) -> None:
    n_elements = p.numel()

    def grid(meta: dict[str, Any]) -> tuple[int, ...]:
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    block_size = min(triton.next_power_of_2(n_elements), 1024)

    try:
        update_fn_kernel[grid](p, grad, exp_avg, lr, wd, beta1, beta2, n_elements, BLOCK_SIZE=block_size)

    except triton.CompilationError as e:
        raise RuntimeError("Error while running Triton kernel") from e
