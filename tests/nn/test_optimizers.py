"""Tests the Lion optimizer Triton kernel matches the vanilla implementation."""

import pytest
import torch
from torch import nn

import mlfab
from mlfab.nn.optimizers import get_lion_update_fn


@pytest.mark.has_triton()
def test_triton_vs_vanilla_update_funcs() -> None:
    vanilla_fn = get_lion_update_fn(True)
    triton_fn = get_lion_update_fn(False)

    p = torch.randn(10, 10).cuda()
    grad = torch.randn(10, 10).cuda()
    exp_avg = torch.randn(10, 10).cuda()

    # Checks that the Triton and vanilla update functions match.
    p_t, exp_avg_t = nn.Parameter(p.clone(), requires_grad=True), exp_avg.clone()
    triton_fn(p_t, grad, exp_avg_t, 1e-2, 1e-4, 0.9, 0.999)
    p_v, exp_avg_v = nn.Parameter(p.clone(), requires_grad=True), exp_avg.clone()
    vanilla_fn(p_v, grad, exp_avg_v, 1e-2, 1e-4, 0.9, 0.999)

    assert torch.allclose(p_t - p, p_v - p, atol=1e-6)
    assert torch.allclose(exp_avg_t, exp_avg_v, atol=1e-6)


@pytest.mark.has_triton()
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_triton_vs_vanilla(dtype: torch.dtype) -> None:
    x = torch.randn(1, 10, dtype=dtype).cuda()

    # Builds two identical models.
    weight, bias = torch.randn(10, 10, dtype=dtype).cuda(), torch.randn(10, dtype=dtype).cuda()
    model_1 = nn.Linear(10, 10).cuda().to(dtype)
    model_2 = nn.Linear(10, 10).cuda().to(dtype)
    model_1.load_state_dict({"weight": weight.clone(), "bias": bias.clone()})
    model_2.load_state_dict({"weight": weight.clone(), "bias": bias.clone()})

    # Steps optimizers with Triton enabled and disabled.
    lr = 1e-2
    opt_vanilla = mlfab.Lion(model_1.parameters(), use_triton=False, lr=lr)
    opt_triton = mlfab.Lion(model_2.parameters(), use_triton=True, lr=lr)

    opt_vanilla.zero_grad()
    model_1(x.clone()).sum().backward()
    opt_vanilla.step()

    opt_triton.zero_grad()
    model_2(x.clone()).sum().backward()
    opt_triton.step()

    # Checks that the parameters are the same after being updated.
    w1, b1 = model_1.weight, model_1.bias
    w2, b2 = model_2.weight, model_2.bias
    sw1, sb1 = opt_vanilla.state[w1]["exp_avg"], opt_vanilla.state[b1]["exp_avg"]
    sw2, sb2 = opt_triton.state[w2]["exp_avg"], opt_triton.state[b2]["exp_avg"]

    assert torch.allclose(w1, w2)
    assert torch.allclose(b1, b2)
    assert torch.allclose(sw1, sw2)
    assert torch.allclose(sb1, sb2)


if __name__ == "__main__":
    # python -m tests.optimizers.test_lion
    test_triton_vs_vanilla_update_funcs()
