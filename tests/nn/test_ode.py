"""Tests ordinary differential equation components."""

from typing import get_args

import pytest
import torch
from torch import Tensor

import mlfab


@pytest.mark.parametrize("s", get_args(mlfab.ODESolverType))
def test_ode_solver_quadratic(s: mlfab.ODESolverType) -> None:
    solver = mlfab.get_ode_solver(s)

    # We're approximating the function x(t) = t^2 + c, which has the derivative
    # x'(t) = 2t. We'll use the initial condition x(0) = c.
    def quadratic(x: Tensor, t: Tensor) -> Tensor:
        return 2 * t

    c = torch.tensor([1.0, 2.0, 3.0]).unsqueeze(-1)
    t = torch.linspace(0.0, 1.0, 100)

    # Initial condition.
    x = c

    # Runs the solver.
    for i in range(len(t) - 1):
        ti, ti_next = t[i : i + 1], t[i + 1 : i + 2]
        x = solver(x, ti, ti_next, quadratic)

    # Checks against the analytical solution.
    x_ref = t[-1] ** 2 + c
    assert torch.allclose(x, x_ref, atol=0.5)

    # Runs the solver in reverse.
    for i in range(len(t) - 2, -1, -1):
        ti, ti_prev = t[i + 1 : i + 2], t[i : i + 1]
        x = solver(x, ti, ti_prev, quadratic)

    # Checks the initial value.
    assert torch.allclose(x, c, atol=0.1)


@pytest.mark.parametrize("s", get_args(mlfab.ODESolverType))
def test_ode_solver_dynamic(s: mlfab.ODESolverType) -> None:
    solver = mlfab.get_ode_solver(s)

    # This time, use the function x(t) = x(t - 1) + t, with initial condition
    # x(0) = [1, 2, 3].
    def dynamic(x: Tensor, t: Tensor) -> Tensor:
        return x + t

    c = torch.tensor([1.0, 2.0, 3.0]).unsqueeze(-1)
    t = torch.linspace(0.0, 1.0, 100)

    # Initial condition.
    x = c

    # Runs the solver.
    for i in range(len(t) - 1):
        ti, ti_next = t[i : i + 1], t[i + 1 : i + 2]
        x = solver(x, ti, ti_next, dynamic)

    # Checks against the analytical solution.
    x_ref = (c + 1) * torch.exp(t[-1]) - t[-1] - 1
    assert torch.allclose(x, x_ref, atol=0.1)

    # Runs the solver in reverse.
    for i in range(len(t) - 2, -1, -1):
        ti, ti_prev = t[i + 1 : i + 2], t[i : i + 1]
        x = solver(x, ti, ti_prev, dynamic)

    # Checks the initial value.
    assert torch.allclose(x, c, atol=0.1)
