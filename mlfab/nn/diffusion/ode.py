"""Defines some components for dealing with ordinary differential equations."""

from abc import ABC, abstractmethod
from typing import Callable, Literal, cast, get_args

import torch
from torch import Tensor

from mlfab.nn.functions import append_dims

ODESolverType = Literal["euler", "heun", "rk4"]


def cast_solver_type(s: str) -> ODESolverType:
    assert s in get_args(ODESolverType), f"Unknown solver {s}"
    return cast(ODESolverType, s)


def vanilla_add_fn(a: Tensor, b: Tensor, ta: Tensor, tb: Tensor) -> Tensor:
    dt = append_dims(tb - ta, a.dim())
    return a + b * dt


class BaseODESolver(ABC):
    @abstractmethod
    def step(
        self,
        samples: Tensor,
        t: Tensor,
        next_t: Tensor,
        func: Callable[[Tensor, Tensor], Tensor],
        add_fn: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor] = vanilla_add_fn,
    ) -> Tensor:
        """Steps the current state forward in time.

        Args:
            samples: The current samples, with shape ``(N, *)``.
            t: The current time step, with shape ``(N)``.
            next_t: The next time step, with shape ``(N)``.
            func: The function to use to compute the derivative, with signature
                ``(samples, t) -> deriv``.
            add_fn: The addition function to use, which has the signature
                ``(a, b, ta, tb) -> a + b * (tb - ta)``.

        Returns:
            The next sample, with shape ``(N, *)``.
        """

    def __call__(
        self,
        samples: Tensor,
        t: Tensor,
        next_t: Tensor,
        func: Callable[[Tensor, Tensor], Tensor],
        add_fn: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor] = vanilla_add_fn,
    ) -> Tensor:
        return self.step(samples, t, next_t, func, add_fn)


class EulerODESolver(BaseODESolver):
    """The Euler method for solving ODEs."""

    @torch.no_grad()
    def step(
        self,
        samples: Tensor,
        t: Tensor,
        next_t: Tensor,
        func: Callable[[Tensor, Tensor], Tensor],
        add_fn: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor] = vanilla_add_fn,
    ) -> Tensor:
        x = func(samples, t)
        return add_fn(samples, x, t, next_t)


class HeunODESolver(BaseODESolver):
    """The Heun method for solving ODEs."""

    @torch.no_grad()
    def step(
        self,
        samples: Tensor,
        t: Tensor,
        next_t: Tensor,
        func: Callable[[Tensor, Tensor], Tensor],
        add_fn: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor] = vanilla_add_fn,
    ) -> Tensor:
        k1 = func(samples, t)
        k2 = func(add_fn(samples, k1, t, next_t), next_t)
        x = (k1 + k2) / 2
        return add_fn(samples, x, t, next_t)


class RK4ODESolver(BaseODESolver):
    """The fourth-order Runge-Kutta method for solving ODEs."""

    @torch.no_grad()
    def step(
        self,
        samples: Tensor,
        t: Tensor,
        next_t: Tensor,
        func: Callable[[Tensor, Tensor], Tensor],
        add_fn: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor] = vanilla_add_fn,
    ) -> Tensor:
        dt = next_t - t
        half_t = t + dt / 2
        k1 = func(samples, t)
        k2 = func(add_fn(samples, k1 / 2, t, half_t), half_t)
        k3 = func(add_fn(samples, k2, t, half_t), half_t)
        k4 = func(add_fn(samples, k3, t, next_t), next_t)
        x = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return add_fn(samples, x, t, next_t)


def get_ode_solver(s: ODESolverType) -> BaseODESolver:
    """Returns an ODE solver for a given key.

    Args:
        s: The solver key to retrieve.

    Returns:
        The solver object.
    """
    match s:
        case "euler":
            return EulerODESolver()
        case "heun":
            return HeunODESolver()
        case "rk4":
            return RK4ODESolver()
