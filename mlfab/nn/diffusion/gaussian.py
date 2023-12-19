# mypy: disable-error-code="import"
"""Defines the API for Gaussian diffusion.

This is largely take from `here <https://github.com/tonyduan/diffusion>`_.

This module can be used to train a Gaussian diffusion model as follows.

.. code-block:: python

    # Instantiate the beta schedule and diffusion module.
    diff = GaussianDiffusion()

    # Pseudo-training loop.
    for _ in range(1000):
        images = ds[index]  # Get some image from the dataset
        loss = diff.loss(images, model)
        loss.backward()
        optimizer.step()

    # Sample from the model.
    init_noise = torch.randn_like(images)
    generated = diff.sample(model, init_noise)
    show_image(generated[-1])

Choices for the beta schedule are:

- ``"linear"``: Linearly increasing beta.
- ``"quad"``: Quadratically increasing beta.
- ``"warmup"``: Linearly increasing beta with a warmup period.
- ``"const"``: Constant beta.
- ``"cosine"``: Cosine annealing schedule.
- ``"jsd"``: Jensen-Shannon divergence schedule.
"""

import functools
import math
from pathlib import Path
from typing import Callable, Literal, cast, get_args

import torch
from torch import Tensor, nn

from mlfab.nn.diffusion.ode import BaseODESolver, ODESolverType, get_ode_solver
from mlfab.nn.functions import append_dims, device_fn
from mlfab.nn.losses import pseudo_huber_loss

DiffusionLossFn = Literal["mse", "l1", "pseudo-huber"]
DiffusionPredMode = Literal["pred_x_0", "pred_eps", "pred_v"]
SigmaType = Literal["upper_bound", "lower_bound"]
DiffusionBetaSchedule = Literal["linear", "quad", "warmup", "const", "cosine", "jsd"]


def get_diffusion_loss_fn(
    loss_fn: DiffusionLossFn,
    dim: int = -1,
    factor: float = 0.00054,
) -> Callable[[Tensor, Tensor], Tensor]:
    match loss_fn:
        case "mse":
            return nn.MSELoss(reduction="none")
        case "l1":
            return nn.L1Loss(reduction="none")
        case "pseudo-huber":
            return functools.partial(pseudo_huber_loss, dim=dim, factor=factor)
        case _:
            raise ValueError(f"Unexpected loss function: {loss_fn}")


def _warmup_beta_schedule(
    beta_start: float,
    beta_end: float,
    num_timesteps: int,
    warmup: float,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    betas = beta_end * torch.ones(num_timesteps, dtype=dtype)
    warmup_time = int(num_timesteps * warmup)
    betas[:warmup_time] = torch.linspace(beta_start, beta_end, warmup_time, dtype=dtype)
    return betas


def _cosine_beta_schedule(
    num_timesteps: int,
    offset: float = 0.008,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    rng = torch.arange(num_timesteps, dtype=dtype)
    f_t = torch.cos((rng / (num_timesteps - 1) + offset) / (1 + offset) * math.pi / 2) ** 2
    bar_alpha = f_t / f_t[0]
    beta = torch.zeros_like(bar_alpha)
    beta[1:] = (1 - (bar_alpha[1:] / bar_alpha[:-1])).clip(0, 0.999)
    return beta


def cast_beta_schedule(schedule: str) -> DiffusionBetaSchedule:
    assert schedule in get_args(DiffusionBetaSchedule), f"Unknown schedule type: {schedule}"
    return cast(DiffusionBetaSchedule, schedule)


def get_diffusion_beta_schedule(
    schedule: DiffusionBetaSchedule,
    num_timesteps: int,
    *,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    warmup: float = 0.1,
    cosine_offset: float = 0.008,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Returns a beta schedule for the given schedule type.

    Args:
        schedule: The schedule type.
        num_timesteps: The total number of timesteps.
        beta_start: The initial beta value, for linear, quad, and warmup
            schedules.
        beta_end: The final beta value, for linear, quad, warmup and const
            schedules.
        warmup: The fraction of timesteps to use for the warmup schedule
            (between 0 and 1).
        cosine_offset: The cosine offset, for cosine schedules.
        dtype: The dtype of the returned tensor.

    Returns:
        The beta schedule, a tensor with shape ``(num_timesteps)``.
    """
    match schedule:
        case "linear":
            return torch.linspace(beta_start, beta_end, num_timesteps, dtype=dtype)
        case "quad":
            return torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps, dtype=dtype) ** 2
        case "warmup":
            return _warmup_beta_schedule(beta_start, beta_end, num_timesteps, warmup, dtype=dtype)
        case "const":
            return torch.full((num_timesteps,), beta_end, dtype=dtype)
        case "cosine":
            return _cosine_beta_schedule(num_timesteps, cosine_offset, dtype=dtype)
        case "jsd":
            return torch.linspace(num_timesteps, 1, num_timesteps, dtype=dtype) ** -1.0
        case _:
            raise NotImplementedError(f"Unknown schedule type: {schedule}")


class GaussianDiffusion(nn.Module):
    """Defines a module which provides utility functions for Gaussian diffusion.

    Parameters:
        beta_schedule: The beta schedule type to use.
        num_beta_steps: The number of beta steps to use.
        pred_mode: The prediction mode, which determines what the model should
            predict. Can be one of:

            - ``"pred_x_0"``: Predicts the initial noise.
            - ``"pred_eps"``: Predicts the noise at the current timestep.
            - ``"pred_v"``: Predicts the velocity of the noise.

        loss: The type of loss to use. Can be one of:

                - ``"mse"``: Mean squared error.
                - ``"l1"``: Mean absolute error.

        sigma_type: The type of sigma to use. Can be one of:

                - ``"upper_bound"``: The upper bound of the posterior noise.
                - ``"lower_bound"``: The lower bound of the posterior noise.

        solver: The ODE solver to use for running incremental model steps.
            If not set, will default to using the built-in ODE solver.
        beta_start: The initial beta value, for linear, quad, and warmup
            schedules.
        beta_end: The final beta value, for linear, quad, warmup and const
            schedules.
        warmup: The fraction of timesteps to use for the warmup schedule
            (between 0 and 1).
        cosine_offset: The cosine offset, for cosine schedules.
        loss_dim: The dimension over which to compute the loss. This should
            typically be the channel dimension.
        loss_factor: The factor to use for the pseudo-Huber loss. The default
            value comes from the Consistency Models improvements paper.
    """

    __constants__ = ["num_timesteps", "pred_mode", "sigma_type"]

    @device_fn("cpu")
    def __init__(
        self,
        beta_schedule: DiffusionBetaSchedule = "linear",
        num_beta_steps: int = 1000,
        pred_mode: DiffusionPredMode = "pred_x_0",
        loss: DiffusionLossFn = "mse",
        sigma_type: SigmaType = "upper_bound",
        solver: ODESolverType = "euler",
        *,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        warmup: float = 0.1,
        cosine_offset: float = 0.008,
        loss_dim: int = -1,
        loss_factor: float = 0.00054,
    ) -> None:
        super().__init__()

        # Gets the beta schedule from the given parameters.
        betas = get_diffusion_beta_schedule(
            schedule=beta_schedule,
            num_timesteps=num_beta_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            warmup=warmup,
            cosine_offset=cosine_offset,
        )

        assert betas.dim() == 1

        self.num_timesteps = betas.shape[0] - 1
        self.pred_mode = pred_mode
        self.sigma_type = sigma_type
        self.loss_fn = get_diffusion_loss_fn(loss, dim=loss_dim, factor=loss_factor)

        assert not (betas < 0).any(), "Betas must be non-negative."
        assert not (betas > 1).any(), "Betas must be less than or equal to 1."

        bar_alpha = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("bar_alpha", bar_alpha, persistent=False)

        # The ODE solver to use.
        self.solver = get_ode_solver(solver)

    bar_alpha: Tensor

    def loss(self, model: Callable[[Tensor, Tensor], Tensor], x: Tensor) -> Tensor:
        """Computes the loss for a given sample.

        Args:
            model: The model forward process, which takes a tensor with the
                same shape as the input data plus a timestep and returns the
                predicted noise or target, with shape ``(*)``.
            x: The input data, with shape ``(*)``

        Returns:
            The loss, with shape ``(*)``.
        """
        bsz = x.shape[0]
        t_sample = torch.randint(1, self.num_timesteps + 1, size=(bsz,), device=x.device)
        eps = torch.randn_like(x)
        bar_alpha = self.bar_alpha[t_sample].view(-1, *[1] * (x.dim() - 1)).expand(x.shape)
        x_t = torch.sqrt(bar_alpha) * x + torch.sqrt(1 - bar_alpha) * eps
        pred_target = model(x_t, t_sample)
        match self.pred_mode:
            case "pred_x_0":
                gt_target = x
            case "pred_eps":
                gt_target = eps
            case "pred_v":
                gt_target = torch.sqrt(bar_alpha) * eps - torch.sqrt(1 - bar_alpha) * x
            case _:
                raise NotImplementedError(f"Unknown pred_mode: {self.pred_mode}")
        return self.loss_fn(pred_target, gt_target)

    @torch.no_grad()
    def partial_sample(
        self,
        model: Callable[[Tensor, Tensor], Tensor],
        reference_sample: Tensor,
        start_percent: float,
        sampling_timesteps: int | None = None,
        solver: BaseODESolver | None = None,
    ) -> Tensor:
        """Samples from the model, starting from a given reference sample.

        Partial sampling takes a reference sample, adds some noise to it, then
        denoises the sample using the model. This can be used for doing
        style transfer, where the reference sample is the source image which
        the model redirects to look more like some target style.

        Args:
            model: The model forward process, which takes a tensor with the
                same shape as the input data plus a timestep and returns the
                predicted noise or target, with shape ``(*)``.
            reference_sample: The reference sample, with shape ``(*)``.
            start_percent: What percent of the diffusion process to start from;
                0 means that all of the diffusion steps will be used, while 1
                means that none of the diffusion steps will be used.
            sampling_timesteps: The number of timesteps to sample for. If
                ``None``, then the full number of timesteps will be used.
            solver: The ODE solver to use for running incremental model steps.
                If not set, will default to using the built-in ODE solver.

        Returns:
            The samples, with shape ``(sampling_timesteps + 1, *)``.
        """
        assert 0.0 <= start_percent <= 1.0
        num_timesteps = round(self.num_timesteps * start_percent)
        scalar_t_start = num_timesteps
        noise = torch.randn_like(reference_sample)
        bar_alpha = self.bar_alpha[scalar_t_start].view(-1, *[1] * (noise.dim() - 1)).expand(noise.shape)
        x = torch.sqrt(bar_alpha) * reference_sample + torch.sqrt(1 - bar_alpha) * noise
        return self._sample_common(
            model=model,
            x=x,
            solver=solver,
            sampling_timesteps=sampling_timesteps,
            start_percent=start_percent,
        )

    @torch.no_grad()
    def sample(
        self,
        model: Callable[[Tensor, Tensor], Tensor],
        shape: tuple[int, ...],
        device: torch.device,
        sampling_timesteps: int | None = None,
        solver: BaseODESolver | None = None,
    ) -> Tensor:
        """Samples from the model.

        Args:
            model: The model forward process, which takes a tensor with the
                same shape as the input data plus a timestep and returns the
                predicted noise or target, with shape ``(*)``.
            shape: The shape of the samples.
            device: The device to put the samples on.
            sampling_timesteps: The number of timesteps to sample for. If
                ``None``, then the full number of timesteps will be used.
            solver: The ODE solver to use for running incremental model steps.
                If not set, will default to using the built-in ODE solver.

        Returns:
            The samples, with shape ``(sampling_timesteps + 1, *)``.
        """
        return self._sample_common(
            model=model,
            x=torch.randn(shape, device=device),
            solver=solver,
            sampling_timesteps=sampling_timesteps,
            start_percent=0.0,
        )

    @torch.no_grad()
    def _get_t_tensor(self, t: int, x: Tensor) -> Tensor:
        return torch.empty([x.shape[0]], dtype=torch.int64, device=x.device).fill_(t)

    @torch.no_grad()
    def _get_bar_alpha(self, t: Tensor, x: Tensor) -> Tensor:
        # When using non-integer timesteps, like when using the RK4 ODE solver,
        # we interpolate the `bar_alpha` values. Since `bar_alpha` is a
        # cumultive product we need to do a weighted geometric mean rather than
        # a linear mean. Side note: This code works for both the case where
        # `t_max - t_min` is 1 and where it is 0.
        if t.is_floating_point():
            t_min, t_max = t.floor().to(torch.int64), t.ceil().to(torch.int64)
            bar_alpha_min, bar_alpha_max = self.bar_alpha[t_min], self.bar_alpha[t_max]
            w_min = t - t_min.to(t)
            factor = bar_alpha_max / bar_alpha_min
            bar_alpha = torch.pow(factor, w_min) * bar_alpha_min
        else:
            bar_alpha = self.bar_alpha[t]
        return append_dims(bar_alpha, x.dim())

    @torch.no_grad()
    def _run_model(self, model: Callable[[Tensor, Tensor], Tensor], x: Tensor, t: Tensor, bar_alpha: Tensor) -> Tensor:
        # Use model to predict x_0.
        match self.pred_mode:
            case "pred_x_0":
                return model(x, t)
            case "pred_eps":
                pred_eps = model(x, t)
                return (x - torch.sqrt(1 - bar_alpha) * pred_eps) / torch.sqrt(bar_alpha)
            case "pred_v":
                pred_v = model(x, t)
                return torch.sqrt(bar_alpha) * x - torch.sqrt(1 - bar_alpha) * pred_v
            case _:
                raise AssertionError(f"Invalid {self.pred_mode=}.")

    @torch.no_grad()
    def _sample_step(
        self,
        model: Callable[[Tensor, Tensor], Tensor],
        x: Tensor,
        solver: BaseODESolver,
        scalar_t_start: int,
        scalar_t_end: int,
    ) -> Tensor:
        def func(x: Tensor, t: Tensor) -> Tensor:
            return self._run_model(model, x, t, self._get_bar_alpha(t, x))

        def add_fn(x: Tensor, pred_x: Tensor, t: Tensor, next_t: Tensor) -> Tensor:
            bar_alpha, bar_alpha_next = self._get_bar_alpha(t, x), self._get_bar_alpha(next_t, x)
            lhs_factor = torch.sqrt(bar_alpha / bar_alpha_next) * (1 - bar_alpha_next)
            rhs_factor = torch.sqrt(bar_alpha_next) * (1 - bar_alpha / bar_alpha_next)
            lhs = lhs_factor * x
            rhs = rhs_factor * pred_x
            return (lhs + rhs) / (1 - bar_alpha)

        t_start, t_end = self._get_t_tensor(scalar_t_start, x), self._get_t_tensor(scalar_t_end, x)

        # Forward model posterior mean given x_0, x_t
        # When t_start = t_end + 1, bar_alpha_start / bar_alpha_end = 1 / alpha_end
        # Here is the non-abstracted code for a simple Euler solver.
        # bar_alpha_start, bar_alpha_end = self._get_bar_alpha(t_start, x), self._get_bar_alpha(t_end, x)
        # pred_x = self._run_model(model, x, t_start, bar_alpha_start)
        # lhs = torch.sqrt(bar_alpha_start / bar_alpha_end) * (1 - bar_alpha_end) * x
        # rhs = torch.sqrt(bar_alpha_end) * (1 - bar_alpha_start / bar_alpha_end) * pred_x
        # x_next = (lhs + rhs) / (1 - bar_alpha_start)

        return solver.step(x, t_start, t_end, func, add_fn)

    @torch.no_grad()
    def _add_noise(self, x: Tensor, scalar_t_start: int, scalar_t_end: int) -> Tensor:
        t_start, t_end = self._get_t_tensor(scalar_t_start, x), self._get_t_tensor(scalar_t_end, x)
        bar_alpha_start, bar_alpha_end = self._get_bar_alpha(t_start, x), self._get_bar_alpha(t_end, x)

        # Forward model posterior noise
        match self.sigma_type:
            case "upper_bound":
                std = torch.sqrt(1 - bar_alpha_start / bar_alpha_end)
                noise = std * torch.randn_like(x)
            case "lower_bound":
                std = torch.sqrt((1 - bar_alpha_start / bar_alpha_end) * (1 - bar_alpha_end) / (1 - bar_alpha_start))
                noise = std * torch.randn_like(x)
            case _:
                raise AssertionError(f"Invalid {self.sigma_type=}.")

        return x + noise

    @torch.no_grad()
    def _sample_common(
        self,
        model: Callable[[Tensor, Tensor], Tensor],
        x: Tensor,
        solver: BaseODESolver | None = None,
        sampling_timesteps: int | None = None,
        start_percent: float = 0.0,
    ) -> Tensor:
        assert 0.0 <= start_percent <= 1.0
        if solver is None:
            solver = self.solver

        sampling_timesteps = self.num_timesteps if sampling_timesteps is None else sampling_timesteps
        assert 1 <= sampling_timesteps <= self.num_timesteps

        # Start sampling at `start_percent` instead of at zero.
        num_timesteps = round(self.num_timesteps * (1 - start_percent))
        sampling_timesteps = round(sampling_timesteps * (1 - start_percent))

        subseq = torch.linspace(num_timesteps, 0, sampling_timesteps + 1).round()
        samples = torch.empty((sampling_timesteps + 1, *x.shape), device=x.device)
        samples[-1] = x

        for idx, (t_start, t_end) in enumerate(zip(subseq[:-1], subseq[1:])):
            x = self._sample_step(model, x, solver, t_start, t_end)
            if t_end != 0:
                x = self._add_noise(x, t_start, t_end)
            samples[-1 - idx - 1] = x
        return samples


def plot_schedules(*, num_timesteps: int = 100, output_file: str | Path | None = None) -> None:
    """Plots all of the schedules together on one graph.

    Args:
        num_timesteps: The number of timesteps to plot
        output_file: The file to save the plot to. If ``None``, then the plot
            will be shown instead.
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Please install matplotlib to use this script: `pip install matplotlib`") from e

    # Computes the beta values for each schedule.
    schedules = get_args(DiffusionBetaSchedule)
    ts = torch.arange(num_timesteps)
    betas = torch.empty((len(schedules), num_timesteps))
    stds = torch.empty((len(schedules), num_timesteps - 1))
    for i, schedule in enumerate(schedules):
        betas[i] = beta = get_diffusion_beta_schedule(schedule, num_timesteps=num_timesteps)
        bar_alpha = torch.cumprod(1.0 - beta, dim=0)
        frac = bar_alpha[1:] / bar_alpha[:-1]
        std = torch.sqrt(1 - frac)
        stds[i] = std

    plt.figure(figsize=(8, 12))

    # Plots the Beta schedule values.
    plt.subplot(2, 1, 1)
    for i, schedule in enumerate(schedules):
        plt.plot(ts, betas[i], label=schedule)
    plt.legend()
    plt.title("Betas")
    plt.xlabel("Time")
    plt.ylabel("Beta")
    plt.yscale("log")
    plt.grid(True)

    # Plots the corresponding sigma values.
    plt.subplot(2, 1, 2)
    for i, schedule in enumerate(schedules):
        plt.plot(ts[:-1], stds[i], label=schedule)
    plt.legend()
    plt.title("Standard Deviations")
    plt.xlabel("Time")
    plt.ylabel("Standard Deviation")
    plt.grid(True)

    plt.tight_layout()
    if output_file is None:
        plt.show()
    else:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file)


if __name__ == "__main__":
    # python -m mlfab.nn.diffusion
    plot_schedules()
