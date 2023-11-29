"""Defines the API for training consistency models.

This code largely references the `OpenAI implementation <https://github.com/openai/consistency_models>`_,
as well as `Simo Ryu's implementation <https://github.com/cloneofsimo/consistency_models/tree/master>`_.

.. code-block:: python

    # Instantiates the consistency model module.
    diff = ConsistencyModel(sigmas)

    # The forward pass should take a noisy tensor and the current timestep
    # and return the denoised tensor. Can add class conditioning as well but
    # you need a function which satisfies this signature.
    def forward_pass(x: Tensor, t: Tensor) -> Tensor:
        ...

    # Compute ths loss.
    loss = diff.loss(forward_pass, x, state)
    loss.sum().backward()

    # Sample from the model. Consistency models produce good samples even with
    # a small number of steps.
    samples = diff.sample(forward_pass, x.shape, x.device, num_steps=4)
"""

import math
from typing import Callable

import torch
from torch import Tensor, nn

from mlfab.nn.diffusion.gaussian import DiffusionLossFn, get_diffusion_loss_fn
from mlfab.nn.functions import append_dims


class ConsistencyModel(nn.Module):
    """Defines a module which implements consistency diffusion models.

    This model introduces an auxiliary consistency penalty to the loss function
    to encourage the ODE to be smooth, allowing for few-step inference.

    This also implements the improvements to vanilla diffusion described in
    ``Elucidating the Design Space of Diffusion-Based Generative Models``.

    Parameters:
        total_steps: The maximum number of training steps, used for determining
            the discretization step schedule.
        sigma_data: The standard deviation of the data.
        sigma_max: The maximum standard deviation for the diffusion process.
        sigma_min: The minimum standard deviation for the diffusion process.
        rho: The rho constant for the noise schedule.
        p_mean: A constant which controls the distribution of timesteps to
            sample for training. Training biases towards sampling timesteps
            from the less noisy end of the spectrum to improve convergence.
        p_std: Another constant that controls the distribution of timesteps for
            training, used in conjunction with ``p_mean``.
        start_scales: The number of different discretization scales to use at
            the start of training. At the start of training, a small number of
            scales is used to encourage the model to learn more quickly, which
            is increased over time.
        end_scales: The number of different discretization scales to use at the
            end of training.
        loss_dim: The dimension over which to compute the loss. This should
            typically be the channel dimension.
        loss_factor: The factor to use for the pseudo-Huber loss. The default
            value comes from the Consistency Models improvements paper.
    """

    __constants__ = [
        "total_steps",
        "sigma_data",
        "sigma_max",
        "sigma_min",
        "rho",
        "p_mean",
        "p_std",
        "start_scales",
        "end_scales",
    ]

    def __init__(
        self,
        total_steps: int | None = None,
        sigma_data: float = 0.5,
        sigma_max: float = 80.0,
        sigma_min: float = 0.002,
        rho: float = 7.0,
        p_mean: float = -1.1,
        p_std: float = 2.0,
        start_scales: int = 20,
        end_scales: int = 1280,
        loss: DiffusionLossFn = "pseudo-huber",
        loss_dim: int = -1,
        loss_factor: float = 0.00054,
    ) -> None:
        super().__init__()

        self.total_steps = total_steps
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = rho
        self.p_mean = p_mean
        self.p_std = p_std
        self.start_scales = start_scales
        self.end_scales = end_scales

        self.loss_fn = get_diffusion_loss_fn(loss, dim=loss_dim, factor=loss_factor)

    def loss(self, model: Callable[[Tensor, Tensor], Tensor], x: Tensor, step: int) -> Tensor:
        """Computes the consistency model loss.

        Args:
            model: The model forward process, which takes a tensor with the
                same shape as the input data plus a timestep and returns the
                predicted noise or target, with shape ``(*)``.
            x: The input data, with shape ``(*)``
            step: The current training step, used to determine the number of
                discretization steps to use.

        Returns:
            The loss for supervising the model.
        """
        dims = x.ndim

        # The number of discretization steps runs on a schedule.
        num_scales = self._get_num_scales(step)

        # Rather than randomly sampling some timesteps for training, we bias the
        # samples to be closer to the less noisy end, which improves training
        # stability. This distribution is defined as a function of the standard
        # deviations.
        # timesteps = torch.randint(0, num_scales - 1, (x.shape[0],), device=x.device)
        timesteps = self._sample_timesteps(x, num_scales)
        t_current, t_next = timesteps / (num_scales - 1), (timesteps + 1) / (num_scales - 1)

        # Converts timesteps to sigmas.
        sigma_next, sigma_current = self._get_sigmas(torch.stack((t_next, t_current))).unbind(0)

        noise = torch.randn_like(x)
        dropout_state = torch.get_rng_state()
        x_current = x + noise * append_dims(sigma_current, dims)
        y_current = self._call_model(model, x_current, sigma_current)

        # Resets the dropout state and runs the target model.
        torch.set_rng_state(dropout_state)
        with torch.no_grad():
            x_next = x + noise * append_dims(sigma_next, dims)
            y_next = self._call_model(model, x_next, sigma_next).detach()

        loss = self._get_loss(y_current, y_next, sigma_next, sigma_current)

        return loss

    @torch.no_grad()
    def partial_sample(
        self,
        model: Callable[[Tensor, Tensor], Tensor],
        reference_sample: Tensor,
        start_percent: float,
        num_steps: int,
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
            start_percent: The percentage of timesteps to start sampling from.
            num_steps: The number of sampling steps to use.

        Returns:
            The samples, with shape ``(num_steps + 1, *)``, with the first
            sample (i.e., ``samples[0]``) as the denoised output and the last
            sample (i.e., ``samples[-1]``) as the reference sample.
        """
        assert 0.0 <= start_percent <= 1.0
        device = reference_sample.device
        timesteps = torch.linspace(start_percent, 1, num_steps + 1, device=device, dtype=torch.float32)
        sigmas = self._get_sigmas(timesteps)
        x = reference_sample
        x = x + torch.randn_like(x) * sigmas[None, 0]
        samples = torch.empty((num_steps + 1, *x.shape), device=x.device)
        samples[num_steps] = x
        for i in range(num_steps):
            x = self._call_model(model, x, sigmas[None, i])
            samples[num_steps - 1 - i] = x
            if i < num_steps - 1:
                x = x + torch.randn_like(x) * sigmas[None, i + 1]
        return samples

    @torch.no_grad()
    def sample(
        self,
        model: Callable[[Tensor, Tensor], Tensor],
        shape: tuple[int, ...],
        device: torch.device,
        num_steps: int,
    ) -> Tensor:
        """Samples from the model.

        Args:
            model: The model forward process, which takes a tensor with the
                same shape as the input data plus a timestep and returns the
                predicted noise or target, with shape ``(*)``.
            shape: The shape of the samples.
            device: The device to put the samples on.
            num_steps: The number of sampling steps to use.

        Returns:
            The samples, with shape ``(num_steps + 1, *)``, with the first
            sample (i.e., ``samples[0]``) as the denoised output and the last
            sample (i.e., ``samples[-1]``) as the random noise.
        """
        timesteps = torch.linspace(0, 1, num_steps + 1, device=device, dtype=torch.float32)
        sigmas = self._get_sigmas(timesteps)
        x = torch.randn(shape, device=device) * sigmas[0]
        samples = torch.empty((num_steps + 1, *x.shape), device=x.device)
        samples[num_steps] = x
        for i in range(num_steps):
            x = self._call_model(model, x, sigmas[None, i])
            samples[num_steps - 1 - i] = x
            if i < num_steps - 1:
                x = x + torch.randn_like(x) * sigmas[None, i + 1]
        return samples

    def _get_scalings(self, sigma: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def _call_model(self, model: Callable[[Tensor, Tensor], Tensor], x_t: Tensor, sigmas: Tensor) -> Tensor:
        c_skip, c_out, c_in = (append_dims(x, x_t.ndim) for x in self._get_scalings(sigmas))
        timesteps = 1000 * 0.25 * torch.log(sigmas + 1e-44)
        model_output = model(c_in * x_t, timesteps)
        denoised = c_out * model_output + c_skip * x_t
        return denoised

    def _get_loss(self, y_hat: Tensor, y: Tensor, sigma_next: Tensor, sigma_current: Tensor) -> Tensor:
        weights = 1 / (sigma_current - sigma_next)
        loss = self.loss_fn(y_hat, y)
        weights = weights.view(-1, *([1] * (loss.dim() - 1)))
        return loss * weights

    @torch.no_grad()
    def _get_sigmas(self, timesteps: Tensor) -> Tensor:
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas: Tensor = (max_inv_rho + timesteps * (min_inv_rho - max_inv_rho)) ** self.rho
        sigmas[timesteps == 1.0] = 0.0
        return sigmas

    def _get_noise_distribution(self, sigma_next: Tensor, sigma_current: Tensor) -> Tensor:
        denom = math.sqrt(2) * self.p_std
        lhs = torch.erf((torch.log(sigma_next) - self.p_mean) / denom)
        rhs = torch.erf((torch.log(sigma_current) - self.p_mean) / denom)
        # return lhs - rhs
        return rhs - lhs

    def _sample_timesteps(self, x: Tensor, num_scales: int) -> Tensor:
        timesteps = torch.linspace(0, 1, num_scales, device=x.device, dtype=torch.float32)
        sigmas = self._get_sigmas(timesteps)
        noise_dist = self._get_noise_distribution(sigmas[1:], sigmas[:-1])
        timesteps = torch.multinomial(noise_dist, x.shape[0], replacement=True)
        return timesteps

    def _get_num_scales(self, step: int) -> int:
        if self.total_steps is None:
            return self.end_scales + 1
        num_steps = min(self.total_steps, step)
        k_prime = math.floor(self.total_steps / (math.log2(self.end_scales / self.start_scales) + 1))
        return min(self.start_scales * 2 ** math.floor(num_steps / k_prime), self.end_scales) + 1
