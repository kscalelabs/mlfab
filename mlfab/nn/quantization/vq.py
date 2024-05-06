"""Defines modules for doing codebook learning via nearest neighbors."""

import copy
from typing import Callable, cast

import torch
import torch.distributed
import torch.nn.functional as F
from torch import Tensor, nn

from mlfab.nn.functions import swap_grads


def _ema_inplace(moving_avg: Tensor, new: Tensor, decay: float) -> None:
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def _laplace_smoothing(x: Tensor, n_categories: int, epsilon: float = 1e-5) -> Tensor:
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def _sample_vectors(samples: Tensor, num: int) -> Tensor:
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)
    return samples[indices]


def _kmeans(samples: Tensor, num_clusters: int, num_iters: int = 10) -> tuple[Tensor, Tensor]:
    dim, dtype = samples.shape[-1], samples.dtype

    means = _sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        diffs = samples.unsqueeze(1) - means.unsqueeze(0)
        dists = -(diffs**2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, buckets.unsqueeze(1).repeat(1, dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


def _identity(x: Tensor) -> Tensor:
    return x


class _EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance.

    Parameters:
        dim: Dimension.
        codebook_size: Codebook size (i.e., the number of codes).
        kmeans_init: Whether to use k-means to initialize the codebooks. If set
            to true, run the k-means algorithm on the first training batch and
            use the learned centroids as initialization.
        kmeans_iters: Number of iterations used for k-means algorithm at
            initialization.
        decay: Decay for exponential moving average over the codebooks.
        epsilon: Epsilon value for numerical stability.
        threshold_ema_dead_code: Threshold for dead code expiration. Replace
            any codes that have an exponential moving average cluster size less
            than the specified threshold with randomly selected vector from the
            current batch.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
    ) -> None:
        super().__init__()

        self.decay = decay

        embed = torch.empty(codebook_size, dim)
        if not kmeans_init:
            nn.init.kaiming_uniform_(embed)

        self.codebook_size = codebook_size

        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.all_reduce_fn = (
            cast(Callable[[Tensor], Tensor], torch.distributed.all_reduce)
            if torch.distributed.is_initialized()
            else _identity
        )

        self.register_buffer("inited", Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

    inited: Tensor
    cluster_size: Tensor
    embed: Tensor
    embed_avg: Tensor

    @torch.jit.ignore
    def init_embed_(self, data: Tensor) -> None:
        if self.inited:
            return

        embed, cluster_size = _kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(Tensor([True]))

    def replace_(self, samples: Tensor, mask: Tensor) -> None:
        modified_codebook = torch.where(mask[..., None], _sample_vectors(samples, self.codebook_size), self.embed)
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples: Tensor) -> None:
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return

        batch_samples = batch_samples.flatten(0, -2)
        self.replace_(batch_samples, mask=expired_codes)

    def preprocess(self, x: Tensor) -> Tensor:
        return x.flatten(0, -2)

    def quantize(self, x: Tensor) -> Tensor:
        embed = self.embed.t()
        dist = -(x.pow(2).sum(1, keepdim=True) - 2 * x @ embed + embed.pow(2).sum(0, keepdim=True))
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    def postprocess_emb(self, embed_ind: Tensor, shape: torch.Size) -> Tensor:
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind: Tensor) -> Tensor:
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    def encode(self, x: Tensor) -> Tensor:
        shape = x.shape
        x = self.preprocess(x)
        embed_ind = self.quantize(x)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind: Tensor) -> Tensor:
        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x)

        self.init_embed_(x)

        embed_ind = self.quantize(x)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        quantize = self.dequantize(embed_ind)

        if self.training:
            # We do the expiry of code at that point as buffers are in sync
            # and all the workers will take the same decision.
            cluster_size = embed_onehot.sum(0)
            self.all_reduce_fn(cluster_size)
            _ema_inplace(self.cluster_size, cluster_size, self.decay)
            embed_sum = x.t() @ embed_onehot
            self.all_reduce_fn(embed_sum)
            _ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            smoothed_sizes = _laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
            cluster_size = smoothed_sizes * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)
            self.expire_codes_(x)

        return quantize, embed_ind


class VectorQuantization(nn.Module):
    """Vector quantization implementation.

    The codebook itself doesn't learn any parameters using backpropagation.
    Instead, it uses an exponential moving average to update the codebooks. For
    a given batch, we assign each vector to the nearest codebook item. We then
    get the mean of each cluster and update the codebook towards that vector
    using an exponential moving average.

    Parameters:
        dim: Dimension.
        codebook_size: Codebook size (i.e., the number of codes).
        codebook_dim: Codebook dimension. If not defined, uses the specified
            dimension in dim.
        decay: Decay for exponential moving average over the codebooks.
        epsilon: Epsilon value for numerical stability.
        kmeans_init: Whether to use kmeans to initialize the codebooks.
        kmeans_iters: Number of iterations used for kmeans initialization.
        threshold_ema_dead_code: Threshold for dead code expiration. Replace
            any codes that have an exponential moving average cluster size less
            than the specified threshold with randomly selected vector from the
            current batch.
        commitment_weight: Weight for commitment loss.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: int | None = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = False,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        commitment_weight: float = 1.0,
    ) -> None:
        super().__init__()

        _codebook_dim = dim if codebook_dim is None else codebook_dim
        requires_projection = _codebook_dim != dim
        self.project_in = nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity()

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight

        self._codebook = _EuclideanCodebook(
            dim=_codebook_dim,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            epsilon=epsilon,
            threshold_ema_dead_code=threshold_ema_dead_code,
        )
        self.codebook_size = codebook_size

    @property
    def codebook(self) -> Tensor:
        return self._codebook.embed

    def encode(self, x: Tensor) -> Tensor:
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind: Tensor) -> Tensor:
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        return quantize

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        device = x.device
        x = self.project_in(x)

        quantize, embed_ind = self._codebook(x)

        loss = torch.tensor([0.0], device=device, requires_grad=self.training)

        if self.training and self.commitment_weight > 0:
            commit_loss = F.mse_loss(quantize.detach(), x)
            loss = loss + commit_loss * self.commitment_weight

        # Backpropagates gradients from `quantize` to `x`. Same as
        # ``quantize = x + (quantize - x).detach()`` but uses an autograd
        # function instead. Note that the original `quantize` doesn't require
        # gradients since it is the EMA of the codebook.
        quantize, _ = swap_grads(quantize, x)

        quantize = self.project_out(quantize)
        return quantize, embed_ind, loss


class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization impementation.

    This module is a wrapper around multiple vector quantization modules. It
    applies the quantization sequentially and adds the quantized output to the
    residual.

    Parameters:
        vq_module: Vector quantization module to wrap.
        num_quantizers: Number of quantizers to use.

    Example::

        vq_module = VectorQuantization(128, 512)
        rvq_module = ResidualVectorQuantization(vq_module, 4)
        x = torch.randn(1, 128, 32)
        quantized, indices, loss = rvq_module(x)

    Input:
        x: Tensor of shape ``(batch_size, seq_len, dim)``.

    Output:
        quantized: Tensor of shape ``(batch_size, seq_len, dim)``.
        indices: Tensor of shape ``(batch_size, seq_len)``.
        loss: Tensor of shape ``(codebook_size)``.
    """

    __constants__ = ["codebook_size", "num_quantizers"]

    def __init__(self, vq_module: VectorQuantization, num_quantizers: int) -> None:
        super().__init__()

        self.codebook_size = vq_module.codebook_size
        self.num_quantizers = num_quantizers

        self.layers = cast(
            list[VectorQuantization],
            nn.ModuleList([self._get_copy(vq_module) for _ in range(num_quantizers)]),
        )

    def _get_copy(self, vq_module: VectorQuantization) -> VectorQuantization:
        return copy.deepcopy(vq_module)

    def forward(self, x: Tensor, n_q: int | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        quantized_out: Tensor | None = None
        residual = x

        all_losses = []
        all_indices = []
        all_quantized = []

        n_q = n_q or len(self.layers)

        for layer in self.layers[:n_q]:
            quantized, indices, loss = layer(residual)
            residual = residual - quantized.detach()
            quantized_out = quantized if quantized_out is None else quantized_out + quantized

            all_indices.append(indices)
            all_losses.append(loss)
            all_quantized.append(quantized)

        out_losses = torch.stack(all_losses)
        out_indices = torch.stack(all_indices)
        out_quant = torch.stack(all_quantized)
        assert quantized_out is not None
        return quantized_out, out_indices, out_losses, out_quant

    def encode(self, x: Tensor, n_q: int | None = None) -> Tensor:
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized.detach()
            all_indices.append(indices)
        out_indices = torch.stack(all_indices, dim=-1)
        return out_indices

    def decode(self, q_indices: Tensor) -> Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices.unbind(-1)):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out
