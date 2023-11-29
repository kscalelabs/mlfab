# mypy: disable-error-code="import, no-untyped-def"
# ruff: noqa: ANN001, ANN201, ANN202, N803, N806
"""Implements a Triton kernel for computing K-Means cluster IDs."""

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def kmeans_kernel(
    x_ptr,
    centers_ptr,
    centers_norm_ptr,
    cluster_ids_ptr,
    n_features,
    n_clusters,
    BLOCK_SIZE_F: tl.constexpr,
):
    e_idx = tl.program_id(0)

    fs = tl.arange(0, BLOCK_SIZE_F)
    fmask = fs < n_features

    x = tl.load(x_ptr + (e_idx * n_features) + fs, mask=fmask, other=0).to(tl.float32)  # (F)
    x_norm = tl.sum(x * x, 0)  # (1)

    min_dist = float("inf")
    min_dist_idx = 0

    for c in range(0, n_clusters):
        centers = tl.load(centers_ptr + (c * n_features) + fs, mask=fmask, other=0).to(tl.float32)  # (F)
        centers_norm = tl.load(centers_norm_ptr + c).to(tl.float32)  # (1)
        dist = x_norm - 2 * tl.sum(x * centers, 0) + centers_norm  # (1)
        if dist < min_dist:
            min_dist = dist
            min_dist_idx = c

    tl.store(cluster_ids_ptr + e_idx, min_dist_idx)


def kmeans(x: Tensor, centers: Tensor, centers_norm: Tensor) -> Tensor:
    assert x.device == centers.device == centers_norm.device, "Expected all tensors to be on the same device"
    assert x.is_cuda, "Expected input tensor to be on CUDA"

    x_shape = x.shape

    x = x.reshape(-1, x_shape[-1])
    (n_elements, n_features), (n_clusters, n_features_2) = x.shape, centers.shape

    assert n_features == n_features_2, f"Expected {n_features} features in input tensor, got {n_features_2}"
    cluster_ids = x.new_empty(n_elements, dtype=torch.long)

    BLOCK_SIZE_F = triton.next_power_of_2(n_features)

    kmeans_kernel[(n_elements,)](
        x,
        centers,
        centers_norm,
        cluster_ids,
        n_features,
        n_clusters,
        BLOCK_SIZE_F=BLOCK_SIZE_F,
    )

    return cluster_ids.reshape(x_shape[:-1])
