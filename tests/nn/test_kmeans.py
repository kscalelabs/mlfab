"""Tests the K-Means PyTorch module.

This tests that the forward pass is computing the centroid IDs correctly.
"""

import pytest
import torch

import mlfab


def test_kmeans() -> None:
    centers = torch.randn(4, 12)
    kmeans = mlfab.KMeans(centers.clone())
    vals = centers[None].repeat(3, 2, 1)

    # Checks that clusters are closest to themselves.
    clusters = kmeans(vals)
    assert (clusters == torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])).all()


@pytest.mark.has_triton()
def test_kmeans_against_triton() -> None:
    centers = torch.randn(4, 12, device="cuda")
    vals = centers[None].repeat(3, 2, 1)

    vanilla_fn = mlfab.kmeans_fn(True)
    triton_fn = mlfab.kmeans_fn(False)

    vanilla_clusters = vanilla_fn(vals, centers, (centers**2).sum(-1))
    triton_clusters = triton_fn(vals, centers, (centers**2).sum(-1))

    assert (vanilla_clusters == triton_clusters).all()


if __name__ == "__main__":
    # python -m tests.models.test_kmeans
    # test_kmeans()
    test_kmeans_against_triton()
