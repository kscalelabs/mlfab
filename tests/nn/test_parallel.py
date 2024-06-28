"""Tests model parallelism primitives."""

import logging

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import mlfab


@pytest.mark.parametrize("n", [13, 101])
@pytest.mark.parametrize("num_workers", [5, 7, 1])
def test_split_n_items_across_workers(n: int, num_workers: int) -> None:
    starts, ends = zip(*(mlfab.split_n_items_across_workers(n, i, num_workers) for i in range(num_workers)))
    lengths = [e - s for s, e in zip(starts, ends)]
    assert sum(lengths) == n
    assert starts[0] == 0
    assert ends[-1] == n
    assert all(starts[i] == ends[i - 1] for i in range(1, num_workers))


class DummyModel(nn.Module):
    def __init__(self, lora_rank: int | None) -> None:
        super().__init__()

        # A simple embedding layer plus two-layer MLP.
        self.emb = mlfab.maybe_lora(mlfab.ParallelEmbedding(10, 12), lora_rank, freeze=False)
        self.l1 = mlfab.maybe_lora(mlfab.ColumnParallelLinear(12, 16, bias=False), lora_rank, freeze=False)
        self.l2 = mlfab.maybe_lora(mlfab.RowParallelLinear(16, 8, bias=False), lora_rank, freeze=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        y1 = self.l2(self.l1(self.emb(x)))
        y2 = F.linear(F.linear(F.embedding(x, self.emb.master_weight), self.l1.master_weight), self.l2.master_weight)
        return y1, y2


def setup() -> None:
    # Hides some nuisance logs.
    logging.getLogger("torch.distributed").setLevel(logging.ERROR)
    logging.getLogger("torch.nn.parallel.distributed").setLevel(logging.ERROR)

    # Setting the seed across all processes to make sure that the weights
    # initialize to the same values (needed to make the test pass).
    torch.manual_seed(1337)


def get_grad(g: Tensor | None) -> Tensor:
    assert g is not None
    return g.clone()


def assert_close(a: Tensor, b: Tensor) -> None:
    assert torch.allclose(a, b, atol=1e-3)


def func() -> None:
    base_model = DummyModel(None)
    model = mlfab.ddp(base_model)

    x = torch.randint(0, 10 - 1, (4, 12))

    # Tests that the forward passes for both models match.
    output_parallel, output_full = model(x)
    assert torch.allclose(output_parallel, output_full, atol=1e-3)

    # Backpropagates the parallel outputs.
    output_parallel, _ = model(x)
    output_parallel.sum().backward()
    emb_grad_parallel = get_grad(base_model.emb.weight.grad)
    l1_grad_parallel = get_grad(base_model.l1.weight.grad)
    l2_grad_parallel = get_grad(base_model.l2.weight.grad)
    model.zero_grad()

    # Backpropagates the full outputs.
    _, output_full = model(x)
    output_full.sum().backward()
    emb_grad_full = get_grad(base_model.emb.weight.grad)
    l1_grad_full = get_grad(base_model.l1.weight.grad)
    l2_grad_full = get_grad(base_model.l2.weight.grad)

    # Checks that the gradients for the parallel outputs and the full outputs
    # match - this is effectively checking that the implementations of the
    # model parallel modules are correct.
    assert_close(emb_grad_parallel, emb_grad_full)
    assert_close(l1_grad_parallel, l1_grad_full)
    assert_close(l2_grad_parallel, l2_grad_full)


def lora_func() -> None:
    base_model = DummyModel(2)
    model = mlfab.ddp(base_model)

    x = torch.randint(0, 10 - 1, (4, 12))

    # Tests that the forward passes for both models match.
    output_parallel, output_full = model(x)
    assert_close(output_parallel, output_full)


@pytest.mark.slow
@pytest.mark.parametrize("use_lora", [True, False])
def test_parallel_model(use_lora: bool) -> None:
    """Tests model parallelism primitives.

    This function launches 4 processes, partitioned into 2 model parallel and
    2 data parallel groups. We check that the partitioned model outputs and
    gradients match the full model.

    Args:
        use_lora: Whether to use LoRA or not.
    """
    mlfab.configure_logging()

    port = mlfab.get_unused_port()

    config = mlfab.MultiProcessConfig(
        world_size=4,
        master_addr="127.0.0.1",
        master_port=port,
        init_method=f"tcp://127.0.0.1:{port}",
        distributed_backend="gloo",
        model_parallelism=2,
        pipeline_parallelism=1,
    )

    mlfab.launch_subprocesses(lora_func if use_lora else func, config, setup=setup)


if __name__ == "__main__":
    # python -m tests.nn.test_parallel
    test_parallel_model(False)
