# mypy: disable-error-code="override"
"""Defines primitive model parallel layers.

Before using this module, you should initialize the parallel process groups
using :func:`mlfab.nn.parallel.init_parallelism`. This will create
three process group for model parallelism, pipeline parallelism, and data
parallelism. The process group information can be accessed using
:func:`mlfab.nn.parallel.parallel_group_info`.

The following layers are defined:

- :class:`ParallelEmbedding`: A model-parallel embedding layer.
- :class:`ColumnParallelLinear`: A column model-parallel linear layer.
- :class:`RowParallelLinear`: A row model-parallel linear layer.

The :class:`RowParallelLinear` and :class:`ColumnParallelLinear` layers can
be used to create a model parallel two-layer MLP, as shown below.

.. code-block:: python

    # Create a parallel embedding layer.
    parallel_embedding = ParallelEmbedding(
        num_embeddings=vocab_size,
        embedding_dim=in_features,
    )

    # Create a column parallel linear layer.
    column_parallel_linear = ColumnParallelLinear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        gather_output=False,
    )

    # Create a row parallel linear layer.
    row_parallel_linear = RowParallelLinear(
        in_features=out_features,
        out_features=out_features,
        bias=bias,
        input_is_parallel=True,
    )

    # Applies the two linear layers together.
    x = torch.randint(0, vocab_size - 1, (bsz, tsz))
    y = row_parallel_linear(column_parallel_linear(parallel_embedding(x)))

This is equivalent to the following single-process implementation.

.. code-block:: python

    # Create a sequential model.
    model = nn.Sequential(
        nn.Embedding(vocab_size, in_features),
        nn.Linear(in_features, out_features, bias=bias),
        nn.Linear(out_features, out_features, bias=bias),
    )

    # Applies the sequential model.
    x = torch.randint(0, vocab_size - 1, (bsz, tsz))
    y = model(x)
"""

import functools
import logging
import math
import os
import socket
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Literal, ParamSpec, Sequence, TypeVar, cast, overload

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from omegaconf import II, Container as OmegaConfContainer, OmegaConf
from torch import Tensor, nn
from torch.autograd.function import Function, FunctionCtx
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import Backend, ReduceOp, Work, _get_default_group, is_initialized
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.api import ShardingStrategy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.dataloader import get_worker_info as _get_worker_info_base

from mlfab.core.conf import field, load_user_config
from mlfab.nn.init import InitializationType, init_
from mlfab.utils.logging import LOG_INFO_ALL, configure_logging
from mlfab.utils.text import colored

logger = logging.getLogger(__name__)

DEFAULT_PORT = 29500

P = ParamSpec("P")
T = TypeVar("T", bound=nn.Module)

_RANK: int | None = None
_LOCAL_RANK: int | None = None
_WORLD_SIZE: int | None = None
_LOCAL_WORLD_SIZE: int | None = None
_MASTER_ADDR: str | None = None
_MASTER_PORT: int | None = None
_INIT_METHOD: str | None = None


def set_rank(rank: int) -> None:
    global _RANK

    if rank != _RANK:
        _RANK = rank
        os.environ["RANK"] = str(rank)
    else:
        raise ValueError(f"Rank {rank} is already set")


def get_rank_optional() -> int | None:
    return _RANK


def get_rank() -> int:
    return 0 if _RANK is None else _RANK


def clear_rank() -> None:
    global _RANK

    _RANK = None
    os.environ.pop("RANK", None)


def set_local_rank(rank: int) -> None:
    global _LOCAL_RANK

    if rank != _LOCAL_RANK:
        _LOCAL_RANK = rank
        os.environ["LOCAL_RANK"] = str(rank)
    else:
        raise ValueError(f"Local rank {rank} is already set")


def get_local_rank_optional() -> int | None:
    return _LOCAL_RANK


def get_local_rank() -> int:
    return 0 if _LOCAL_RANK is None else _LOCAL_RANK


def clear_local_rank() -> None:
    global _LOCAL_RANK

    _LOCAL_RANK = None
    os.environ.pop("LOCAL_RANK", None)


def set_world_size(world_size: int) -> None:
    global _WORLD_SIZE

    if world_size != _WORLD_SIZE:
        _WORLD_SIZE = world_size
        os.environ["WORLD_SIZE"] = str(world_size)
    else:
        raise ValueError(f"World size {world_size} is already set")


def get_world_size_optional() -> int | None:
    return _WORLD_SIZE


def get_world_size() -> int:
    return 1 if _WORLD_SIZE is None else _WORLD_SIZE


def clear_world_size() -> None:
    global _WORLD_SIZE

    _WORLD_SIZE = None
    os.environ.pop("WORLD_SIZE", None)


def set_local_world_size(local_world_size: int) -> None:
    global _LOCAL_WORLD_SIZE

    if local_world_size != _LOCAL_WORLD_SIZE:
        _LOCAL_WORLD_SIZE = local_world_size
        os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)
    else:
        raise ValueError(f"World size {local_world_size} is already set")


def get_local_world_size_optional() -> int | None:
    return _LOCAL_WORLD_SIZE


def get_local_world_size() -> int:
    return 1 if _LOCAL_WORLD_SIZE is None else _LOCAL_WORLD_SIZE


def clear_local_world_size() -> None:
    global _LOCAL_WORLD_SIZE

    _LOCAL_WORLD_SIZE = None
    os.environ.pop("LOCAL_WORLD_SIZE", None)


def set_master_addr(master_addr: str) -> None:
    global _MASTER_ADDR

    if master_addr != _MASTER_ADDR:
        os.environ["MASTER_ADDR"] = _MASTER_ADDR = master_addr
    else:
        raise ValueError(f"Master address {master_addr} is already set")


def get_master_addr() -> str:
    assert _MASTER_ADDR is not None, "Master address is not yet set"
    return _MASTER_ADDR


def clear_master_addr() -> None:
    global _MASTER_ADDR

    _MASTER_ADDR = None
    os.environ.pop("MASTER_ADDR", None)


def set_master_port(port: int) -> None:
    global _MASTER_PORT

    if port != _MASTER_PORT:
        _MASTER_PORT = port
        os.environ["MASTER_PORT"] = str(port)
    else:
        raise ValueError(f"Master port {port} is already set")


def get_master_port() -> int:
    assert _MASTER_PORT is not None, "Master port is not yet set"
    return _MASTER_PORT


def clear_master_port() -> None:
    global _MASTER_PORT

    _MASTER_PORT = None
    os.environ.pop("MASTER_PORT", None)


def is_master() -> bool:
    return get_rank() == 0


def is_distributed() -> bool:
    return _INIT_METHOD is not None


def set_init_method(init_method: str) -> None:
    global _INIT_METHOD

    if init_method != _INIT_METHOD:
        os.environ["INIT_METHOD"] = _INIT_METHOD = init_method
    else:
        raise ValueError(f"Init method {init_method} is already set")


def get_init_method() -> str:
    assert _INIT_METHOD is not None, "Init method is not yet set"
    return _INIT_METHOD


def clear_init_method() -> None:
    global _INIT_METHOD

    _INIT_METHOD = None
    os.environ.pop("INIT_METHOD", None)


def set_dist(
    rank: int,
    local_rank: int,
    world_size: int,
    local_world_size: int,
    master_addr: str,
    master_port: int,
    init_method: str,
) -> None:
    set_rank(rank)
    set_local_rank(local_rank)
    set_world_size(world_size)
    set_local_world_size(local_world_size)
    set_master_addr(master_addr)
    set_master_port(master_port)
    set_init_method(init_method)


def clear_dist() -> None:
    clear_rank()
    clear_local_rank()
    clear_world_size()
    clear_local_world_size()
    clear_master_addr()
    clear_master_port()
    clear_init_method()


@dataclass
class _GroupInfo:
    """Information and helper functions for a process group.

    This is a singleton which can be accessed via ``group_info()``. For example,
    to do a model parallel reduction, you can do:

    .. code-block:: python

        group_info().mp.reduce(tensor)

    Attributes:
        group: The process group.
        global_ranks: The global ranks of all processes in the group.
        rank: The rank of the current process in the group.
        world_size: The number of processes in the group.
    """

    group: ProcessGroup
    global_ranks: list[int]
    rank: int
    world_size: int

    @overload
    def reduce(
        self,
        tensor: Tensor,
        op: Any = ReduceOp.SUM,  # noqa: ANN401
        *,
        async_op: Literal[False] = False,
    ) -> Tensor: ...

    @overload
    def reduce(
        self,
        tensor: Tensor,
        op: Any = ReduceOp.SUM,  # noqa: ANN401
        *,
        async_op: Literal[True],
    ) -> Work: ...

    def reduce(
        self,
        tensor: Tensor,
        op: Any = ReduceOp.SUM,
        *,
        async_op: bool = False,
    ) -> Tensor | Work:  # noqa: ANN401
        """Reduces the tensor across all processes in the group.

        Consider two tensors in the same process group on different processes,
        with values ``[1, 2, 3]`` and ``[4, 5, 6]``. After calling this
        function, both tensors will have the value ``[5, 7, 9]``.

        Args:
            tensor: The tensor to reduce.
            op: The reduction operation to perform.
            async_op: Whether to perform the operation asynchronously.

        Returns:
            The reduced tensor.
        """
        if self.world_size == 1:
            return tensor
        work = dist.all_reduce(tensor, op=op, group=self.group, async_op=async_op)
        return work if async_op else tensor

    def split(self, tensor: Tensor, dim: int = 0) -> Tensor:
        """Splits the tensor across all processes in the group.

        Consider a tensor with shape ``[8, 4]`` split across 4 processes. After
        calling this function, each process will have a tensor with shape
        ``[2, 4]``.

        Args:
            tensor: The tensor to split.
            dim: The dimension to split along.

        Returns:
            The split tensor.
        """
        if self.world_size == 1:
            return tensor
        slice_len = tensor.shape[dim] // self.world_size
        return tensor.narrow(dim, self.rank * slice_len, slice_len)

    @overload
    def gather(self, tensor: Tensor, dim: int = -1, *, async_op: Literal[False] = False) -> Tensor: ...

    @overload
    def gather(self, tensor: Tensor, dim: int = -1, *, async_op: Literal[True]) -> Work: ...

    def gather(self, tensor: Tensor, dim: int = -1, *, async_op: bool = False) -> Tensor | Work:
        """Gathers the tensor across all processes in the group.

        Consider a tensor with shape ``[2, 4]`` split across 4 processes. After
        calling this function, the process with rank 0 will have a tensor with
        shape ``[8, 4]``.

        Args:
            tensor: The tensor to gather.
            dim: The dimension to gather along.
            async_op: Whether to perform the operation asynchronously.

        Returns:
            The gathered tensor, or a work pointer if async.
        """
        if self.world_size == 1:
            return tensor
        output = [torch.empty_like(tensor) for _ in range(self.world_size)]
        work = dist.all_gather(output, tensor, group=self.group, async_op=async_op)
        return work if async_op else torch.cat(output, dim=dim)


@dataclass
class _GroupsInfos:
    mp: _GroupInfo
    pp: _GroupInfo
    dp: _GroupInfo


_parallel_group_info: _GroupsInfos | None = None
_default_group_info: _GroupInfo | None = None


def parallel_group_info() -> _GroupsInfos:
    assert _parallel_group_info is not None
    return _parallel_group_info


def default_group_info() -> _GroupInfo | None:
    global _default_group_info
    if _default_group_info is None and is_initialized():
        rank, world_size = dist.get_rank(), dist.get_world_size()
        _default_group_info = _GroupInfo(_get_default_group(), list(range(world_size)), rank, world_size)
    return _default_group_info


class ParallismError(Exception):
    pass


def init_parallelism(
    model_parallelism: int = 1,
    pipeline_parallelism: int = 1,
    *,
    mp_backend: str | Backend | None = None,
    pp_backend: str | Backend | None = None,
    dp_backend: str | Backend | None = None,
) -> None:
    """Initializes parallelism groups and parameters.

    Args:
        model_parallelism: Number of model parallel GPUs. Each layer of
            computation will simultaneously run on this many GPUs.
        pipeline_parallelism: Number of pipeline parallel layers. The total
            number of GPUs processing a single input will be the product
            of ``model_parallelism`` and ``pipeline_parallelism``.
        mp_backend: Backend to use for model parallelism.
        pp_backend: Backend to use for pipeline parallelism.
        dp_backend: Backend to use for data parallelism.

    Raises:
        ParallismError: If some settings are invalid.
    """
    global _parallel_group_info

    if _parallel_group_info is not None:
        raise ParallismError("Parallelism is already initialized; call `reset_parallelism` first.")

    if not dist.is_initialized():
        raise ParallismError("Distributed training is not initialized.")

    rank, world_size = dist.get_rank(), dist.get_world_size()

    # This is specific behavior - if model parallelism is too large for the
    # current machine, we just clamp it to whatever the world size is. We
    # don't do this for pipeline parallelism because there are fewer use cases
    # where it is necessary.
    if model_parallelism > world_size:
        logger.warning(
            "Model parallelism %d is greater than world size %d, setting to %d",
            model_parallelism,
            world_size,
            world_size,
        )
        model_parallelism = world_size

    # Validates parallelism for current world size.
    if world_size % model_parallelism != 0:
        raise ParallismError(f"{world_size=} is not divisible by {model_parallelism=}")
    if world_size % (model_parallelism * pipeline_parallelism) != 0:
        pipeline_size = model_parallelism * pipeline_parallelism
        raise ParallismError(f"{world_size=} is not divisible by {pipeline_size=}")

    data_parallelism = world_size // (model_parallelism * pipeline_parallelism)

    logger.info(
        "Parallism configuration\n ↪ %s parallelism %s\n ↪ %s parallelism %s\n ↪ %s parallelism %s",
        colored("Model", "light-green"),
        colored(str(model_parallelism), "light-cyan", bold=True),
        colored("Pipeline", "light-green"),
        colored(str(pipeline_parallelism), "light-cyan", bold=True),
        colored("Data", "light-green"),
        colored(str(data_parallelism), "light-cyan", bold=True),
    )

    # [[[0, 1],
    #   [2, 3]],
    #  [[4, 5],
    #   [6, 7]]]
    groups = torch.arange(world_size).view(data_parallelism, pipeline_parallelism, model_parallelism)

    # We split this way so that two near-by GPUs are more likely to be in the
    # same model parallel group than data parallel group. This is because for
    # typical environments we have data parallel groups that are on separate
    # devices.
    dp_rank = rank % (model_parallelism * pipeline_parallelism)
    pp_rank = (rank // pipeline_parallelism) % model_parallelism
    mp_rank = rank // (model_parallelism * pipeline_parallelism)

    def get_groups(groups: Sequence[Tensor], backend: str | Backend | None) -> list[tuple[ProcessGroup, list[int]]]:
        return [(dist.new_group(group.tolist(), backend=backend), group.tolist()) for group in groups]

    # [[0, 4], [1, 5], [2, 6], [3, 7]].
    dp_groups = get_groups(groups.flatten(1).unbind(1), dp_backend)
    # [[0, 2], [1, 3], [4, 6], [5, 7]
    pp_groups = get_groups(groups.transpose(0, 1).flatten(1).unbind(1), pp_backend)
    # [[0, 1], [2, 3], [4, 5], [6, 7]]
    mp_groups = get_groups(groups.flatten(0, 1).unbind(0), mp_backend)

    # We need to initialize all groups across all devices, but then we choose
    # the specific group for this device.
    dp_group, dp_ids = dp_groups[dp_rank]
    pp_group, pp_ids = pp_groups[pp_rank]
    mp_group, mp_ids = mp_groups[mp_rank]

    # Sets the group info now that it is initialized.
    _parallel_group_info = _GroupsInfos(
        mp=_GroupInfo(mp_group, mp_ids, mp_rank, model_parallelism),
        pp=_GroupInfo(pp_group, pp_ids, pp_rank, pipeline_parallelism),
        dp=_GroupInfo(dp_group, dp_ids, dp_rank, data_parallelism),
    )


def parallelism_is_initialized() -> bool:
    return _parallel_group_info is not None


def reset_parallelism() -> None:
    global _parallel_group_info
    _parallel_group_info = None


class _ModelParallelCopy(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: Tensor,
        op: Any,  # noqa: ANN401
    ) -> Tensor:
        ctx.op = op
        return x

    @staticmethod
    def backward(ctx: FunctionCtx, grad: Tensor) -> tuple[Tensor, None]:
        return parallel_group_info().mp.reduce(grad, op=ctx.op), None


def mp_copy(x: Tensor, op: Any = ReduceOp.SUM) -> Tensor:  # noqa: ANN401
    """Copies the input to the model parallel region.

    Forward this is a no-op, but backward it reduces the gradient across
    model parallel replicas (i.e., it is a cross-replica sum).

    Args:
        x: Input tensor, with shape ``(*)``.
        op: Reduction operation to use when reducing the gradient.

    Returns:
        Output tensor, with shape ``(*)``.
    """
    return _ModelParallelCopy.apply(x, op)


class _ModelParallelReduce(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: Tensor,
        op: Any,  # noqa: ANN401
    ) -> Tensor:
        ctx.mark_dirty(x)
        return parallel_group_info().mp.reduce(x, op=op)

    @staticmethod
    def backward(ctx: FunctionCtx, grad: Tensor) -> tuple[Tensor, None]:
        return grad, None


def mp_reduce(x: Tensor, op: Any = ReduceOp.SUM) -> Tensor:  # noqa: ANN401
    """Reduces the input from the model parallel region.

    Forward this reduces the input across model parallel replicas (i.e., it is
    a cross-replica sum), but backward it is a no-op.

    Args:
        x: Input tensor, with shape ``(*)``.
        op: Reduction operation to use when reducing the gradient.

    Returns:
        Output tensor, with shape ``(*)``.
    """
    return _ModelParallelReduce.apply(x, op)


class _ModelParallelScatter(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor, dim: int) -> Tensor:
        ctx.dim = dim
        return parallel_group_info().mp.split(x, dim=dim)

    @staticmethod
    def backward(ctx: FunctionCtx, grad: Tensor) -> tuple[Tensor, None]:
        return parallel_group_info().mp.gather(grad, dim=ctx.dim), None


def mp_scatter(x: Tensor, dim: int = -1) -> Tensor:
    """Scatters the input across model parallel regions.

    Args:
        x: Input tensor, with shape ``(..., N, ...)``.
        dim: Dimension to scatter along.

    Returns:
        Output tensor, with shape ``(..., N // world_size, ...)``.
    """
    return _ModelParallelScatter.apply(x, dim)


class _ModelParallelGather(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor, dim: int) -> Tensor:
        ctx.dim = dim
        return parallel_group_info().mp.gather(x, dim=dim)

    @staticmethod
    def backward(ctx: FunctionCtx, grad: Tensor) -> tuple[Tensor, None]:
        return parallel_group_info().mp.split(grad, dim=ctx.dim), None


def mp_gather(x: Tensor, dim: int = -1) -> Tensor:
    """Gathers the input from model parallel regions.

    Args:
        x: Input tensor, with shape ``(..., N, ...)``.
        dim: Dimension to gather along.

    Returns:
        Output tensor, with shape ``(..., N * world_size, ...)``.
    """
    return _ModelParallelGather.apply(x, dim)


def initialize_model_parallel_affine_weight_(
    weight: Tensor,
    out_features: int,
    in_features: int,
    per_partition_size: int,
    partition_dim: int,
    init_type: InitializationType = "xavier_normal",
    stride: int = 1,
) -> None:
    """Initializes an affine weight tensor for model-parallel training.

    Args:
        weight: Weight tensor to initialize.
        out_features: Number of output features.
        in_features: Number of input features.
        per_partition_size: Size of each partition.
        partition_dim: Partition dimension.
        init_type: Initialization type.
        stride: Stride for the initialization.
    """
    # Skip meta weights.
    if weight.is_meta:
        return

    mp_info = parallel_group_info().mp
    rank, world_size = mp_info.rank, mp_info.world_size

    # For single GPU cases, just initialize normally.
    if world_size == 1:
        init_(weight, None, init_type)
        return

    # Initializes the master weight.
    master_weight = weight.new_empty(out_features, in_features, requires_grad=False)
    init_(master_weight, None, init_type)

    # Splits the master weight by the world size.
    assert per_partition_size % stride == 0, f"{per_partition_size=} is not divisible by {stride=}"
    per_partition_per_stride_size = per_partition_size // stride
    weight_list = torch.split(master_weight, per_partition_per_stride_size, dim=partition_dim)

    # Copies the rank weight to the model parallel weight.
    rank_weight_list = weight_list[rank::world_size]
    with torch.no_grad():
        torch.cat(rank_weight_list, dim=partition_dim, out=weight)


class ParallelEmbedding(nn.Module):
    __constants__ = ["num_embeddings", "embedding_dim", "padding_idx", "max_norm", "scale_grad_by_freq", "sparse"]

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        init_type: InitializationType = "xavier_normal",
    ) -> None:
        """Model-parallel embeddings.

        Embeddings are partitioned along the ``embedding_dim`` dimension.

        Args:
            num_embeddings: Number of embeddings (vocabulary size).
            embedding_dim: Embedding dimension; must be divisible by the
                model-parallel size.
            padding_idx: See ``nn.Embedding``.
            max_norm: See ``nn.Embedding``.
            norm_type: See ``nn.Embedding``.
            scale_grad_by_freq: See ``nn.Embedding``.
            sparse: See ``nn.Embedding``.
            init_type: Initialization type.
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.init_type = init_type
        self._weight = None

        # Splits by world size.
        world_size = parallel_group_info().mp.world_size
        assert embedding_dim % world_size == 0, f"{embedding_dim=} not divisible by {world_size=}"
        self.embedding_dim_per_rank = embedding_dim // world_size

        # Allocate weights for current rank.
        self.weight = nn.Parameter(torch.empty(num_embeddings, self.embedding_dim_per_rank))

        self.reset_parameters()

    @property
    def master_weight(self) -> Tensor:
        return mp_gather(self.weight, dim=1)

    def reset_parameters(self) -> None:
        initialize_model_parallel_affine_weight_(
            weight=self.weight,
            out_features=self.num_embeddings,
            in_features=self.embedding_dim,
            per_partition_size=self.embedding_dim_per_rank,
            partition_dim=1,
            init_type=self.init_type,
            stride=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = mp_copy(x)

        output_parallel = F.embedding(
            x,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

        return mp_gather(output_parallel)


class ColumnParallelLinear(nn.Module):
    __constants__ = ["in_features", "out_features", "gather_output", "init_type", "stride"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
        init_type: InitializationType = "xavier_normal",
        stride: int = 1,
    ) -> None:
        """A column parallel linear layer.

        This layer splits the weight matrix along the output feature dimension,
        and each rank is only responsible for ``out_features // world_size``
        number of output features.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            bias: Whether to include a bias term.
            gather_output: Whether to gather the output from all the model
                parallel GPUs.
            init_type: Initialization type.
            stride: Stride for the initialization.
            lora_rank: The LoRA rank to use, if any.
        """
        super().__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.init_type = init_type
        self.stride = stride

        # Splits by world size.
        world_size = parallel_group_info().mp.world_size
        assert out_features % world_size == 0, f"{out_features=} not divisible by {world_size=}"
        self.output_size_per_partition = out_features // world_size

        # Initializes the per-rank weight.
        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, self.in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        initialize_model_parallel_affine_weight_(
            weight=self.weight,
            out_features=self.out_features,
            in_features=self.in_features,
            per_partition_size=self.output_size_per_partition,
            partition_dim=0,
            init_type=self.init_type,
            stride=self.stride,
        )

    @property
    def master_weight(self) -> Tensor:
        return mp_gather(self.weight, dim=0)

    @property
    def master_bias(self) -> Tensor | None:
        return None if self.bias is None else mp_gather(self.bias, dim=0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward method.

        Args:
            x: input tensor of size ``(*, in_features)``

        Returns:
            Output tensor of size ``(*, out_features // world_size)``, or
            ``(*, out_features)`` if ``gather_output`` is set to ``True``.
        """
        input_parallel = mp_copy(x)
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        return mp_gather(output_parallel) if self.gather_output else output_parallel


class RowParallelLinear(nn.Module):
    __constants__ = ["in_features", "out_features", "input_is_parallel", "init_type", "stride"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        init_type: InitializationType = "xavier_normal",
        stride: int = 1,
    ) -> None:
        """A row parallel linear layer.

        This layer splits the weight matrix along the input feature dimension,
        and each rank is only responsible for ``in_features // world_size``
        number of input features.

        This can be paired with a column parallel layer to create a model
        parallel two-stage linear layer.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            bias: Whether to include a bias term.
            input_is_parallel: Whether the input tensor is already split
                along the feature dimension.
            init_type: Initialization type.
            stride: Stride for the initialization.
        """
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.init_type = init_type
        self.stride = stride

        # Splits by world size.
        world_size = parallel_group_info().mp.world_size
        assert in_features % world_size == 0, f"{in_features=} not divisible by {world_size=}"
        self.input_size_per_partition = in_features // world_size

        # Initializes the per-rank weight.
        self.weight = nn.Parameter(Tensor(self.out_features, self.input_size_per_partition))
        if bias:
            self.bias = nn.Parameter(Tensor(self.out_features))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        initialize_model_parallel_affine_weight_(
            weight=self.weight,
            out_features=self.out_features,
            in_features=self.in_features,
            per_partition_size=self.input_size_per_partition,
            partition_dim=-1,
            init_type=self.init_type,
            stride=self.stride,
        )

    @property
    def master_weight(self) -> Tensor:
        return mp_gather(self.weight, dim=-1)

    @property
    def master_bias(self) -> Tensor | None:
        return None if self.bias is None else mp_gather(self.bias, dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward method.

        Args:
            x: input tensor of size ``(*, in_features)``, or
                ``(*, in_features // world_size)`` if ``input_is_parallel``
                is set to ``True``.

        Returns:
            Output tensor of size ``(*, out_features)``.
        """
        input_parallel = x if self.input_is_parallel else mp_scatter(x)
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        output = mp_reduce(output_parallel)
        return output if self.bias is None else output + self.bias


@dataclass
class WorkerInfo:
    worker_id: int
    num_workers: int
    in_worker: bool


def get_data_worker_info() -> WorkerInfo:
    if (worker_info := _get_worker_info_base()) is None:
        return WorkerInfo(
            worker_id=0,
            num_workers=1,
            in_worker=False,
        )

    return WorkerInfo(
        worker_id=worker_info.id,
        num_workers=worker_info.num_workers,
        in_worker=True,
    )


def split_n_items_across_workers(n: int, worker_id: int, num_workers: int) -> tuple[int, int]:
    """Computes offsets for splitting N items across K workers.

    This returns the start and end indices for the items to be processed by the
    given worker. The end index is exclusive.

    Args:
        n: The number of items to process.
        worker_id: The ID of the current worker.
        num_workers: The total number of workers.

    Returns:
        The start and end index for the items in the current worker.
    """
    assert n >= num_workers, f"n ({n}) must be >= num_workers ({num_workers})"
    assert 0 <= worker_id < num_workers, f"worker_id ({worker_id}) must be >= 0 and < num_workers ({num_workers})"

    # The number of items to process per worker.
    items_per_worker = math.ceil(n / num_workers)

    # The start and end indices for the items to process.
    start = worker_id * items_per_worker
    end = min(start + items_per_worker, n)

    return start, end


def num_workers(default: int) -> int:
    max_workers = load_user_config().experiment.max_workers
    if hasattr(os, "sched_getaffinity"):
        try:
            return min(len(os.sched_getaffinity(0)), max_workers)
        except Exception:
            pass
    if (cpu_count := os.cpu_count()) is not None:
        return min(cpu_count, max_workers)
    return min(default, max_workers)


OmegaConf.register_new_resolver("mlfab.num_workers", num_workers, replace=True)


def get_unused_port(default: int | None = None) -> int:
    """Returns an unused port number on the local machine.

    Args:
        default: A default port to try before trying other ports.

    Returns:
        A port number which is currently unused
    """
    if default is not None:
        sock = socket.socket()
        try:
            sock.bind(("", default))
            return default
        except OSError:
            pass
        finally:
            sock.close()

    sock = socket.socket()
    sock.bind(("", 0))
    return sock.getsockname()[1]


OmegaConf.register_new_resolver("mlfab.unused_port", get_unused_port, replace=True)


def port_is_busy(port: int) -> int:
    """Checks whether a port is busy.

    Args:
        port: The port to check.

    Returns:
        Whether the port is busy.
    """
    sock = socket.socket()
    try:
        sock.bind(("", port))
        return False
    except OSError:
        return True
    finally:
        sock.close()


def get_device_count(default: int) -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else default


OmegaConf.register_new_resolver("mlfab.device_count", get_device_count, replace=True)


@dataclass
class ParallelConfig:
    use_fsdp: bool = field(False, help="If set, use FSDP; otherwise, use DDP")
    cpu_offload: bool = field(False, help="CPU offloading for FSDP")
    sharding_strategy: ShardingStrategy = field(ShardingStrategy.HYBRID_SHARD, help="Sharding strategy")
    sync_module_states: bool = field(True, help="Whether to sync module states on initialization")


def ddp(model: nn.Module, cfg: ParallelConfig) -> DDP:
    group_info = parallel_group_info()

    return DDP(model, process_group=group_info.dp.group)


def _all_params_are_cuda(model: nn.Module) -> bool:
    return all(p.is_cuda for p in model.parameters())


def fsdp(model: nn.Module, cfg: ParallelConfig) -> FSDP:
    group_info = parallel_group_info()

    process_group: tuple[ProcessGroup, ProcessGroup] | ProcessGroup
    if cfg.sharding_strategy in (ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2):
        process_group = group_info.mp.group, group_info.dp.group
    else:
        process_group = group_info.mp.group

    if cfg.cpu_offload:
        logger.warning("CPU offloading doesn't support gradient accumulation")

    return FSDP(
        model,
        process_group=process_group,
        sharding_strategy=cfg.sharding_strategy,
        sync_module_states=cfg.sync_module_states and _all_params_are_cuda(model),
        cpu_offload=CPUOffload(cfg.cpu_offload),
    )


def dp(model: T, cfg: ParallelConfig) -> T | DDP | FSDP:
    """Wraps a model for data parallel training, if necessary.

    Args:
        model: The model to wrap.
        cfg: The model configuration.

    Returns:
        The wrapped model.
    """
    if get_world_size() <= 1:
        return model
    return fsdp(model, cfg) if cfg.use_fsdp else ddp(model, cfg)


@dataclass
class MultiProcessConfig:
    rank: int = field(-1, help="The rank of the process")
    local_rank: int = field(-1, help="The local rank of the process")
    world_size: int = field(II("mlfab.device_count:1"), help="The total number of processes")
    local_world_size: int = field(II("world_size"), help="The number of processes per machine")
    master_addr: str = field("127.0.0.1", help="The address of the master process")
    master_port: int = field(II("mlfab.unused_port:29500"), help="The port of the master process")
    init_method: str = field("env://", help="The initialization method")
    model_parallelism: int = field(1, help="The number of model parallel processes")
    pipeline_parallelism: int = field(1, help="The number of pipeline parallel processes")
    backend: str | None = field(None, help="The distributed backend")
    model_parallel_backend: str | None = field(None, help="The model parallel backend")
    pipeline_parallel_backend: str | None = field(None, help="The pipeline parallel backend")
    data_parallel_backend: str | None = field(None, help="The data parallel backend")
    launch_method: str = field("spawn", help="The launch method for multiprocessing")


def init_process_group_from_backend(backend: str | dist.Backend | None = None) -> None:
    if backend is None:
        backend = get_distributed_backend()
    init_method, world_size, rank = get_init_method(), get_world_size(), get_rank()

    logger.log(LOG_INFO_ALL, "Initializing %d / %d using %s - %s", rank, world_size, init_method, backend)
    dist.init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=rank)

    if torch.cuda.is_available():
        torch.cuda.set_device(get_local_rank())

    logger.info("Initialized process group; running dummy all-reduce")
    dist.all_reduce(torch.zeros(1, device="cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Dummy all-reduce succeeded")


def init_dist(
    rank: int,
    local_rank: int,
    world_size: int,
    local_world_size: int,
    master_addr: str,
    master_port: int,
    init_method: str,
    backend: str | dist.Backend | None = None,
) -> None:
    """Initializes distributed environment.

    Args:
        rank: The rank of the current process.
        local_rank: The local rank of the current process.
        world_size: The total number of processes.
        local_world_size: The number of processes per machine.
        master_addr: The address of the master process.
        master_port: The port of the master process.
        init_method: The initialization method.
        backend: The distributed backend.
    """
    set_dist(rank, local_rank, world_size, local_world_size, master_addr, master_port, init_method)
    init_process_group_from_backend(backend)


@functools.lru_cache(maxsize=None)
def default_backend() -> str:
    if torch.cuda.is_available():
        return "nccl"
    return "gloo"


def get_distributed_backend() -> dist.Backend:
    # Used to change the distributed backend to something other than NCCL.
    # For example, if you're on a system with some strange NCCL errors, you
    # can try changing this environment variable to `gloo`.
    return dist.Backend(os.environ.get("TORCH_DISTRIBUTED_BACKEND", default_backend()))


def set_distributed_backend(backend: str) -> None:
    os.environ["TORCH_DISTRIBUTED_BACKEND"] = backend


def init_and_run(
    func: Callable[P, None],
    cfg: MultiProcessConfig,
    *args: P.args,
    **kwargs: P.kwargs,
) -> None:
    configure_logging(rank=cfg.rank, world_size=cfg.world_size)

    init_dist(
        rank=cfg.rank,
        local_rank=cfg.local_rank,
        world_size=cfg.world_size,
        local_world_size=cfg.local_world_size,
        master_addr=cfg.master_addr,
        master_port=cfg.master_port,
        init_method=cfg.init_method,
        backend=cfg.backend,
    )

    init_parallelism(
        model_parallelism=cfg.model_parallelism,
        pipeline_parallelism=cfg.pipeline_parallelism,
        mp_backend=cfg.backend if cfg.model_parallel_backend is None else cfg.model_parallel_backend,
        pp_backend=cfg.backend if cfg.pipeline_parallel_backend is None else cfg.pipeline_parallel_backend,
        dp_backend=cfg.backend if cfg.data_parallel_backend is None else cfg.data_parallel_backend,
    )

    func(*args, **kwargs)


def _func_wrapped(
    func: Callable[P, None],
    setup: Callable[[], None] | None,
    cfg: MultiProcessConfig,
    error_queue: "mp.SimpleQueue[str | None]",
    *args: P.args,
    **kwargs: P.kwargs,
) -> None:
    try:
        if setup is not None:
            setup()

        init_and_run(func, cfg, *args, **kwargs)

    except KeyboardInterrupt:
        logger.info("Caught KeyboardInterrupt; exiting")

    except Exception:
        error_queue.put(traceback.format_exc())
        sys.exit(1)

    error_queue.put(None)


def cleanup() -> None:
    if dist.GroupMember.WORLD is not None:
        dist.destroy_process_group()
    clear_dist()
    reset_parallelism()


def launch_subprocesses(
    func: Callable[P, None],
    cfg: MultiProcessConfig | None = None,
    setup: Callable[[], None] | None = None,
    rank_offset: int = 0,
    *args: P.args,
    **kwargs: P.kwargs,
) -> None:
    """Launches a function in multiple subprocesses.

    Args:
        func: The function to launch.
        cfg: The configuration for the function.
        args: The positional arguments to pass to the function.
        setup: A function to run before launching the subprocesses.
        rank_offset: The offset to add to the rank of each subprocess.
        kwargs: The keyword arguments to pass to the function.

    Raises:
        RuntimeError: If the function fails in any subprocess.
    """
    if cfg is None:
        cfg = MultiProcessConfig()

    # Runs OmegaConf resolve to resolve any variables.
    cfg = cast(MultiProcessConfig, OmegaConf.merge(OmegaConf.structured(MultiProcessConfig), cfg))
    OmegaConf.resolve(cast(OmegaConfContainer, cfg))

    if cfg.world_size <= 1:
        cfg.rank = 0
        cfg.local_rank = 0
        init_and_run(func, cfg, *args, **kwargs)
        cleanup()
        return

    logger.info("Launching %d training workers", cfg.world_size)
    ctx = mp.get_context(cfg.launch_method)
    error_queues: list["mp.SimpleQueue[str | None]"] = []
    procs = []
    for rank in range(cfg.world_size):
        rank = rank + rank_offset
        error_queue = ctx.SimpleQueue()
        cfg.rank = rank
        cfg.local_rank = rank % cfg.local_world_size
        proc = ctx.Process(
            target=_func_wrapped,
            args=[func, setup, cfg, error_queue, *args],
            kwargs=kwargs,
            daemon=False,
            name=f"worker-{rank}",
        )
        logger.debug("Started process %d", rank)
        proc.start()
        error_queues.append(error_queue)
        procs.append(proc)

    pctx = mp.ProcessContext(procs, error_queues)
    while not pctx.join():
        pass

    for rank, error_queue in enumerate(error_queues):
        error = error_queue.get()
        if error:
            raise RuntimeError(f"Process {rank} failed with error:\n{error}")

    cleanup()


class _AllToAll(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, group: dist.ProcessGroup, input: Tensor) -> Tensor:
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        if dist.is_initialized():
            dist.all_to_all_single(output, input, group=group)
        else:
            assert group is None
            output = input
        return output

    @staticmethod
    def backward(ctx: FunctionCtx, *grad_output: Tensor) -> tuple[None, Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output))


def all_to_all(input: Tensor, group: dist.ProcessGroup | None) -> Tensor:
    """Performs an all-to-all operation on the input tensor.

    Args:
        input: The input tensor.
        group: The process group to use for the all-to-all operation.

    Returns:
        The output tensor.
    """
    if group is None:
        group = dist.group.WORLD
    return _AllToAll.apply(group, input)
