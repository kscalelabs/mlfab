# mypy: disable-error-code="override, misc"
"""Defines primitive model parallel layers.

Before using this module, you should initialize all the process groups
using :func:`mlfab.nn.parallel.init_dist`. This will create two process group
for model parallelism and data parallelism. The process group information can
be accessed using :func:`mlfab.nn.parallel.parallel_group_info`.
"""

import datetime
import functools
import logging
import math
import os
import pickle as pkl
import socket
import sys
import tempfile
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Literal, NotRequired, ParamSpec, TypedDict, TypeVar, Unpack, cast, overload

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import II, Container as OmegaConfContainer, OmegaConf
from torch import Tensor, nn
from torch.autograd.function import Function, FunctionCtx
from torch.distributed import ProcessGroup
from torch.distributed._tensor import DeviceMesh
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.distributed_c10d import ReduceOp, Work
from torch.utils.data.dataloader import get_worker_info as _get_worker_info_base

from mlfab.core.conf import field, load_user_config
from mlfab.nn.init import InitializationType, init_
from mlfab.utils.logging import LOG_DEBUG_ALL, LOG_INFO_ALL, configure_logging
from mlfab.utils.text import colored

logger = logging.getLogger(__name__)

DEFAULT_PORT = 29500

PROCESS_GROUP_TIMEOUT = datetime.timedelta(minutes=5)

P = ParamSpec("P")
T = TypeVar("T", bound=nn.Module)


class MultiProcessKwargs(TypedDict):
    rank: NotRequired[int]
    local_rank: NotRequired[int]
    world_size: NotRequired[int]
    local_world_size: NotRequired[int]
    master_addr: NotRequired[str]
    master_port: NotRequired[int]
    tensor_parallelism: NotRequired[int | str]
    multiprocess_launch_method: NotRequired[str]


@dataclass(kw_only=True)
class MultiProcessConfig:
    rank: int = field(-1, help="The rank of the process")
    local_rank: int = field(-1, help="The local rank of the process")
    world_size: int = field(II("mlfab.device_count:1"), help="The total number of processes")
    local_world_size: int = field(II("world_size"), help="The number of processes per machine")
    master_addr: str = field("127.0.0.1", help="The address of the master process")
    master_port: int = field(II("mlfab.unused_port:29500"), help="The port of the master process")
    tensor_parallelism: int | str = field(1, help="The number of tensor parallel processes")
    multiprocess_launch_method: str = field("forkserver", help="The launch method for multiprocessing")

    @classmethod
    def default_config(cls, **kwargs: Unpack[MultiProcessKwargs]) -> "MultiProcessConfig":
        kwargs.setdefault("rank", 0)
        kwargs.setdefault("local_rank", kwargs["rank"])
        kwargs.setdefault("world_size", 1)
        kwargs.setdefault("local_world_size", kwargs["world_size"])
        kwargs.setdefault("master_port", get_unused_port())
        return MultiProcessConfig(**kwargs)


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def is_master() -> bool:
    return get_rank() == 0


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
class _GroupsInfos:
    tp: _GroupInfo
    dp: _GroupInfo

    def device_mesh(self, device_type: str) -> DeviceMesh:
        return init_device_mesh(
            device_type,
            (self.dp.world_size, self.tp.world_size),
            mesh_dim_names=("dp", "tp"),
        )


_parallel_group_info: _GroupsInfos | None = None


@overload
def parallel_group_info(required: Literal[True] = True) -> _GroupsInfos: ...


@overload
def parallel_group_info(required: Literal[False]) -> _GroupsInfos | None: ...


def parallel_group_info(required: bool = True) -> _GroupsInfos | None:
    if required:
        assert _parallel_group_info is not None
    return _parallel_group_info


@functools.lru_cache(None)
def device_mesh(device_type: str) -> DeviceMesh:
    return parallel_group_info().device_mesh(device_type)


def tp_info() -> _GroupInfo:
    if _parallel_group_info is None:
        raise RuntimeError("Parallel process groups have not been initialized!")
    return _parallel_group_info.tp


def dp_info() -> _GroupInfo:
    if _parallel_group_info is None:
        raise RuntimeError("Parallel process groups have not been initialized!")
    return _parallel_group_info.dp


def tp_rank() -> int:
    return 0 if _parallel_group_info is None else tp_info().rank


def tp_world_size() -> int:
    return 1 if _parallel_group_info is None else tp_info().world_size


def tp_ranks() -> list[int]:
    return tp_info().global_ranks


def tp_group() -> ProcessGroup:
    return tp_info().group


def tp_group_nullable() -> ProcessGroup | None:
    return None if _parallel_group_info is None else tp_group()


def dp_rank() -> int:
    return 0 if _parallel_group_info is None else dp_info().rank


def dp_world_size() -> int:
    return 1 if _parallel_group_info is None else dp_info().world_size


def dp_ranks() -> list[int]:
    return dp_info().global_ranks


def dp_group() -> ProcessGroup:
    return dp_info().group


def dp_group_nullable() -> ProcessGroup | None:
    return None if _parallel_group_info is None else dp_group()


class ParallismError(Exception):
    pass


class _TensorParallelCopy(Function):
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
        return grad if _parallel_group_info is None else tp_info().reduce(grad, op=ctx.op), None


def tp_copy(x: Tensor, op: Any = ReduceOp.SUM) -> Tensor:  # noqa: ANN401
    """Copies the input to the model parallel region.

    Forward this is a no-op, but backward it reduces the gradient across
    model parallel replicas (i.e., it is a cross-replica sum).

    Args:
        x: Input tensor, with shape ``(*)``.
        op: Reduction operation to use when reducing the gradient.

    Returns:
        Output tensor, with shape ``(*)``.
    """
    return _TensorParallelCopy.apply(x, op)


class _TensorParallelReduce(Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: Tensor,
        op: Any,  # noqa: ANN401
    ) -> Tensor:
        ctx.mark_dirty(x)
        return x if _parallel_group_info is None else tp_info().reduce(x, op=op)

    @staticmethod
    def backward(ctx: FunctionCtx, grad: Tensor) -> tuple[Tensor, None]:
        return grad, None


def tp_reduce(x: Tensor, op: Any = ReduceOp.SUM) -> Tensor:  # noqa: ANN401
    """Reduces the input from the model parallel region.

    Forward this reduces the input across model parallel replicas (i.e., it is
    a cross-replica sum), but backward it is a no-op.

    Args:
        x: Input tensor, with shape ``(*)``.
        op: Reduction operation to use when reducing the gradient.

    Returns:
        Output tensor, with shape ``(*)``.
    """
    return _TensorParallelReduce.apply(x, op)


class _TensorParallelScatter(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor, dim: int) -> Tensor:
        ctx.dim = dim
        return x if _parallel_group_info is None else tp_info().split(x, dim=dim)

    @staticmethod
    def backward(ctx: FunctionCtx, grad: Tensor) -> tuple[Tensor, None]:
        return grad if _parallel_group_info is None else tp_info().gather(grad, dim=ctx.dim), None


def tp_scatter(x: Tensor, dim: int = -1) -> Tensor:
    """Scatters the input across model parallel regions.

    Args:
        x: Input tensor, with shape ``(..., N, ...)``.
        dim: Dimension to scatter along.

    Returns:
        Output tensor, with shape ``(..., N // world_size, ...)``.
    """
    return _TensorParallelScatter.apply(x, dim)


class _TensorParallelGather(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor, dim: int) -> Tensor:
        ctx.dim = dim
        return x if _parallel_group_info is None else tp_info().gather(x, dim=dim)

    @staticmethod
    def backward(ctx: FunctionCtx, grad: Tensor) -> tuple[Tensor, None]:
        return grad if _parallel_group_info is None else tp_info().split(grad, dim=ctx.dim), None


def tp_gather(x: Tensor, dim: int = -1) -> Tensor:
    """Gathers the input from model parallel regions.

    Args:
        x: Input tensor, with shape ``(..., N, ...)``.
        dim: Dimension to gather along.

    Returns:
        Output tensor, with shape ``(..., N * world_size, ...)``.
    """
    return _TensorParallelGather.apply(x, dim)


def initialize_tensor_parallel_affine_weight_(
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

    rank, world_size = tp_rank(), tp_world_size()

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


@dataclass(kw_only=True)
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


def all_params_are_cuda(model: nn.Module) -> bool:
    return all(p.is_cuda for p in model.parameters())


def init_dist(cfg: MultiProcessConfig | None = None, all_reduce: bool = True) -> None:
    """Initializes distributed environment.

    Args:
        cfg: The multi-processs configuration.
        all_reduce: If set, run a dummy all-reduce after initialization.
    """
    global _parallel_group_info

    if _parallel_group_info is not None:
        raise ParallismError("Parallelism is already initialized; call `reset_parallelism` first.")

    if cfg is None:
        cfg = MultiProcessConfig.default_config()

    os.environ["MASTER_ADDR"] = cfg.master_addr
    os.environ["MASTER_PORT"] = str(cfg.master_port)

    device_id: torch.device | None = None
    if torch.cuda.is_available():
        dev_id = (local_rank := cfg.local_rank) % (dev_cnt := torch.cuda.device_count())
        logger.log(LOG_DEBUG_ALL, "Setting device %d (local rank %d with %d device(s))", dev_id, local_rank, dev_cnt)
        torch.cuda.set_device(dev_id)
        device_id = torch.device("cuda", dev_id)

    init_method = "env://"
    logger.log(LOG_INFO_ALL, "Initializing %d / %d using %s", cfg.rank, cfg.world_size, init_method)
    dist.init_process_group(
        backend=get_distributed_backend(),
        init_method=init_method,
        world_size=cfg.world_size,
        rank=cfg.rank,
        timeout=PROCESS_GROUP_TIMEOUT,
        device_id=device_id,
    )

    logger.debug("Initialized process group")
    if all_reduce:
        dist.all_reduce(torch.zeros(1, device="cuda" if torch.cuda.is_available() else "cpu"))
        logger.debug("Dummy all-reduce succeeded")

    if not dist.is_initialized():
        raise ParallismError("Distributed training is not initialized.")

    global_rank, global_world_size = dist.get_rank(), dist.get_world_size()
    tensor_parallelism = cfg.tensor_parallelism

    # Tries to parse to int.
    if isinstance(tensor_parallelism, str):
        try:
            tensor_parallelism = int(tensor_parallelism)
        except ValueError:
            pass

    # Converts special keys.
    special_values: dict[str, int] = {
        "local": cfg.local_world_size,
        "global": global_world_size,
    }
    if isinstance(tensor_parallelism, str):
        tensor_parallelism = tensor_parallelism.lower()
        if tensor_parallelism in special_values:
            tensor_parallelism = special_values[tensor_parallelism]
        else:
            try:
                tensor_parallelism = int(tensor_parallelism)
            except ValueError:
                special_str = "[" + ", ".join(sorted(special_values.keys())) + "]"
                raise NotImplementedError(
                    f"Invalid value for model parallelism: {tensor_parallelism}, "
                    f"should either be an integer or one of {special_str}"
                )

    if tensor_parallelism <= 0:
        raise ValueError(f"Tensor parallelism must be positive, got {tensor_parallelism}")

    # This is specific behavior - if model parallelism is too large for the
    # current machine, we just clamp it to whatever the world size is.
    if tensor_parallelism > global_world_size:
        logger.warning(
            "Tensor parallelism %d is greater than world size %d, setting to %d",
            tensor_parallelism,
            global_world_size,
            global_world_size,
        )
        tensor_parallelism = global_world_size

    # Validates parallelism for current world size.
    if global_world_size % tensor_parallelism != 0:
        raise ParallismError(f"{global_world_size=} is not divisible by {tensor_parallelism=}")
    data_parallelism = global_world_size // tensor_parallelism

    logger.log(
        LOG_INFO_ALL,
        ("Parallism configuration\n ↪ %s parallelism %s\n ↪ %s parallelism %s"),
        colored("Tensor", "light-green"),
        colored(str(tensor_parallelism), "light-cyan", bold=True),
        colored("Data", "light-green"),
        colored(str(data_parallelism), "light-cyan", bold=True),
    )

    # We split this way so that two near-by GPUs are more likely to be in the
    # same model parallel group than data parallel group. This is because for
    # typical environments we have data parallel groups that are on separate
    # devices.
    groups_dm = torch.arange(global_world_size).view(data_parallelism, tensor_parallelism)

    def get_group(groups_nd: Tensor) -> tuple[ProcessGroup, list[int]]:
        assert groups_nd.dim() == 2
        group: tuple[ProcessGroup, list[int]] | None = None
        for i in range(groups_nd.size(0)):
            group_ranks = groups_nd[i].tolist()
            group_i = dist.new_group(
                group_ranks,
                timeout=PROCESS_GROUP_TIMEOUT,
                backend=get_distributed_backend(),
            )
            if global_rank in group_ranks:
                group = (group_i, group_ranks)
        if group is None:
            raise RuntimeError(f"{global_rank=} not found in {groups_nd}")
        return group

    # We need to initialize all groups across all devices, but then we choose
    # the specific group for this device.
    dp_group, dp_ids = get_group(groups_dm.permute(1, 0))
    tp_group, tp_ids = get_group(groups_dm)

    assert isinstance(dp_group, ProcessGroup), dp_group
    assert isinstance(tp_group, ProcessGroup), tp_group

    assert len(dp_ids) == data_parallelism, f"{len(dp_ids)=} != {data_parallelism=}"
    assert len(tp_ids) == tensor_parallelism, f"{len(tp_ids)=} != {tensor_parallelism=}"

    dp_rank = global_rank // tensor_parallelism
    tp_rank = global_rank % tensor_parallelism

    # Sets the group info now that it is initialized.
    _parallel_group_info = _GroupsInfos(
        tp=_GroupInfo(
            group=tp_group,
            global_ranks=tp_ids,
            rank=tp_rank,
            world_size=tensor_parallelism,
        ),
        dp=_GroupInfo(
            group=dp_group,
            global_ranks=dp_ids,
            rank=dp_rank,
            world_size=data_parallelism,
        ),
    )


def cleanup_dist() -> None:
    global _parallel_group_info
    if _parallel_group_info is not None:
        dist.destroy_process_group(_parallel_group_info.dp.group)
        dist.destroy_process_group(_parallel_group_info.tp.group)
    if (pg := dist.GroupMember.WORLD) is not None:
        dist.destroy_process_group(pg)
    _parallel_group_info = None


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


def init_and_run(
    func: Callable[P, None],
    cfg: MultiProcessConfig | None = None,
    *args: P.args,
    **kwargs: P.kwargs,
) -> None:
    if cfg is None:
        cfg = MultiProcessConfig.default_config()
    configure_logging(rank=cfg.rank, world_size=cfg.world_size)
    init_dist(cfg)
    func(*args, **kwargs)
    cleanup_dist()


def _func_wrapped(
    func: Callable[P, None],
    setup: Callable[[], None] | None,
    cfg: MultiProcessConfig,
    error_file: str,
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
        with open(error_file, "wb") as fh:
            pkl.dump(traceback.format_exc(), fh)
        sys.exit(1)


def launch_subprocesses(
    func: Callable[P, None],
    cfg: MultiProcessConfig | None = None,
    setup: Callable[[], None] | None = None,
    rank_offset: int = 0,
    daemon: bool = False,
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
        daemon: The spawned processes' daemon flag. If set to True, daemonic
            processes will be created.
        kwargs: The keyword arguments to pass to the function.
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
        return

    logger.info("Launching %d training workers", cfg.world_size)
    ctx = mp.get_context(cfg.multiprocess_launch_method)
    error_files: list[str | None] = []
    procs = []
    for rank in range(cfg.world_size):
        rank = rank + rank_offset
        cfg.rank = rank
        cfg.local_rank = rank % cfg.local_world_size

        # Using a tempfile to write error logs to.
        tf = tempfile.NamedTemporaryFile(prefix="mlfab-errorfile-", suffix=".pickle", delete=False)
        tf.close()
        os.unlink(tf.name)

        proc = ctx.Process(
            target=_func_wrapped,
            args=[func, setup, cfg, tf.name, *args],
            kwargs=kwargs,
            daemon=daemon,
            name=f"worker-{rank}",
        )
        logger.debug("Started process %d", rank)
        proc.start()
        error_files.append(tf.name)
        procs.append(proc)

    pctx = mp.ProcessContext(procs, error_files)
    while not pctx.join():
        pass


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
