"""Defines a mixin for handling model checkpointing."""

import contextlib
import json
import logging
import pickle
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, ContextManager, Generic, Literal, Self, TypeVar, cast, overload

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.distributed._tensor.api import DTensor
from torch.distributed.checkpoint import load as load_ckpt, save as save_ckpt
from torch.distributed.checkpoint.filesystem import FileSystemReader, FileSystemWriter
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    StateDictType,
)
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.optim.optimizer import Optimizer

from mlfab.core.conf import field
from mlfab.core.state import State
from mlfab.nn.functions import recursive_apply_all
from mlfab.nn.parallel import device_mesh, dp_rank, get_rank, mp_group
from mlfab.task.mixins.artifacts import ArtifactsConfig, ArtifactsMixin
from mlfab.utils.experiments import diff_configs, get_diff_string

logger = logging.getLogger(__name__)


def dtensor_to_tensor(t: Any) -> Any:  # noqa: ANN401
    if isinstance(t, DTensor):
        return t.to_local()
    return t


def dtensors_to_tensors(t: Any) -> Any:  # noqa: ANN401
    return recursive_apply_all(t, dtensor_to_tensor)


def tensor_to_dtensor(t: Any) -> Any:  # noqa: ANN401
    if isinstance(t, Tensor):
        return DTensor.from_local(t, device_mesh=device_mesh(t.device.type)["tp"])
    return t


def tensors_to_dtensors(t: Any) -> Any:  # noqa: ANN401
    return recursive_apply_all(t, tensor_to_dtensor)


@dataclass(kw_only=True)
class CheckpointingConfig(ArtifactsConfig):
    save_every_n_steps: int | None = field(None, help="Save a checkpoint every N steps")
    save_every_n_seconds: float | None = field(60.0 * 60.0, help="Save a checkpoint every N seconds")
    load_from_ckpt_path: str | None = field(None, help="If set, load initial model weights from this path")


Config = TypeVar("Config", bound=CheckpointingConfig)


class CustomPickler(pickle.Pickler):
    def persistent_id(self, obj: Any) -> Any:  # noqa: ANN401
        return None


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> type:
        try:
            return super().find_class(module, name)
        except AttributeError:
            return lambda *args, **kwargs: None  # type: ignore[return-value]


class CustomPickleModule:
    Pickler = CustomPickler
    Unpickler = CustomUnpickler


class CheckpointingMixin(ArtifactsMixin[Config], Generic[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.__last_ckpt_time = 0.0

    def get_ckpt_path(self) -> Path:
        return self.exp_dir / "checkpoints"

    @classmethod
    def read_state_dict(cls, path: str | Path) -> dict:
        """Reads a state dict from a checkpoint file.

        Args:
            path: The path to the checkpoint file
            map_location: The device to map the state dict to
            mmap: Whether to map the checkpoint to memory

        Returns:
            The state dict loaded from the checkpoint
        """
        ckpt_path = Path(path)
        weight_dict: dict = {}
        load_ckpt(
            state_dict=weight_dict,
            storage_reader=FileSystemReader(ckpt_path),
            # process_group=mp_group(),
        )
        state_dict = torch.load(ckpt_path / "state_dict.pth", map_location="cpu", pickle_module=CustomPickleModule)
        return {**weight_dict, **state_dict}

    @overload
    @classmethod
    def load_raw_checkpoint(
        cls,
        path: str | Path,
        *,
        missing_ok: Literal[True],
        raw: Literal[True],
        use_cli: bool | list[str] = False,
        config_fn: Callable[[DictConfig], DictConfig] = lambda x: x,
    ) -> tuple[DictConfig, dict]: ...

    @overload
    @classmethod
    def load_raw_checkpoint(
        cls,
        path: str | Path,
        *,
        missing_ok: Literal[False] = False,
        raw: Literal[True],
        use_cli: bool | list[str] = False,
        config_fn: Callable[[DictConfig], DictConfig] = lambda x: x,
    ) -> tuple[DictConfig, dict]: ...

    @overload
    @classmethod
    def load_raw_checkpoint(
        cls,
        path: str | Path,
        *,
        missing_ok: Literal[True],
        raw: Literal[False] = False,
        use_cli: bool | list[str] = False,
        config_fn: Callable[[DictConfig], DictConfig] = lambda x: x,
    ) -> tuple[Config | None, dict]: ...

    @overload
    @classmethod
    def load_raw_checkpoint(
        cls,
        path: str | Path,
        *,
        missing_ok: Literal[False] = False,
        raw: Literal[False] = False,
        use_cli: bool | list[str] = False,
        config_fn: Callable[[DictConfig], DictConfig] = lambda x: x,
    ) -> tuple[Config, dict]: ...

    @classmethod
    def load_raw_checkpoint(
        cls,
        path: str | Path,
        *,
        missing_ok: bool = False,
        raw: bool = False,
        use_cli: bool | list[str] = False,
        config_fn: Callable[[DictConfig], DictConfig] = lambda x: x,
    ) -> tuple[Config | DictConfig | None, dict]:
        """Loads a raw checkpoint from a file.

        Args:
            path: The path to the checkpoint file
            missing_ok: Whether it's okay for the checkpoint to be missing
            raw: If set, return the raw config, otherwise parse against the
                config dataclass
            use_cli: Whether to use CLI overrides
            config_fn: A function to apply to the loaded config, to help with
                versioning checkpoints

        Returns:
            The raw config and state dict loaded from the checkpoint
        """
        state_dict = cls.read_state_dict(path)
        raw_config = state_dict.pop("config", None)
        if raw_config is None:
            if missing_ok:
                return None, state_dict
            raise RuntimeError(f"Could not find config in checkpoint at {path}!")
        raw_config = config_fn(raw_config)
        if raw:
            return raw_config, state_dict
        cfg = cls.get_config(OmegaConf.create(raw_config), use_cli=use_cli)
        return cfg, state_dict

    @classmethod
    def get_task_from_ckpt(
        cls,
        path: str | Path,
        *,
        strict: bool = True,
        assign: bool = False,
        use_cli: bool | list[str] = False,
        config_fn: Callable[[DictConfig], DictConfig] = lambda x: x,
    ) -> Self:
        """Loads a task from a checkpoint file.

        Args:
            path: The path to the checkpoint file
            strict: Whether to strictly load the checkpoint
            assign: Whether to assign the checkpoint to the task
            use_cli: Whether to use CLI overrides
            config_fn: A function to apply to the loaded config

        Returns:
            The task loaded from the checkpoint
        """
        cfg, state_dict = cls.load_raw_checkpoint(
            path,
            use_cli=use_cli,
            config_fn=config_fn,
        )
        task = cls(cfg)
        task.load_task_state_dict_(
            state_dict,
            strict=strict,
            assign=assign,
        )
        return task

    def get_init_ckpt_path(self) -> Path | None:
        ckpt_path = self.get_ckpt_path()
        if ckpt_path.exists():
            return ckpt_path
        if self.config.load_from_ckpt_path is not None:
            ckpt_path = Path(self.config.load_from_ckpt_path)
            assert ckpt_path.exists(), f"Checkpoint path {ckpt_path} does not exist."
            return ckpt_path
        return None

    @classmethod
    def state_dict_context(cls, mod: nn.Module, opt: Optimizer) -> ContextManager:
        if isinstance(mod, FSDP):
            return FSDP.state_dict_type(
                module=mod,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
                optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),
            )
        return contextlib.nullcontext()

    def load_checkpoint_(
        self,
        module: nn.Module,
        optimizer: Optimizer,
        ckpt_path: str | Path | None = None,
        strict: bool = True,
        assign: bool = False,
    ) -> State:
        if ckpt_path is None:
            ckpt_path = self.get_init_ckpt_path()
            if ckpt_path is None:
                return State.init_state()
        else:
            ckpt_path = Path(ckpt_path)
        raw_config, state_dict = self.load_raw_checkpoint(ckpt_path, missing_ok=False, raw=True)
        raw_state = state_dict.pop("state", None)
        if raw_config is not None:
            config_diff = get_diff_string(diff_configs(cast(DictConfig, self.config), OmegaConf.create(raw_config)))
            if config_diff:
                logger.warning("Loaded config differs from current config:\n%s", config_diff)

        with self.state_dict_context(module, optimizer):
            if (module_state_dict := state_dict.pop("model", None)) is not None:
                consume_prefix_in_state_dict_if_present(module_state_dict, "module.")
                module_state_dict = tensors_to_dtensors(module_state_dict)
                module.load_state_dict(module_state_dict)
            if (optimizer_state_dict := state_dict.pop("optimizer", None)) is not None:
                optimizer_state_dict = tensors_to_dtensors(optimizer_state_dict)
                if isinstance(module, FSDP):
                    optimizer_state_dict = FSDP.optim_state_dict_to_load(module, optimizer, optimizer_state_dict)
                optimizer.load_state_dict(optimizer_state_dict)

        self.load_task_state_dict_(state_dict, strict, assign)
        if raw_state is not None:
            return State(**json.loads(raw_state))

        warnings.warn("No state found in checkpoint! Using default initial state.")
        return State.init_state()

    def should_checkpoint(self, state: State) -> bool:
        if self.config.save_every_n_steps is not None:
            if state.num_steps % self.config.save_every_n_steps == 0:
                return True
        if self.config.save_every_n_seconds is not None:
            last_time, cur_time = self.__last_ckpt_time, state.elapsed_time_s
            if cur_time - last_time >= self.config.save_every_n_seconds:
                self.__last_ckpt_time = cur_time
                return True
        return False

    def save_checkpoint(
        self,
        state: State,
        module: nn.Module,
        optimizer: Optimizer,
        ckpt_path: str | Path | None = None,
    ) -> Path:
        ckpt_path = self.get_ckpt_path() if ckpt_path is None else Path(ckpt_path)
        self.on_before_save_checkpoint(ckpt_path)

        # Gets the path to the last checkpoint.
        logger.info("Saving checkpoint to %s", ckpt_path)
        ckpt_path.mkdir(exist_ok=True, parents=True)

        # Saves the complete state dict to the checkpoint.
        weight_dict: dict = {}
        with self.state_dict_context(module, optimizer):
            module_state_dict = module.state_dict()
            # module_state_dict = dtensors_to_tensors(module.state_dict())
            weight_dict["model"] = module_state_dict

            if isinstance(module, FSDP):
                optimizer_state_dict = FSDP.optim_state_dict(module, optimizer)
            else:
                optimizer_state_dict = optimizer.state_dict()
            # optimizer_state_dict = dtensors_to_tensors(optimizer_state_dict)
            weight_dict["optimizer"] = optimizer_state_dict

        if dp_rank() == 0:
            save_ckpt(
                state_dict=weight_dict,
                storage_writer=FileSystemWriter(ckpt_path),
                process_group=mp_group(),
            )

        if get_rank() == 0:
            state_dict: dict = {}
            state_dict["task"] = self.task_state_dict()
            state_dict["state"] = json.dumps(asdict(state))
            state_dict["config"] = OmegaConf.to_yaml(self.config)
            torch.save(state_dict, ckpt_path / "state_dict.pth", pickle_module=CustomPickleModule)

            # Marks directory with artifacts which shouldn't be overwritten.
            self.add_lock_file("ckpt", exists_ok=True)

        self.on_after_save_checkpoint(ckpt_path)

        return ckpt_path
