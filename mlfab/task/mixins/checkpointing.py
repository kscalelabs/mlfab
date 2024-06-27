"""Defines a mixin for handling model checkpointing."""

import json
import logging
import pickle
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Generic, Literal, Self, TypeVar, cast, overload

import torch
from omegaconf import DictConfig, OmegaConf
from torch.serialization import MAP_LOCATION

from mlfab.core.conf import field
from mlfab.core.state import State
from mlfab.nn.parallel import is_dp_master, parallel_group_info
from mlfab.task.mixins.artifacts import ArtifactsConfig, ArtifactsMixin
from mlfab.utils.experiments import diff_configs, get_diff_string

logger = logging.getLogger(__name__)


def get_ckpt_path(exp_dir: Path, state: State | None = None) -> Path:
    """Defines the path to the checkpoint for a given state.

    Args:
        exp_dir: The experiment directory
        state: The current trainer state

    Returns:
        The path to the PyTorch checkpoint to save or load
    """
    name = "ckpt"
    ginfo = parallel_group_info(required=False)
    if ginfo is not None:
        world_size = ginfo.mp.world_size * ginfo.pp.world_size
        if world_size > 1:
            rank = ginfo.mp.rank * ginfo.pp.world_size + ginfo.pp.rank
            name += f"_{rank}"
    if state is not None:
        name += f".{state.num_steps}"
    return exp_dir / "checkpoints" / f"{name}.pt"


@dataclass(kw_only=True)
class CheckpointingConfig(ArtifactsConfig):
    save_every_n_steps: int | None = field(None, help="Save a checkpoint every N steps")
    save_every_n_seconds: float | None = field(60.0 * 60.0, help="Save a checkpoint every N seconds")
    only_save_most_recent: bool = field(True, help="Only keep the most recent checkpoint")
    load_from_ckpt_path: str | None = field(None, help="If set, load initial model weights from this path")
    load_ckpt_strict: bool = field(True, help="If set, only load weights for which have a matching key in the model")


Config = TypeVar("Config", bound=CheckpointingConfig)


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> type:
        try:
            return super().find_class(module, name)
        except AttributeError:
            return lambda *args, **kwargs: None  # type: ignore[return-value]


class CustomPickleModule:
    Unpickler = CustomUnpickler


class CheckpointingMixin(ArtifactsMixin[Config], Generic[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.__last_ckpt_time = 0.0

    def get_ckpt_path(self, state: State | None = None) -> Path:
        return get_ckpt_path(self.exp_dir, state)

    @classmethod
    def read_state_dict(
        cls,
        path: str | Path,
        map_location: MAP_LOCATION = None,
        mmap: bool | None = None,
    ) -> dict:
        """Reads a state dict from a checkpoint file.

        Args:
            path: The path to the checkpoint file
            map_location: The device to map the state dict to
            mmap: Whether to map the checkpoint to memory

        Returns:
            The state dict loaded from the checkpoint
        """
        return torch.load(path, map_location=map_location, mmap=mmap, pickle_module=CustomPickleModule)

    @overload
    @classmethod
    def load_raw_checkpoint(
        cls,
        path: str | Path,
        *,
        missing_ok: Literal[True],
        raw: Literal[True],
        use_cli: bool | list[str] = False,
        map_location: MAP_LOCATION = None,
        mmap: bool | None = None,
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
        map_location: MAP_LOCATION = None,
        mmap: bool | None = None,
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
        map_location: MAP_LOCATION = None,
        mmap: bool | None = None,
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
        map_location: MAP_LOCATION = None,
        mmap: bool | None = None,
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
        map_location: MAP_LOCATION = None,
        mmap: bool | None = None,
        config_fn: Callable[[DictConfig], DictConfig] = lambda x: x,
    ) -> tuple[Config | DictConfig | None, dict]:
        """Loads a raw checkpoint from a file.

        Args:
            path: The path to the checkpoint file
            missing_ok: Whether it's okay for the checkpoint to be missing
            raw: If set, return the raw config, otherwise parse against the
                config dataclass
            use_cli: Whether to use CLI overrides
            map_location: The device to map the state dict to
            mmap: Whether to map the checkpoint to memory
            config_fn: A function to apply to the loaded config, to help with
                versioning checkpoints

        Returns:
            The raw config and state dict loaded from the checkpoint
        """
        state_dict = cls.read_state_dict(path, map_location=map_location, mmap=mmap)
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
        map_location: MAP_LOCATION = None,
        mmap: bool | None = None,
        config_fn: Callable[[DictConfig], DictConfig] = lambda x: x,
    ) -> Self:
        """Loads a task from a checkpoint file.

        Args:
            path: The path to the checkpoint file
            strict: Whether to strictly load the checkpoint
            assign: Whether to assign the checkpoint to the task
            use_cli: Whether to use CLI overrides
            map_location: The device to map the state dict to
            mmap: Whether to map the checkpoint to memory
            config_fn: A function to apply to the loaded config

        Returns:
            The task loaded from the checkpoint
        """
        cfg, state_dict = cls.load_raw_checkpoint(
            path,
            use_cli=use_cli,
            map_location=map_location,
            mmap=mmap,
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

    def load_checkpoint_(
        self,
        ckpt_path: str | Path | None = None,
        map_location: MAP_LOCATION = None,
        mmap: bool | None = None,
        strict: bool = True,
        assign: bool = False,
    ) -> State:
        if ckpt_path is None:
            ckpt_path = self.get_init_ckpt_path()
            if ckpt_path is None:
                return State.init_state()
        else:
            ckpt_path = Path(ckpt_path)
        raw_config, state_dict = self.load_raw_checkpoint(
            ckpt_path,
            missing_ok=False,
            raw=True,
            map_location=map_location,
            mmap=mmap,
        )
        raw_state = state_dict.pop("state", None)
        if raw_config is not None:
            config_diff = get_diff_string(diff_configs(cast(DictConfig, self.config), OmegaConf.create(raw_config)))
            if config_diff:
                logger.warning("Loaded config differs from current config:\n%s", config_diff)
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

    def save_checkpoint(self, state: State, ckpt_path: str | Path | None = None) -> Path:
        ckpt_path = self.get_ckpt_path(state) if ckpt_path is None else Path(ckpt_path)
        self.on_before_save_checkpoint(ckpt_path)

        if not is_dp_master():
            return ckpt_path

        # Gets the path to the last checkpoint.
        logger.info("Saving checkpoint to %s", ckpt_path)
        last_ckpt_path = self.get_ckpt_path()
        ckpt_path.parent.mkdir(exist_ok=True, parents=True)

        # Potentially removes the last checkpoint.
        if last_ckpt_path.exists() and self.config.only_save_most_recent:
            if (base_ckpt := last_ckpt_path.resolve()).is_file():
                base_ckpt.unlink()

        # Saves the complete state dict to the checkpoint.
        state_dict = self.task_state_dict()
        state_dict["state"] = json.dumps(asdict(state))
        state_dict["config"] = OmegaConf.to_yaml(self.config)
        torch.save(state_dict, ckpt_path)

        # Updates the symlink to the new checkpoint.
        last_ckpt_path.unlink(missing_ok=True)
        try:
            last_ckpt_path.symlink_to(ckpt_path.relative_to(last_ckpt_path.parent))
        except FileExistsError:
            logger.exception("Exception while trying to update %s", ckpt_path)
        except ValueError:
            logger.warning("Could not create symlink to %s", ckpt_path)

        # Marks directory as having artifacts which shouldn't be overwritten.
        self.add_lock_file("ckpt", exists_ok=True)
        self.on_after_save_checkpoint(ckpt_path)

        return ckpt_path
