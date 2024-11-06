"""Defines a mixin for handling model checkpointing."""

import json
import logging
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Generic, Iterable, Literal, Self, TypeVar, overload

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.checkpoint.state_dict_saver import save as dcp_save
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.optim.optimizer import Optimizer

from mlfab.core.conf import field
from mlfab.core.state import State
from mlfab.nn.parallel import is_master
from mlfab.task.mixins.artifacts import ArtifactsConfig, ArtifactsMixin
from mlfab.task.mixins.trainable import TrainableModule
from mlfab.utils.experiments import diff_configs, get_diff_string
from mlfab.utils.sugar import default

logger = logging.getLogger(__name__)

STATE_FILE_NAME = "state.pt"
CKPT_FILE_NAME = "ckpt.pt"


@dataclass(kw_only=True)
class CheckpointingConfig(ArtifactsConfig):
    save_every_n_steps: int | None = field(None, help="Save a checkpoint every N steps")
    save_every_n_seconds: float | None = field(60.0 * 60.0, help="Save a checkpoint every N seconds")
    load_from_ckpt_path: str | None = field(None, help="If set, load initial model weights from this path")
    ckpt_ignore_frozen_params: bool = field(False, help="Whether to ignore frozen parameters when loading checkpoints")
    ckpt_strict: bool = field(True, help="Use strict mode when loading checkpoints")


Config = TypeVar("Config", bound=CheckpointingConfig)


def _maybe_barrier() -> None:
    if dist.is_initialized():
        dist.barrier()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class CheckpointingMixin(ArtifactsMixin[Config], Generic[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.__last_ckpt_time = 0.0

    def get_ckpt_path(self) -> Path:
        return self.exp_dir / "ckpt"

    @classmethod
    def read_state_dict(cls, path: str | Path) -> dict:
        """Reads a state dict from a checkpoint file.

        Args:
            path: The path to the checkpoint file

        Returns:
            The state dict loaded from the checkpoint. This just contains the
            task information, not the model weights.
        """
        ckpt_path = Path(path)
        state_dict = torch.load(ckpt_path / STATE_FILE_NAME, map_location="cpu", weights_only=True)
        return state_dict

    @overload
    @classmethod
    def load_raw_ckpt(
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
    def load_raw_ckpt(
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
    def load_raw_ckpt(
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
    def load_raw_ckpt(
        cls,
        path: str | Path,
        *,
        missing_ok: Literal[False] = False,
        raw: Literal[False] = False,
        use_cli: bool | list[str] = False,
        config_fn: Callable[[DictConfig], DictConfig] = lambda x: x,
    ) -> tuple[Config, dict]: ...

    @classmethod
    def load_raw_ckpt(
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
        cfg, state_dict = cls.load_raw_ckpt(path, use_cli=use_cli, config_fn=config_fn)
        task = cls(cfg)
        task.load_task_state_dict_(state_dict, strict=strict, assign=assign)
        task.load_ckpt_(task, ckpt_path=path, strict=strict, assign=assign)
        return task

    def get_init_ckpt_path(self) -> Path | None:
        ckpt_path = self.get_ckpt_path()
        if ckpt_path.exists():
            if any(ckpt_path.iterdir()):
                return ckpt_path
        elif is_master():
            ckpt_path.mkdir(parents=True)
        if self.config.load_from_ckpt_path is not None:
            ckpt_path = Path(self.config.load_from_ckpt_path)
            assert ckpt_path.exists(), f"Checkpoint path {ckpt_path} does not exist."
            return ckpt_path
        return None

    def _ckpt_options(self) -> StateDictOptions:
        return StateDictOptions(
            full_state_dict=False,
            cpu_offload=True,
            ignore_frozen_params=self.config.ckpt_ignore_frozen_params,
            keep_submodule_prefixes=True,
            strict=self.config.ckpt_strict,
        )

    def load_ckpt_(
        self,
        model: nn.Module,
        *,
        optimizer: Optimizer | None = None,
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

        raw_config, state_dict = self.load_raw_ckpt(ckpt_path, missing_ok=False, raw=True)
        raw_state = state_dict.pop("state", None)
        if raw_config is not None:
            base_config = OmegaConf.create(self.config)  # type: ignore[call-overload]
            diff = get_diff_string(diff_configs(base_config, OmegaConf.create(raw_config)))
            if diff:
                logger.warning("Loaded config differs from current config:\n%s", diff)
        self.load_task_state_dict_(state_dict, strict, assign)

        if isinstance(model, FSDP):
            _maybe_barrier()
            options = self._ckpt_options()
            fsdp_optimizer: Optimizer | Iterable[Optimizer] = [] if optimizer is None else optimizer
            model_state_dict, optimizer_state_dict = get_state_dict(model, fsdp_optimizer, options=options)
            dcp_state_dict = {
                "model": model_state_dict,
                "optimizer": optimizer_state_dict,
            }
            dcp_load(dcp_state_dict, checkpoint_id=ckpt_path)
            _maybe_barrier()
            set_state_dict(
                model,
                fsdp_optimizer,
                model_state_dict=model_state_dict,
                optim_state_dict=optimizer_state_dict,
                options=options,
            )
        else:
            ckpt_dict = torch.load(ckpt_path / CKPT_FILE_NAME, map_location="cpu", weights_only=True)
            model_ckpt_dict = ckpt_dict["model"]
            if not isinstance(model, TrainableModule):
                consume_prefix_in_state_dict_if_present(model_ckpt_dict, "base_mod.")
            model.load_state_dict(model_ckpt_dict)
            if optimizer is not None:
                optimizer_ckpt_dict = ckpt_dict["optimizer"]
                optimizer.load_state_dict(optimizer_ckpt_dict)
        _maybe_barrier()

        if raw_state is not None:
            state = State(**json.loads(raw_state))
            state.start_time_s = time.time()
            state.elapsed_time_s = 0.0
            return state

        warnings.warn("No state found in checkpoint! Using default initial state.")
        return State.init_state()

    def should_save_ckpt(self, state: State) -> bool:
        if self.config.save_every_n_steps is not None:
            if state.num_steps % self.config.save_every_n_steps == 0:
                return True
        if self.config.save_every_n_seconds is not None:
            last_time, cur_time = self.__last_ckpt_time, state.elapsed_time_s
            if cur_time - last_time >= self.config.save_every_n_seconds:
                self.__last_ckpt_time = cur_time
                return True
        return False

    def save_ckpt(
        self,
        state: State,
        model: nn.Module,
        *,
        optimizer: Optimizer | None = None,
        ckpt_path: str | Path | None = None,
    ) -> Path:
        ckpt_path = default(ckpt_path, self.get_ckpt_path, lambda p: Path(p))

        self.on_before_save_ckpt(ckpt_path)

        # Gets the path to the last checkpoint.
        logger.info("Saving checkpoint to %s", ckpt_path)

        if isinstance(model, FSDP):
            _maybe_barrier()
            options = self._ckpt_options()
            fsdp_optimizer: Optimizer | Iterable[Optimizer] = [] if optimizer is None else optimizer
            dcp_model_state_dict, dcp_optimizer_state_dict = get_state_dict(model, fsdp_optimizer, options=options)
            dcp_state_dict = {
                "model": dcp_model_state_dict,
                "optimizer": dcp_optimizer_state_dict,
            }
            dcp_save(dcp_state_dict, checkpoint_id=ckpt_path)
        elif is_master():
            nn_model_state_dict = model.state_dict()
            nn_optimizer_state_dict = optimizer.state_dict() if optimizer is not None else None
            dcp_state_dict = {
                "model": nn_model_state_dict,
                "optimizer": nn_optimizer_state_dict,
            }
            torch.save(dcp_state_dict, ckpt_path / CKPT_FILE_NAME)
        _maybe_barrier()

        if is_master():
            state_dict: dict = {}
            state_dict["task"] = self.task_state_dict()
            state_dict["state"] = json.dumps(asdict(state))
            state_dict["config"] = OmegaConf.to_yaml(self.config)
            torch.save(state_dict, ckpt_path / STATE_FILE_NAME)

            # Marks directory with artifacts which shouldn't be overwritten.
            self.add_lock_file("ckpt", exists_ok=True)
        _maybe_barrier()

        self.on_after_save_ckpt(ckpt_path)
        return ckpt_path
