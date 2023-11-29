"""Defines a mixin for handling model checkpointing."""

import json
import logging
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Generic, TypeVar, cast

import torch
from omegaconf import DictConfig, OmegaConf

from mlfab.core.conf import field
from mlfab.core.state import State
from mlfab.nn.parallel import is_master
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
    if state is None:
        return exp_dir / "checkpoints" / "ckpt.pt"
    return exp_dir / "checkpoints" / f"ckpt.{state.num_steps}.pt"


@dataclass
class CheckpointingConfig(ArtifactsConfig):
    save_every_n_steps: int | None = field(None, help="Save a checkpoint every N steps")
    save_every_n_seconds: float | None = field(60.0 * 60.0, help="Save a checkpoint every N seconds")
    only_save_most_recent: bool = field(True, help="Only keep the most recent checkpoint")
    load_from_ckpt_path: str | None = field(None, help="If set, load initial model weights from this path")
    load_ckpt_strict: bool = field(True, help="If set, only load weights for which have a matching key in the model")


Config = TypeVar("Config", bound=CheckpointingConfig)


class CheckpointingMixin(ArtifactsMixin[Config], Generic[Config]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.__last_ckpt_time = 0.0

    def get_ckpt_path_optional(self, state: State | None = None) -> Path | None:
        if self._exp_dir is None:
            return None
        return get_ckpt_path(self.exp_dir, state)

    def get_ckpt_path(self, state: State | None = None) -> Path:
        return get_ckpt_path(self.exp_dir, state)

    def read_state_dict(self, path: Path) -> dict:
        return torch.load(path)

    def get_init_ckpt_path(self) -> Path | None:
        if (ckpt_path := self.get_ckpt_path_optional()) is not None and ckpt_path.exists():
            return ckpt_path
        if self.config.load_from_ckpt_path is not None:
            ckpt_path = Path(self.config.load_from_ckpt_path)
            assert ckpt_path.exists(), f"Checkpoint path {ckpt_path} does not exist."
            return ckpt_path
        return None

    def load_initial_state(self) -> State:
        init_ckpt_path = self.get_init_ckpt_path()
        if init_ckpt_path is not None:
            state_dict = self.read_state_dict(init_ckpt_path)
            raw_state = state_dict.pop("state", None)
            raw_config = state_dict.pop("config", None)
            if raw_config is not None:
                config_diff = get_diff_string(diff_configs(OmegaConf.create(raw_config), cast(DictConfig, self.config)))
                if config_diff:
                    logger.warning("Loaded config differs from current config:\n%s", config_diff)
            self.load_task_state_dict(state_dict)
            if raw_state is None:
                warnings.warn("No state found in checkpoint! Using default initial state.")
            else:
                return State(**json.loads(raw_state))
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

    def save_checkpoint(self, state: State) -> Path:
        ckpt_path = self.get_ckpt_path(state)
        self.on_before_save_checkpoint(ckpt_path)

        if not is_master():
            return ckpt_path

        # Gets the path to the last checkpoint.
        logger.info("Saving checkpoint to %s", ckpt_path)
        last_ckpt_path = self.get_ckpt_path()
        ckpt_path.parent.mkdir(exist_ok=True, parents=True)

        # Gets the complete state dict.
        state_dict = self.task_state_dict()
        state_dict["state"] = json.dumps(asdict(state))
        state_dict["config"] = OmegaConf.to_yaml(self.config)
        torch.save(state_dict, ckpt_path)

        if last_ckpt_path.exists():
            if self.config.only_save_most_recent:
                base_ckpt = last_ckpt_path.resolve()
                if base_ckpt.is_file():
                    base_ckpt.unlink()
            last_ckpt_path.unlink()

        try:
            last_ckpt_path.symlink_to(ckpt_path)
        except FileExistsError:
            logger.exception("Exception while trying to update %s", ckpt_path)
        self.add_lock_file("ckpt", exists_ok=True)
        self.on_after_save_checkpoint(ckpt_path)

        return ckpt_path
