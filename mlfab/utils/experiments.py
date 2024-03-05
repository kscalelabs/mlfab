"""Functions for managing experiments."""

import datetime
import enum
import functools
import hashlib
import inspect
import itertools
import logging
import os
import random
import re
import shutil
import sys
import tempfile
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, TypeVar, cast

import git
import torch
from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf
from torch import Tensor, inf, nn
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype, _has_foreach_support
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset
from torchvision.datasets.utils import download_url

from mlfab.core.conf import get_pretrained_models_dir
from mlfab.core.state import State
from mlfab.utils.text import colored

logger = logging.getLogger(__name__)

GradDict = dict[tuple[torch.device, torch.dtype], tuple[list[list[Tensor]], list[int]]]

# Date format for staging environments.
DATE_FORMAT = "%Y-%m-%d"

T = TypeVar("T")


class CumulativeTimer:
    """Defines a simple timer to track an average value."""

    def __init__(self) -> None:
        self.steps = 0
        self.elapsed_time = 0.0

    @functools.cached_property
    def start_time(self) -> float:
        return time.time()

    def step(self, steps: int, cur_time: float) -> None:
        if steps != self.steps:
            self.steps = steps
            self.elapsed_time = cur_time - self.start_time

    @property
    def steps_per_second(self) -> float:
        return 0.0 if self.elapsed_time < 1e-4 else self.steps / self.elapsed_time

    @property
    def steps_per_hour(self) -> float:
        return self.steps_per_second * 60 * 60

    @property
    def seconds_per_step(self) -> float:
        return 0.0 if self.steps <= 0 else self.elapsed_time / self.steps

    @property
    def hours_per_step(self) -> float:
        return self.seconds_per_step / (60 * 60)


class IterationTimer:
    """Defines a simple timer to track consecutive values."""

    def __init__(self) -> None:
        self.iteration_time = 0.0
        self.last_time = time.time()

    def step(self, cur_time: float) -> None:
        self.iteration_time = cur_time - self.last_time
        self.last_time = cur_time

    @property
    def iter_seconds(self) -> float:
        return self.iteration_time

    @property
    def iter_hours(self) -> float:
        return self.iter_seconds / (60 * 60)


class StateTimer:
    """Defines a timer for all state information."""

    def __init__(self) -> None:
        self.step_timer = CumulativeTimer()
        self.sample_timer = CumulativeTimer()
        self.iter_timer = IterationTimer()

    def step(self, state: State) -> None:
        cur_time = time.time()
        self.step_timer.step(state.num_steps, cur_time)
        self.sample_timer.step(state.num_samples, cur_time)
        self.iter_timer.step(cur_time)

    def log_dict(self) -> dict[str, dict[str, int | float]]:
        logs: dict[str, dict[str, int | float]] = {}

        # Logs step statistics.
        logs["⏰ steps"] = {
            "total": self.step_timer.steps,
            "per-second": self.step_timer.steps_per_second,
        }

        # Logs sample statistics.
        logs["⏰ samples"] = {
            "total": self.sample_timer.steps,
            "per-second": self.sample_timer.steps_per_second,
        }

        # Logs full iteration statistics.
        logs["🔧 dt"] = {
            "iter": self.iter_timer.iter_seconds,
        }

        return logs


class IntervalTicker:
    def __init__(self, interval: float) -> None:
        self.interval = interval
        self.last_tick_time: float | None = None

    def tick(self, elapsed_time: float) -> bool:
        if self.last_tick_time is None or elapsed_time - self.last_tick_time > self.interval:
            self.last_tick_time = elapsed_time
            return True
        return False


def abs_path(path: str) -> str:
    return str(Path(path).resolve())


OmegaConf.register_new_resolver("ml.abs_path", abs_path, replace=True)


def cpu_count(default: int) -> int:
    if (cpu_count := os.cpu_count()) is not None:
        return cpu_count
    return default


OmegaConf.register_new_resolver("ml.cpu_count", cpu_count, replace=True)


def date_str(_: str) -> str:
    return time.strftime("%Y-%m-%d")


OmegaConf.register_new_resolver("ml.date_str", date_str, replace=True)


def get_random_port(default: int = 1337) -> int:
    try:
        return (hash(time.time()) + random.randint(0, 100000)) % (65_535 - 10_000) + 10_000
    except Exception:
        return default


OmegaConf.register_new_resolver("mlfab.get_random_port", get_random_port, replace=True)


@torch.no_grad()
def get_weight_norm(
    parameters: Iterable[nn.Parameter],
    norm_type: float = 2.0,
    foreach: bool | None = None,
) -> Tensor:
    """Computes the norm of an iterable of parameters.

    The norm is computed over all parameters together, as if they were
    concatenated into a single vector.

    Args:
        parameters: An iterable of the model parameters.
        norm_type: The type of the used p-norm.
        foreach: Use the faster foreach-based implementation.

    Returns:
        The total norm of the parameters (viewed as a single vector).
    """
    parameters = list(parameters)
    if len(parameters) == 0:
        return torch.tensor([0.0])

    first_device = parameters[0].device
    grouped_params = cast(GradDict, _group_tensors_by_device_and_dtype([[p.detach() for p in parameters]]))

    if norm_type == inf:
        norms = [p.detach().abs().max().to(first_device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        norms = []
        for (device, _), ([param], _) in grouped_params.items():
            if (foreach is None or foreach) and _has_foreach_support(param, device=device):
                norms.extend(torch._foreach_norm(param, norm_type))
            else:
                norms.extend([torch.norm(g, norm_type) for g in param])
        total_norm = torch.norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)

    return total_norm


@torch.no_grad()
def get_grad_norm(
    parameters: Iterable[nn.Parameter],
    norm_type: float = 2.0,
    foreach: bool | None = None,
) -> tuple[Tensor, GradDict]:
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return torch.tensor([0.0]), {}

    first_device = grads[0].device
    grouped_grads = cast(GradDict, _group_tensors_by_device_and_dtype([[g.detach() for g in grads]]))

    if norm_type == inf:
        norms = [g.detach().abs().max().to(first_device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        norms = []
        for (device, _), ([grads], _) in grouped_grads.items():
            if (foreach is None or foreach) and _has_foreach_support(grads, device=device):
                norms.extend(torch._foreach_norm(grads, norm_type))
            else:
                norms.extend([torch.norm(g, norm_type) for g in grads])
        total_norm = torch.norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)

    return total_norm, grouped_grads


@torch.no_grad()
def clip_grad_norm_(
    parameters: Iterable[nn.Parameter],
    max_norm: float,
    norm_type: float = 2.0,
    foreach: bool | None = None,
) -> tuple[Tensor, bool]:
    """Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters: An iterable of the model parameters.
        max_norm: The maximum norm of the gradients.
        norm_type: The type of the used p-norm.
        foreach: Use the faster foreach-based implementation. If ``None``, use
            the foreach implementation for CUDA and CPU native tensors and
            silently fall back to the slow implementation for other device
            types. If ``True`` or ``False``, use the foreach or non-foreach
            implementation, respectively, and raise an error if the chosen
            implementation is not available.

    Returns:
        The total norm of the parameters (viewed as a single vector) and
        whether the parameters were successfully clipped.
    """
    total_norm, grouped_grads = get_grad_norm(parameters, norm_type, foreach)

    if not torch.isfinite(total_norm):
        return total_norm, False

    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for (device, _), ([grads], _) in grouped_grads.items():
        if (foreach is None or foreach) and _has_foreach_support(grads, device=device):
            torch._foreach_mul_(grads, clip_coef_clamped.to(device))
        else:
            clip_coef_clamped_device = clip_coef_clamped.to(device)
            for g in grads:
                g.detach().mul_(clip_coef_clamped_device)

    return total_norm, True


class NaNError(Exception):
    """Raised when NaNs are detected in the model parameters."""


class TrainingFinishedError(Exception):
    """Raised when training is finished."""


class MinGradScaleError(TrainingFinishedError):
    """Raised when the minimum gradient scale is reached.

    This is a subclass of :class:`TrainingFinishedError` because it indicates
    that training is finished and causes the post-training hooks to be run.
    """


def diff_configs(
    first: ListConfig | DictConfig,
    second: ListConfig | DictConfig,
    prefix: str | None = None,
) -> tuple[list[str], list[str]]:
    """Returns the difference between two configs.

    Args:
        first: The first (original) config
        second: The second (new) config
        prefix: The prefix to check (used for recursion, not main call)

    Returns:
        Two lists of lines describing the diff between the two configs
    """

    def get_diff_string(prefix: str | None, val: Any) -> str:  # noqa: ANN401
        if isinstance(val, (str, float, int)):
            return f"{prefix}={val}"
        return f"{prefix}= ... ({type(val)})"

    def cast_enums(k: Any) -> Any:  # noqa: ANN401
        return k.name if isinstance(k, enum.Enum) else k

    new_first: list[str] = []
    new_second: list[str] = []

    any_config = (ListConfig, DictConfig)

    if isinstance(first, DictConfig) and isinstance(second, DictConfig):
        first_keys, second_keys = cast(set[str], set(first.keys())), cast(set[str], set(second.keys()))

        # Gets the new keys in each config.
        new_first += [f"{prefix}.{key}" for key in first_keys.difference(second_keys)]
        new_second += [f"{prefix}.{key}" for key in second_keys.difference(first_keys)]

        # Gets the new sub-keys in each config.
        for key in first_keys.intersection(second_keys):
            sub_prefix = key if prefix is None else f"{prefix}.{key}"
            if OmegaConf.is_missing(first, key) or OmegaConf.is_missing(second, key):
                if not OmegaConf.is_missing(first, key):
                    new_first += [get_diff_string(sub_prefix, first[key])]
                if not OmegaConf.is_missing(second, key):
                    new_second += [get_diff_string(sub_prefix, second[key])]
            elif isinstance(first[key], any_config) and isinstance(second[key], any_config):
                sub_new_first, sub_new_second = diff_configs(first[key], second[key], prefix=sub_prefix)
                new_first, new_second = new_first + sub_new_first, new_second + sub_new_second
            elif cast_enums(first[key]) != cast_enums(second[key]):
                first_val, second_val = first[key], second[key]
                new_first += [get_diff_string(sub_prefix, first_val)]
                new_second += [get_diff_string(sub_prefix, second_val)]

    elif isinstance(first, ListConfig) and isinstance(second, ListConfig):
        if len(first) > len(second):
            for i in range(len(second), len(first)):
                new_first += [get_diff_string(prefix, first[i])]
        elif len(second) > len(first):
            for i in range(len(first), len(second)):
                new_second += [get_diff_string(prefix, second[i])]

        for i in range(min(len(first), len(second))):
            sub_prefix = str(i) if prefix is None else f"{prefix}.{i}"
            if isinstance(first[i], any_config) and isinstance(second[i], any_config):
                sub_new_first, sub_new_second = diff_configs(first[i], second[i], prefix=sub_prefix)
                new_first, new_second = new_first + sub_new_first, new_second + sub_new_second
    else:
        new_first += [get_diff_string(prefix, first)]
        new_second += [get_diff_string(prefix, second)]

    return new_first, new_second


def get_diff_string(config_diff: tuple[list[str], list[str]]) -> str | None:
    added_keys, deleted_keys = config_diff
    if not added_keys and not deleted_keys:
        return None
    change_lines: list[str] = []
    change_lines += [f" ↪ {colored('+', 'green')} {added_key}" for added_key in added_keys]
    change_lines += [f" ↪ {colored('-', 'red')} {deleted_key}" for deleted_key in deleted_keys]
    change_summary = "\n".join(change_lines)
    return change_summary


def save_config(config_path: Path, raw_config: DictConfig) -> None:
    if config_path.exists():
        config_diff = diff_configs(raw_config, cast(DictConfig, OmegaConf.load(config_path)))
        diff_string = get_diff_string(config_diff)
        if diff_string is not None:
            logger.warning("Overwriting config %s:\n%s", config_path, diff_string)
            OmegaConf.save(raw_config, config_path)
    else:
        config_path.parent.mkdir(exist_ok=True, parents=True)
        OmegaConf.save(raw_config, config_path)
        logger.info("Saved config to %s", config_path)


def to_markdown_table(config: DictConfig) -> str:
    """Converts a config to a markdown table string.

    Args:
        config: The config to convert to a table.

    Returns:
        The config, formatted as a Markdown string.
    """

    def format_as_string(value: Any) -> str:  # noqa: ANN401
        if isinstance(value, str):
            return value
        if isinstance(value, Tensor):
            value = value.detach().float().cpu().item()
        if isinstance(value, (int, float)):
            return f"{value:.4g}"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, datetime.datetime):
            return value.isoformat()
        if isinstance(value, datetime.timedelta):
            return f"{value.total_seconds():.4g}s"
        if value is None:
            return ""
        if value is MISSING:
            return ""
        return str(value)

    def iter_flat(config: dict) -> Iterator[tuple[list[str | None], str]]:
        for key, value in reversed(config.items()):
            if isinstance(value, dict):
                is_first = True
                for sub_key_list, sub_value in iter_flat(value):
                    yield [format_as_string(key) if is_first else None] + sub_key_list, sub_value
                    is_first = False
            elif isinstance(value, (list, tuple)):
                is_first = True
                for i, sub_value in enumerate(value):
                    for sub_key_list, sub_sub_value in iter_flat({f"{i}": sub_value}):
                        yield [format_as_string(key) if is_first else None] + sub_key_list, sub_sub_value
                        is_first = False
            else:
                yield [format_as_string(key)], format_as_string(value)

    config_dict = cast(dict, OmegaConf.to_container(config, resolve=True, throw_on_missing=False, enum_to_str=True))
    config_flat = list(iter_flat(config_dict))

    # Gets rows of strings.
    rows: list[list[str]] = []
    for key_list, value in config_flat:
        row = ["" if key is None else key for key in key_list] + [value]
        rows.append(row)

    # Pads all rows to the same length.
    max_len = max(len(row) for row in rows)
    rows = [row[:-1] + [""] * (max_len - len(row)) + row[-1:] for row in rows]

    # Converts to a markdown table.
    header_str = "| " + " | ".join([f"key_{i}" for i in range(max_len - 1)]) + " | value |"
    header_sep_str = "|-" + "-|-" * (max_len - 1) + "-|"
    rows_str = "\n".join(["| " + " | ".join(row) + " |" for row in rows])
    return "\n".join([header_str, header_sep_str, rows_str])


def stage_environment(obj: object, root: Path) -> None:
    """Stages the current task to a staging directory.

    Args:
        obj: The object with the module to stage.
        root: The root directory to stage to.
    """
    root.mkdir(exist_ok=True, parents=True)

    # Gets the path to the root module. This is done heuristically, so it may
    # not work in all cases, but it should generally work.
    if (mod := inspect.getmodule(obj.__class__)) is None:
        raise RuntimeError(f"Could not find module for task {obj.__class__}!")
    if (spec := mod.__spec__) is None:
        raise RuntimeError(f"Could not find spec for module {mod}!")
    if spec.origin is None:
        raise RuntimeError(f"Could not find origin for spec {spec}!")
    root_mod = spec.name.split(".", 1)[0]
    path_parts = Path(spec.origin).parts[:-1]
    if root_mod not in path_parts:
        raise RuntimeError(f"Could not find root module {root_mod} in path {path_parts}!")
    root_path = Path(*path_parts[: path_parts.index(root_mod) + 1])

    # Gets files to stage.
    fpaths: set[tuple[Path, Path]] = set()
    for module in sys.modules.values():
        if (fpath_str := getattr(module, "__file__", None)) is None:
            continue
        fpath = Path(fpath_str).resolve()
        try:
            rel_fpath = fpath.relative_to(root_path)
            fpaths.add((fpath, rel_fpath))
        except ValueError:
            pass

    # Computes hash of all files and return if it matches the previous hash.
    hashobj = hashlib.md5()
    for fpath, _ in fpaths:
        with open(fpath, "rb") as f:
            while data := f.read(65536):
                hashobj.update(data)
    hashval = hashobj.hexdigest()
    prev_hashval: str | None = None
    hash_file = root / ".hash"
    if hash_file.exists():
        prev_hashval = hash_file.read_text().strip()
    if prev_hashval == hashval:
        return
    hash_file.write_text(hashval)

    # Copies all files to the staging directory.
    if (root / root_mod).exists():
        shutil.rmtree(root / root_mod, ignore_errors=True)
    for fpath, rel_fpath in fpaths:
        new_fpath = root / root_mod / rel_fpath
        new_fpath.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(fpath, new_fpath)


def get_git_state(obj: object) -> str:
    """Gets the state of the Git repo that an object is in as a string.

    Args:
        obj: The object which is in the target Git repo.
        width: The width of the text blocks.

    Returns:
        A nicely-formatted string showing the current task's Git state.
    """
    try:
        task_file = inspect.getfile(type(obj))
        repo = git.Repo(task_file, search_parent_directories=True)
        branch = repo.active_branch
        commit = repo.head.commit
        status = textwrap.indent(str(repo.git.status()), "    ")
        diff = textwrap.indent(str(repo.git.diff(color=False)), "    ")
        return "\n".join(
            [
                f"Path: {task_file}",
                f"Branch: {branch}",
                f"Commit: {commit}",
                "Status:",
                status,
                "Diff:",
                diff,
            ]
        )

    except Exception:
        return traceback.format_exc()


def get_training_code(obj: object) -> str:
    """Gets the text from the file containing the provided object.

    Args:
        obj: The object to get the file from.

    Returns:
        The text from the file containing the object.
    """
    try:
        task_file = inspect.getfile(type(obj))
        with open(task_file, "r") as f:
            return f.read()
    except Exception:
        return traceback.format_exc()


def test_dataset(
    ds: Dataset[T] | IterableDataset[T] | DataLoader[T],
    max_samples: int = 3,
    log_interval: int = 10,
    callback: Callable[[T], None] | None = None,
) -> None:
    """Iterates through a dataset.

    Args:
        ds: The dataset to iterate through
        max_samples: Maximum number of samples to loop through
        log_interval: How often to log the time it takes to load a sample
        callback: A callback to run on each sample
    """
    start_time = time.time()

    if isinstance(ds, (IterableDataset, DataLoader)):
        logger.info("Iterating samples in %s", "dataloader" if isinstance(ds, DataLoader) else "dataset")
        for i, sample in enumerate(itertools.islice(ds, max_samples)):
            if callback is not None:
                callback(sample)
            if i % log_interval == 0:
                logger.info("Sample %d in %.2g seconds", i, time.time() - start_time)
    else:
        samples = len(ds)  # type: ignore[arg-type]
        logger.info("Dataset has %d items", samples)
        for i in range(min(samples, max_samples)):
            sample = ds[i]
            if callback is not None:
                callback(sample)
            if i % log_interval == 0:
                logger.info("Sample %d in %.2g seconds", i, time.time() - start_time)


def check_md5(file_path: str | Path, hash_str: str | None, chunk_size: int = 2**16) -> bool:
    """Checks the MD5 of the downloaded file.

    Args:
        file_path: Path to the downloaded file.
        hash_str: Expected MD5 of the file; if None, return True.
        chunk_size: Size of the chunks to read from the file.

    Returns:
        True if the MD5 matches, False otherwise.
    """
    if hash_str is None:
        return True

    md5 = hashlib.md5()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)

    return md5.hexdigest() == hash_str


def check_sha256(file_path: str | Path, hash_str: str | None, chunk_size: int = 2**16) -> bool:
    """Checks the SHA256 of the downloaded file.

    Args:
        file_path: Path to the downloaded file.
        hash_str: Expected SHA256 of the file; if None, return True.
        chunk_size: Size of the chunks to read from the file.

    Returns:
        True if the SHA256 matches, False otherwise.
    """
    if hash_str is None:
        return True

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)

    return sha256.hexdigest() == hash_str


def ensure_downloaded(
    url: str,
    *dnames: str,
    md5: str | None = None,
    sha256: str | None = None,
    is_tmp: bool = False,
    recheck_hash: bool = False,
) -> Path:
    """Ensures that a checkpoint URL has been downloaded.

    This basically just provides a nice way of organizing pre-trained models,
    by saving them to a consistent location.

    Args:
        url: The URL to download.
        dnames: The directory to download to (note that this is relative to the
            model directory). The final name should be the file name
        md5: The MD5 hash of the file, if known.
        sha256: The SHA256 hash of the file, if known.
        is_tmp: If set, use ``tmp/`` instead of ``get_pretrained_models_dir()``
        recheck_hash: Whether to recheck the hash of the file if it already
            exists.

    Returns:
        The path to the downloaded file.
    """
    assert len(dnames) >= 1, "Must provide at least 1 directory name"
    filepath = Path(tempfile.mkdtemp("models")) if is_tmp else get_pretrained_models_dir()
    for dname in dnames:
        filepath = filepath / dname
    (root := filepath.parent).mkdir(parents=True, exist_ok=True)

    def check_hashes() -> bool:
        return filepath.is_file() and check_sha256(filepath, sha256) and check_md5(filepath, md5)

    def download_file() -> None:
        download_url(url, root=root, filename=filepath.name)
        assert filepath.is_file(), f"Failed to download {url} to {filepath}"
        if not check_hashes():
            filepath.unlink()
            raise RuntimeError(f"Hashes for {url} do not match")

    # If the file does not exist, download it and check the hashes.
    if not filepath.exists():
        download_file()

    # By default, assume the downloaded file hash is correct.
    if not recheck_hash:
        return filepath

    # Check the file hashes again, to ensure the file was not corrupted.
    if not check_hashes():
        filepath.unlink()
        download_file()

    return filepath


def get_state_dict_prefix(
    ckpt: dict[str, T],
    prefix: str | None = None,
    suffix: str | None = None,
    regexp: re.Pattern[str] | None = None,
) -> dict[str, T]:
    """Returns the parts of a checkpoint which begin with a prefix.

    Args:
        ckpt: The checkpoint to modify
        prefix: The prefix to clip
        suffix: The suffix to clip
        regexp: The regexp to search for (doesn't modify any keys)

    Returns:
        The modified checkpoint
    """
    if prefix is not None:
        ckpt = {k[len(prefix) :]: v for k, v in ckpt.items() if k.startswith(prefix)}
    if suffix is not None:
        ckpt = {k[: -len(suffix)]: v for k, v in ckpt.items() if k.endswith(suffix)}
    if regexp is not None:
        ckpt = {k: v for k, v in ckpt.items() if regexp.match(k)}
    return ckpt
