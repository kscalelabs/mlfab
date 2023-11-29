"""Defines a logger which logs JSON lines to a file."""

import json
from dataclasses import asdict
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from mlfab.task.logger import LoggerImpl, LogLine
from mlfab.utils.text import TextBlock, render_text_blocks


def get_json_value(value: Any) -> Any:  # noqa: ANN401
    if isinstance(value, Tensor):
        value = value.detach().float().cpu().item()
    return value


class JsonLogger(LoggerImpl):
    def __init__(
        self,
        run_directory: str | Path,
        log_file_name: str = "log.jsonl",
        git_state_name: str = "git_state.txt",
        config_name: str = "config.yaml",
        flush_immediately: bool = False,
        open_mode: Literal["w", "a"] = "w",
        line_sep: str = "\n",
        remove_unicode_from_namespaces: bool = True,
        log_interval_seconds: float = 10.0,
    ) -> None:
        """Defines a simpler logger which logs to stdout.

        Args:
            run_directory: The directory to log to.
            log_file_name: The name of the log file.
            git_state_name: The name of the git state file.
            config_name: The name of the config file.
            flush_immediately: Whether to flush the file after every write.
            open_mode: The file open mode.
            line_sep: The line separator to use.
            remove_unicode_from_namespaces: Whether to remove unicode from
                namespaces. This is the typical behavior for namespaces that
                use ASCII art for visibility in other logs, but in the JSON
                log file should be ignored.
            log_interval_seconds: The interval between successive log lines.
        """
        super().__init__(log_interval_seconds)

        self.log_file = Path(run_directory).expanduser().resolve() / log_file_name
        self.git_state_file = Path(run_directory).expanduser().resolve() / git_state_name
        self.config_file = Path(run_directory).expanduser().resolve() / config_name
        self.flush_immediately = flush_immediately
        self.open_mode = open_mode
        self.line_sep = line_sep
        self.remove_unicode_from_namespaces = remove_unicode_from_namespaces

        self.__fp: TextIOWrapper | None = None

    def start(self) -> None:
        self.__fp = open(self.log_file, self.open_mode)
        return super().start()

    def stop(self) -> None:
        if self.__fp is not None:
            self.__fp.close()
            self.__fp = None
        return super().stop()

    @property
    def fp(self) -> TextIOWrapper:
        assert self.__fp is not None, "Not started!"
        return self.__fp

    def get_json(self, line: LogLine) -> str:
        data: dict = {"state": asdict(line.state)}
        for log in (line.scalars, line.strings):
            for namespace, values in log.items():
                if self.remove_unicode_from_namespaces:
                    namespace = namespace.encode("ascii", errors="ignore").decode("ascii").strip()
                if namespace not in data:
                    data[namespace] = {}
                for k, v in values.items():
                    data[namespace][k] = get_json_value(v)
        return json.dumps(data)

    def write(self, line: LogLine) -> None:
        self.fp.write(self.get_json(line))
        self.fp.write(self.line_sep)
        if self.flush_immediately:
            self.fp.flush()

    def log_git_state(self, git_state: list[TextBlock]) -> None:
        state_str = render_text_blocks([[t] for t in git_state])
        with open(self.git_state_file, "w") as f:
            f.write(state_str)

    def log_config(self, config: DictConfig) -> None:
        OmegaConf.save(config, self.config_file)
