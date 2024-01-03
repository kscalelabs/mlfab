"""Defines a logger which logs JSON lines to a file."""

import json
import sys
from dataclasses import asdict
from typing import Any, Literal, TextIO

from torch import Tensor

from mlfab.task.logger import LoggerImpl, LogLine


def get_json_value(value: Any) -> Any:  # noqa: ANN401
    if isinstance(value, Tensor):
        value = value.detach().float().cpu().item()
    return value


class JsonLogger(LoggerImpl):
    def __init__(
        self,
        log_stream: TextIO = sys.stdout,
        flush_immediately: bool = False,
        open_mode: Literal["w", "a"] = "w",
        line_sep: str = "\n",
        remove_unicode_from_namespaces: bool = True,
        log_interval_seconds: float = 10.0,
    ) -> None:
        """Defines a simpler logger which logs to stdout.

        Args:
            log_stream: The stream to log to.
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

        self.log_stream = log_stream
        self.flush_immediately = flush_immediately
        self.open_mode = open_mode
        self.line_sep = line_sep
        self.remove_unicode_from_namespaces = remove_unicode_from_namespaces

    @property
    def fp(self) -> TextIO:
        return self.log_stream

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
