"""Defines a logger that logs to stdout."""

import datetime
import functools
import logging
import sys
from collections import deque
from typing import Any, Deque, TextIO

from torch import Tensor
from wcwidth import wcswidth

from mlfab.task.logger import LoggerImpl, LogLine
from mlfab.utils.experiments import ToastKind, Toasts
from mlfab.utils.text import Color, TextBlock, colored, format_timedelta, render_text_blocks


def format_number(value: int | float, precision: int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.{precision}g}"


def as_str(value: Any, precision: int) -> str:  # noqa: ANN401
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, Tensor):
        value = value.detach().float().cpu().item()
    if isinstance(value, (int, float)):
        return format_number(value, precision)
    raise TypeError(f"Unexpected log type: {type(value)}")


class StdoutLogger(LoggerImpl):
    def __init__(
        self,
        debug: bool = False,
        debug_fp: TextIO = sys.stdout,
        precision: int = 4,
        log_timers: bool = True,
        log_perf: bool = False,
        log_optim: bool = False,
        max_toasts: int = 10,
        log_interval_seconds: float = 1.0,
    ) -> None:
        """Defines a logger which shows a pop-up using Curses.

        Args:
            debug: Whether to enable debugging mode. If debugging, the
                terminal will not be modified; instead, we just print to stdout
                normally.
            debug_fp: The file to write debug logs to.
            precision: The integer precision to use when logging scalars.
            log_timers: Whether to log timers.
            log_perf: Whether to log performance metrics.
            log_optim: Whether to log optimizer parameters.
            max_toasts: The maximum number of toasts to display.
            log_interval_seconds: The interval between successive log lines.
        """
        super().__init__(log_interval_seconds)

        self.debug = debug
        self.debug_fp = debug_fp
        self.log_timers = log_timers
        self.log_perf = log_perf
        self.log_optim = log_optim
        self.precision = precision
        self.logger = logging.getLogger("stdout")

        self.persistent_toast_queue: Deque[tuple[str, ToastKind]] = deque()
        self.temporary_toast_queue: Deque[tuple[str, ToastKind]] = deque(maxlen=max_toasts)

        Toasts.register_callback("error", functools.partial(self.handle_toast, persistent=False, kind="error"))
        Toasts.register_callback("warning", functools.partial(self.handle_toast, persistent=False, kind="warning"))
        Toasts.register_callback("info", functools.partial(self.handle_toast, persistent=False, kind="info"))
        Toasts.register_callback("status", functools.partial(self.handle_toast, persistent=True, kind="status"))

    def start(self) -> None:
        return super().start()

    def stop(self) -> None:
        self.write_queue_window(self.persistent_toast_queue, True)
        self.write_queue_window(self.temporary_toast_queue, False)
        return super().stop()

    def handle_toast(self, msg: str, persistent: bool, kind: ToastKind) -> None:
        if persistent:
            self.persistent_toast_queue.append((msg, kind))
        else:
            self.temporary_toast_queue.append((msg, kind))

    def write_state_window(self, line: LogLine) -> None:
        elapsed_time = format_timedelta(datetime.timedelta(seconds=line.state.elapsed_time_s), short=True)
        state_info = {
            "Steps": f"{line.state.num_steps}",
            "Samples": f"{line.state.num_samples}",
            "Elapsed Time": f"{elapsed_time}",
        }
        if line.state.num_epochs > 0:
            state_info["Epochs"] = f"{line.state.num_epochs}"

        block = render_text_blocks(
            [
                [
                    TextBlock("Phase", no_sep=True),
                    TextBlock(
                        line.state.phase,
                        color="green" if line.state.phase == "train" else "yellow",
                        no_sep=True,
                        bold=True,
                    ),
                ],
                *([TextBlock(k, no_sep=True), TextBlock(v, color="cyan", no_sep=True)] for k, v in state_info.items()),
            ],
            align_all_blocks=True,
        )
        self.debug_fp.write(block)
        self.debug_fp.write("\n")

    def write_log_window(self, line: LogLine) -> None:
        namespace_to_lines: dict[str, dict[str, str]] = {}
        num_lines = 0
        max_width = 0
        for log in (line.scalars, line.strings):
            for namespace, values in log.items():
                if not self.log_timers and namespace.startswith("⏰"):
                    continue
                if not self.log_perf and namespace.startswith("🔧"):
                    continue
                if not self.log_optim and namespace.startswith("📉"):
                    continue
                if namespace not in namespace_to_lines:
                    namespace_to_lines[namespace] = {}
                num_lines += 2 if num_lines > 0 else 1
                max_width = max(max_width, wcswidth(namespace) + 3)
                for k, v in values.items():
                    v_str = as_str(v, self.precision)
                    namespace_to_lines[namespace][k] = v_str
                    num_lines += 1
                    max_width = max(max_width, wcswidth(k) + wcswidth(v_str) + 2)

        block = render_text_blocks(
            [
                [
                    TextBlock(namespace if i == 0 else "", no_sep=True, color="cyan", bold=True),
                    TextBlock(k, no_sep=True),
                    TextBlock(v, no_sep=True),
                ]
                for namespace, lines in sorted(namespace_to_lines.items())
                for i, (k, v) in enumerate(lines.items())
            ],
            align_all_blocks=True,
        )
        self.debug_fp.write(block)
        self.debug_fp.write("\n")

    def write_queue_window(self, q: Deque[tuple[str, ToastKind]], bold: bool) -> None:
        if not q:
            return

        def get_color(kind: ToastKind) -> Color:
            match kind:
                case "error":
                    return "red"
                case "warning":
                    return "yellow"
                case "info":
                    return "cyan"
                case "status":
                    return "green"
                case _:
                    return "magenta"

        if q:
            if bold:
                block = render_text_blocks(
                    [[TextBlock(msg, color=get_color(kind), no_sep=True)] for msg, kind in reversed(q)],
                    align_all_blocks=True,
                )
                self.debug_fp.write(block)
                self.debug_fp.write("\n")
            else:
                self.debug_fp.write("\n".join(f" ↪ {colored(msg, get_color(kind))}" for msg, kind in reversed(q)))
                self.debug_fp.write("\n")

    def write(self, line: LogLine) -> None:
        self.write_state_window(line)
        self.write_log_window(line)
        self.write_queue_window(self.persistent_toast_queue, True)
        self.write_queue_window(self.temporary_toast_queue, False)
        self.temporary_toast_queue.clear()
        sys.stdout.flush()
