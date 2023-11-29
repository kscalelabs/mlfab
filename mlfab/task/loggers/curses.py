"""Defines a logger that shows a pop-up using Curses."""

import bdb
import curses
import datetime
import functools
import logging
import sys
from collections import deque
from typing import TYPE_CHECKING, Any, Deque, TextIO

from torch import Tensor
from wcwidth import wcswidth

from mlfab.task.logger import LoggerImpl, LogLine
from mlfab.utils.experiments import ToastKind, Toasts
from mlfab.utils.text import format_timedelta

if TYPE_CHECKING:
    from _curses import _CursesWindow


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


class CursesLogger(LoggerImpl):
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

        self.persistent_toast_queue: Deque[tuple[str, ToastKind]] = deque()
        self.temporary_toast_queue: Deque[tuple[str, ToastKind]] = deque(maxlen=max_toasts)

        Toasts.register_callback("error", functools.partial(self.handle_toast, persistent=False, kind="error"))
        Toasts.register_callback("warning", functools.partial(self.handle_toast, persistent=False, kind="warning"))
        Toasts.register_callback("info", functools.partial(self.handle_toast, persistent=False, kind="info"))
        Toasts.register_callback("status", functools.partial(self.handle_toast, persistent=True, kind="status"))

        self.__stdscr: "_CursesWindow | None" = None

    def _get_window(self) -> "_CursesWindow":
        stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        curses.curs_set(0)
        stdscr.keypad(True)
        if curses.has_colors():
            curses.start_color()
        return stdscr

    def start(self) -> None:
        if self.__stdscr is None:
            self.__stdscr = self._get_window()
            curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
            curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
            curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
            curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
            curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
        return super().start()

    def stop(self) -> None:
        if self.__stdscr is not None:
            self.__stdscr.keypad(False)
            curses.curs_set(1)
            curses.echo()
            curses.nocbreak()
            curses.endwin()
            self.__stdscr = None
        sys.stdout.write("")
        sys.stdout.flush()
        return super().stop()

    def handle_toast(self, msg: str, persistent: bool, kind: ToastKind) -> None:
        if persistent:
            self.persistent_toast_queue.append((msg, kind))
        else:
            self.temporary_toast_queue.append((msg, kind))

    @property
    def stdscr(self) -> "_CursesWindow":
        assert self.__stdscr is not None, "Not started!"
        return self.__stdscr

    def write_state_window(self, line: LogLine, window: "_CursesWindow") -> None:
        elapsed_time = format_timedelta(datetime.timedelta(seconds=line.state.elapsed_time_s), short=True)
        state_info = {
            "Steps": f"{line.state.num_steps}",
            "Samples": f"{line.state.num_samples}",
            "Elapsed Time": f"{elapsed_time}",
        }
        if line.state.num_epochs > 0:
            state_info["Epochs"] = f"{line.state.num_epochs}"
        max_width = max(wcswidth(k) + wcswidth(v) for k, v in state_info.items()) + 2

        try:
            window.resize(len(state_info) + 3, max_width + 4)
        except Exception:
            return
        window.border()
        window.addstr(1, 2, "Phase")
        ph = line.state.phase
        window.addstr(1, max_width - wcswidth(ph) + 2, ph, curses.color_pair(2 if ph == "train" else 3))
        for i, (k, v) in enumerate(state_info.items()):
            window.addstr(i + 2, 2, k)
            window.addstr(i + 2, max_width - wcswidth(v) + 2, v, curses.color_pair(4))

    def write_log_window(self, line: LogLine, window: "_CursesWindow") -> None:
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

        try:
            window.resize(num_lines + 2, max_width + 6)
        except Exception:
            return
        window.border()
        line = 1
        for namespace, lines in sorted(namespace_to_lines.items()):
            if line > 1:
                line += 1
            window.addstr(line, 2, namespace, curses.color_pair(4) | curses.A_BOLD)
            line += 1
            for k, v in lines.items():
                window.addstr(line, 2, k)
                window.addstr(line, max_width - wcswidth(v) + 4, v, curses.color_pair(5))
                line += 1

    def write_queue_window(self, q: Deque[tuple[str, ToastKind]], window: "_CursesWindow", width: int) -> None:
        if not q:
            return

        def clip_left(s: str, length: int) -> str:
            s = s.replace("\n", " ")
            return s if len(s) <= length else "..." + s[-length + 3 :]

        def get_color(kind: ToastKind) -> int:
            match kind:
                case "error":
                    return 1
                case "warning":
                    return 3
                case "info":
                    return 4
                case "status":
                    return 2
                case _:
                    return 5

        width = max(len(clip_left(msg, width - 4)) for msg, _ in q) + 4
        try:
            window.resize(len(q) + 2, width)
        except Exception:
            return
        window.border()
        for i, (msg, kind) in enumerate(reversed(q)):
            window.addstr(i + 1, 2, clip_left(msg, width - 4), curses.color_pair(get_color(kind)))

    def write(self, line: LogLine) -> None:
        try:
            self.stdscr.clear()
            _, win_width = self.stdscr.getmaxyx()

            # Writes the state window.
            try:
                state_window = self.stdscr.subwin(0, 0)
            except Exception:
                self.stdscr.refresh()
                return
            self.write_state_window(line, state_window)
            sw_height, sw_width = state_window.getmaxyx()

            # Writes the log window.
            try:
                log_window = self.stdscr.subwin(0, sw_width)
            except Exception:
                self.stdscr.refresh()
                return
            self.write_log_window(line, log_window)
            lw_height, _ = log_window.getmaxyx()

            # Writes the persistent window.
            start_y, start_x = max(sw_height, lw_height), 0
            if self.persistent_toast_queue:
                try:
                    persistent_window = self.stdscr.subwin(start_y, start_x)
                except Exception:
                    self.stdscr.refresh()
                    return
                self.write_queue_window(self.persistent_toast_queue, persistent_window, min(win_width, 100))
                pw_height, _ = persistent_window.getmaxyx()
                start_y += pw_height

            # Writes the warning window.
            if self.temporary_toast_queue:
                try:
                    temporary_window = self.stdscr.subwin(start_y, start_x)
                except Exception:
                    self.stdscr.refresh()
                    return
                self.write_queue_window(self.temporary_toast_queue, temporary_window, min(win_width, 100))
                tw_height, _ = temporary_window.getmaxyx()
                start_y += tw_height

            self.stdscr.refresh()

        except (KeyboardInterrupt, bdb.BdbQuit):
            raise

        except Exception:
            logging.exception("Error while writing log line")
