"""A simple mixin for monitoring if the main training job is still alive.

If this mixin detects that the training job has died then it will kill the
current process.
"""

import logging
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from multiprocessing.managers import SyncManager
from multiprocessing.synchronize import Event
from typing import Callable, TypeVar

import psutil
import setproctitle

from mlfab.core.conf import field
from mlfab.core.state import State
from mlfab.task.mixins.monitor_process import MonitorProcessConfig, MonitorProcessMixin

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class HeartbeatConfig(MonitorProcessConfig):
    heartbeat_ping_interval: float = field(60.0 * 30.0, help="How often to check for liveness (in seconds)")


Config = TypeVar("Config", bound=HeartbeatConfig)


def worker(
    heartbeat_interval: float,
    heartbeat_event: Event,
    start_event: Event,
    pid: int,
    on_heartbeat: Callable[[int, Event], None],
) -> None:
    setproctitle.setproctitle("mlfab-heartbeat")

    start_event.set()

    logger.debug("Starting heartbeat monitor for PID %d with PID %d", pid, os.getpid())

    while True:
        try:
            heartbeat_event.set()
            time.sleep(heartbeat_interval)
            if heartbeat_event.is_set():
                on_heartbeat(pid, heartbeat_event)

        except psutil.NoSuchProcess:
            logger.info("No parent process; probably cleaning up")


def kill_process(pid: int, heartbeat_event: Event) -> None:
    logger.warning("Heartbeat not responding; killing process %d", pid)
    proc = psutil.Process(pid)
    proc.kill()

    cur_pid = os.getpid()
    cur_proc = psutil.Process(cur_pid)
    cur_proc.kill()


class HeartbeatMonitor:
    def __init__(
        self,
        heartbeat_interval: float,
        manager: SyncManager,
        on_heartbeat: Callable[[int, Event], None] | None,
    ) -> None:
        self._heartbeat_interval = heartbeat_interval
        self._on_heartbeat = on_heartbeat
        self._manager = manager
        self._heartbeat_event = manager.Event()
        self._start_event = manager.Event()
        self._proc: mp.Process | None = None

    def beat(self) -> None:
        if self._heartbeat_event.is_set():
            self._heartbeat_event.clear()

    def start(self, wait: bool = False) -> None:
        if self._proc is not None:
            raise RuntimeError("Heartbeat already started")
        if self._heartbeat_event.is_set():
            self._heartbeat_event.clear()
        if self._start_event.is_set():
            self._start_event.clear()
        self._proc = mp.Process(
            target=worker,
            args=(self._heartbeat_interval, self._heartbeat_event, self._start_event, os.getpid(), self._on_heartbeat),
            daemon=True,
        )
        self._proc.start()
        if wait:
            self._start_event.wait()

    def stop(self) -> None:
        if self._proc is None:
            raise RuntimeError("Heartbeat not started")
        if self._proc.is_alive():
            self._proc.terminate()
            logger.debug("Terminated heartbeat process; joining...")
            self._proc.join()
        self._proc = None


class HeartbeatMonitorMixin(MonitorProcessMixin[Config]):
    """Defines a trainer mixin for running a heartbeat process."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self._heartbeat_monitor = HeartbeatMonitor(self.config.heartbeat_ping_interval, self._mp_manager, kill_process)

    def on_training_start(self, state: State) -> None:
        super().on_training_start(state)

        self._heartbeat_monitor.start()

    def on_training_end(self, state: State) -> None:
        super().on_training_end(state)

        self._heartbeat_monitor.stop()

    def on_step_start(self, state: State) -> None:
        super().on_step_start(state)

        self._heartbeat_monitor.beat()
