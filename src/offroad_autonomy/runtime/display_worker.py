"""Render completed dashcam inference snapshots without blocking control."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from offroad_autonomy.runtime.timing import RuntimeStats
from offroad_autonomy.types import PathPlan, PipelineStepResult

logger = logging.getLogger("offroad_autonomy.runtime.display")


@dataclass
class DisplayState:
    """Everything the dashboard draws, as of one control-loop iteration.

    This is a *snapshot to read*, never a channel back into the stack: the
    worker consumes it and produces pixels. The arrays inside are the ones
    the loop has already finished with, and each iteration produces fresh
    ones, so the worker can hold a reference without copying. A redraw that
    straddles two iterations shows a one-frame-old overlay, which is a
    cosmetic outcome and the reason no lock is held while drawing.
    """

    result: PipelineStepResult
    telemetry: object
    plan: PathPlan | None = None
    valid_roi: np.ndarray | None = None
    debug_view: str = "default"
    timing_overlay: bool = False


class DisplayWorker:
    def __init__(
        self,
        render: Callable[[DisplayState], np.ndarray],
        show: Callable[[np.ndarray], bool],
        read_key: Callable[[], int],
        rate_hz: float = 20.0,
        asynchronous: bool = True,
        stats: RuntimeStats | None = None,
    ) -> None:
        self._render = render
        self._show = show
        self._read_key = read_key
        self._period = 0.0
        if rate_hz > 0.0:
            self._period = 1.0 / rate_hz
        self._async = bool(asynchronous)

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._pending: DisplayState | None = None
        self._keys: deque[int] = deque(maxlen=32)
        self._stop = False
        self._closed = False
        self._thread: threading.Thread | None = None
        self._last_draw_time = 0.0
        if stats is None:
            stats = RuntimeStats()
        self.stats = stats
        self.rendered = 0
        self.dropped = 0
        self.failed = 0
        self._last_error_log = 0.0

    @property
    def asynchronous(self) -> bool:
        return self._async

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def fps(self) -> float:
        return self.stats.fps()

    def start(self) -> None:
        if not self._async or self._thread is not None:
            return
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="display-worker", daemon=True)
        self._thread.start()
        logger.info("Dashboard worker started (period %.3f s, 0 = unthrottled)", self._period)

    def stop(self, timeout: float = 2.0) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def publish(self, state: DisplayState) -> None:
        """Overwrites rather than queues, so a slow dashboard goes stale
        instead of building latency."""
        if not self._async:
            self._draw(state)
            return
        with self._cond:
            if self._pending is not None:
                self.dropped += 1
            self._pending = state
            self._cond.notify()

    def drain_keys(self) -> list[int]:
        with self._lock:
            keys = list(self._keys)
            self._keys.clear()
        return keys

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                remaining = self._period - (time.perf_counter() - self._last_draw_time)
                if remaining > 0.0:
                    self._cond.wait(remaining)
                if self._stop:
                    return
                state = self._pending
                self._pending = None
            if state is not None:
                self._draw(state)
                if self.closed:
                    return

    def _draw(self, state: DisplayState) -> None:
        t0 = time.perf_counter()
        self._last_draw_time = t0
        try:
            canvas = self._render(state)
            t2 = time.perf_counter()
            alive = self._show(canvas)
            t3 = time.perf_counter()
        except Exception:
            self.failed += 1
            if t0 - self._last_error_log > 5.0:
                logger.exception("Dashboard rendering failed")
                self._last_error_log = t0
            return

        self.stats.record("dashboard_render", (t2 - t0) * 1000.0)
        self.stats.record("dashboard_show", (t3 - t2) * 1000.0)
        self.stats.record("dashboard_total", (t3 - t0) * 1000.0)
        self.stats.tick()
        self.rendered += 1

        key = self._read_key()
        with self._lock:
            if key is not None and key >= 0:
                self._keys.append(int(key))
            if not alive:
                self._closed = True
