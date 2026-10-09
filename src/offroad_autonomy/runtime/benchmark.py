"""Per-run scores from simulator ground truth, the same for every stack.

Nothing here reads what a stack says about itself, so a stereo stack, an
end-to-end stack and the mono_yoloe stack are scored alike:

* **Driving** - distance, completion and cross-track error come from the
  vehicle's true position against a recorded reference route
  (``runtime/route.py``). A run without a route reports those as null.
  They stop at the first safe stop: what the operator drives afterwards is
  not the stack's.
* **Departures** - how many times the true position went more than
  ``departure_cross_track_m`` off the route.
* **Smoothness** - steering rate and sign changes, from the commands sent.
* **Timing** - loop rate, capture-to-actuation latency and the time spent
  in the stack's ``step``, measured by the harness around it.

Whatever the stack puts in ``StackOutput.metrics`` is averaged into a
separate ``stack_metrics`` section that never feeds the comparison.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from offroad_autonomy.contract import finite_metrics
from offroad_autonomy.runtime.route import Route
from offroad_autonomy.runtime.timing import MAIN_STAGES, RuntimeStats
from offroad_autonomy.types import ControlCommand, HarnessConfig, VehicleState

#: (column heading, report key) for the side-by-side comparison.
COMPARISON_COLUMNS = (
    ("run", "label"),
    ("stack", "stack"),
    ("FPS", "main_fps"),
    ("lat p95", "primary_latency_p95_ms"),
    ("step p95", "step_p95_ms"),
    ("over budget", "step_over_budget"),
    ("dist m", "distance_m"),
    ("done %", "completion_pct"),
    ("xte mean", "cross_track_mean_m"),
    ("xte max", "cross_track_max_m"),
    ("departures", "departures"),
    ("safe stops", "safe_stops"),
    ("steer rate", "steering_rate_mean_per_s"),
    ("sign chg/min", "steering_sign_changes_per_min"),
)


class BenchmarkRecorder:
    def __init__(
        self,
        label: str,
        config_path: str,
        config: HarnessConfig,
        route: Route | None = None,
        warmup_s: float = 5.0,
    ) -> None:
        self.label = label
        route_file = None
        if route is not None and route.path is not None:
            route_file = str(route.path)
        self._header = {
            "label": label,
            "config": config_path,
            "stack": config.stack,
            "map": config.beamng_map,
            "spawn_index": config.beamng_spawn_index,
            "route": route_file,
        }
        self._route = route
        self._warmup_s = float(warmup_s)
        self._departure_m = config.benchmark_departure_cross_track_m
        self._deadband = config.benchmark_steer_sign_deadband
        self._budget_ms = config.dashboard_thresholds.latency_budget_ms

        # Warm-up counts from the first frame, not from construction: the
        # recorder exists before BeamNG has even launched.
        self._t_origin: float | None = None
        self._t_first: float | None = None
        self._t_last: float | None = None
        self.frames = 0
        self.over_budget = 0
        self._primary_ms: list[float] = []
        self._step_ms: list[float] = []
        self._full_ms: list[float] = []
        self._metric_sums: dict[str, float] = {}
        self._metric_counts: dict[str, int] = {}

        # Driving is scored from the first frame, warm-up included: the
        # vehicle may already be moving, and completion counts from the spawn.
        self._scoring = True
        self._was_active = True
        self.safe_stops = 0
        self._last_position: tuple[float, float] | None = None
        self.distance_m = 0.0
        self._cross_track: list[float] = []
        self.departures = 0
        self._departed = False
        self._progress_m = 0.0
        self._last_steer: tuple[float, float] | None = None
        self._steer_rates: list[float] = []
        self._last_sign = 0
        self.sign_changes = 0
        self._t_drive_first: float | None = None
        self._t_drive_last: float | None = None

    def add(
        self,
        now: float,
        state: VehicleState,
        command: ControlCommand,
        autopilot_active: bool,
        primary_ms: float,
        step_ms: float,
        full_ms: float,
        stack_metrics: dict[str, float] | None = None,
    ) -> None:
        if self._was_active and not autopilot_active:
            self.safe_stops += 1
            self._scoring = False
        self._was_active = autopilot_active
        if self._scoring:
            self._score_driving(now, state, command)

        if self._t_origin is None:
            self._t_origin = now
        if now - self._t_origin < self._warmup_s:
            return
        if self._t_first is None:
            self._t_first = now
        self._t_last = now
        self.frames += 1
        self._primary_ms.append(primary_ms)
        self._step_ms.append(step_ms)
        self._full_ms.append(full_ms)
        if step_ms > self._budget_ms:
            self.over_budget += 1
        if stack_metrics:
            for name, value in finite_metrics(stack_metrics).items():
                self._metric_sums[name] = self._metric_sums.get(name, 0.0) + value
                self._metric_counts[name] = self._metric_counts.get(name, 0) + 1

    def _score_driving(self, now: float, state: VehicleState, command: ControlCommand) -> None:
        if self._t_drive_first is None:
            self._t_drive_first = now
        self._t_drive_last = now

        steer = float(command.steering)
        if self._last_steer is not None:
            last_t, last_steer = self._last_steer
            dt = now - last_t
            if dt > 0.0:
                self._steer_rates.append(abs(steer - last_steer) / dt)
        self._last_steer = (now, steer)
        sign = 0
        if steer > self._deadband:
            sign = 1
        elif steer < -self._deadband:
            sign = -1
        if sign != 0:
            if self._last_sign != 0 and sign != self._last_sign:
                self.sign_changes += 1
            self._last_sign = sign

        if not state.valid:
            return
        x, y = float(state.position[0]), float(state.position[1])
        if self._last_position is not None:
            self.distance_m += math.hypot(x - self._last_position[0], y - self._last_position[1])
        self._last_position = (x, y)
        if self._route is None:
            return
        cross_track, progress = self._route.locate(x, y)
        self._cross_track.append(cross_track)
        self._progress_m = max(self._progress_m, progress)
        if cross_track > self._departure_m:
            if not self._departed:
                self.departures += 1
                self._departed = True
        else:
            self._departed = False

    def report(self, main: RuntimeStats) -> dict:
        duration = 0.0
        if self._t_first is not None and self._t_last is not None:
            duration = self._t_last - self._t_first
        main_fps = None
        if duration > 0:
            main_fps = round(self.frames / duration, 2)
        drive_s = 0.0
        if self._t_drive_first is not None and self._t_drive_last is not None:
            drive_s = self._t_drive_last - self._t_drive_first

        report = dict(self._header)
        report.update(
            {
                "duration_s": round(duration, 1),
                "frames": self.frames,
                "main_fps": main_fps,
                "primary_latency_mean_ms": _round(_stat(self._primary_ms, np.mean)),
                "primary_latency_p95_ms": _round(_stat(self._primary_ms, _p95)),
                "step_mean_ms": _round(_stat(self._step_ms, np.mean)),
                "step_p95_ms": _round(_stat(self._step_ms, _p95)),
                "step_over_budget": self.over_budget,
                "full_loop_mean_ms": _round(_stat(self._full_ms, np.mean)),
                "full_loop_p95_ms": _round(_stat(self._full_ms, _p95)),
                "distance_m": _round(self.distance_m, 1),
                "route_length_m": None,
                "completion_pct": None,
                "cross_track_mean_m": None,
                "cross_track_max_m": None,
                "departures": None,
                "safe_stops": self.safe_stops,
                "ended_in_safe_stop": not self._was_active,
                "steering_rate_mean_per_s": _round(_stat(self._steer_rates, np.mean), 3),
                "steering_sign_changes_per_min": None,
                "main_stages": {
                    name: {"mean_ms": round(s.mean_ms, 2), "p95_ms": round(s.p95_ms, 2)}
                    for name, s in main.summary(MAIN_STAGES).items()
                },
                "stack_metrics": {
                    name: round(total / self._metric_counts[name], 4)
                    for name, total in sorted(self._metric_sums.items())
                },
                "notes": (
                    "Driving metrics are from simulator ground truth and stop at the first "
                    "safe stop; stack_metrics are the stack's own and are not compared."
                ),
            }
        )
        if drive_s > 0.0:
            report["steering_sign_changes_per_min"] = _round(self.sign_changes / drive_s * 60.0)
        if self._route is not None:
            report["route_length_m"] = _round(self._route.length_m, 1)
            report["completion_pct"] = _round(
                100.0 * self._progress_m / max(self._route.length_m, 1e-6), 1
            )
            report["cross_track_mean_m"] = _round(_stat(self._cross_track, np.mean), 3)
            report["cross_track_max_m"] = _round(_stat(self._cross_track, np.max), 3)
            report["departures"] = self.departures
        return report

    def save(self, path: str | Path, report: dict) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return out


def comparison_table(
    reports: list[dict],
    markdown: bool = False,
    columns: tuple[tuple[str, str], ...] = COMPARISON_COLUMNS,
) -> list[str]:
    """One row per run; values a run could not measure print as "-"."""
    headers = [name for name, _ in columns]
    rows = [[_cell(report.get(key)) for _, key in columns] for report in reports]
    if markdown:
        lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
        lines += ["| " + " | ".join(row) + " |" for row in rows]
        return lines
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths))]
    lines.append("  ".join("-" * w for w in widths))
    lines += ["  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in rows]
    return lines


def _cell(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        if value:
            return "yes"
        return "no"
    if isinstance(value, float):
        if abs(value) < 10:
            return f"{value:.2f}"
        return f"{value:.1f}"
    return str(value)


def _p95(values: np.ndarray) -> float:
    return float(np.percentile(values, 95))


def _stat(values: list[float], fn) -> float | None:
    if not values:
        return None
    return float(fn(np.asarray(values)))


def _round(value: float | None, digits: int = 2) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)
