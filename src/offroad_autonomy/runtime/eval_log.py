"""Evaluation run log for scored drives (``--eval-log``).

Records, per control frame: the dashcam image, the model's road mask, the
perception gate's own decision and the road-area fraction. Separately, the
vehicle's true world position, sampled at a fixed rate on its own connection,
which nothing that steers ever reads. ``run_metadata.json`` pins down exactly
what ran and how the run ended, so a result can be reproduced or rejected.

PNG encoding takes tens of milliseconds per image, so it runs on a writer
thread. The control loop never waits on disk: if the writer falls behind, an
image is dropped and counted rather than stalling steering.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import importlib.metadata
import json
import logging
import queue
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from offroad_autonomy.planning.perception_gate import GateDecision
from offroad_autonomy.types import PathPlan

logger = logging.getLogger("offroad_autonomy.evaluation")

_FRAME_FIELDS = (
    "frame",
    "capture_time_s",
    "log_time_s",
    "gate_evaluated",
    "gate_ok",
    "gate_reason",
    "gate_area_fraction",
    "road_fraction",
    "confidence",
    "plan_status",
    "fallback_reason",
    "image_saved",
)

# Outcomes that mean the route was driven end to end under autopilot.
COMPLETED = "completed"


class EvalLogger:
    def __init__(self, directory: str | Path, metadata: dict, queue_size: int = 256) -> None:
        self.directory = Path(directory)
        if self.directory.exists() and any(self.directory.iterdir()):
            raise FileExistsError(
                f"Evaluation log folder {self.directory} is not empty; "
                "earlier results are never overwritten, so choose a new folder"
            )
        (self.directory / "frames").mkdir(parents=True, exist_ok=True)
        (self.directory / "masks").mkdir(exist_ok=True)
        self._metadata = dict(metadata)
        self._t0 = time.perf_counter()
        self.dropped_images = 0
        self.frames_logged = 0

        self._frames_file = (self.directory / "frames.csv").open("w", newline="", encoding="utf-8")
        self._frames = csv.DictWriter(self._frames_file, fieldnames=_FRAME_FIELDS)
        self._frames.writeheader()
        self._pose_file = (self.directory / "pose.csv").open("w", newline="", encoding="utf-8")
        self._pose = csv.writer(self._pose_file)
        self._pose.writerow(("log_time_s", "x", "y", "z", "travelled_m"))
        self._events_file = (self.directory / "events.csv").open("w", newline="", encoding="utf-8")
        self._events = csv.writer(self._events_file)
        self._events.writerow(("log_time_s", "kind", "detail"))

        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._writer = threading.Thread(target=self._write_images, name="eval-log", daemon=True)
        self._writer.start()

    def elapsed(self, now: float) -> float:
        return now - self._t0

    def write_json(self, name: str, data) -> None:
        (self.directory / name).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def frame(
        self,
        index: int,
        capture_time: float,
        image: np.ndarray,
        mask: np.ndarray | None,
        gate: GateDecision | None,
        road_fraction: float,
        confidence: float,
        plan: PathPlan,
    ) -> None:
        saved = 1
        name = f"{index:06d}.png"
        mask_image = None
        if mask is not None:
            mask_image = mask.astype(np.uint8) * 255
        try:
            self._queue.put_nowait((name, image, mask_image))
        except queue.Full:
            self.dropped_images += 1
            saved = 0
        # A stale capture skips the planner, so its gate never ran.
        evaluated = gate is not None and plan.status != "stale"
        row = {
            "frame": index,
            "capture_time_s": round(self.elapsed(capture_time), 4),
            "log_time_s": round(self.elapsed(time.perf_counter()), 4),
            "gate_evaluated": int(evaluated),
            "gate_ok": "",
            "gate_reason": "",
            "gate_area_fraction": "",
            "road_fraction": round(float(road_fraction), 5),
            "confidence": round(float(confidence), 4),
            "plan_status": plan.status,
            "fallback_reason": plan.fallback_reason,
            "image_saved": saved,
        }
        if evaluated:
            row["gate_ok"] = int(gate.ok)
            row["gate_area_fraction"] = round(gate.area_fraction, 5)
            # Planners write their own failures into an accepted decision's
            # reason; only a rejected one still holds the gate's reason.
            if not gate.ok:
                row["gate_reason"] = gate.reason
        self._frames.writerow(row)
        self.frames_logged += 1

    def pose(self, now: float, position: tuple[float, float, float], travelled_m: float) -> None:
        x, y, z = position
        self._pose.writerow(
            (
                round(self.elapsed(now), 4),
                round(x, 4),
                round(y, 4),
                round(z, 4),
                round(travelled_m, 3),
            )
        )

    def event(self, now: float, kind: str, detail: str = "") -> None:
        logger.info("Evaluation event: %s %s", kind, detail)
        self._events.writerow((round(self.elapsed(now), 4), kind, detail))

    def close(self, outcome: dict) -> None:
        self._queue.put(None)
        self._writer.join()
        for stream in (self._frames_file, self._pose_file, self._events_file):
            stream.close()
        self._metadata.update(outcome)
        self._metadata["frames_logged"] = self.frames_logged
        self._metadata["images_dropped"] = self.dropped_images
        self._metadata["ended"] = now_iso()
        self.write_json("run_metadata.json", self._metadata)
        logger.info("Evaluation log written to %s", self.directory)

    def _write_images(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            name, image, mask = item
            cv2.imwrite(str(self.directory / "frames" / name), image)
            if mask is not None:
                cv2.imwrite(str(self.directory / "masks" / name), mask)


class EvalRun:
    """Tracks true distance driven and decides when a scored run is over."""

    def __init__(self, log: EvalLogger, route_length_m: float, max_duration_s: float) -> None:
        self.log = log
        self.route_length_m = route_length_m
        self.max_duration_s = max_duration_s
        self.travelled_m = 0.0
        self.interventions = 0
        self._last: tuple[float, float] | None = None
        self._start: float | None = None

    def step(self, now: float, samples: list[tuple[float, float, float, float]]) -> str | None:
        """Log position samples, each (time, x, y, z); return why the run ends, or None."""
        if self._start is None:
            self._start = now
        for sample_time, x, y, z in samples:
            if self._last is not None:
                self.travelled_m += float(np.hypot(x - self._last[0], y - self._last[1]))
            self._last = (x, y)
            self.log.pose(sample_time, (x, y, z), self.travelled_m)
        if self.travelled_m >= self.route_length_m:
            return COMPLETED
        if now - self._start >= self.max_duration_s:
            return "timeout"
        return None

    def intervention(self, now: float, kind: str, detail: str = "") -> None:
        self.interventions += 1
        self.log.event(now, kind, detail)

    def finish(self, outcome: str, detail: str = "", extra: dict | None = None) -> None:
        summary = {
            "outcome": outcome,
            "outcome_detail": detail,
            "run_completed": outcome == COMPLETED and self.interventions == 0,
            "travelled_m": round(self.travelled_m, 2),
            "interventions": self.interventions,
        }
        summary.update(extra or {})
        self.log.close(summary)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5, check=False
    )
    if dirty.stdout.strip():
        # Results from uncommitted code cannot be reproduced from the hash alone.
        commit += "-dirty"
    return commit


def now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
