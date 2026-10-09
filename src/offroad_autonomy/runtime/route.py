"""Reference routes for scoring a run from the vehicle's true position.

A route is a recorded clean drive: the simulator's ground-truth positions,
one point every ``spacing_m``. Off-road trails are usually missing from
BeamNG's road network, so a recording is the one source that works on every
map. Routes are stored as JSON, one per map and spawn.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from offroad_autonomy.types import VehicleState

#: How far along the route, in points either side of the last match, the next
#: position is looked for. A full search could snap to another part of a
#: route that loops back near itself, such as the end of a closed circuit.
_SEARCH_BEHIND = 20
_SEARCH_AHEAD = 200


def route_path(routes_dir: str | Path, map_name: str, spawn_index: int) -> Path:
    return Path(routes_dir) / f"{map_name}_spawn{spawn_index}.json"


class Route:
    def __init__(self, points: np.ndarray) -> None:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("A route needs at least two (x, y) points")
        self.points = points
        segments = np.diff(points, axis=0)
        self._lengths = np.linalg.norm(segments, axis=1)
        self._cumulative = np.concatenate(([0.0], np.cumsum(self._lengths)))
        self.length_m = float(self._cumulative[-1])
        #: The file it was loaded from, named in benchmark reports.
        self.path: Path | None = None
        # A route is recorded from the same spawn the run starts at, so the
        # first search starts at its beginning, never at its far end.
        self._last = 0

    @classmethod
    def load(cls, path: str | Path) -> Route:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        route = cls(np.asarray(data["points"], dtype=np.float64))
        route.path = Path(path)
        return route

    def reset(self) -> None:
        self._last = 0

    def locate(self, x: float, y: float) -> tuple[float, float]:
        """``(cross_track_m, progress_m)`` of a position against the route.

        Cross-track is unsigned: scoring asks how far off the route the
        vehicle is, not which side.
        """
        count = len(self.points) - 1
        start = max(0, self._last - _SEARCH_BEHIND)
        stop = min(count, self._last + _SEARCH_AHEAD)
        a = self.points[start:stop]
        seg = np.diff(self.points[start : stop + 1], axis=0)
        lengths_sq = np.maximum((seg**2).sum(axis=1), 1e-12)
        position = np.array([x, y])
        t = np.clip(((position - a) * seg).sum(axis=1) / lengths_sq, 0.0, 1.0)
        nearest = a + seg * t[:, None]
        distances = np.linalg.norm(nearest - position, axis=1)
        best = int(np.argmin(distances))
        index = start + best
        self._last = index
        progress = self._cumulative[index] + t[best] * self._lengths[index]
        return float(distances[best]), float(progress)


class RouteRecorder:
    """Records a route from a drive, dropping points closer than ``spacing_m``
    so a vehicle standing still does not pile up points."""

    def __init__(self, spacing_m: float) -> None:
        self._spacing_m = float(spacing_m)
        self.points: list[tuple[float, float]] = []

    def add(self, state: VehicleState) -> None:
        if not state.valid:
            return
        x, y = float(state.position[0]), float(state.position[1])
        if self.points:
            last_x, last_y = self.points[-1]
            if math.hypot(x - last_x, y - last_y) < self._spacing_m:
                return
        self.points.append((x, y))

    def save(self, path: str | Path, map_name: str, spawn_index: int) -> Path:
        if len(self.points) < 2:
            raise ValueError("The drive was too short to record a route")
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "map": map_name,
            "spawn_index": spawn_index,
            "spacing_m": self._spacing_m,
            "points": [[round(x, 3), round(y, 3)] for x, y in self.points],
        }
        out.write_text(json.dumps(data, indent=1), encoding="utf-8")
        return out
