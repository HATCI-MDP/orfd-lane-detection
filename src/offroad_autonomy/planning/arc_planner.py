"""Pick the best constant-curvature arc through the traversability grid.

This is the "tentacles" approach used on off-road research vehicles: a fan of
arcs the vehicle can actually steer, each scored on the grid, best one wins.
It never reports a path the grid does not support, and unlike a row-by-row
walk it cannot end early because one stretch of image was misleading: every
arc is judged on the whole corridor at once.

Arcs start at the rear axle, because that is the point a bicycle-model
vehicle turns about, and are expressed in the camera's ground frame that the
grid and the controllers use. No packaged library scores arcs on a custom
grid, so this part is written for this stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from offroad_autonomy.planning.bev_grid import TraversabilityGrid
from offroad_autonomy.planning.grid_config import GridPlannerConfig


@dataclass
class ArcChoice:
    curvature: float
    forward_m: np.ndarray
    right_m: np.ndarray
    length_m: float
    min_clearance_m: float
    score: float


class ArcPlanner:
    def __init__(
        self,
        config: GridPlannerConfig,
        wheelbase_m: float,
        max_wheel_angle_deg: float,
        vehicle_half_width_m: float,
        rear_axle_behind_camera_m: float,
    ) -> None:
        self.config = config
        self._max_curvature = math.tan(math.radians(max_wheel_angle_deg)) / max(wheelbase_m, 0.1)
        # Positive curvature turns right, matching BeamNG steering.
        self.curvatures = np.linspace(-self._max_curvature, self._max_curvature, config.arc_count)
        self._required_clearance = float(vehicle_half_width_m) + config.clearance_margin_m

        s = np.arange(0.0, config.lookahead_m + rear_axle_behind_camera_m, config.step_m)
        k = self.curvatures[:, None]
        straight = np.abs(k) < 1e-9
        safe_k = np.where(straight, 1.0, k)
        forward = np.where(straight, s, np.sin(safe_k * s) / safe_k) - rear_axle_behind_camera_m
        right = np.where(straight, 0.0, (1.0 - np.cos(safe_k * s)) / safe_k)
        self._forward = forward
        self._right = right
        self._previous: float | None = None

    def reset(self) -> None:
        self._previous = None

    def choose(self, grid: TraversabilityGrid) -> ArcChoice | None:
        cfg = self.config
        clearance = grid.clearance_m()
        road = grid.road()
        rows, cols = grid.to_cell(self._forward, self._right)
        inside = (rows >= 0) & (rows < grid.rows) & (cols >= 0) & (cols < grid.cols)
        r = np.clip(rows, 0, grid.rows - 1)
        c = np.clip(cols, 0, grid.cols - 1)
        arc_clearance = np.where(inside, clearance[r, c], 0.0)
        arc_road = inside & road[r, c]
        ahead = self._forward >= cfg.start_m

        best: ArcChoice | None = None
        for i, curvature in enumerate(self.curvatures):
            candidate = self._score(
                i, float(curvature), arc_clearance[i], arc_road[i], ahead[i], inside[i]
            )
            if candidate is not None and (best is None or candidate.score > best.score):
                best = candidate
        if best is not None:
            self._previous = best.curvature
        return best

    def _score(
        self,
        index: int,
        curvature: float,
        clearance: np.ndarray,
        road: np.ndarray,
        ahead: np.ndarray,
        inside: np.ndarray,
    ) -> ArcChoice | None:
        cfg = self.config
        # The body would leave the road, or the arc leaves the grid: the arc
        # ends there, however good the ground beyond might look.
        blocked = (ahead & (clearance < self._required_clearance)) | ~inside
        end = len(clearance)
        if blocked.any():
            end = int(np.argmax(blocked))
        drivable = ahead.copy()
        drivable[end:] = False
        supported = np.flatnonzero(drivable & road)
        if len(supported) == 0:
            return None
        first = int(np.argmax(drivable))
        last = int(supported[-1])
        # Length counts only ground the grid has seen as road, so the path
        # (and with it the controller's path-end speed) never runs into the unknown.
        length = (last - first) * cfg.step_m
        if length < cfg.min_path_m:
            return None

        span = slice(first, last + 1)
        reach = min(length / cfg.lookahead_m, 1.0)
        capped = np.minimum(clearance[span], cfg.clearance_cap_m) / cfg.clearance_cap_m
        bend = abs(curvature) / self._max_curvature
        change = 0.0
        if self._previous is not None:
            change = abs(curvature - self._previous) / (2.0 * self._max_curvature)
        score = (
            cfg.weight_length * reach
            + cfg.weight_clearance * float(capped.mean())
            - cfg.weight_consistency * change
            - cfg.weight_curvature * bend
        )
        return ArcChoice(
            curvature=curvature,
            forward_m=self._forward[index, span].copy(),
            right_m=self._right[index, span].copy(),
            length_m=length,
            min_clearance_m=float(clearance[span].min()),
            score=score,
        )
