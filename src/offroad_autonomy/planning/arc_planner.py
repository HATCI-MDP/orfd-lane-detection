"""Pick the best constant-curvature arc through the traversability grid.

This is the "tentacles" approach used on off-road research vehicles: a fan of
arcs the vehicle can actually steer, each scored on the grid, best one wins.
It never reports a path the grid does not support, and unlike a row-by-row
walk it cannot end early because one stretch of image was misleading: every
arc is judged on the whole corridor at once.

Arcs start at the rear axle, because that is the point a bicycle-model
vehicle turns about, and are expressed in the camera's ground frame that the
grid and Stanley use. No packaged library scores arcs on a custom
grid, so this part is written for this stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from offroad_autonomy.planning.bev_grid import TraversabilityGrid
from offroad_autonomy.planning.grid_config import GridPlannerConfig
from offroad_autonomy.planning.trajectory_continuity import trajectory_distance, transform_path


@dataclass
class ArcChoice:
    curvature: float
    forward_m: np.ndarray
    right_m: np.ndarray
    length_m: float
    min_clearance_m: float
    score: float
    continuity_distance_m: float = 0.0
    #: Mean distance from the arc to ground seen as not road, before scoring.
    centring_m: float = 0.0


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
        self._previous_path: np.ndarray | None = None
        self._previous_pose = None
        self.branch_count = 0
        self.branch_locked = False

    def reset(self) -> None:
        self._previous = None
        self._previous_path = None
        self._previous_pose = None
        self.branch_count = 0
        self.branch_locked = False

    def choose(self, grid: TraversabilityGrid, pose=None) -> ArcChoice | None:
        cfg = self.config
        clearance = grid.clearance_m()
        road = grid.road()
        rows, cols = grid.to_cell(self._forward, self._right)
        inside = (rows >= 0) & (rows < grid.rows) & (cols >= 0) & (cols < grid.cols)
        r = np.clip(rows, 0, grid.rows - 1)
        c = np.clip(cols, 0, grid.cols - 1)
        arc_clearance = np.where(inside, clearance[r, c], 0.0)
        arc_road = inside & road[r, c]
        # Farther than the grid's half width, an obstacle is out of the grid,
        # so clearance beyond it says nothing about centring.
        arc_centring = np.minimum(arc_clearance, grid.half_width_m)
        ahead = self._forward >= cfg.start_m

        candidates = []
        reference = self._previous_path
        if reference is not None:
            if pose is not None and self._previous_pose is not None:
                reference = transform_path(reference, self._previous_pose, pose)
            else:
                # Without registration, do not pretend old ego-frame points
                # are still where the vehicle saw them.
                reference = None
                self.branch_locked = False
        for i, curvature in enumerate(self.curvatures):
            candidate = self._score(
                i,
                float(curvature),
                arc_clearance[i],
                arc_road[i],
                arc_centring[i],
                ahead[i],
                inside[i],
            )
            if candidate is not None and reference is not None:
                candidate.continuity_distance_m = trajectory_distance(
                    np.column_stack((candidate.forward_m, candidate.right_m)), reference
                )
            if candidate is not None:
                candidates.append(candidate)
        # Centring and separation from the previous path are both measured in
        # units of the corridor, the best arc's clearance this frame, so they
        # weigh the same on a 3 m trail as on a 14 m road. In metres, a 4 m
        # clearance cap stopped centring on the wide road, and the separation
        # penalty held a straight arc into the outside of its bend.
        corridor = max((candidate.centring_m for candidate in candidates), default=0.0)
        if corridor > 0.0:
            for candidate in candidates:
                candidate.score += cfg.weight_clearance * candidate.centring_m / corridor
                candidate.score -= (
                    cfg.weight_trajectory_distance * candidate.continuity_distance_m / corridor
                )
        candidates = self._keep_branch(candidates, grid, road, clearance, reference)
        best = max(candidates, key=lambda candidate: candidate.score, default=None)
        if best is not None:
            self._previous = best.curvature
            self._previous_path = np.column_stack((best.forward_m, best.right_m))
            self._previous_pose = pose
        else:
            self.reset()
        return best

    def _split_options(self, candidates, grid, road, clearance):
        """Find distinct vehicle-width corridors reached by valid arcs.

        A fan of arcs on one wide road is one option, not dozens of branches.
        Arcs ending in the common trunk before the split cannot vote for a branch.
        """
        if len(candidates) < 2:
            return None
        cfg = self.config
        distances = np.arange(cfg.start_m, cfg.lookahead_m, cfg.step_m)
        rows, _ = grid.to_cell(distances, np.zeros_like(distances))
        in_grid = (rows >= 0) & (rows < grid.rows)
        distances = distances[in_grid]
        rows = rows[in_grid]
        # Unseen ground (especially the hood exclusion) is not a divider.
        # Only a known obstacle/boundary can separate two road branches.
        safe = clearance[rows] >= self._required_clearance
        starts = safe.copy()
        starts[:, 1:] &= ~safe[:, :-1]
        split = np.flatnonzero(np.count_nonzero(starts, axis=1) >= 2)
        if len(split) == 0:
            return None

        # Where each arc is sideways at every distance with two corridors;
        # NaN where the arc does not reach that far.
        lateral = np.full((len(candidates), len(split)), np.nan)
        for j, candidate in enumerate(candidates):
            forward, indices = np.unique(candidate.forward_m, return_index=True)
            within = (distances[split] >= forward[0]) & (distances[split] <= forward[-1])
            lateral[j, within] = np.interp(
                distances[split][within], forward, candidate.right_m[indices]
            )

        best = None
        most = 1
        for n, i in enumerate(split):
            labels = np.where(safe[i], np.cumsum(starts[i]), 0)
            reached = np.flatnonzero(np.isfinite(lateral[:, n]))
            _, cols = grid.to_cell(distances[i], lateral[reached, n])
            groups = {}
            for j, col in zip(reached, cols):
                if 0 <= col < grid.cols and labels[col] != 0 and road[rows[i], col]:
                    groups.setdefault(int(labels[col]), []).append(candidates[j])
            if len(groups) > most:
                most = len(groups)
                best = float(distances[i]), labels, groups
        return best

    def _keep_branch(self, candidates, grid, road, clearance, reference):
        split = self._split_options(candidates, grid, road, clearance)
        if split is None:
            self.branch_count = int(bool(candidates))
            self.branch_locked = False
            return candidates

        distance, labels, groups = split
        self.branch_count = len(groups)
        selected = None
        if self.branch_locked and reference is not None:
            forward, indices = np.unique(reference[:, 0], return_index=True)
            if forward[0] <= distance <= forward[-1]:
                lateral = np.interp(distance, forward, reference[indices, 1])
                _, col = grid.to_cell(np.asarray(distance), np.asarray(lateral))
                if 0 <= col < grid.cols:
                    # Match corridor membership, not its changing list index or
                    # score. Losing another branch must not change our branch.
                    selected = groups.get(int(labels[col]))

        if selected is None:
            # First split (or the chosen branch no longer has a safe candidate):
            # choose the closest branch before considering arc-quality scores.
            target = reference
            if target is None or np.max(target[:, 0]) <= self.config.start_m:
                target = np.array([[0.0, 0.0], [self.config.lookahead_m, 0.0]])

            def nearest(group):
                separation = min(
                    trajectory_distance(
                        np.column_stack((candidate.forward_m, candidate.right_m)), target
                    )
                    for candidate in group
                )
                return separation, -max(candidate.score for candidate in group)

            selected = min(groups.values(), key=nearest)
        self.branch_locked = True
        return selected

    def _score(
        self,
        index: int,
        curvature: float,
        clearance: np.ndarray,
        road: np.ndarray,
        centring: np.ndarray,
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
        reach = min(length / cfg.reach_m, 1.0)
        # Centring is averaged over the same reach_m for every arc, with
        # ground an arc does not reach counting as no clearance; averaged over
        # its own length, an arc ending before a narrow stretch would look
        # more central than one that drives through it.
        horizon = int(round(cfg.reach_m / cfg.step_m)) + 1
        near = centring[first : min(first + horizon, last + 1)]
        bend = abs(curvature) / self._max_curvature
        change = 0.0
        if self._previous is not None:
            change = abs(curvature - self._previous) / (2.0 * self._max_curvature)
        score = (
            cfg.weight_length * reach
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
            centring_m=float(near.sum()) / horizon,
        )
