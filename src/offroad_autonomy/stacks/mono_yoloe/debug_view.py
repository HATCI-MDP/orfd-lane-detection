"""The mono_yoloe pipeline page: what each stage saw, in one 2 x 2 grid.

Drawn by the stack because only it knows its stages, but only when the
dashboard calls for the page, and on the dashboard's thread.
"""

from __future__ import annotations

import cv2
import numpy as np

from offroad_autonomy.types import PipelineStepResult
from offroad_autonomy.visualization.widgets import as_bgr, as_mask

_TITLES = (
    ("1 Raw RGB", 0, 0),
    ("2 Segmentation (Model)", 1, 0),
    ("3 Planner Input", 0, 1),
    ("4 Trajectory + Control", 1, 1),
)


def _level_color(
    value: float,
    warn: float,
    bad: float,
    ok: tuple[int, int, int],
) -> tuple[int, int, int]:
    if value > bad:
        return (0, 0, 255)
    if value > warn:
        return (0, 165, 255)
    return ok


def _tint(image: np.ndarray, mask: np.ndarray, color, alpha: float = 0.45) -> np.ndarray:
    out = image.copy()
    mask = as_mask(mask, image.shape[:2])
    out[mask] = (out[mask] * (1.0 - alpha) + np.array(color) * alpha).astype(np.uint8)
    return out


def _title(grid: np.ndarray, text: str, x: int, y: int) -> None:
    # Large and outlined: the grid is scaled down roughly by half to fit the
    # viewport, and thin text vanished at that size.
    origin = (x + 12, y + 34)
    cv2.putText(grid, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(grid, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)


def pipeline_grid(result: PipelineStepResult) -> np.ndarray:
    """Uses ``result.plan``, which is kept even under safe stop, so this
    page always shows what the planner decided."""
    frame = result.frame.preprocessed
    h, w = frame.shape[:2]
    raw = cv2.resize(as_bgr(result.frame.raw), (w, h), interpolation=cv2.INTER_AREA)
    plan = result.plan

    seg = _tint(frame, result.perception.mask, (0, 200, 255))

    processed = (frame * 0.4).astype(np.uint8)
    processed = _tint(processed, result.stabilized.mask, (90, 90, 90), 0.6)
    if plan.planner_mask is not None:
        processed = _tint(processed, plan.planner_mask, (0, 255, 0), 0.7)
    cv2.line(processed, (0, plan.roi_top), (w - 1, plan.roi_top), (255, 255, 0), 1)

    planner_mask = plan.planner_mask
    if planner_mask is None:
        planner_mask = np.zeros((h, w), bool)
    traj = _tint(frame, planner_mask, (0, 255, 0), 0.25)
    path_color = (255, 200, 0)
    if plan.fallback_active:
        path_color = (0, 165, 255)
    if len(plan.centerline) >= 2:
        pts = np.round(plan.centerline).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(traj, [pts], False, path_color, 4, cv2.LINE_AA)
    cv2.line(traj, (w // 2, h - 1), (w // 2, h - 30), (255, 255, 255), 2)
    dbg = result.command.debug
    if dbg is not None:
        edge_color = _level_color(dbg.edge_risk, warn=0.0, bad=0.66, ok=(0, 255, 255))
        for (ul, vl), (ur, vr) in dbg.boundary_px:
            for u, v in ((ul, vl), (ur, vr)):
                if 3 < u < w - 4:
                    cv2.drawMarker(
                        traj, (int(u), int(v)), edge_color, cv2.MARKER_TILTED_CROSS, 9, 2
                    )
        if dbg.lookahead_px is not None:
            lx, ly = (int(round(v)) for v in dbg.lookahead_px)
            cv2.circle(traj, (lx, ly), 9, (0, 0, 0), 3, cv2.LINE_AA)
            marker = _level_color(dbg.saturation, warn=0.5, bad=0.85, ok=(255, 0, 255))
            cv2.circle(traj, (lx, ly), 9, marker, 2, cv2.LINE_AA)
            cv2.line(traj, (w // 2, h - 1), (lx, ly), (255, 0, 255), 1, cv2.LINE_AA)

    grid = np.vstack([np.hstack([raw, seg]), np.hstack([processed, traj])])
    cv2.line(grid, (w, 0), (w, 2 * h), (255, 255, 255), 2)
    cv2.line(grid, (0, h), (2 * w, h), (255, 255, 255), 2)
    for text, column, row in _TITLES:
        _title(grid, text, column * w, row * h)
    return grid
