"""Scoring a logged run against the simulator's authored road geometry.

Used only offline on evaluation logs (``--eval-log``), never by the driving
stack. Offsets are measured from the vehicle's reference node to the nearest
segment of an authored road's middle line, so they score the drive against
the road as built rather than against what perception believed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Centerline:
    """Every authored road's middle line as independent 2D segments."""

    start: np.ndarray  # (M, 2)
    end: np.ndarray  # (M, 2)
    half_width: np.ndarray  # (M,)


def _edge_points(road: dict, side: str) -> np.ndarray:
    return np.array([edge[side][:2] for edge in road.get("edges") or []], dtype=np.float64)


def roads_near(roads: dict, centre: tuple[float, float], radius_m: float) -> dict:
    """Keep roads with any middle point within ``radius_m`` of ``centre``."""
    kept = {}
    for road_id, road in roads.items():
        middle = _edge_points(road, "middle")
        if len(middle) < 2:
            continue
        if np.min(np.hypot(*(middle - np.asarray(centre)).T)) <= radius_m:
            kept[road_id] = road
    return kept


def build_centerline(roads: dict) -> Centerline:
    starts, ends, widths = [], [], []
    for road in roads.values():
        middle = _edge_points(road, "middle")
        if len(middle) < 2:
            continue
        half = np.hypot(*(_edge_points(road, "left") - _edge_points(road, "right")).T) / 2.0
        starts.append(middle[:-1])
        ends.append(middle[1:])
        widths.append((half[:-1] + half[1:]) / 2.0)
    if not starts:
        raise ValueError("No authored road with at least two points to score against")
    return Centerline(np.concatenate(starts), np.concatenate(ends), np.concatenate(widths))


def travel_headings(xy: np.ndarray) -> np.ndarray:
    """Unit direction of travel at each sample, from its neighbours."""
    heading = np.gradient(np.asarray(xy, dtype=np.float64), axis=0)
    norm = np.hypot(heading[:, 0], heading[:, 1])
    norm[norm == 0] = 1.0
    return heading / norm[:, None]


def lateral_offsets(
    xy: np.ndarray, centerline: Centerline, headings: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Signed perpendicular distance to the nearest centreline segment.

    Positive is left of the direction of travel. A road's nodes may run
    against the drive, so each segment is oriented by the vehicle's heading
    before the cross product sets the sign. Returns (offset, half_width).
    """
    a, b = centerline.start, centerline.end
    ab = b - a
    length_sq = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12)
    offsets = np.empty(len(xy))
    half_widths = np.empty(len(xy))
    for i, point in enumerate(np.asarray(xy, dtype=np.float64)):
        t = np.clip(np.einsum("ij,ij->i", point - a, ab) / length_sq, 0.0, 1.0)
        nearest = a + t[:, None] * ab
        distance = np.hypot(*(point - nearest).T)
        k = int(np.argmin(distance))
        direction = ab[k]
        if np.dot(direction, headings[i]) < 0:
            direction = -direction
        rel = point - nearest[k]
        cross = direction[0] * rel[1] - direction[1] * rel[0]
        sign = 1.0
        if cross < 0:
            sign = -1.0
        offsets[i] = sign * distance[k]
        half_widths[i] = centerline.half_width[k]
    return offsets, half_widths


def count_departures(offsets: np.ndarray, half_widths: np.ndarray) -> int:
    """Times the reference point crossed from on the road to off it."""
    outside = np.abs(offsets) > half_widths
    entries = outside[1:] & ~outside[:-1]
    return int(entries.sum() + bool(len(outside) and outside[0]))


@dataclass
class RejectionStats:
    evaluated: int
    rejected: int
    rate_pct: float | None
    longest_streak: int


def rejection_stats(accepted: list[bool | None]) -> RejectionStats:
    """Gate rejections over the frames the gate actually evaluated.

    ``None`` marks a frame the gate never saw (a stale capture). It neither
    counts nor breaks a streak, because the car was no better informed then.
    """
    evaluated = rejected = streak = longest = 0
    for flag in accepted:
        if flag is None:
            continue
        evaluated += 1
        if flag:
            streak = 0
            continue
        rejected += 1
        streak += 1
        longest = max(longest, streak)
    rate = None
    if evaluated:
        rate = 100.0 * rejected / evaluated
    return RejectionStats(evaluated, rejected, rate, longest)
