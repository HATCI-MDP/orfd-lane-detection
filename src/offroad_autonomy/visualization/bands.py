"""Health levels for the dashboard readouts.

Each level is also the name of the colour that paints it.
"""

from __future__ import annotations

GOOD = "GOOD"
WARN = "WARN"
BAD = "BAD"


def fps_level(fps: float, target_fps: float, warn_fraction: float) -> str:
    if fps >= target_fps:
        return GOOD
    if fps >= target_fps * warn_fraction:
        return WARN
    return BAD


def latency_level(p95_ms: float, budget_ms: float) -> str:
    if p95_ms <= budget_ms:
        return GOOD
    return BAD


def confidence_level(confidence: float, floor: float, good: float) -> str:
    if confidence < floor:
        return BAD
    if confidence < good:
        return WARN
    return GOOD


def road_level(road_fraction: float, floor: float) -> str:
    if road_fraction < floor:
        return BAD
    return GOOD
