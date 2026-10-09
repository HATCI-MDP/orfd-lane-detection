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


def gauge_level(value: float, thresholds: tuple[float, ...]) -> str | None:
    """A stack gauge's level: below the first threshold is bad, below the
    second a warning. ``None`` for a gauge with no thresholds, which is a
    plain reading rather than a health measure."""
    if not thresholds:
        return None
    if value < thresholds[0]:
        return BAD
    if len(thresholds) > 1 and value < thresholds[1]:
        return WARN
    return GOOD


def fps_note(level: str, target_fps: float) -> str:
    if level == GOOD:
        return "On Target"
    if level == WARN:
        return f"Slightly Below {target_fps:g} Fps Target"
    return f"Below {target_fps:g} Fps Target"


def latency_note(level: str, budget_ms: float) -> str:
    if level == GOOD:
        return f"Within {budget_ms:.0f} Ms Budget"
    return f"Over {budget_ms:.0f} Ms Budget"
