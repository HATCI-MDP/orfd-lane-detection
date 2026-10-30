"""Daylight percentage to BeamNG time of day.

The team names its lighting conditions by daylight percentage (Noon 100,
Evening 74, Night 0). BeamNG's own value is a fraction of the day starting at
noon (0 is 12:00, 0.25 is 18:00, 0.5 is midnight), so 0-1 on the command line
would be easy to confuse with it. The percentage walks the afternoon from
noon to midnight in equal clock steps: the team's Evening is a dusk setting,
and the morning half would light the trail from the opposite side.
"""

from __future__ import annotations

SECONDS_PER_DAY = 86400.0


def validate_daylight(percent: float) -> float:
    value = float(percent)
    if not 0.0 <= value <= 100.0:
        raise ValueError(f"daylight must be between 0 and 100 percent, got {percent}")
    return value


def daylight_to_tod(percent: float) -> float:
    return 0.5 * (1.0 - validate_daylight(percent) / 100.0)


def tod_to_clock(tod: float) -> str:
    seconds = int(round(((tod + 0.5) % 1.0) * SECONDS_PER_DAY)) % int(SECONDS_PER_DAY)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def tod_distance(a: float, b: float) -> float:
    # 0 and 1 are both noon, so distance is measured around the circle.
    return abs((a - b + 0.5) % 1.0 - 0.5)
