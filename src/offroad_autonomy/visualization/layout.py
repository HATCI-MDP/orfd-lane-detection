"""Where every dashboard region sits on the canvas.

Panels never move between frames or states, so the rectangles are computed once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

PAD = 24
GAP = 16
HEADER_H = 56
SIDEBAR_W = 400
RUNTIME_H = 112
RUNTIME_GAP = 14
CARD_GAP = 14
VEHICLE_CARD_H = 300
DASHBOARD_TILE_W = 230

#: Share of the rows lost to the cover fit that come off the top. Even, because
#: the far end of the planned path and the hood both carry information and
#: cropping only one side would clip one of them.
COVER_CROP_ANCHOR = 0.5


class Rect(NamedTuple):
    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h


@dataclass(frozen=True)
class Layout:
    width: int
    height: int
    header: Rect
    viewport: Rect
    runtime: Rect
    vehicle: Rect
    perception: Rect
    autonomy_fps: Rect
    latency: Rect
    dashboard_fps: Rect


def build_layout(width: int, height: int) -> Layout:
    header = Rect(PAD, PAD, width - 2 * PAD, HEADER_H)
    content_top = header.bottom + GAP
    content_h = height - PAD - content_top
    main_w = width - 2 * PAD - GAP - SIDEBAR_W

    runtime = Rect(PAD, height - PAD - RUNTIME_H, main_w, RUNTIME_H)
    viewport = Rect(PAD, content_top, main_w, runtime.y - RUNTIME_GAP - content_top)

    sidebar_x = PAD + main_w + GAP
    vehicle = Rect(sidebar_x, content_top, SIDEBAR_W, VEHICLE_CARD_H)
    perception = Rect(
        sidebar_x,
        vehicle.bottom + CARD_GAP,
        SIDEBAR_W,
        content_h - VEHICLE_CARD_H - CARD_GAP,
    )

    tile_w = (main_w - 2 * RUNTIME_GAP - DASHBOARD_TILE_W) // 2
    autonomy_fps = Rect(runtime.x, runtime.y, tile_w, RUNTIME_H)
    latency = Rect(autonomy_fps.right + RUNTIME_GAP, runtime.y, tile_w, RUNTIME_H)
    dashboard_fps = Rect(
        latency.right + RUNTIME_GAP,
        runtime.y,
        runtime.right - latency.right - RUNTIME_GAP,
        RUNTIME_H,
    )
    return Layout(
        width=width,
        height=height,
        header=header,
        viewport=viewport,
        runtime=runtime,
        vehicle=vehicle,
        perception=perception,
        autonomy_fps=autonomy_fps,
        latency=latency,
        dashboard_fps=dashboard_fps,
    )
