"""Drawing primitives shared by the dashboard panels."""

from __future__ import annotations

import math

import cv2
import numpy as np

from offroad_autonomy.visualization.layout import Rect
from offroad_autonomy.visualization.text import BODY_STRONG, TextRenderer, TextStyle

Color = tuple[int, int, int]


def fill_rect(canvas: np.ndarray, rect: Rect, color: Color) -> None:
    cv2.rectangle(canvas, (rect.x, rect.y), (rect.right - 1, rect.bottom - 1), color, -1)


def outline_rect(canvas: np.ndarray, rect: Rect, color: Color, thickness: int = 1) -> None:
    cv2.rectangle(canvas, (rect.x, rect.y), (rect.right - 1, rect.bottom - 1), color, thickness)


def blend_rect(canvas: np.ndarray, rect: Rect, color: Color, alpha: float) -> None:
    roi = canvas[rect.y : rect.bottom, rect.x : rect.right]
    if roi.size == 0:
        return
    cv2.addWeighted(np.full_like(roi, color), alpha, roi, 1.0 - alpha, 0, roi)


def draw_bar(
    canvas: np.ndarray,
    rect: Rect,
    fraction: float,
    color: Color,
    track: Color,
    tick: float | None = None,
    tick_color: Color = (255, 255, 255),
) -> None:
    """A fill bar. ``tick`` is a position as a fraction of the bar's width."""
    fraction = _clamp01(fraction)
    fill_rect(canvas, rect, track)
    fill_w = int(round(rect.w * fraction))
    if fill_w > 0:
        fill_rect(canvas, Rect(rect.x, rect.y, fill_w, rect.h), color)
    if tick is not None:
        tick_x = rect.x + int(round(rect.w * _clamp01(tick)))
        fill_rect(canvas, Rect(tick_x - 1, rect.y - 4, 2, rect.h + 8), tick_color)


def draw_centered_bar(
    canvas: np.ndarray,
    rect: Rect,
    value: float,
    color: Color,
    track: Color,
    mark: Color,
) -> None:
    """A bar from -1 to +1 that fills outwards from its centre tick."""
    if not math.isfinite(value):
        value = 0.0
    value = max(-1.0, min(1.0, value))
    fill_rect(canvas, rect, track)
    half = rect.w // 2
    fill_w = int(round(half * abs(value)))
    if value >= 0.0:
        fill_rect(canvas, Rect(rect.x + half, rect.y, fill_w, rect.h), color)
    else:
        fill_rect(canvas, Rect(rect.x + half - fill_w, rect.y, fill_w, rect.h), color)
    fill_rect(canvas, Rect(rect.x + half - 1, rect.y - 4, 2, rect.h + 8), mark)


def chip_width(text_renderer: TextRenderer, text: str, style: TextStyle = BODY_STRONG) -> int:
    return text_renderer.width(text, style) + 28


def chip_height(text_renderer: TextRenderer, style: TextStyle = BODY_STRONG) -> int:
    return text_renderer.cap_height(style) + 20


def draw_chip(
    canvas: np.ndarray,
    text_renderer: TextRenderer,
    x: int,
    y: int,
    text: str,
    color: Color,
    ink: Color,
    solid: bool = False,
    style: TextStyle = BODY_STRONG,
    align: str = "left",
) -> Rect:
    """A labelled box: outlined in ``color``, or filled with ``ink`` text.

    ``y`` is the top edge. ``x`` is the left or right edge per ``align``.
    """
    cap = text_renderer.cap_height(style)
    w = chip_width(text_renderer, text, style)
    h = chip_height(text_renderer, style)
    if align == "right":
        x -= w
    rect = Rect(x, y, w, h)
    if solid:
        fill_rect(canvas, rect, color)
        text_color = ink
    else:
        outline_rect(canvas, rect, color, 2)
        text_color = color
    text_renderer.draw(canvas, text, x + 14, y + 10 + cap, style, text_color)
    return rect


def wrap_text(
    text_renderer: TextRenderer, text: str, style: TextStyle, max_width: int
) -> list[str]:
    """Greedy word wrap. A word wider than the line stands on its own line."""
    lines: list[str] = []
    line = ""
    for word in text.split():
        candidate = word
        if line:
            candidate = f"{line} {word}"
        if line and text_renderer.width(candidate, style) > max_width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    return lines


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))
