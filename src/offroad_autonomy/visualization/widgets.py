"""Drawing primitives shared by the dashboard panels."""

from __future__ import annotations

import math

import cv2
import numpy as np

from offroad_autonomy.visualization.layout import COVER_CROP_ANCHOR, Rect
from offroad_autonomy.visualization.text import BODY_STRONG, TextRenderer, TextStyle

Color = tuple[int, int, int]

#: Scale on each axis and crop offsets from ``blit_cover``.
CoverTransform = tuple[float, float, int, int]


def fill_rect(canvas: np.ndarray, rect: Rect, color: Color) -> None:
    cv2.rectangle(canvas, (rect.x, rect.y), (rect.right - 1, rect.bottom - 1), color, -1)


def outline_rect(canvas: np.ndarray, rect: Rect, color: Color, thickness: int = 1) -> None:
    cv2.rectangle(canvas, (rect.x, rect.y), (rect.right - 1, rect.bottom - 1), color, thickness)


def blend_rect(canvas: np.ndarray, rect: Rect, color: Color, alpha: float) -> None:
    roi = canvas[rect.y : rect.bottom, rect.x : rect.right]
    if roi.size == 0:
        return
    cv2.addWeighted(np.full_like(roi, color), alpha, roi, 1.0 - alpha, 0, roi)


def blit_cover(canvas: np.ndarray, image: np.ndarray, rect: Rect) -> CoverTransform:
    """Fills ``rect`` with the image scaled to cover it, cropping the overflow.

    Returns the transform, so points in the source can be mapped onto the
    canvas with ``cover_point``.
    """
    src_h, src_w = image.shape[:2]
    scale = max(rect.w / src_w, rect.h / src_h)
    new_w = max(rect.w, int(round(src_w * scale)))
    new_h = max(rect.h, int(round(src_h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    off_x = (new_w - rect.w) // 2
    off_y = int(round((new_h - rect.h) * COVER_CROP_ANCHOR))
    canvas[rect.y : rect.bottom, rect.x : rect.right] = resized[
        off_y : off_y + rect.h, off_x : off_x + rect.w
    ]
    return new_w / src_w, new_h / src_h, off_x - rect.x, off_y - rect.y


def cover_point(transform: CoverTransform, point: tuple[int, int]) -> tuple[int, int]:
    scale_x, scale_y, off_x, off_y = transform
    return int(round(point[0] * scale_x)) - off_x, int(round(point[1] * scale_y)) - off_y


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


def draw_autonomy_chip(
    canvas: np.ndarray,
    text_renderer: TextRenderer,
    colors: dict[str, Color],
    header: Rect,
    inset: int,
    active: bool,
) -> None:
    """The header's autonomy state, solid red under safe stop so it reads at a glance."""
    text, color, solid = "AUTONOMY ACTIVE", colors["GOOD"], False
    if not active:
        text, color, solid = "SAFE STOP | MANUAL CONTROL", colors["BAD"], True
    draw_chip(
        canvas,
        text_renderer,
        header.right - inset,
        header.y + (header.h - chip_height(text_renderer)) // 2,
        text,
        color,
        colors["BG"],
        solid=solid,
        align="right",
    )


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


def as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def as_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """A boolean mask at ``shape``, resized without blending class edges."""
    if mask.shape[:2] == shape and mask.dtype == bool:
        return mask
    mask_u8 = mask.astype(np.uint8)
    if mask_u8.shape[:2] != shape:
        mask_u8 = cv2.resize(mask_u8, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask_u8.astype(bool)


def smooth_points(points: np.ndarray, samples: int = 120) -> np.ndarray:
    if len(points) < 2:
        return points
    diffs = np.diff(points, axis=0)
    lengths = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(arc[-1])
    if total <= 1e-6:
        return points
    q = np.linspace(0.0, total, max(samples, len(points)))
    xs = np.interp(q, arc, points[:, 0])
    ys = np.interp(q, arc, points[:, 1])
    return np.stack([xs, ys], axis=1)


def fit_image(
    image: np.ndarray,
    target_w: int,
    target_h: int,
    fill: Color,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """The whole image scaled to fit and centred, with ``fill`` around it."""
    src_h, src_w = image.shape[:2]
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    out = np.full((target_h, target_w, 3), fill, dtype=np.uint8)
    off_x = (target_w - new_w) // 2
    off_y = (target_h - new_h) // 2
    out[off_y : off_y + new_h, off_x : off_x + new_w] = resized
    return out
