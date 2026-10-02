"""Text for the dashboard, drawn with bundled fonts through Pillow.

OpenCV's own text is limited to its stroke fonts. Each distinct string is
rasterised once into an alpha mask and blended onto the canvas in the colour
asked for, so a repeated value costs a small array blend, not a Pillow call.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = Path(__file__).resolve().parent / "fonts"

SANS = "IBMPlexSans-Regular.ttf"
SANS_SEMIBOLD = "IBMPlexSans-SemiBold.ttf"
MONO = "IBMPlexMono-Medium.ttf"

#: Rasterised strings kept before the cache is dropped. Live values come from a
#: small set (a speed has a few hundred distinct readings), so the cache is
#: cleared whole rather than tracking recency.
_CACHE_LIMIT = 1024


class TextStyle(NamedTuple):
    font: str
    size: int


DISPLAY = TextStyle(MONO, 80)
HERO = TextStyle(MONO, 48)
HERO_SANS = TextStyle(SANS_SEMIBOLD, 48)
LARGE = TextStyle(MONO, 30)
LARGE_SANS = TextStyle(SANS_SEMIBOLD, 30)
TITLE = TextStyle(SANS_SEMIBOLD, 22)
VALUE = TextStyle(MONO, 20)
BODY = TextStyle(SANS, 16)
BODY_STRONG = TextStyle(SANS_SEMIBOLD, 16)
LABEL = TextStyle(SANS_SEMIBOLD, 13)
CAPTION = TextStyle(SANS, 13)
CAPTION_MONO = TextStyle(MONO, 13)


class _Run(NamedTuple):
    alpha: np.ndarray
    ascent: int


class TextRenderer:
    def __init__(self, font_dir: Path = FONT_DIR) -> None:
        self._font_dir = font_dir
        self._fonts: dict[TextStyle, ImageFont.FreeTypeFont] = {}
        self._runs: dict[tuple[str, TextStyle], _Run] = {}

    def _font(self, style: TextStyle) -> ImageFont.FreeTypeFont:
        font = self._fonts.get(style)
        if font is None:
            path = self._font_dir / style.font
            # A missing face must stop startup: a silent fallback would change
            # every width the layout was built around.
            if not path.is_file():
                raise FileNotFoundError(f"Dashboard font is missing: {path}")
            font = ImageFont.truetype(str(path), style.size)
            self._fonts[style] = font
        return font

    def preload(self, styles: tuple[TextStyle, ...]) -> None:
        for style in styles:
            self._font(style)

    def _run(self, text: str, style: TextStyle) -> _Run:
        key = (text, style)
        run = self._runs.get(key)
        if run is None:
            if len(self._runs) >= _CACHE_LIMIT:
                self._runs.clear()
            font = self._font(style)
            ascent, descent = font.getmetrics()
            width = max(1, math.ceil(font.getlength(text)) + 2)
            image = Image.new("L", (width, ascent + descent), 0)
            ImageDraw.Draw(image).text((0, ascent), text, fill=255, font=font, anchor="ls")
            run = _Run(np.asarray(image), ascent)
            self._runs[key] = run
        return run

    def width(self, text: str, style: TextStyle) -> int:
        return math.ceil(self._font(style).getlength(text))

    def cap_height(self, style: TextStyle) -> int:
        return -self._font(style).getbbox("H", anchor="ls")[1]

    def draw(
        self,
        canvas: np.ndarray,
        text: str,
        x: int,
        baseline: int,
        style: TextStyle,
        color: tuple[int, int, int],
        align: str = "left",
    ) -> int:
        """Draws ``text`` with its baseline at ``baseline`` and returns its width.

        ``x`` is the left edge, the right edge or the centre, per ``align``.
        """
        if not text:
            return 0
        run = self._run(text, style)
        width = run.alpha.shape[1]
        if align == "right":
            x -= width
        elif align == "center":
            x -= width // 2
        top = baseline - run.ascent
        _blend_alpha(canvas, run.alpha, x, top, color)
        return width


def _blend_alpha(
    canvas: np.ndarray, alpha: np.ndarray, x: int, y: int, color: tuple[int, int, int]
) -> None:
    height, width = alpha.shape
    x0 = max(x, 0)
    y0 = max(y, 0)
    x1 = min(x + width, canvas.shape[1])
    y1 = min(y + height, canvas.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    patch = alpha[y0 - y : y1 - y, x0 - x : x1 - x]
    roi = canvas[y0:y1, x0:x1]
    weight = patch[:, :, None].astype(np.uint16)
    color_arr = np.array(color, dtype=np.uint16)
    roi[:] = ((roi.astype(np.uint16) * (255 - weight) + color_arr * weight) // 255).astype(np.uint8)
