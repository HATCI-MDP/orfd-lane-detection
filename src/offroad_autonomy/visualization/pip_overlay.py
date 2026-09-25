"""Picture-in-picture compositing for the recorded dashboard video.

The orbit camera exists only to be seen in the corner of a recording; it is
composited onto the already-rendered dashboard canvas rather than drawn by
``AutonomyDashboard`` itself, so recording touches neither perception nor
the main dashboard layout.
"""

from __future__ import annotations

import cv2
import numpy as np

_SCALE = 0.28
_MARGIN_PX = 16
_BORDER_PX = 2
_BORDER_COLOR = (255, 255, 255)


def composite_pip(canvas: np.ndarray, inset: np.ndarray | None) -> np.ndarray:
    """Pastes ``inset`` into the bottom-right corner of ``canvas``.

    ``inset`` is scaled to a fixed fraction of the canvas width with its
    aspect ratio preserved. Modifies ``canvas`` in place and returns it; a
    missing inset, or one that would not fit, leaves it untouched.
    """
    if inset is None or inset.size == 0:
        return canvas

    canvas_h, canvas_w = canvas.shape[:2]
    inset_w = max(1, int(canvas_w * _SCALE))
    inset_h = max(1, int(inset_w * inset.shape[0] / inset.shape[1]))
    if inset_w >= canvas_w or inset_h >= canvas_h:
        return canvas

    resized = cv2.resize(inset, (inset_w, inset_h), interpolation=cv2.INTER_AREA)
    x0 = max(0, canvas_w - _MARGIN_PX - inset_w)
    y0 = max(0, canvas_h - _MARGIN_PX - inset_h)

    cv2.rectangle(
        canvas,
        (x0 - _BORDER_PX, y0 - _BORDER_PX),
        (x0 + inset_w + _BORDER_PX, y0 + inset_h + _BORDER_PX),
        _BORDER_COLOR,
        thickness=_BORDER_PX,
    )
    canvas[y0 : y0 + inset_h, x0 : x0 + inset_w] = resized
    return canvas
