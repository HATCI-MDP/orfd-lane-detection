"""Tests for the recording-only picture-in-picture compositing."""

import numpy as np

from offroad_autonomy.visualization.pip_overlay import composite_pip


def test_inset_lands_in_the_bottom_right_corner():
    canvas = np.zeros((200, 400, 3), dtype=np.uint8)
    inset = np.full((60, 80, 3), 255, dtype=np.uint8)

    result = composite_pip(canvas, inset)

    assert result is canvas
    assert canvas[-30, -30].tolist() == [255, 255, 255]
    assert canvas[5, 5].tolist() == [0, 0, 0]


def test_missing_inset_leaves_canvas_untouched():
    canvas = np.zeros((200, 400, 3), dtype=np.uint8)

    result = composite_pip(canvas, None)

    assert result is canvas
    assert not canvas.any()


def test_oversized_inset_is_skipped_rather_than_crashing():
    # A very short, wide canvas: the inset's scaled height, following its
    # own tall aspect ratio, ends up taller than the canvas itself.
    canvas = np.zeros((5, 400, 3), dtype=np.uint8)
    inset = np.full((800, 600, 3), 255, dtype=np.uint8)

    result = composite_pip(canvas, inset)

    assert result is canvas
    assert not canvas.any()
