"""Tests for the dashboard's layout, text and drawing helpers."""

import numpy as np
import pytest

from offroad_autonomy.visualization import bands
from offroad_autonomy.visualization.layout import Rect, build_layout
from offroad_autonomy.visualization.text import BODY, DISPLAY, LABEL, LABEL_SPACED, TextRenderer
from offroad_autonomy.visualization.widgets import (
    draw_bar,
    draw_centered_bar,
    wrap_text,
)


@pytest.mark.parametrize(
    ("fps", "expected"),
    [(8.0, bands.GOOD), (7.0, bands.GOOD), (6.9, bands.WARN), (5.7, bands.WARN), (5.5, bands.BAD)],
)
def test_fps_bands_around_the_target(fps, expected):
    assert bands.fps_level(fps, 7.0, 0.8) == expected


def test_latency_is_good_up_to_the_budget_and_bad_after():
    assert bands.latency_level(143.0, 143.0) == bands.GOOD
    assert bands.latency_level(143.1, 143.0) == bands.BAD


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [(0.17, bands.BAD), (0.18, bands.WARN), (0.49, bands.WARN), (0.5, bands.GOOD)],
)
def test_confidence_bands_follow_the_gate_floor(confidence, expected):
    assert bands.confidence_level(confidence, 0.18, 0.5) == expected


def test_road_fraction_below_the_floor_is_bad():
    assert bands.road_level(0.014, 0.015) == bands.BAD
    assert bands.road_level(0.015, 0.015) == bands.GOOD


def test_layout_regions_fit_the_canvas_and_do_not_overlap():
    layout = build_layout(1600, 900)
    regions = [
        layout.header,
        layout.viewport,
        layout.runtime,
        layout.vehicle,
        layout.perception,
    ]

    for rect in regions:
        assert rect.x >= 0 and rect.y >= 0
        assert rect.right <= layout.width and rect.bottom <= layout.height
    for index, a in enumerate(regions):
        for b in regions[index + 1 :]:
            separate = a.right <= b.x or b.right <= a.x or a.bottom <= b.y or b.bottom <= a.y
            assert separate, (a, b)


def test_runtime_tiles_tile_the_runtime_strip():
    layout = build_layout(1600, 900)

    assert layout.autonomy_fps.x == layout.runtime.x
    assert layout.dashboard_fps.right == layout.runtime.right
    assert layout.autonomy_fps.right < layout.latency.x < layout.latency.right
    assert layout.latency.right < layout.dashboard_fps.x


def test_missing_font_names_the_file(tmp_path):
    renderer = TextRenderer(font_dir=tmp_path)

    with pytest.raises(FileNotFoundError, match="IBMPlexSans-Regular.ttf"):
        renderer.preload((BODY,))


def test_text_draws_in_the_requested_colour_and_clips_at_the_edge():
    renderer = TextRenderer()
    canvas = np.zeros((60, 120, 3), dtype=np.uint8)

    width = renderer.draw(canvas, "9.4", 5, 50, DISPLAY, (10, 20, 30))
    renderer.draw(canvas, "9.4", 100, 50, DISPLAY, (10, 20, 30))
    renderer.draw(canvas, "off", -500, 50, BODY, (10, 20, 30))

    assert width > 0
    assert (canvas == np.array([10, 20, 30])).all(axis=2).any()
    assert canvas.max() == 30


def test_repeated_text_is_rasterised_once():
    renderer = TextRenderer()
    canvas = np.zeros((60, 120, 3), dtype=np.uint8)

    renderer.draw(canvas, "8.6", 0, 40, BODY, (255, 255, 255))
    first = renderer._runs[("8.6", BODY)]
    renderer.draw(canvas, "8.6", 10, 40, BODY, (255, 0, 0))

    assert renderer._runs[("8.6", BODY)] is first


def test_wrap_text_keeps_every_line_within_the_width():
    renderer = TextRenderer()
    text = "Planner missed three frames and is holding the last accepted path"

    lines = wrap_text(renderer, text, BODY, 200)

    assert len(lines) > 1
    assert " ".join(lines) == text
    assert all(renderer.width(line, BODY) <= 200 for line in lines)


def test_wrap_text_keeps_an_overlong_word_on_its_own_line():
    renderer = TextRenderer()

    lines = wrap_text(renderer, "ok " + "x" * 80, BODY, 100)

    assert lines[0] == "ok"
    assert lines[1] == "x" * 80


def test_bar_fill_and_tick_positions():
    canvas = np.zeros((20, 100, 3), dtype=np.uint8)
    rect = Rect(0, 5, 100, 10)

    draw_bar(canvas, rect, 0.5, (0, 255, 0), (40, 40, 40), tick=0.8, tick_color=(255, 255, 255))

    assert tuple(canvas[10, 10]) == (0, 255, 0)
    assert tuple(canvas[10, 70]) == (40, 40, 40)
    assert tuple(canvas[10, 80]) == (255, 255, 255)


def test_bar_ignores_non_finite_values():
    canvas = np.zeros((20, 100, 3), dtype=np.uint8)

    draw_bar(canvas, Rect(0, 5, 100, 10), float("nan"), (0, 255, 0), (40, 40, 40))

    assert tuple(canvas[10, 50]) == (40, 40, 40)


def test_centered_bar_fills_outwards_from_the_middle():
    canvas = np.zeros((20, 100, 3), dtype=np.uint8)
    rect = Rect(0, 5, 100, 10)

    draw_centered_bar(canvas, rect, -0.5, (255, 255, 255), (40, 40, 40), (200, 200, 200))

    assert tuple(canvas[10, 35]) == (255, 255, 255)
    assert tuple(canvas[10, 70]) == (40, 40, 40)


def test_letter_spacing_widens_the_run_by_the_gaps():
    text = TextRenderer()
    word = "AUTONOMY"

    plain = text.width(word, LABEL)
    spaced = text.width(word, LABEL_SPACED)

    assert spaced - plain == pytest.approx(LABEL_SPACED.tracking * (len(word) - 1), abs=1)


def test_spaced_text_draws_inside_its_measured_width():
    text = TextRenderer()
    canvas = np.zeros((40, 200, 3), dtype=np.uint8)

    drawn = text.draw(canvas, "ORBIT", 10, 30, LABEL_SPACED, (255, 255, 255))

    columns = np.flatnonzero(canvas.any(axis=(0, 2)))
    assert columns.min() >= 10
    assert columns.max() < 10 + drawn
