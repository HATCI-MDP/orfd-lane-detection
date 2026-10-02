"""Tests for the presentation video frame."""

from dataclasses import replace

import numpy as np
import pytest

from offroad_autonomy.main import parse_args
from offroad_autonomy.types import (
    DEFAULT_DASHBOARD_COLORS,
    ControlCommand,
    FramePacket,
    PathPlan,
    PerceptionResult,
    PipelineStepResult,
    StabilizedResult,
)
from offroad_autonomy.visualization.dashboard import DashboardTelemetry
from offroad_autonomy.visualization.layout import Rect
from offroad_autonomy.visualization.presentation import (
    DASHCAM_CARD,
    ORBIT_VIEW,
    STATS_CARD,
    PresentationRenderer,
)

COLORS = DEFAULT_DASHBOARD_COLORS

# The autonomy FPS block, the stats card's top-left metric.
FPS_BLOCK = Rect(STATS_CARD.x + 24, STATS_CARD.y + 134, 266, 102)


@pytest.fixture(scope="module")
def renderer() -> PresentationRenderer:
    # Shared: building the fonts and the static layer is the slow part.
    return PresentationRenderer()


def _result(plan: PathPlan | None = None) -> PipelineStepResult:
    """A flat grey frame, so any pixel in a palette colour was drawn on it."""
    frame = np.full((465, 720, 3), 90, dtype=np.uint8)
    mask = np.zeros((465, 720), dtype=bool)
    if plan is None:
        plan = PathPlan(centerline=np.array([[360.0, 440.0], [360.0, 300.0], [370.0, 200.0]]))
    return PipelineStepResult(
        frame=FramePacket(raw=frame, preprocessed=frame, timestamp=0.0, height=465, width=720),
        perception=PerceptionResult(mask=mask, confidences=[0.71]),
        stabilized=StabilizedResult(mask=mask),
        plan=plan,
        command=ControlCommand(),
    )


def _telemetry(**overrides) -> DashboardTelemetry:
    fields = {
        "speed_mph": 9.4,
        "steering": 0.14,
        "throttle": 0.34,
        "brake": 0.0,
        "perception_confidence": 0.71,
        "stability_score": 1.0,
        "kalman_active": False,
        "fps": 8.6,
        "latency_ms": 96.0,
        "latency_p95_ms": 118.0,
        "road_fraction": 0.08,
    }
    fields.update(overrides)
    return DashboardTelemetry(**fields)


def _orbit() -> np.ndarray:
    return np.full((728, 960, 3), 200, dtype=np.uint8)


def _crop(image: np.ndarray, rect: Rect) -> np.ndarray:
    return image[rect.y : rect.bottom, rect.x : rect.right]


def _count_color(image: np.ndarray, color: tuple[int, int, int]) -> int:
    return int((image == np.array(color, dtype=np.uint8)).all(axis=2).sum())


@pytest.mark.parametrize("state", ["active", "safe_stop", "fallback"])
def test_every_state_renders_a_full_hd_frame(renderer, state):
    result = _result()
    plan = result.plan
    telemetry = _telemetry()
    if state == "safe_stop":
        plan = None
        telemetry = _telemetry(autopilot_active=False)
    elif state == "fallback":
        plan = replace(plan, fallback_active=True, fallback_reason="low confidence")
        result = replace(result, plan=plan)

    frame = renderer.render(result, telemetry, _orbit(), plan, None)

    assert frame.shape == (1080, 1920, 3)
    assert frame.dtype == np.uint8


def test_missing_orbit_frame_shows_the_placeholder(renderer):
    result = _result()

    frame = renderer.render(result, _telemetry(), None, result.plan, None)

    view = _crop(frame, ORBIT_VIEW)
    centre = view[view.shape[0] // 2 - 60 : view.shape[0] // 2 - 30, 100:-100]
    assert _count_color(centre, COLORS["CARD_BG"]) == centre.shape[0] * centre.shape[1]
    # The placeholder text sits on the view's centre line.
    middle = view[view.shape[0] // 2 - 20 : view.shape[0] // 2 + 10]
    assert (middle != np.array(COLORS["CARD_BG"], dtype=np.uint8)).any()


def test_orbit_frame_fills_the_view(renderer):
    result = _result()

    frame = renderer.render(result, _telemetry(), _orbit(), result.plan, None)

    view = _crop(frame, ORBIT_VIEW)
    assert _count_color(view, (200, 200, 200)) > 0.9 * view.shape[0] * view.shape[1]


def test_safe_stop_draws_no_path_in_the_dashcam_card(renderer):
    result = _result()
    path = COLORS["PATH_CORE"]

    driving = renderer.render(result, _telemetry(), _orbit(), result.plan, None)
    stopped = renderer.render(result, _telemetry(autopilot_active=False), _orbit(), None, None)

    assert _count_color(_crop(driving, DASHCAM_CARD), path) > 0
    assert _count_color(_crop(stopped, DASHCAM_CARD), path) == 0
    assert _count_color(_crop(stopped, ORBIT_VIEW), COLORS["BAD"]) > 0


def test_safe_stop_hides_a_plan_even_if_one_is_passed(renderer):
    result = _result()

    stopped = renderer.render(
        result, _telemetry(autopilot_active=False), _orbit(), result.plan, None
    )

    assert _count_color(_crop(stopped, DASHCAM_CARD), COLORS["PATH_CORE"]) == 0


@pytest.mark.parametrize(
    ("fps", "expected"),
    [(7.0, "GOOD"), (6.9, "WARN"), (3.0, "BAD")],
)
def test_fps_colour_follows_the_dashboard_bands(renderer, fps, expected):
    result = _result()

    frame = renderer.render(result, _telemetry(fps=fps), _orbit(), result.plan, None)

    block = _crop(frame, FPS_BLOCK)
    for level in ("GOOD", "WARN", "BAD"):
        count = _count_color(block, COLORS[level])
        if level == expected:
            assert count > 0
        else:
            assert count == 0


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"latency_p95_ms": 143.0}, "GOOD"),
        ({"latency_p95_ms": 150.0}, "BAD"),
        ({"road_fraction": 0.01}, "BAD"),
        ({"perception_confidence": 0.3}, "WARN"),
        ({"perception_confidence": 0.1}, "BAD"),
    ],
)
def test_other_metric_colours_follow_the_dashboard_bands(renderer, overrides, expected):
    result = _result()

    frame = renderer.render(result, _telemetry(**overrides), _orbit(), result.plan, None)

    assert _count_color(_crop(frame, STATS_CARD), COLORS[expected]) > 0


def test_static_layer_is_built_once_and_never_drawn_on(monkeypatch):
    calls = []
    original = PresentationRenderer._build_static

    def counting(self):
        calls.append(1)
        return original(self)

    monkeypatch.setattr(PresentationRenderer, "_build_static", counting)
    renderer = PresentationRenderer()
    before = renderer._static.copy()
    result = _result()

    renderer.render(result, _telemetry(), _orbit(), result.plan, None)
    renderer.render(result, _telemetry(autopilot_active=False), None, None, None)

    assert len(calls) == 1
    assert np.array_equal(renderer._static, before)


def test_presentation_implies_recording():
    assert parse_args(["--presentation"]).record_video is True
    assert parse_args([]).record_video is False
