"""Tests for the presentation video frame."""

import numpy as np
import pytest

from offroad_autonomy.contract import Gauge, ImageLayer, MaskOverlay, Polyline, StackOutput
from offroad_autonomy.main import build_parser, video_paths
from offroad_autonomy.types import DEFAULT_DASHBOARD_COLORS, ControlCommand
from offroad_autonomy.visualization.dashboard import DashboardTelemetry
from offroad_autonomy.visualization.layout import Rect
from offroad_autonomy.visualization.presentation import (
    DASHCAM_CARD,
    ORBIT_VIEW,
    STATS_CARD,
    PresentationRenderer,
)
from tests.conftest import count_color

COLORS = DEFAULT_DASHBOARD_COLORS

# The autonomy FPS block, the stats card's top-left metric.
FPS_BLOCK = Rect(STATS_CARD.x + 24, STATS_CARD.y + 134, 266, 102)


@pytest.fixture(scope="module")
def renderer() -> PresentationRenderer:
    # Shared: building the fonts and the static layer is the slow part.
    return PresentationRenderer()


def _output(style: str = "path", road: float = 0.08, confidence: float = 0.71) -> StackOutput:
    """A flat grey frame, so any pixel in a palette colour was drawn on it."""
    frame = np.full((465, 720, 3), 90, dtype=np.uint8)
    mask = np.zeros((465, 720), dtype=bool)
    path = np.array([[360.0, 440.0], [360.0, 300.0], [370.0, 200.0]])
    return StackOutput(
        command=ControlCommand(),
        visuals=[
            ImageLayer("Dashcam", frame, camera="dashcam"),
            MaskOverlay("Dashcam", mask, "traversable", "Traversable Mask"),
            Polyline("Dashcam", path, "image", style, "Planned Path"),
            Gauge("Road / Valid Px", road, (0.0, 0.2), (0.015,), text=f"{road:.3f}"),
            Gauge("Segmentation Confidence", confidence, (0.0, 1.0), (0.18, 0.5)),
        ],
    )


def _telemetry(**overrides) -> DashboardTelemetry:
    fields = {
        "speed_mph": 9.4,
        "steering": 0.14,
        "throttle": 0.34,
        "brake": 0.0,
        "fps": 8.6,
        "latency_ms": 96.0,
        "latency_p95_ms": 118.0,
    }
    fields.update(overrides)
    return DashboardTelemetry(**fields)


def _orbit() -> np.ndarray:
    return np.full((728, 960, 3), 200, dtype=np.uint8)


def _crop(image: np.ndarray, rect: Rect) -> np.ndarray:
    return image[rect.y : rect.bottom, rect.x : rect.right]


@pytest.mark.parametrize("state", ["active", "safe_stop", "held"])
def test_every_state_renders_a_full_hd_frame(renderer, state):
    output = _output()
    telemetry = _telemetry()
    if state == "safe_stop":
        telemetry = _telemetry(autopilot_active=False)
    elif state == "held":
        output = _output(style="held")

    frame = renderer.render(output, telemetry, _orbit())

    assert frame.shape == (1080, 1920, 3)
    assert frame.dtype == np.uint8


def test_a_stack_with_no_image_or_gauges_still_renders(renderer):
    frame = renderer.render(StackOutput(command=ControlCommand()), _telemetry(), _orbit())

    assert frame.shape == (1080, 1920, 3)


def test_missing_orbit_frame_shows_the_placeholder(renderer):
    frame = renderer.render(_output(), _telemetry(), None)

    view = _crop(frame, ORBIT_VIEW)
    centre = view[view.shape[0] // 2 - 60 : view.shape[0] // 2 - 30, 100:-100]
    assert count_color(centre, COLORS["CARD_BG"]) == centre.shape[0] * centre.shape[1]
    # The placeholder text sits on the view's centre line.
    middle = view[view.shape[0] // 2 - 20 : view.shape[0] // 2 + 10]
    assert (middle != np.array(COLORS["CARD_BG"], dtype=np.uint8)).any()


def test_orbit_frame_fills_the_view(renderer):
    frame = renderer.render(_output(), _telemetry(), _orbit())

    view = _crop(frame, ORBIT_VIEW)
    assert count_color(view, (200, 200, 200)) > 0.9 * view.shape[0] * view.shape[1]


def test_safe_stop_draws_no_path_in_the_dashcam_card(renderer):
    path = COLORS["PATH_CORE"]

    driving = renderer.render(_output(), _telemetry(), _orbit())
    stopped = renderer.render(_output(), _telemetry(autopilot_active=False), _orbit())

    assert count_color(_crop(driving, DASHCAM_CARD), path) > 0
    assert count_color(_crop(stopped, DASHCAM_CARD), path) == 0
    assert count_color(_crop(stopped, ORBIT_VIEW), COLORS["BAD"]) > 0


@pytest.mark.parametrize(
    ("fps", "expected"),
    [(7.0, "GOOD"), (6.9, "WARN"), (3.0, "BAD")],
)
def test_fps_colour_follows_the_dashboard_bands(renderer, fps, expected):
    frame = renderer.render(_output(), _telemetry(fps=fps), _orbit())

    block = _crop(frame, FPS_BLOCK)
    for level in ("GOOD", "WARN", "BAD"):
        count = count_color(block, COLORS[level])
        if level == expected:
            assert count > 0
        else:
            assert count == 0


@pytest.mark.parametrize(
    ("telemetry", "gauges", "expected"),
    [
        ({"latency_p95_ms": 143.0}, {}, "GOOD"),
        ({"latency_p95_ms": 150.0}, {}, "BAD"),
        ({}, {"road": 0.01}, "BAD"),
        ({}, {"confidence": 0.3}, "WARN"),
        ({}, {"confidence": 0.1}, "BAD"),
    ],
)
def test_other_metric_colours_follow_the_bands(renderer, telemetry, gauges, expected):
    frame = renderer.render(_output(**gauges), _telemetry(**telemetry), _orbit())

    assert count_color(_crop(frame, STATS_CARD), COLORS[expected]) > 0


def test_static_layer_is_built_once_and_never_drawn_on(monkeypatch):
    calls = []
    original = PresentationRenderer._build_static

    def counting(self):
        calls.append(1)
        return original(self)

    monkeypatch.setattr(PresentationRenderer, "_build_static", counting)
    renderer = PresentationRenderer()
    before = renderer._static.copy()

    renderer.render(_output(), _telemetry(), _orbit())
    renderer.render(_output(), _telemetry(autopilot_active=False), None)

    assert len(calls) == 1
    assert np.array_equal(renderer._static, before)


def test_presentation_is_recorded_apart_from_the_dashboard():
    args = build_parser().parse_args(["--record-video", "--presentation"])

    dashboard, presentation = video_paths(args, "run")

    assert dashboard.parent.as_posix() == "output/videos"
    assert presentation.parent.as_posix() == "output/presentations"


def test_presentation_alone_does_not_record_the_dashboard():
    args = build_parser().parse_args(["--presentation"])

    assert args.presentation is True
    assert args.record_video is False


@pytest.mark.parametrize("record", [False, True])
def test_live_presentation_displays_orbit_and_handles_stop_resume_and_close(run_harness, record):
    from unittest.mock import Mock

    from offroad_autonomy.contract import camera_sensor
    from offroad_autonomy.types import DEFAULT_CAMERA, CameraFrame, HarnessConfig, VehicleState

    cfg = HarnessConfig(ui_display_async=False)
    client, stack, window, video = Mock(), Mock(), Mock(), Mock()
    sensors = [camera_sensor(DEFAULT_CAMERA)]
    stack.sensors.return_value = sensors
    stack.step.return_value = _output()
    client.capture.return_value = {"dashcam": CameraFrame(image=np.zeros((620, 960, 3), np.uint8))}
    client.capture_orbit.return_value = _orbit()
    client.get_vehicle_state.return_value = VehicleState()
    window.backend = "tkinter"
    window.held_keys = set()
    canvases = []

    def show(canvas):
        canvases.append(canvas)
        window.last_key = [ord("e"), ord("p"), -1][len(canvases) - 1]
        return len(canvases) < 3

    window.show.side_effect = show
    client_factory = Mock(return_value=client)
    open_window = Mock(return_value=window)
    open_video = Mock(return_value=video)
    argv = ["--presentation-view"]
    if record:
        argv.append("--presentation")
    run_harness(
        stack,
        client,
        argv,
        cfg,
        BeamNGClient=client_factory,
        _open_window=open_window,
        _open_video=open_video,
        _build_dashboard_telemetry=lambda *args: _telemetry(),
    )

    client_factory.assert_called_once_with(cfg, sensors, orbit=True)
    open_window.assert_called_once_with(1920, 1080)
    assert len(canvases) == 3
    assert all(canvas.shape == (1080, 1920, 3) for canvas in canvases)
    orbit = _crop(canvases[0], ORBIT_VIEW)
    assert count_color(orbit, (200, 200, 200)) > 0.9 * orbit.shape[0] * orbit.shape[1]
    client.park.assert_called_once()
    client.release_park.assert_called_once()
    # Resuming hands the stack a fresh start: the vehicle may have been moved by hand.
    stack.reset.assert_called_once()
    window.close.assert_called_once()
    client.disconnect.assert_called_once()
    if record:
        open_video.assert_called_once()
        assert video.write.call_count == 3
        video.close.assert_called_once()
    else:
        open_video.assert_not_called()


def test_one_path_for_both_recordings_is_refused(tmp_path):
    out = str(tmp_path / "run.mp4")
    args = build_parser().parse_args(
        ["--record-video", "--presentation", "--record-video-out", out, "--presentation-out", out]
    )

    with pytest.raises(SystemExit):
        video_paths(args, "run")
