"""The mono_yoloe stack behind the contract: sensors in, command and visuals out."""

from unittest.mock import patch

import numpy as np
import pytest

from offroad_autonomy.contract import (
    DrivingStack,
    Gauge,
    ImageLayer,
    MaskOverlay,
    Observation,
    Polyline,
    Status,
    camera_spec,
)
from offroad_autonomy.stacks.mono_yoloe.config import load_pipeline_config
from offroad_autonomy.types import PerceptionResult, VehicleState

H, W = 465, 720


def _segment_road(road: bool = True):
    def segment(frame, valid_roi):
        mask = np.zeros((frame.height, frame.width), dtype=bool)
        if road:
            middle = frame.width // 2
            mask[frame.height // 2 :, middle - 150 : middle + 150] = True
        mask &= valid_roi
        return PerceptionResult(
            mask=mask,
            confidences=[0.9],
            num_detections=1,
            valid_roi=valid_roi,
            road_fraction=float(mask.sum() / valid_roi.sum()),
        )

    return segment


@pytest.fixture
def make_stack():
    """A real stack with only the segmenter faked, so no model is loaded."""
    patcher = patch("offroad_autonomy.stacks.mono_yoloe.pipeline.RoadSegmenter")
    segmenter = patcher.start()

    def build(road: bool = True, **stack_config):
        from offroad_autonomy.stacks.mono_yoloe.stack import build as build_stack

        segmenter.return_value.predict.side_effect = _segment_road(road)
        return build_stack(stack_config)

    yield build
    patcher.stop()


def _observation(t: float = 0.0, speed: float = 1.0) -> Observation:
    image = np.zeros((620, 960, 3), dtype=np.uint8)
    return Observation(
        sensors={"dashcam": image},
        vehicle=VehicleState(speed_mps=speed),
        timestamp=t,
        frame_id=1,
    )


def _layers(output) -> list[ImageLayer]:
    return [v for v in output.visuals if isinstance(v, ImageLayer)]


def test_stack_meets_the_contract_and_declares_its_dashcam(make_stack):
    stack = make_stack()

    assert isinstance(stack, DrivingStack)
    sensors = stack.sensors()
    assert [(s.name, s.kind) for s in sensors] == [("dashcam", "camera")]
    expected = load_pipeline_config({}).camera
    spec = camera_spec(sensors[0])
    assert spec.pos == expected.pos
    assert spec.sensor == expected.sensor


def test_step_returns_the_command_and_describes_itself_with_primitives(make_stack):
    output = make_stack().step(_observation())

    assert np.isfinite(output.command.steering)
    assert output.command.throttle > 0.0
    layers = _layers(output)
    assert [layer.name for layer in layers] == [
        "Dashcam",
        "Raw Dashcam",
        "Segmentation Mask",
        "Pipeline Debug",
    ]
    assert layers[0].camera == "dashcam"
    assert layers[0].pixels().shape == (H, W, 3)
    paths = [v for v in output.visuals if isinstance(v, Polyline)]
    assert len(paths) == 1 and paths[0].style == "path" and paths[0].layer == "Dashcam"
    roles = {v.role for v in output.visuals if isinstance(v, MaskOverlay)}
    assert roles == {"traversable", "excluded", "info"}
    gauges = [v.name for v in output.visuals if isinstance(v, Gauge)]
    assert gauges[:2] == ["Road / Valid Px", "Segmentation Confidence"]
    assert output.metrics["segmentation_confidence"] == pytest.approx(0.9)
    assert {"segmentation_ms", "control_ms", "cross_track_m"} <= set(output.metrics)
    assert output.safe_stop == ""


def test_pipeline_page_is_drawn_only_when_the_dashboard_asks(make_stack):
    with patch("offroad_autonomy.stacks.mono_yoloe.stack.pipeline_grid") as grid:
        grid.return_value = np.zeros((2 * H, 2 * W, 3), np.uint8)
        output = make_stack().step(_observation())
        grid.assert_not_called()
        layer = next(v for v in _layers(output) if v.name == "Pipeline Debug")
        assert layer.pixels().shape == (2 * H, 2 * W, 3)
        grid.assert_called_once()


def test_real_pipeline_page_renders_the_four_stages(make_stack):
    output = make_stack().step(_observation())
    layer = next(v for v in _layers(output) if v.name == "Pipeline Debug")

    assert layer.pixels().shape == (2 * H, 2 * W, 3)


def test_no_road_asks_for_a_safe_stop_and_reset_clears_the_timer(make_stack):
    stack = make_stack(road=False, safety={"no_road_time_s": 1.0})

    assert stack.step(_observation(0.0)).safe_stop == ""
    output = stack.step(_observation(1.5))
    assert "no traversable road" in output.safe_stop
    statuses = [v for v in output.visuals if isinstance(v, Status)]
    assert any(s.level != "good" for s in statuses)

    stack.step(_observation(2.0))
    stack.reset()
    assert stack.step(_observation(2.5)).safe_stop == ""


def test_stack_config_moves_the_declared_camera_and_the_stages_with_it(make_stack):
    stack = make_stack(camera={"height_m": 2.1})

    assert camera_spec(stack.sensors()[0]).pos[2] == pytest.approx(2.1)
    assert stack.pipeline.view.camera.spec.pos[2] == pytest.approx(2.1)


@pytest.mark.parametrize("page", [0, 1, 2, 3])
def test_dashboard_and_presentation_draw_every_page_of_the_real_stack(make_stack, page):
    from offroad_autonomy.visualization.dashboard import AutonomyDashboard, DashboardTelemetry
    from offroad_autonomy.visualization.presentation import PresentationRenderer

    stack = make_stack()
    output = stack.step(_observation())
    telemetry = DashboardTelemetry(
        speed_mph=5.0, steering=0.1, throttle=0.2, brake=0.0, fps=7.0, latency_ms=80.0
    )

    frame = AutonomyDashboard(sensors=stack.sensors()).render(output, telemetry, page=page)
    video = PresentationRenderer(sensors=stack.sensors()).render(output, telemetry, None)

    assert frame.shape == (900, 1600, 3)
    assert video.shape == (1080, 1920, 3)


def test_harness_loads_and_drives_the_stack_from_the_shipped_config(make_stack, run_harness):
    from pathlib import Path
    from unittest.mock import Mock

    import offroad_autonomy.main as app
    from offroad_autonomy.contract import load_stack
    from offroad_autonomy.types import CameraFrame
    from offroad_autonomy.utils.config import load_config

    make_stack()  # installs the segmenter fake
    config = load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml")
    stack = load_stack(config.stack, config.stack_config)
    frame = CameraFrame(image=np.zeros((620, 960, 3), np.uint8))
    client = Mock()
    captures = iter([{"dashcam": frame}] * 3 + [None])

    def capture():
        result = next(captures, None)
        if result is None:
            app._shutdown = True
        return result

    client.capture.side_effect = capture
    client.get_vehicle_state.return_value = VehicleState(speed_mps=1.0)
    config.ui_headless = True

    run_harness(stack, client, config=config)

    sent = [c.args[0] for c in client.send_controls.call_args_list]
    assert len(sent) == 3
    assert all(command.throttle > 0.0 for command in sent)
    client.park.assert_not_called()
