"""Unit tests for the pipeline with mocked stages."""

import time
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pytest

from offroad_autonomy.types import (
    CameraFrame,
    ControlCommand,
    FramePacket,
    PathPlan,
    PerceptionResult,
    PipelineConfig,
    PipelineStepResult,
    StabilizedResult,
    VehicleSignals,
)

H, W = 465, 720


def _make_config(**overrides) -> PipelineConfig:
    return PipelineConfig(
        model_weights="dummy.pt",
        preprocess_width=W,
        preprocess_height=H,
        **overrides,
    )


_STAGE_PATCHES = (
    "offroad_autonomy.stacks.mono_yoloe.pipeline.ImagePreprocessor",
    "offroad_autonomy.stacks.mono_yoloe.pipeline.RoadSegmenter",
    "offroad_autonomy.stacks.mono_yoloe.pipeline.TemporalStabilizer",
    "offroad_autonomy.stacks.mono_yoloe.pipeline.CenterlinePlanner",
    "offroad_autonomy.stacks.mono_yoloe.pipeline.StanleyController",
)


def _stage_outputs():
    """Canned outputs for every mocked stage, shaped like the real ones."""
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    mask = np.zeros((H, W), dtype=bool)
    mask[200:300, 200:440] = True

    return (
        FramePacket(raw=frame, preprocessed=frame, timestamp=0.0, height=H, width=W),
        PerceptionResult(mask=mask, confidences=[0.9], num_detections=1),
        StabilizedResult(mask=mask, stability_score=0.95),
        PathPlan(centerline=np.array([[320.0, 300.0], [320.0, 100.0]])),
        ControlCommand(steering=0.0, throttle=0.3, brake=0.0),
    )


def _wire(mocks):
    """Point the five mocked stages at the canned outputs."""
    packet, perception, stabilized, plan, command = _stage_outputs()
    pre, seg, stab, planner, ctrl = mocks
    pre.return_value.process.return_value = packet
    seg.return_value.predict.return_value = perception
    stab.return_value.stabilize.return_value = stabilized
    planner.return_value.plan.return_value = plan
    ctrl.return_value.compute.return_value = command
    return packet, perception, stabilized, plan, command


def _capture(**overrides) -> CameraFrame:
    frame = np.zeros((620, 960, 3), dtype=np.uint8)
    fields = {
        "image": frame,
        "timestamp": time.perf_counter(),
        "frame_id": 1,
        "is_new": True,
    }
    fields.update(overrides)
    return CameraFrame(**fields)


def _pipeline(stack, config):
    mocks = [stack.enter_context(patch(target)) for target in _STAGE_PATCHES]
    outputs = _wire(mocks)
    from offroad_autonomy.stacks.mono_yoloe.pipeline import AutonomyPipeline

    return AutonomyPipeline(config), mocks, outputs


def test_pipeline_uses_stanley_with_the_perception_camera():
    from offroad_autonomy.control.stanley_controller import StanleyController
    from offroad_autonomy.stacks.mono_yoloe.pipeline import AutonomyPipeline

    with ExitStack() as stack:
        for target in _STAGE_PATCHES[:-1]:
            stack.enter_context(patch(target))
        pipeline = AutonomyPipeline(_make_config())

    assert isinstance(pipeline.controller, StanleyController)
    assert pipeline.controller._camera is pipeline.view.camera


def test_each_stage_receives_the_previous_stage_output():
    with ExitStack() as stack:
        pipeline, mocks, outputs = _pipeline(stack, _make_config())
        packet, perception, stabilized, plan, command = outputs
        capture = _capture()
        state = VehicleSignals(speed_mps=4.0)

        result = pipeline.step_result(capture, state)

        mocks[0].return_value.process.assert_called_once_with(capture.image)
        mocks[1].return_value.predict.assert_called_once_with(packet, pipeline.valid_roi)
        mocks[2].return_value.stabilize.assert_called_once_with(perception)
        mocks[3].return_value.plan.assert_called_once_with(stabilized, signals=state)
        mocks[4].return_value.compute.assert_called_once_with(plan, state)
        assert isinstance(result, PipelineStepResult)
        assert (result.frame, result.stabilized, result.plan) == (packet, stabilized, plan)
        assert result.command is command
        assert set(result.timings_ms) == {
            "preprocess",
            "segmentation",
            "postprocess",
            "planning",
            "control",
        }


def test_reset_clears_the_remaining_stateful_stages():
    with ExitStack() as stack:
        pipeline, mocks, _ = _pipeline(stack, _make_config())
        pipeline.reset()
        for mock in mocks[2:]:
            mock.return_value.reset.assert_called_once_with()


def test_missing_dashcam_is_not_accepted():
    with ExitStack() as stack:
        pipeline, _, _ = _pipeline(stack, _make_config())
        capture = _capture(image=None)
        with pytest.raises(ValueError, match="No frame available"):
            pipeline.step_result(capture, VehicleSignals())


@pytest.mark.parametrize("planner_mode", ["baseline", "advanced"])
@pytest.mark.parametrize("road_half_width", [120, 300])
def test_rgb_pipeline_plans_and_controls_in_every_mode(planner_mode, road_half_width):
    from offroad_autonomy.stacks.mono_yoloe.pipeline import AutonomyPipeline

    config = _make_config(planner_mode=planner_mode)

    def segment(frame, valid_roi):
        mask = np.zeros((frame.height, frame.width), dtype=bool)
        middle = frame.width // 2
        mask[frame.height // 2 :, middle - road_half_width : middle + road_half_width] = True
        mask &= valid_roi
        return PerceptionResult(
            mask=mask,
            confidences=[0.9],
            num_detections=1,
            valid_roi=valid_roi,
            road_fraction=float(mask.sum() / valid_roi.sum()),
        )

    with patch("offroad_autonomy.stacks.mono_yoloe.pipeline.RoadSegmenter") as segmenter:
        segmenter.return_value.predict.side_effect = segment
        pipeline = AutonomyPipeline(config)
        result = pipeline.step_result(_capture(), VehicleSignals(speed_mps=1.0))

    assert not result.plan.fallback_active, result.plan.fallback_reason
    assert len(result.plan.centerline) >= 2
    assert result.plan.min_clearance_m == float("inf")
    assert np.isfinite(result.command.steering)
    assert result.command.throttle > 0.0
    assert result.command.debug is not None
    assert result.stabilized.mask.shape == pipeline.valid_roi.shape
    # A symmetric road must not produce a turn around the hood's silhouette.
    np.testing.assert_allclose(result.plan.centerline[:, 0], W / 2, atol=8)
    assert abs(result.command.steering) < 0.1


def test_dashcam_geometry_and_capture_reach_all_stages():
    with ExitStack() as stack:
        cfg = _make_config()
        pipeline, mocks, _ = _pipeline(stack, cfg)
        capture = _capture()
        result = pipeline.step_result(capture, VehicleSignals())
        assert result.capture is capture
        assert pipeline.view.camera.spec is cfg.camera
        assert mocks[3].call_args.kwargs["camera"] is pipeline.view.camera
        assert mocks[4].call_args.kwargs["camera"] is pipeline.view.camera
        assert mocks[0].return_value.process.call_args.args[0] is capture.image
