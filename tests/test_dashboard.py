"""Tests for the dashcam dashboard."""

from dataclasses import replace

import numpy as np
import pytest

from offroad_autonomy.types import (
    DEBUG_VIEWS,
    CameraFrame,
    ControlCommand,
    FramePacket,
    PathPlan,
    PerceptionResult,
    PipelineStepResult,
    StabilizedResult,
)
from offroad_autonomy.visualization.dashboard import AutonomyDashboard, DashboardTelemetry


def _result() -> PipelineStepResult:
    frame = np.random.default_rng(0).integers(0, 255, (465, 720, 3), dtype=np.uint8)
    mask = np.zeros((465, 720), dtype=bool)
    mask[250:400, 250:470] = True
    raw = np.zeros((620, 960, 3), dtype=np.uint8)
    return PipelineStepResult(
        frame=FramePacket(raw=raw, preprocessed=frame, timestamp=0.0, height=465, width=720),
        perception=PerceptionResult(mask=mask, confidences=[0.8]),
        stabilized=StabilizedResult(mask=mask),
        plan=PathPlan(centerline=np.array([[360.0, 440.0], [360.0, 300.0], [370.0, 200.0]])),
        command=ControlCommand(),
        capture=CameraFrame(image=raw),
    )


@pytest.mark.parametrize("view", DEBUG_VIEWS)
def test_every_debug_view_renders(view):
    dashboard = AutonomyDashboard()
    result = _result()
    telemetry = DashboardTelemetry(
        speed_mph=10.0,
        steering=0.1,
        throttle=0.2,
        brake=0.0,
        perception_confidence=0.8,
        stability_score=0.9,
        kalman_active=False,
        fps=22.0,
        latency_ms=35.0,
        latency_p95_ms=44.0,
        timing_lines=["capture  1.0 ms"],
    )
    roi = np.ones((465, 720), dtype=bool)
    roi[420:] = False

    frame = dashboard.render(
        result,
        telemetry,
        plan=result.plan,
        valid_roi=roi,
        debug_view=view,
        timing_overlay=True,
    )

    assert frame.shape == (900, 1600, 3)


def test_dashboard_survives_missing_camera_frames():
    dashboard = AutonomyDashboard()
    result = replace(_result(), capture=None)
    telemetry = DashboardTelemetry(
        speed_mph=0.0,
        steering=0.0,
        throttle=0.0,
        brake=0.0,
        perception_confidence=0.0,
        stability_score=1.0,
        kalman_active=True,
        fps=0.0,
        latency_ms=0.0,
        autopilot_active=False,
    )

    for view in DEBUG_VIEWS:
        frame = dashboard.render(result, telemetry, debug_view=view)
        assert frame.shape == (900, 1600, 3)


def test_raw_view_is_the_frame_used_for_inference():
    result = _result()
    image = AutonomyDashboard()._main_view("raw", result, None, None)
    assert image is result.frame.raw
