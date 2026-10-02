"""Unit tests for the dashcam geometry."""

import math
from dataclasses import replace

import numpy as np
import pytest

from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.types import DEFAULT_CAMERA, GMSL2_CAPTURE_SENSOR, GMSL2_SENSOR


def test_camera_axes_follow_vehicle_convention():
    """+X is the vehicle's left, so camera-right must be vehicle -X."""
    model = CameraModel(
        replace(DEFAULT_CAMERA, dir=(0.0, -1.0, 0.0), up=(0.0, 0.0, 1.0)),
        width=720,
        height=465,
    )

    right, down, forward = model.rotation
    assert np.allclose(right, [-1.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(down, [0.0, 0.0, -1.0], atol=1e-6)
    assert np.allclose(forward, [0.0, -1.0, 0.0], atol=1e-6)


def test_focal_length_is_derived_from_horizontal_fov():
    """f = (w/2) / tan(hfov/2) - never a hard-coded constant."""
    sensor = replace(GMSL2_SENSOR, width=640, height=360, fov_x_deg=90.0)
    model = CameraModel(replace(DEFAULT_CAMERA, sensor=sensor), 640, 360)

    # At 90 degrees tan(45) == 1, so f is exactly half the width.
    assert model.focal_px == pytest.approx(320.0)
    assert model.cx == pytest.approx(320.0)
    assert model.cy == pytest.approx(180.0)


def test_gmsl2_focal_length_matches_the_sensor_geometry():
    """The 2880x1860 @ 120 deg imager, at full resolution."""
    model = CameraModel(replace(DEFAULT_CAMERA, sensor=GMSL2_SENSOR))

    expected = 1440.0 / math.tan(math.radians(60.0))
    assert model.focal_px == pytest.approx(expected, rel=1e-9)
    assert model.focal_px == pytest.approx(831.38, abs=0.01)
    assert model.horizontal_fov_deg == pytest.approx(120.0, abs=1e-6)
    # Vertical angle is a consequence of the aspect ratio, not a free knob.
    assert model.vertical_fov_deg == pytest.approx(96.41, abs=0.01)


def test_capture_sensor_is_an_exact_fraction_of_the_imager():
    """Same lens at 1/3 resolution: same angles, focal scaled exactly."""
    assert GMSL2_CAPTURE_SENSOR.width * 3 == GMSL2_SENSOR.width
    assert GMSL2_CAPTURE_SENSOR.height * 3 == GMSL2_SENSOR.height
    assert GMSL2_CAPTURE_SENSOR.fov_y_deg == pytest.approx(GMSL2_SENSOR.fov_y_deg)
    assert GMSL2_CAPTURE_SENSOR.focal_px == pytest.approx(GMSL2_SENSOR.focal_px / 3.0)


def test_focal_length_scales_exactly_with_downsampling():
    """Downscaling must scale the intrinsics, not invalidate them."""
    full = CameraModel(DEFAULT_CAMERA)
    work = CameraModel(DEFAULT_CAMERA, 720, 465)

    assert work.scale == pytest.approx(0.75)
    assert work.focal_px == pytest.approx(full.focal_px * 0.75)
    assert work.cx == pytest.approx(full.cx * 0.75)
    assert work.cy == pytest.approx(full.cy * 0.75)
    assert work.horizontal_fov_deg == pytest.approx(full.horizontal_fov_deg)


def test_beamngpy_receives_the_derived_vertical_fov():
    """beamngpy takes fov_y; passing the horizontal angle would widen the lens."""
    sensor = DEFAULT_CAMERA.sensor

    assert sensor.fov_x_deg == 120.0
    assert sensor.fov_y_deg == pytest.approx(96.41, abs=0.01)


def test_sensor_frame_interval_matches_target_rate():
    sensor = GMSL2_CAPTURE_SENSOR
    assert sensor.frame_interval_s == pytest.approx(1.0 / sensor.target_fps)
    assert sensor.frame_interval_s <= 1.0 / 28.0


def test_project_backproject_round_trip():
    model = CameraModel(DEFAULT_CAMERA, 720, 465)
    points = np.array([[-1.0, -10.0, 0.0], [2.0, -25.0, 0.6], [0.0, -4.0, -0.3]], dtype=np.float32)

    recovered = model.to_vehicle(model.backproject(*model.project(model.from_vehicle(points))))

    assert np.allclose(recovered, points, atol=1e-4)


def test_point_to_the_right_projects_right_of_centre():
    model = CameraModel(DEFAULT_CAMERA, 720, 465)
    # +X is the vehicle's left, so -X is to the right (and the left camera
    # itself sits at +0.3).
    right_point = np.array([[-2.0, -12.0, 0.0]], dtype=np.float32)

    u, _, _ = model.project(model.from_vehicle(right_point))

    assert u[0] > model.cx


def test_vertical_fov_follows_aspect_ratio():
    sensor = replace(GMSL2_SENSOR, width=640, height=360, fov_x_deg=90.0)
    model = CameraModel(replace(DEFAULT_CAMERA, sensor=sensor), 640, 360)
    expected = math.degrees(2.0 * math.atan(180.0 / 320.0))

    assert model.vertical_fov_deg == pytest.approx(expected)
