"""Sensor attachment from a stack's declaration, and shared capture."""

from dataclasses import replace
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from offroad_autonomy.contract import SensorSpec, camera_sensor
from offroad_autonomy.simulation.beamng_client import BeamNGClient, BeamNGConnectionError
from offroad_autonomy.types import DEFAULT_CAMERA, HarnessConfig

DASHCAM = camera_sensor(DEFAULT_CAMERA)


def _small_dashcam(name: str = "dashcam") -> SensorSpec:
    camera = replace(
        DEFAULT_CAMERA, name=name, sensor=replace(DEFAULT_CAMERA.sensor, width=4, height=2)
    )
    return camera_sensor(camera)


@pytest.mark.parametrize(
    "host,launch,hint",
    [
        ("localhost", False, "BEAMNG_HOME"),
        ("192.168.1.50", False, "firewall"),
        ("localhost", True, "Automatic launch"),
    ],
)
@pytest.mark.parametrize("error_type", ["disconnected", "socket"])
def test_connection_failure_explains_endpoint_and_recovery(host, launch, hint, error_type):
    from beamngpy.logging import BNGDisconnectedError

    cfg = HarnessConfig(beamng_host=host, beamng_port=65432, beamng_launch=launch)
    client = BeamNGClient(cfg, [DASHCAM])
    error = BNGDisconnectedError("refused") if error_type == "disconnected" else OSError("refused")
    with patch("beamngpy.BeamNGpy") as bng, patch("beamngpy.Scenario") as scenario:
        bng.return_value.open.side_effect = error
        with pytest.raises(BeamNGConnectionError) as caught:
            client.connect()
        message = str(caught.value)
        assert f"{host}:65432" in message
        assert "-tcom -tport 65432" in message
        assert hint in message
        assert caught.value.__cause__ is error
        scenario.assert_not_called()
        client.disconnect()
        if launch:
            bng.return_value.close.assert_called_once()
        else:
            bng.return_value.disconnect.assert_called_once()


@pytest.mark.parametrize("headless", [True, False])
@pytest.mark.parametrize("transport", ["shared_memory", "socket"])
def test_connect_attaches_only_the_declared_dashcam(headless, transport):
    cfg = HarnessConfig(ui_headless=headless, beamng_camera_transport=transport)
    client = BeamNGClient(cfg, [DASHCAM])
    with (
        patch("beamngpy.BeamNGpy"),
        patch("beamngpy.Scenario"),
        patch("beamngpy.Vehicle") as vehicle_cls,
        patch("beamngpy.sensors.camera.Camera") as camera,
        patch("offroad_autonomy.simulation.beamng_client.time.sleep"),
    ):
        vehicle_cls.return_value.queue_lua_command.return_value = "manualGearbox"
        client.connect()
        camera.assert_called_once()
        args = camera.call_args.kwargs
        assert args["name"] == "dashcam"
        assert args["pos"] == DEFAULT_CAMERA.pos
        assert args["resolution"] == (960, 620)
        assert args["field_of_view_y"] == pytest.approx(DEFAULT_CAMERA.fov_y_deg)
        assert args["is_render_colours"]
        assert not args["is_render_annotations"]
        assert not args["is_render_instance"]
        assert not args["is_render_depth"]
        assert args["is_streaming"] == (transport == "shared_memory")
        assert set(client._cameras) == {"dashcam"}
        client._vehicle.set_shift_mode.assert_called_once_with("realistic_automatic")
        client._vehicle.control.assert_any_call(gear=1)
        client.disconnect()
        camera.return_value.remove.assert_called_once()


def test_connect_attaches_every_camera_a_stack_declares():
    left = camera_sensor(replace(DEFAULT_CAMERA, name="left", pos=(0.4, -0.3, 1.85)))
    right = camera_sensor(replace(DEFAULT_CAMERA, name="right", pos=(-0.4, -0.3, 1.85)))
    client = BeamNGClient(HarnessConfig(), [left, right])
    with (
        patch("beamngpy.BeamNGpy"),
        patch("beamngpy.Scenario"),
        patch("beamngpy.Vehicle") as vehicle_cls,
        patch("beamngpy.sensors.camera.Camera") as camera,
        patch("offroad_autonomy.simulation.beamng_client.time.sleep"),
    ):
        vehicle_cls.return_value.queue_lua_command.return_value = "manualGearbox"
        client.connect()
        attached = {call.kwargs["name"]: call.kwargs["pos"] for call in camera.call_args_list}
        assert attached == {"left": (0.4, -0.3, 1.85), "right": (-0.4, -0.3, 1.85)}
        client.disconnect()


@pytest.mark.parametrize(
    ("sensors", "match"),
    [
        ([SensorSpec("lidar", "lidar", {})], "supports"),
        ([DASHCAM, DASHCAM], "unique"),
        ([camera_sensor(replace(DEFAULT_CAMERA, name="orbit"))], "reserved"),
        ([SensorSpec("dashcam", "camera", {"pos": (0, 0, 1)})], "missing"),
    ],
)
def test_sensors_the_adapter_cannot_attach_fail_before_connecting(sensors, match):
    with pytest.raises(ValueError, match=match):
        BeamNGClient(HarnessConfig(), sensors)


@pytest.mark.parametrize("transport", ["shared_memory", "socket"])
def test_capture_reads_each_sensor_and_owns_its_pixels(transport):
    client = BeamNGClient(HarnessConfig(beamng_camera_transport=transport), [_small_dashcam()])
    sensor = MagicMock()
    rgba = np.zeros((2, 4, 4), dtype=np.uint8)
    rgba[:] = (11, 22, 33, 255)
    buffer = bytearray(rgba.tobytes())
    sensor.colour_shmem.read.return_value = memoryview(buffer)
    sensor.poll_raw.return_value = {"colour": buffer}
    client._cameras = {"dashcam": sensor}

    first = client.capture()["dashcam"]
    assert first.is_new
    assert first.image[0, 0].tolist() == [33, 22, 11]
    assert not client.capture()["dashcam"].is_new
    buffer[0] = 99
    third = client.capture()["dashcam"]
    assert third.is_new
    assert third.frame_id == first.frame_id + 2
    assert first.image[0, 0].tolist() == [33, 22, 11]
    assert third.image[0, 0].tolist() == [33, 22, 99]
    if transport == "socket":
        assert sensor.poll_raw.call_count == 3
        sensor.colour_shmem.read.assert_not_called()
    else:
        assert sensor.colour_shmem.read.call_count == 3
        sensor.poll_raw.assert_not_called()


def test_missing_or_malformed_frame_is_rejected():
    client = BeamNGClient(HarnessConfig(), [DASHCAM])
    assert client.capture() is None
    sensor = MagicMock()
    sensor.colour_shmem.read.return_value = b"bad frame"
    client._cameras = {"dashcam": sensor}
    assert client.capture() is None


def test_one_missing_camera_withholds_the_whole_observation():
    client = BeamNGClient(HarnessConfig(), [_small_dashcam("left"), _small_dashcam("right")])
    good = MagicMock()
    good.colour_shmem.read.return_value = np.zeros((2, 4, 4), dtype=np.uint8).tobytes()
    missing = MagicMock()
    missing.colour_shmem.read.return_value = None
    client._cameras = {"left": good, "right": missing}

    assert client.capture() is None
    client._cameras["right"] = good
    frames = client.capture()
    assert set(frames) == {"left", "right"}


@pytest.mark.parametrize("transport", ["shared_memory", "socket"])
def test_annotation_depth_and_orbit_buffers_cannot_replace_dashcam_rgb(transport):
    client = BeamNGClient(HarnessConfig(beamng_camera_transport=transport), [_small_dashcam()])
    sensor, orbit = MagicMock(), MagicMock()
    rgb = np.full((2, 4, 4), [11, 22, 33, 255], dtype=np.uint8).tobytes()
    labels = np.full((2, 4, 4), 255, dtype=np.uint8).tobytes()
    sensor.colour_shmem.read.return_value = rgb
    sensor.annotation_shmem.read.side_effect = AssertionError("ground-truth annotations read")
    sensor.instance_shmem.read.side_effect = AssertionError("ground-truth instances read")
    sensor.depth_shmem.read.side_effect = AssertionError("ground-truth depth read")
    sensor.poll_raw.return_value = {
        "colour": rgb,
        "annotation": labels,
        "instance": labels,
        "depth": labels,
    }
    client._cameras = {"dashcam": sensor, "orbit": orbit}
    captured = client.capture()["dashcam"]
    np.testing.assert_array_equal(captured.image, np.full((2, 4, 3), [33, 22, 11], dtype=np.uint8))
    sensor.colour_shmem.read.return_value = None
    sensor.poll_raw.return_value.pop("colour")
    assert client.capture() is None  # No fallback to the available privileged buffers.
    sensor.annotation_shmem.read.assert_not_called()
    sensor.instance_shmem.read.assert_not_called()
    sensor.depth_shmem.read.assert_not_called()
    assert not orbit.mock_calls


def test_presentation_camera_also_disables_ground_truth_render_buffers():
    client = BeamNGClient(HarnessConfig(), [DASHCAM])
    camera = MagicMock()
    client._attach_camera(camera, MagicMock(), client._config.orbit_camera)
    flags = camera.call_args.kwargs
    assert flags["is_render_colours"]
    assert not flags["is_render_annotations"]
    assert not flags["is_render_instance"]
    assert not flags["is_render_depth"]


@pytest.mark.parametrize("kind,gear", [("manualGearbox", 1), ("automaticGearbox", 2)])
def test_resume_selects_forward_before_releasing_brakes(kind, gear):
    client = BeamNGClient(HarnessConfig(), [])
    vehicle = MagicMock()
    client._vehicle = vehicle
    vehicle.queue_lua_command.return_value = kind
    client.release_park()
    assert vehicle.mock_calls[:5] == [
        call.control(throttle=0, brake=0, parkingbrake=1),
        call.set_shift_mode("realistic_automatic"),
        call.queue_lua_command(
            'return powertrain.getDevice("gearbox").type',
            response=True,
        ),
        call.control(gear=gear),
        call.control(steering=0, throttle=0, brake=0, parkingbrake=0),
    ]


@pytest.mark.parametrize("failure_at", ["set_shift_mode", "queue_lua_command"])
def test_failed_forward_setup_keeps_parking_brake_applied(failure_at):
    client = BeamNGClient(HarnessConfig(), [])
    vehicle = MagicMock()
    client._vehicle = vehicle
    getattr(vehicle, failure_at).side_effect = RuntimeError("connection lost")
    with pytest.raises(RuntimeError, match="connection lost"):
        client.release_park()
    assert not any(c.kwargs.get("parkingbrake") == 0 for c in vehicle.control.call_args_list)


def test_unknown_gearbox_is_not_released():
    client = BeamNGClient(HarnessConfig(), [])
    client._vehicle = MagicMock()
    client._vehicle.queue_lua_command.return_value = "unsupported"
    with pytest.raises(RuntimeError, match="Unsupported forward-drive gearbox"):
        client.release_park()
    assert not any(
        c.kwargs.get("parkingbrake") == 0 for c in client._vehicle.control.call_args_list
    )
