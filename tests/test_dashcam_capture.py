"""Single-camera attachment and shared inference/display capture."""

from dataclasses import replace
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from offroad_autonomy.simulation.beamng_client import BeamNGClient
from offroad_autonomy.types import PipelineConfig


@pytest.mark.parametrize("headless", [True, False])
@pytest.mark.parametrize("transport", ["shared_memory", "socket"])
def test_connect_attaches_only_the_dashcam(headless, transport):
    cfg = PipelineConfig(ui_headless=headless, beamng_camera_transport=transport)
    client = BeamNGClient(cfg)
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
        assert args["pos"] == cfg.camera.pos
        assert args["resolution"] == (960, 620)
        assert args["field_of_view_y"] == pytest.approx(cfg.camera.fov_y_deg)
        assert args["is_render_colours"]
        assert not args["is_render_depth"]
        assert args["is_streaming"] == (transport == "shared_memory")
        assert set(client._cameras) == {"dashcam"}
        client._vehicle.set_shift_mode.assert_called_once_with("realistic_automatic")
        client._vehicle.control.assert_any_call(gear=1)
        client.disconnect()
        camera.return_value.remove.assert_called_once()


@pytest.mark.parametrize("transport", ["shared_memory", "socket"])
def test_capture_reads_one_sensor_and_owns_its_pixels(transport):
    cfg = PipelineConfig(beamng_camera_transport=transport)
    cfg.camera = replace(cfg.camera, sensor=replace(cfg.camera.sensor, width=4, height=2))
    client = BeamNGClient(cfg)
    sensor = MagicMock()
    rgba = np.zeros((2, 4, 4), dtype=np.uint8)
    rgba[:] = (11, 22, 33, 255)
    buffer = bytearray(rgba.tobytes())
    sensor.colour_shmem.read.return_value = memoryview(buffer)
    sensor.poll_raw.return_value = {"colour": buffer}
    client._cameras = {"dashcam": sensor}

    first = client.capture_frame()
    assert first.is_new
    assert first.image[0, 0].tolist() == [33, 22, 11]
    assert not client.capture_frame().is_new
    buffer[0] = 99
    third = client.capture_frame()
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


def test_missing_or_malformed_dashcam_frame_is_rejected():
    client = BeamNGClient(PipelineConfig())
    assert client.capture_frame() is None
    sensor = MagicMock()
    sensor.colour_shmem.read.return_value = b"bad frame"
    client._cameras = {"dashcam": sensor}
    assert client.capture_frame() is None


@pytest.mark.parametrize("kind,gear", [("manualGearbox", 1), ("automaticGearbox", 2)])
def test_resume_selects_forward_before_releasing_brakes(kind, gear):
    client = BeamNGClient(PipelineConfig())
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
    client = BeamNGClient(PipelineConfig())
    vehicle = MagicMock()
    client._vehicle = vehicle
    getattr(vehicle, failure_at).side_effect = RuntimeError("connection lost")
    with pytest.raises(RuntimeError, match="connection lost"):
        client.release_park()
    assert not any(c.kwargs.get("parkingbrake") == 0 for c in vehicle.control.call_args_list)


def test_unknown_gearbox_is_not_released():
    client = BeamNGClient(PipelineConfig())
    client._vehicle = MagicMock()
    client._vehicle.queue_lua_command.return_value = "unsupported"
    with pytest.raises(RuntimeError, match="Unsupported forward-drive gearbox"):
        client.release_park()
    assert not any(
        c.kwargs.get("parkingbrake") == 0 for c in client._vehicle.control.call_args_list
    )
