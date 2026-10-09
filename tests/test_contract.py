"""The contract between the harness and a driving stack."""

import sys
import types
from dataclasses import replace

import numpy as np
import pytest

from offroad_autonomy.contract import (
    ImageLayer,
    MaskOverlay,
    Polyline,
    SensorSpec,
    StackOutput,
    Status,
    camera_sensor,
    camera_spec,
    finite_metrics,
    load_stack,
)
from offroad_autonomy.types import DEFAULT_CAMERA, ControlCommand


class _Stack:
    def __init__(self, config: dict, sensors=None) -> None:
        self.config = config
        self._sensors = sensors
        if sensors is None:
            self._sensors = [camera_sensor(DEFAULT_CAMERA)]

    def sensors(self):
        return self._sensors

    def step(self, obs):
        return StackOutput(ControlCommand())

    def reset(self):
        pass


@pytest.fixture
def stack_module(monkeypatch):
    module = types.ModuleType("fake_stack_module")
    module.build = _Stack
    module.blind = lambda config: _Stack(config, sensors=[])
    module.lidar = lambda config: _Stack(config, sensors=[SensorSpec("lidar", "lidar")])
    module.not_a_stack = lambda config: object()
    module.value = 3
    monkeypatch.setitem(sys.modules, "fake_stack_module", module)
    return module


def test_load_stack_builds_the_named_function_from_its_config(stack_module):
    stack = load_stack("fake_stack_module:build", {"gain": 2})

    assert isinstance(stack, _Stack)
    assert stack.config == {"gain": 2}


@pytest.mark.parametrize(
    ("path", "error", "match"),
    [
        ("fake_stack_module", ValueError, "package.module:function"),
        ("no_such_module_anywhere:build", ValueError, "cannot be imported"),
        ("fake_stack_module:missing", ValueError, "no callable 'missing'"),
        ("fake_stack_module:value", ValueError, "no callable 'value'"),
        ("fake_stack_module:not_a_stack", TypeError, "lacks sensors/step/reset"),
        ("fake_stack_module:blind", ValueError, "declared no sensors"),
        ("fake_stack_module:lidar", ValueError, "supports"),
    ],
)
def test_load_stack_refuses_what_cannot_run(stack_module, path, error, match):
    with pytest.raises(error, match=match):
        load_stack(path, {})


def test_a_stack_cannot_receive_the_harness_config_by_reference(stack_module):
    original = {"gain": 2}

    stack = load_stack("fake_stack_module:build", original)
    stack.config["gain"] = 5

    assert original == {"gain": 2}


def test_camera_sensor_round_trips_the_mount_and_lens():
    spec = camera_spec(camera_sensor(DEFAULT_CAMERA))

    assert spec.name == DEFAULT_CAMERA.name
    assert spec.pos == DEFAULT_CAMERA.pos
    assert spec.dir == pytest.approx(DEFAULT_CAMERA.dir)
    assert spec.sensor == DEFAULT_CAMERA.sensor


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"width": 0}, "positive"),
        ({"fov_x_deg": 180.0}, "fov_x_deg"),
        ({"target_fps": 0.0}, "target_fps"),
        ({"pos": (0.0, 1.0)}, "3 components"),
        ({"zoom": 2.0}, "unknown"),
    ],
)
def test_invalid_camera_parameters_are_refused(params, match):
    sensor = camera_sensor(DEFAULT_CAMERA)
    sensor = replace(sensor, params={**sensor.params, **params})

    with pytest.raises(ValueError, match=match):
        camera_spec(sensor)


def test_camera_spec_refuses_another_kind():
    with pytest.raises(ValueError, match="not a camera"):
        camera_spec(SensorSpec("lidar", "lidar"))


def test_a_layer_image_can_be_drawn_on_demand():
    calls = []

    def draw():
        calls.append(1)
        return np.zeros((2, 2, 3), np.uint8)

    layer = ImageLayer("Costly", draw)

    assert calls == []
    assert layer.pixels().shape == (2, 2, 3)
    assert calls == [1]


def test_only_finite_metrics_are_averaged():
    assert finite_metrics({"a": 1, "b": float("inf"), "c": float("nan")}) == {"a": 1.0}


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: Polyline("Main", np.zeros((2, 2)), frame="world"), "Polyline.frame"),
        (lambda: Polyline("Main", np.zeros((2, 2)), style="dotted"), "Polyline.style"),
        (lambda: MaskOverlay("Main", np.zeros((2, 2), bool), role="road"), "MaskOverlay.role"),
        (lambda: Status("Odd", level="info"), "Status.level"),
    ],
)
def test_visuals_refuse_values_the_dashboard_cannot_draw(build, match):
    with pytest.raises(ValueError, match=match):
        build()
