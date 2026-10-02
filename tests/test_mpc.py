"""Metric paths exercise the real projection, optimizer and safety interface."""

import math
import time
from unittest.mock import Mock, patch

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

from offroad_autonomy.control.controller_config import MPCConfig
from offroad_autonomy.control.mpc_controller import MPCController, SolveTimeout
from offroad_autonomy.control.vehicle_model import bicycle_step, rollout_with_jacobian
from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.types import PathPlan, PipelineConfig, VehicleState
from offroad_autonomy.utils.config import load_config


def setup_controller(**kwargs):
    cfg = PipelineConfig(controller="mpc", mpc=MPCConfig(max_solve_time_s=1.0), **kwargs)
    return cfg, MPCController(cfg)


def path(cfg, curve):
    forward = np.linspace(1.0, 16.0, 32)
    camera = CameraModel(cfg.camera, cfg.preprocess_width, cfg.preprocess_height)
    return PathPlan(centerline=camera.ground_to_image(forward, curve(forward))[::-1])


@pytest.mark.parametrize(
    "curve,sign",
    [
        (lambda f: f * 0, 0),
        (lambda f: -0.012 * f**2, -1),
        (lambda f: 0.012 * f**2, 1),
        (lambda f: 0.045 * f**2, 1),
        (lambda f: 0.7 * np.sin(f / 4), 1),
        (lambda f: np.ones_like(f), 1),
        (lambda f: 0.2 * f, 1),
        (lambda f: 0.012 * f**2 + 0.04 * np.sin(9 * f), 1),
    ],
    ids=[
        "straight",
        "gentle-left",
        "gentle-right",
        "sharp",
        "changing-curvature",
        "lateral-offset",
        "heading-error",
        "noisy",
    ],
)
def test_tracking_scenarios(curve, sign):
    cfg, ctrl = setup_controller()
    command = ctrl.compute(path(cfg, curve), VehicleState(speed_mps=2.0))
    assert command.debug.solver_success, command.debug.controller_fallback_reason
    if sign:
        assert sign * command.steering > 0
    else:
        assert abs(command.steering) < 1e-4
    assert command.throttle * command.brake == 0
    assert np.all(ctrl.last_prediction[:, 3] >= -cfg.mpc.constraint_tolerance)
    assert np.all(ctrl.last_prediction[:, 3] <= cfg.mpc.max_speed_mps + 1e-5)
    delta = np.diff(np.r_[0.0, ctrl.warm[:, 0]])
    assert np.max(np.abs(delta)) <= cfg.mpc.max_steering_rate_rad_s * cfg.mpc.dt_s + 1e-5
    assert np.max(np.abs(ctrl.warm[:, 0])) <= cfg.mpc.max_steering_rad + 1e-5
    assert np.min(ctrl.warm[:, 1]) >= cfg.mpc.min_accel_mps2 - 1e-5
    assert np.max(ctrl.warm[:, 1]) <= cfg.mpc.max_accel_mps2 + 1e-5


def test_missing_path_stops_and_recovers():
    cfg, ctrl = setup_controller()
    straight = path(cfg, lambda f: f * 0)
    ctrl.compute(straight, VehicleState(speed_mps=2.0))
    missing = ctrl.compute(PathPlan(np.empty((0, 2))), VehicleState(speed_mps=2.0))
    assert missing.throttle == 0 and missing.brake == cfg.max_brake
    assert missing.debug.controller_fallback and ctrl.warm is None
    ctrl.last_time = time.perf_counter() - cfg.mpc.dt_s
    recovered = ctrl.compute(straight, VehicleState(speed_mps=2.0))
    assert recovered.debug.solver_success


@pytest.mark.parametrize(
    "state",
    [
        VehicleState(valid=False),
        VehicleState(speed_mps=float("nan")),
        VehicleState(position=(0, float("inf"), 0)),
    ],
)
def test_invalid_state_brakes(state):
    cfg, ctrl = setup_controller()
    cmd = ctrl.compute(path(cfg, lambda f: f * 0), state)
    assert cmd.throttle == 0 and cmd.brake == cfg.max_brake
    assert cmd.debug.controller_fallback_reason == "invalid vehicle state"


def test_gate_hold_uses_stanley_and_gate_stop_brakes():
    cfg, ctrl = setup_controller()
    plan = path(cfg, lambda f: 0.02 * f**2)
    plan.fallback_active, plan.speed_scale = True, 0.4
    with patch.object(ctrl, "_solve") as solve:
        cmd = ctrl.compute(plan, VehicleState(speed_mps=2.0))
    solve.assert_not_called()
    assert cmd.debug.controller_fallback and cmd.steering > 0
    plan.speed_scale = 0
    cmd = ctrl.compute(plan, VehicleState(speed_mps=2.0))
    assert cmd.throttle == 0 and cmd.brake == cfg.max_brake


@pytest.mark.parametrize("failure", [SolveTimeout("deadline"), RuntimeError("solver exception")])
def test_solver_exceptions_fallback(failure, caplog):
    cfg, ctrl = setup_controller()
    with patch.object(ctrl, "_solve", side_effect=failure):
        cmd = ctrl.compute(path(cfg, lambda f: 0.02 * f**2), VehicleState(speed_mps=2.0))
    assert cmd.debug.controller_fallback and not cmd.debug.solver_success
    assert "MPC fallback" in caplog.text and cmd.steering > 0


def test_unsuccessful_and_nonfinite_solver_result():
    cfg, ctrl = setup_controller()
    plan = path(cfg, lambda f: f * 0)
    with patch.object(
        ctrl, "_solve", return_value=(Mock(success=False, message="iterations"), None)
    ):
        cmd = ctrl.compute(plan, VehicleState(speed_mps=2.0))
    assert cmd.debug.controller_fallback
    result = Mock(success=True, x=np.full(cfg.mpc.horizon * 2, np.nan), fun=0)
    with patch("offroad_autonomy.control.mpc_controller.minimize", return_value=result):
        cmd = ctrl.compute(plan, VehicleState(speed_mps=2.0))
    assert "infeasible" in cmd.debug.controller_fallback_reason


def test_late_success_is_rejected():
    cfg, ctrl = setup_controller()
    ctrl.cfg.max_solve_time_s = 0.001

    def late(*args):
        time.sleep(0.005)
        return Mock(success=True), None

    with patch.object(ctrl, "_solve", side_effect=late):
        cmd = ctrl.compute(path(cfg, lambda f: f * 0), VehicleState(speed_mps=2.0))
    assert cmd.debug.controller_fallback and "time budget" in cmd.debug.solver_status


def test_model_and_analytic_sensitivities():
    np.testing.assert_allclose(bicycle_step([0, 0, 0, 2], [0, 1], 0.25, 2.6), [0.5, 0, 0, 2.25])
    controls = np.array([[0.1, 0.2], [0.15, -0.2], [-0.1, 0.1]])
    initial = np.array([1.0, 2.0, 0.2, 3.0])
    _, jac = rollout_with_jacobian(initial, controls, 0.25, 2.6)
    numerical = approx_derivative(
        lambda u: rollout_with_jacobian(initial, u.reshape(-1, 2), 0.25, 2.6)[0].ravel(),
        controls.ravel(),
    )
    np.testing.assert_allclose(jac.reshape(-1, controls.size), numerical, atol=1e-7)


def test_curvature_reduces_reference_speed():
    cfg, straight_ctrl = setup_controller()
    curve_ctrl = MPCController(cfg)
    straight_ctrl.compute(path(cfg, lambda f: f * 0), VehicleState(speed_mps=3.0))
    curve_ctrl.compute(path(cfg, lambda f: 0.04 * f**2), VehicleState(speed_mps=3.0))
    assert curve_ctrl.last_reference[:, 3].mean() < straight_ctrl.last_reference[:, 3].mean()


def test_acceleration_pedals_and_reset():
    cfg, ctrl = setup_controller()
    for accel in np.linspace(-2, 1, 20):
        throttle, brake = ctrl.acceleration_to_pedals(accel)
        assert throttle * brake == 0
        assert 0 <= throttle <= cfg.max_throttle and 0 <= brake <= cfg.max_brake
        assert (throttle > 0) == (accel >= 0)
    ctrl.compute(path(cfg, lambda f: f * 0.1), VehicleState(speed_mps=2.0))
    ctrl.reset()
    assert ctrl.warm is None and ctrl.last_time is None
    np.testing.assert_array_equal(ctrl.previous, [0, 0])


def test_projection_offset_and_config_switch():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/mpc.yaml")
    assert cfg.controller == "mpc" and cfg.mpc.horizon == 12
    assert load_config(root / "configs/default.yaml").controller == "stanley"
    ctrl = MPCController(cfg)
    plan = path(cfg, lambda f: f * 0)
    ground = ctrl.fallback.tracking_reference(plan)
    assert ground.start == pytest.approx(1.0, abs=1e-4)
    from offroad_autonomy.control.mpc_reference import build_reference

    initial = np.array([3.0, 0, 0, 2.0])
    ref = build_reference(ground, initial, cfg, 2.0)
    assert ref[0, 0] == pytest.approx(3.0)
    assert np.max(ref[:, 0]) <= ground.end + cfg.mpc.camera_ahead_of_rear_axle_m


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt_s": 0},
        {"horizon": 1},
        {"max_speed_mps": math.nan},
        {"rd_steering": -1},
        {"min_accel_mps2": 1},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        MPCConfig(**kwargs)


def test_beamng_failed_poll_marks_state_invalid():
    from offroad_autonomy.simulation.beamng_client import BeamNGClient

    client = BeamNGClient(PipelineConfig())
    client._vehicle = Mock()
    client._vehicle.sensors.poll.side_effect = RuntimeError("connection")
    assert not client.get_vehicle_state().valid


def test_pipeline_selects_mpc():
    from contextlib import ExitStack

    from tests.test_pipeline import _make_config, _pipeline

    with ExitStack() as stack:
        pipe, _, _ = _pipeline(stack, _make_config(controller="mpc"))
        assert isinstance(pipe.controller, MPCController)


def test_fallback_switch_respects_rate():
    cfg, ctrl = setup_controller()
    right = path(cfg, lambda f: 0.02 * f**2)
    first = ctrl.compute(right, VehicleState(speed_mps=2.0))
    ctrl.last_time = time.perf_counter() - cfg.mpc.dt_s
    with patch.object(ctrl, "_solve", side_effect=RuntimeError("injected")):
        second = ctrl.compute(path(cfg, lambda f: -0.03 * f**2), VehicleState(speed_mps=2.0))
    assert abs(second.steering - first.steering) * ctrl.max_wheel <= 0.1 + 1e-5


def test_delayed_closed_loop_recovers_offset_and_completes_s_bend():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    from sbend_sim import simulate

    cfg = load_config(root / "configs/mpc.yaml")
    # A generous test budget avoids CI scheduling affecting trajectory assertions;
    # production's 80 ms deadline is tested separately and measured in benchmarks.
    cfg.mpc.max_solve_time_s = 1.0
    result = simulate(cfg, radius=7.0, start_lateral_m=0.75, completion_margin_m=15.0)
    assert result.completed
    assert result.max_lateral_error < 1.5
    assert result.mean_abs_error < 0.5
    assert sum(row["solver_success"] for row in result.log) > 0.95 * len(result.log)


def test_solver_speed_caps_and_short_path_stop():
    cfg, ctrl = setup_controller()
    short = path(cfg, lambda f: f * 0)
    cam = CameraModel(cfg.camera, cfg.preprocess_width, cfg.preprocess_height)
    short.centerline = cam.ground_to_image(np.linspace(0.5, 1.0, 10), np.zeros(10))
    cmd = ctrl.compute(short, VehicleState(speed_mps=2.0))
    assert cmd.throttle == 0 and cmd.brake > 0
    fast = ctrl.compute(path(cfg, lambda f: f * 0), VehicleState(speed_mps=5.0))
    assert fast.throttle == 0 and fast.brake > 0 and fast.debug.controller_fallback


def test_elapsed_time_limits_first_command():
    cfg, ctrl = setup_controller()
    ctrl.last_time = time.perf_counter() - 0.05
    with patch(
        "offroad_autonomy.control.mpc_controller.time.perf_counter",
        return_value=ctrl.last_time + 0.05,
    ):
        cmd = ctrl.compute(path(cfg, lambda f: 0.03 * f**2), VehicleState(speed_mps=2.0))
    assert cmd.debug.solver_success
    assert abs(cmd.steering) * ctrl.max_wheel <= cfg.mpc.max_steering_rate_rad_s * 0.05 + 1e-5


@pytest.mark.parametrize("repeated", [False, True])
def test_application_camera_watchdog_parks(repeated, monkeypatch):
    import itertools

    import offroad_autonomy.main as app
    from offroad_autonomy.runtime.timing import RuntimeStats
    from offroad_autonomy.types import CameraFrame

    cfg, _ = setup_controller()
    cfg.ui_headless = True
    client, pipeline = Mock(), Mock()
    pipeline.stats = RuntimeStats()
    pipeline.ego_coverage = 0.0
    calls = []

    def capture():
        calls.append(1)
        if len(calls) == 6:
            app._shutdown = True
        return CameraFrame(np.zeros((2, 2, 3)), is_new=False) if repeated else None

    client.capture_frame.side_effect = capture
    monkeypatch.setattr("sys.argv", ["offroad-autonomy", "--headless"])
    monkeypatch.setattr(app, "load_config", lambda _: cfg)
    monkeypatch.setattr(app, "BeamNGClient", lambda _, **__: client)
    monkeypatch.setattr(app, "AutonomyPipeline", lambda _: pipeline)
    monkeypatch.setattr(app, "_log_runtime", Mock())
    monkeypatch.setattr(app.signal, "signal", Mock())
    monkeypatch.setattr(app.time, "sleep", Mock())
    ticks = itertools.count(step=0.5)
    monkeypatch.setattr(app.time, "perf_counter", lambda: next(ticks))
    app.main()
    client.park.assert_called_once()
    client.send_controls.assert_not_called()
    pipeline.step_result.assert_not_called()
    client.disconnect.assert_called_once()
