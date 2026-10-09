"""Tests for the harness loop, capture signatures, timing, routes and ground-truth scoring."""

import json
from unittest.mock import Mock

import numpy as np
import pytest

from offroad_autonomy.contract import SensorSpec, StackOutput, camera_sensor
from offroad_autonomy.runtime.benchmark import BenchmarkRecorder, comparison_table
from offroad_autonomy.runtime.route import Route, RouteRecorder, route_path
from offroad_autonomy.runtime.timing import RuntimeStats
from offroad_autonomy.simulation.beamng_client import frame_signature
from offroad_autonomy.types import (
    DEFAULT_CAMERA,
    CameraFrame,
    ControlCommand,
    HarnessConfig,
    VehicleState,
)


class FakeStack:
    """The smallest thing the harness accepts: no perception, no planner."""

    def __init__(self, outputs) -> None:
        self._outputs = list(outputs)
        self.steps = 0
        self.resets = 0

    def sensors(self) -> list[SensorSpec]:
        return [camera_sensor(DEFAULT_CAMERA)]

    def step(self, obs) -> StackOutput:
        assert set(obs.sensors) == {"dashcam"}
        output = self._outputs[min(self.steps, len(self._outputs) - 1)]
        self.steps += 1
        return output

    def reset(self) -> None:
        self.resets += 1


def _client(steps_before_shutdown: int | None = None):
    import offroad_autonomy.main as app

    client = Mock()
    frame = CameraFrame(image=np.zeros((620, 960, 3), np.uint8), timestamp=0.0, frame_id=1)
    calls = []

    def capture():
        calls.append(1)
        if steps_before_shutdown is not None and len(calls) > steps_before_shutdown:
            app._shutdown = True
            return None
        return {"dashcam": frame}

    client.capture.side_effect = capture
    client.get_vehicle_state.return_value = VehicleState(speed_mps=3.0)
    return client


def test_connection_failure_exits_cleanly_and_disconnects(run_harness, monkeypatch):
    import offroad_autonomy.main as app
    from offroad_autonomy.simulation.beamng_client import BeamNGConnectionError

    client = Mock()
    client.connect.side_effect = BeamNGConnectionError("Set BEAMNG_HOME to start BeamNG")
    stack = FakeStack([StackOutput(ControlCommand())])
    error_log = Mock()
    monkeypatch.setattr(app.logger, "error", error_log)
    with pytest.raises(SystemExit) as caught:
        run_harness(stack, client)
    assert caught.value.code == 1
    assert "Set BEAMNG_HOME" in str(error_log.call_args.args[1])
    client.disconnect.assert_called_once()
    assert stack.steps == 0


def test_a_stack_asking_for_a_safe_stop_parks_without_resending_its_throttle(
    run_harness, monkeypatch
):
    stack = FakeStack([StackOutput(ControlCommand(throttle=0.45), safe_stop="test no-road stop")])
    client = _client(steps_before_shutdown=1)

    run_harness(stack, client)

    client.park.assert_called_once()
    applied = client.send_controls.call_args_list[0].args[0]
    assert applied.throttle == 0 and applied.parkingbrake == 1


def test_the_harness_stuck_detector_applies_to_any_stack(run_harness, monkeypatch):
    stack = FakeStack([StackOutput(ControlCommand(throttle=0.5))])
    client = _client(steps_before_shutdown=3)
    client.get_vehicle_state.return_value = VehicleState(speed_mps=0.0)
    config = HarnessConfig(ui_headless=True, safety_stuck_time_s=1e-9)

    run_harness(stack, client, config=config)

    client.park.assert_called_once()


def test_stuck_detector_fires_after_its_time_without_motion():
    """Pushing on the throttle without moving for the configured time is a stop."""
    from offroad_autonomy.main import _StuckDetector

    detector = _StuckDetector(speed_mps=0.4, throttle_min=0.15, time_s=3.0)

    detector.update(0.0, speed_mps=0.1, throttle=0.5)
    triggered, reason = detector.update(4.0, speed_mps=0.1, throttle=0.5)

    assert triggered
    assert reason == "vehicle stuck"


def test_stack_keeps_stepping_under_safe_stop_and_its_commands_are_ignored(
    run_harness, monkeypatch
):
    outputs = [
        StackOutput(ControlCommand(throttle=0.3), safe_stop="no road"),
        StackOutput(ControlCommand(throttle=0.9)),
    ]
    stack = FakeStack(outputs)
    client = _client(steps_before_shutdown=3)

    run_harness(stack, client)

    assert stack.steps == 3
    assert all(c.args[0].throttle == 0 for c in client.send_controls.call_args_list)


def test_benchmark_run_writes_a_ground_truth_report(run_harness, monkeypatch, tmp_path):
    routes = tmp_path / "routes"
    route_file = route_path(routes, "automation_test_track", 0)
    route_file.parent.mkdir(parents=True)
    route_file.write_text(json.dumps({"points": [[0.0, 0.0], [100.0, 0.0]]}), encoding="utf-8")
    out = tmp_path / "report.json"
    stack = FakeStack([StackOutput(ControlCommand(throttle=0.3), metrics={"seg_ms": 5.0})])
    client = _client(steps_before_shutdown=5)
    positions = iter([(float(x), 0.5, 0.0) for x in range(100)])
    client.get_vehicle_state.side_effect = lambda: VehicleState(
        position=next(positions), speed_mps=3.0
    )
    config = HarnessConfig(ui_headless=True, benchmark_routes_dir=str(routes))

    run_harness(
        stack,
        client,
        argv=("--headless", "--benchmark-out", str(out), "--benchmark-warmup", "0"),
        config=config,
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["stack"] == config.stack
    assert report["route"] == str(route_file)
    assert report["cross_track_mean_m"] == pytest.approx(0.5)
    assert report["departures"] == 0
    assert report["distance_m"] == pytest.approx(4.0)
    assert report["stack_metrics"] == {"seg_ms": 5.0}
    assert report["safe_stops"] == 0


def test_record_route_saves_the_true_path(run_harness, monkeypatch, tmp_path):
    out = tmp_path / "route.json"
    stack = FakeStack([StackOutput(ControlCommand())])
    client = _client(steps_before_shutdown=4)
    positions = iter([(float(x), 0.0, 0.0) for x in range(10)])
    client.get_vehicle_state.side_effect = lambda: VehicleState(position=next(positions))

    run_harness(stack, client, argv=("--headless", "--record-route", str(out)))

    route = Route.load(out)
    assert route.length_m == pytest.approx(3.0)


def test_map_and_spawn_can_be_overridden_for_every_run(run_harness, monkeypatch):
    config = HarnessConfig(ui_headless=True)
    stack = FakeStack([StackOutput(ControlCommand())])
    client = _client(steps_before_shutdown=0)

    run_harness(stack, client, argv=("--headless", "--map", "utah", "--spawn", "1"), config=config)

    assert (config.beamng_map, config.beamng_spawn_index) == ("utah", 1)


def test_frame_signature_detects_a_new_render():
    buffer = bytearray(np.random.default_rng(0).integers(0, 255, 960 * 620 * 4, dtype=np.uint8))
    before = frame_signature(buffer)
    buffer[997 * 10] ^= 0xFF

    assert frame_signature(buffer) != before
    assert frame_signature(None) is None


def test_runtime_stats_mean_p95_and_rate():
    stats = RuntimeStats(window=100)
    for value in range(1, 101):
        stats.record("stage", float(value))
    for tick in range(11):
        stats.tick(now=tick * 0.05)

    summary = stats.stage("stage")
    assert summary.mean_ms == pytest.approx(50.5)
    assert summary.p95_ms == pytest.approx(95.05)
    assert stats.fps() == pytest.approx(20.0)


def test_runtime_stats_window_is_bounded():
    stats = RuntimeStats(window=10)
    for value in range(1000):
        stats.record("stage", float(value))

    assert stats.stage("stage").count == 10
    assert stats.stage("stage").mean_ms == pytest.approx(994.5)


# ------------------------------------------------------------------ routes


def test_route_locates_cross_track_and_progress():
    route = Route(np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]))

    assert route.length_m == pytest.approx(20.0)
    assert route.locate(5.0, 1.0) == pytest.approx((1.0, 5.0))
    assert route.locate(11.0, 5.0) == pytest.approx((1.0, 15.0))


def test_closed_circuit_starts_at_zero_progress_and_ends_complete():
    # A square lap recorded from the spawn ends where it began, so a global
    # search would put the vehicle at the finish before it has moved.
    side = np.arange(0.0, 50.0, 0.5)
    lap = np.concatenate(
        [
            np.stack([side, np.zeros_like(side)], axis=1),
            np.stack([np.full_like(side, 50.0), side], axis=1),
            np.stack([50.0 - side, np.full_like(side, 50.0)], axis=1),
            np.stack([np.zeros_like(side), 50.0 - side], axis=1),
            [[0.0, 0.0]],
        ]
    )
    route = Route(lap)

    assert route.locate(0.0, 0.2)[1] == pytest.approx(0.0)
    progress = 0.0
    for x, y in lap[::4]:
        progress = max(progress, route.locate(float(x), float(y) + 0.2)[1])
    assert progress == pytest.approx(route.length_m, abs=1.0)


def test_route_recorder_drops_points_closer_than_its_spacing(tmp_path):
    recorder = RouteRecorder(spacing_m=0.5)
    for x in np.arange(0.0, 5.0, 0.1):
        recorder.add(VehicleState(position=(float(x), 0.0, 0.0)))
    recorder.add(VehicleState(position=(100.0, 0.0, 0.0), valid=False))

    path = recorder.save(tmp_path / "r.json", "utah", 1)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert (data["map"], data["spawn_index"]) == ("utah", 1)
    assert len(data["points"]) == 10
    assert Route.load(path).length_m == pytest.approx(4.5, abs=0.11)


def test_too_short_a_drive_is_not_saved_as_a_route(tmp_path):
    with pytest.raises(ValueError, match="too short"):
        RouteRecorder(spacing_m=0.5).save(tmp_path / "r.json", "utah", 0)


# --------------------------------------------------------------- benchmark


def _recorder(route=None, warmup_s=0.0, **overrides) -> BenchmarkRecorder:
    from offroad_autonomy.types import DashboardThresholds

    config = HarnessConfig(
        stack="pkg.stack:build",
        beamng_map="utah",
        dashboard_thresholds=DashboardThresholds(latency_budget_ms=100.0),
        **overrides,
    )
    return BenchmarkRecorder("t", "cfg.yaml", config, route, warmup_s)


def _add(recorder, t, x, y, steer=0.0, active=True, step_ms=20.0, metrics=None):
    recorder.add(
        now=t,
        state=VehicleState(position=(x, y, 0.0)),
        command=ControlCommand(steering=steer),
        autopilot_active=active,
        primary_ms=30.0,
        step_ms=step_ms,
        full_ms=40.0,
        stack_metrics=metrics,
    )


def test_benchmark_scores_departures_and_completion_from_the_true_position():
    route = Route(np.array([[0.0, 0.0], [100.0, 0.0]]))
    recorder = _recorder(route)
    lateral = [0.0] * 10 + [3.0] * 5 + [0.0] * 10 + [2.5] * 5 + [0.0] * 10

    for step, offset in enumerate(lateral):
        _add(recorder, step * 0.1, float(step), offset)

    report = recorder.report(RuntimeStats())
    assert report["departures"] == 2
    assert report["completion_pct"] == pytest.approx(39.0)
    assert report["cross_track_max_m"] == pytest.approx(3.0)
    assert report["frames"] == len(lateral)


def test_scoring_stops_at_the_first_safe_stop():
    route = Route(np.array([[0.0, 0.0], [100.0, 0.0]]))
    recorder = _recorder(route)
    for step in range(10):
        _add(recorder, step * 0.1, float(step), 0.0)
    for step in range(10, 20):
        # The operator drives on by hand after the stop: not the stack's distance.
        _add(recorder, step * 0.1, float(step), 5.0, active=False)

    report = recorder.report(RuntimeStats())
    assert report["safe_stops"] == 1
    assert report["ended_in_safe_stop"] is True
    assert report["distance_m"] == pytest.approx(9.0)
    assert report["departures"] == 0
    assert report["frames"] == 20


def test_without_a_route_driving_scores_are_null_but_distance_is_kept():
    recorder = _recorder()
    for step in range(5):
        _add(recorder, step * 0.1, float(step), 0.0)

    report = recorder.report(RuntimeStats())
    assert report["route"] is None
    assert report["cross_track_mean_m"] is None
    assert report["departures"] is None
    assert report["distance_m"] == pytest.approx(4.0)


def test_smoothness_counts_sign_changes_outside_the_deadband():
    recorder = _recorder(benchmark_steer_sign_deadband=0.05)
    steering = [0.2, -0.2, 0.02, -0.02, 0.2, -0.2]
    for step, steer in enumerate(steering):
        _add(recorder, step * 1.0, 0.0, 0.0, steer=steer)

    report = recorder.report(RuntimeStats())
    # 0.2 -> -0.2 -> (dither ignored) -> 0.2 -> -0.2 is three real reversals in 5 s.
    assert report["steering_sign_changes_per_min"] == pytest.approx(36.0)
    assert report["steering_rate_mean_per_s"] == pytest.approx(0.256)


def test_step_budget_overruns_and_stack_metrics_are_reported_apart():
    recorder = _recorder()
    for step in range(4):
        _add(
            recorder,
            step * 0.1,
            0.0,
            0.0,
            step_ms=150.0 * (step % 2),
            metrics={"depth_coverage": 0.5 * step, "edge_m": float("inf")},
        )

    report = recorder.report(RuntimeStats())
    assert report["step_over_budget"] == 2
    assert report["stack_metrics"] == {"depth_coverage": 0.75}


def test_warmup_is_left_out_of_timing_but_not_out_of_driving():
    recorder = _recorder(warmup_s=1.0)
    for step in range(20):
        _add(recorder, step * 0.1, float(step), 0.0)

    report = recorder.report(RuntimeStats())
    assert report["frames"] == 10
    assert report["distance_m"] == pytest.approx(19.0)


def test_comparison_table_lines_up_runs_and_marks_missing_values():
    reports = [
        {"label": "a", "stack": "mono", "main_fps": 7.5, "departures": None},
        {"label": "b", "stack": "stereo", "main_fps": 12.25, "departures": 3},
    ]

    lines = comparison_table(reports)
    markdown = comparison_table(reports, markdown=True)

    assert lines[0].startswith("run")
    assert len({len(line.rstrip()) for line in lines[:2]}) == 1
    assert "-" in lines[2].split()
    assert markdown[1].startswith("|---")
    assert "12.2" in markdown[3]
