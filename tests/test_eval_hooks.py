"""Evaluation hooks: gate verdicts, simulator truth reads and their settings."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from offroad_autonomy.main import build_parser
from offroad_autonomy.planning.centerline_planner import CenterlinePlanner
from offroad_autonomy.simulation.beamng_client import BeamNGClient
from offroad_autonomy.types import PerceptionResult, PipelineConfig, StabilizedResult
from offroad_autonomy.utils.config import load_config


def test_planner_keeps_the_gate_decision_of_its_last_frame():
    planner = CenterlinePlanner(PipelineConfig())
    assert planner.last_gate is None
    mask = np.zeros((465, 720), bool)
    raw = PerceptionResult(mask=mask, confidences=[], inference_time_ms=0.0)
    planner.plan(StabilizedResult(mask=mask, raw_result=raw))
    assert planner.last_gate is not None
    assert not planner.last_gate.ok


def test_true_position_reads_the_vehicle_state():
    client = BeamNGClient(PipelineConfig())
    client._vehicle = SimpleNamespace(sensors=Mock(), state={"pos": (1.0, 2.0, 3.0)})
    assert client.true_position() == (1.0, 2.0, 3.0)
    client._vehicle.sensors.poll.assert_called_once_with("state")


def test_a_failed_true_position_read_does_not_end_the_run():
    client = BeamNGClient(PipelineConfig())
    sensors = Mock()
    sensors.poll.side_effect = OSError("socket closed")
    client._vehicle = SimpleNamespace(sensors=sensors, state={})
    assert client.true_position() is None
    assert BeamNGClient(PipelineConfig()).true_position() is None


def test_simulator_version_is_asked_of_the_game_engine():
    client = BeamNGClient(PipelineConfig())
    control = Mock()
    control.queue_lua_command.return_value = "0.38.5.0.19602"
    client._bng = SimpleNamespace(control=control)
    assert client.simulator_version() == "0.38.5.0.19602"
    control.queue_lua_command.side_effect = OSError("gone")
    assert client.simulator_version() is None


def test_shipped_evaluation_settings():
    config = load_config("configs/default.yaml")
    assert config.eval_route_length_m == 335.0
    assert config.eval_max_duration_s == 400.0
    assert config.eval_trim_start_s == 3.0


def test_invalid_route_length_is_refused(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("evaluation:\n  route_length_m: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="route_length_m"):
        load_config(path)


def test_eval_log_flag_defaults_off():
    assert build_parser().parse_args([]).eval_log == ""
    assert build_parser().parse_args(["--eval-log", "out/r"]).eval_log == "out/r"
