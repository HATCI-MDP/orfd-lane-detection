"""Daylight must reach BeamNG exactly as asked, frozen, or the run must not start."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from offroad_autonomy.main import build_parser
from offroad_autonomy.simulation import beamng_client
from offroad_autonomy.simulation.beamng_client import BeamNGClient
from offroad_autonomy.simulation.daylight import daylight_to_tod, tod_to_clock
from offroad_autonomy.types import PipelineConfig
from offroad_autonomy.utils.config import load_config


@pytest.mark.parametrize(
    ("percent", "tod", "clock"),
    [(100, 0.0, "12:00:00"), (74, 0.13, "15:07:12"), (50, 0.25, "18:00:00"), (0, 0.5, "00:00:00")],
)
def test_daylight_walks_the_afternoon_from_noon_to_midnight(percent, tod, clock):
    assert daylight_to_tod(percent) == pytest.approx(tod)
    assert tod_to_clock(daylight_to_tod(percent)) == clock


@pytest.mark.parametrize("percent", [-1, 100.5, float("nan")])
def test_daylight_outside_zero_to_hundred_is_refused(percent):
    with pytest.raises(ValueError, match="between 0 and 100"):
        daylight_to_tod(percent)


def _client(percent, *readbacks, monkeypatch):
    monkeypatch.setattr(beamng_client.time, "sleep", lambda _s: None)
    client = BeamNGClient(PipelineConfig(beamng_daylight_pct=percent))
    env = Mock()
    env.get_tod.side_effect = [{"time": t} for t in readbacks]
    client._bng = SimpleNamespace(env=env)
    return client, env


def test_daylight_sets_and_freezes_the_clock(monkeypatch):
    client, env = _client(74, 0.13, 0.13, monkeypatch=monkeypatch)
    client._apply_daylight()
    env.set_tod.assert_called_once_with(tod=pytest.approx(0.13), play=False)


def test_noon_read_back_as_one_is_accepted(monkeypatch):
    client, _ = _client(100, 1.0, 1.0, monkeypatch=monkeypatch)
    client._apply_daylight()


def test_no_daylight_leaves_the_map_time_alone(monkeypatch):
    client, env = _client(None, monkeypatch=monkeypatch)
    client._apply_daylight()
    env.set_tod.assert_not_called()


def test_a_time_beamng_did_not_take_stops_the_run(monkeypatch):
    client, _ = _client(74, 0.0, 0.0, monkeypatch=monkeypatch)
    with pytest.raises(RuntimeError, match="expected"):
        client._apply_daylight()


def test_a_running_day_cycle_stops_the_run(monkeypatch):
    client, _ = _client(74, 0.13, 0.1305, monkeypatch=monkeypatch)
    with pytest.raises(RuntimeError, match="still running"):
        client._apply_daylight()


def test_cli_accepts_percent_and_refuses_out_of_range(capsys):
    assert build_parser().parse_args(["--daylight", "74"]).daylight == 74.0
    assert build_parser().parse_args([]).daylight is None
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--daylight", "150"])
    assert "between 0 and 100" in capsys.readouterr().err


def test_shipped_config_keeps_the_map_time_of_day():
    assert load_config("configs/default.yaml").beamng_daylight_pct is None


def test_config_daylight_is_read_and_validated(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("beamng:\n  daylight_pct: 74\n", encoding="utf-8")
    assert load_config(path).beamng_daylight_pct == 74.0
    path.write_text("beamng:\n  daylight_pct: 101\n", encoding="utf-8")
    with pytest.raises(ValueError, match="daylight_pct"):
        load_config(path)
