"""The position sampler must run beside the control loop, not inside it."""

import time
from itertools import pairwise
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from offroad_autonomy.simulation.position_sampler import PositionSampler


def _fake_bng(replies):
    bng = Mock()
    bng.control = SimpleNamespace(queue_lua_command=Mock(side_effect=replies))
    return bng


def _start(bng, rate_hz=200.0):
    sampler = PositionSampler("host", 64256, "ego", rate_hz)
    with patch("beamngpy.BeamNGpy", return_value=bng):
        sampler.start()
    return sampler


def test_samples_arrive_with_host_times_and_drain_empties_them():
    replies = [{"x": float(i), "y": 2.0, "z": 3.0} for i in range(1000)]
    bng = _fake_bng(replies)
    sampler = _start(bng)
    deadline = time.perf_counter() + 2.0
    while len(sampler._samples) < 3 and time.perf_counter() < deadline:
        time.sleep(0.01)
    sampler.stop()
    samples = sampler.drain()
    assert len(samples) >= 3
    assert [s[1] for s in samples] == sorted(s[1] for s in samples)
    assert all(a[0] <= b[0] for a, b in pairwise(samples))
    assert sampler.drain() == []
    bng.open.assert_called_once_with(launch=False)


def test_stop_disconnects_and_never_quits_the_simulator():
    bng = _fake_bng([{"x": 0.0, "y": 0.0, "z": 0.0}] * 1000)
    sampler = _start(bng)
    sampler.stop()
    bng.disconnect.assert_called_once()
    bng.close.assert_not_called()


def test_a_missing_vehicle_fails_at_start_not_silently_later():
    bng = _fake_bng([None])
    with pytest.raises(RuntimeError, match="cannot find the vehicle"):
        _start(bng)
    bng.disconnect.assert_called_once()


def test_failed_reads_are_counted_and_sampling_continues():
    replies = [{"x": 0.0, "y": 0.0, "z": 0.0}, OSError("timeout")] + [
        {"x": 1.0, "y": 0.0, "z": 0.0}
    ] * 1000
    sampler = _start(_fake_bng(replies))
    deadline = time.perf_counter() + 2.0
    while len(sampler._samples) < 2 and time.perf_counter() < deadline:
        time.sleep(0.01)
    sampler.stop()
    assert sampler.failures == 1
    assert sampler.drain()
