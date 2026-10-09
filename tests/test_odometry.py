"""Dead reckoning from wheel speed and yaw rate, and the simulator's emulation of them."""

import math

import numpy as np
import pytest

from offroad_autonomy.planning.odometry import DeadReckoning, GroundPose
from offroad_autonomy.simulation.beamng_client import OdometryEmulator
from offroad_autonomy.types import VehicleSignals


def _reading(t: float, speed: float = 0.0, yaw_rate: float = 0.0) -> VehicleSignals:
    return VehicleSignals(speed_mps=speed, yaw_rate_rps=yaw_rate, timestamp_s=t)


def _heading(pose: GroundPose) -> float:
    return math.atan2(pose.forward_y, pose.forward_x)


def test_the_first_reading_starts_the_local_frame_at_the_origin():
    pose = DeadReckoning(1.0).update(_reading(5.0, speed=3.0, yaw_rate=0.4))

    assert (pose.x, pose.y, _heading(pose)) == (0.0, 0.0, 0.0)


def test_straight_driving_moves_along_the_heading():
    odometry = DeadReckoning(1.0)
    for t in (0.0, 0.25, 0.5):
        pose = odometry.update(_reading(t, speed=2.0))

    assert pose.x == pytest.approx(1.0)
    assert pose.y == pytest.approx(0.0)
    assert odometry.interval_s == pytest.approx(0.25)


def test_a_constant_turn_lands_exactly_on_its_circle():
    """The arc is integrated exactly, so coarse frame intervals add no error."""
    odometry = DeadReckoning(1.0)
    speed, yaw_rate = 2.0, 0.4
    for step in range(10):
        pose = odometry.update(_reading(0.25 * step, speed, yaw_rate))

    radius = speed / yaw_rate
    turned = yaw_rate * 0.25 * 9
    assert pose.x == pytest.approx(radius * math.sin(turned), abs=1e-9)
    # Positive yaw rate turns left, toward +y.
    assert pose.y == pytest.approx(radius * (1.0 - math.cos(turned)), abs=1e-9)
    assert _heading(pose) == pytest.approx(turned)


def test_reversing_moves_backwards():
    odometry = DeadReckoning(1.0)
    odometry.update(_reading(0.0))
    pose = odometry.update(_reading(0.5, speed=-1.0))

    assert pose.x == pytest.approx(-0.5)


@pytest.mark.parametrize(
    "bad",
    [None, VehicleSignals(valid=False), _reading(0.25, speed=float("nan"))],
)
def test_a_missing_or_invalid_reading_breaks_registration_and_restarts(bad):
    odometry = DeadReckoning(1.0)
    odometry.update(_reading(0.0, speed=2.0))
    odometry.update(_reading(0.25, speed=2.0))

    assert odometry.update(bad) is None
    # The next good reading starts a fresh frame rather than integrating across the hole.
    pose = odometry.update(_reading(0.5, speed=2.0))
    assert (pose.x, pose.y) == (0.0, 0.0)


@pytest.mark.parametrize("later", [1.6, -0.1])
def test_a_long_gap_or_a_clock_jump_breaks_registration(later):
    odometry = DeadReckoning(max_gap_s=1.0)
    odometry.update(_reading(0.0, speed=2.0))
    before = odometry.update(_reading(0.5, speed=2.0))

    assert odometry.update(_reading(0.5 + later, speed=2.0)) is None
    # Integration resumes from the reading after the gap, not across it.
    after = odometry.update(_reading(0.75 + later, speed=2.0))
    assert after.x == pytest.approx(before.x + 0.5)


def test_an_offset_is_taken_in_the_vehicle_frame():
    facing_y = GroundPose(x=1.0, y=2.0, forward_x=0.0, forward_y=1.0)

    ahead = facing_y.offset(forward_m=1.4, left_m=0.0)
    left = facing_y.offset(forward_m=0.0, left_m=1.0)

    assert (ahead.x, ahead.y) == pytest.approx((1.0, 3.4))
    assert (left.x, left.y) == pytest.approx((0.0, 2.0))


def _emulate(emulator, position, heading, speed, t):
    direction = (math.cos(heading), math.sin(heading), 0.0)
    velocity = (speed * direction[0], speed * direction[1], 0.0)
    return emulator.read(position, velocity, direction, t)


def test_emulated_speed_is_signed_along_the_vehicle_axis():
    emulator = OdometryEmulator()
    _emulate(emulator, (10.0, 5.0, 0.0), math.pi / 2, -1.0, 0.0)

    signals = _emulate(emulator, (10.0, 4.5, 0.0), math.pi / 2, -1.0, 0.5)

    assert signals.speed_mps == pytest.approx(-1.0)
    assert signals.yaw_rate_rps == pytest.approx(0.0)


def test_emulated_yaw_rate_is_the_heading_change_positive_to_the_left():
    emulator = OdometryEmulator()
    _emulate(emulator, (0.0, 0.0, 0.0), math.pi - 0.05, 0.0, 0.0)

    # Across the +-pi seam, turning left by 0.1 rad.
    signals = _emulate(emulator, (0.0, 0.0, 0.0), -math.pi + 0.05, 0.0, 0.25)

    assert signals.yaw_rate_rps == pytest.approx(0.4)


def test_the_first_reading_and_a_repeated_reading_hold_no_invented_motion():
    emulator = OdometryEmulator()
    first = _emulate(emulator, (0.0, 0.0, 0.0), 0.0, 2.0, 1.0)
    assert first.yaw_rate_rps == 0.0
    assert first.speed_mps == pytest.approx(2.0)

    moved = _emulate(emulator, (0.5, 0.0, 0.0), 0.1, 2.0, 1.25)
    repeated = _emulate(emulator, (0.5, 0.0, 0.0), 0.1, 2.0, 1.25)
    assert (repeated.speed_mps, repeated.yaw_rate_rps) == (moved.speed_mps, moved.yaw_rate_rps)


def test_a_degenerate_direction_is_an_invalid_reading():
    signals = OdometryEmulator().read((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0)

    assert not signals.valid


def test_emulated_signals_reproduce_the_true_motion():
    """A weaving, accelerating vehicle whose reported origin is ahead of its
    rear axle, as BeamNG's is: over the grid's 2 s memory, dead reckoning from
    the emulated signals stays within 2 cm of the true relative motion."""
    emulator = OdometryEmulator()
    odometry = DeadReckoning(1.0)
    x = y = heading = 0.0
    speed, dt, origin_ahead = 0.5, 0.25, 1.1
    truth, estimate = [], []
    for frame in range(60):
        position = (x + origin_ahead * math.cos(heading), y + origin_ahead * math.sin(heading), 0)
        pose = odometry.update(_emulate(emulator, position, heading, speed, frame * dt))
        truth.append((x, y, heading))
        estimate.append((pose.x, pose.y, _heading(pose)))
        steering = 0.6 * math.sin(0.7 * frame * dt)
        for _ in range(40):
            heading += speed * math.tan(steering * math.radians(32.0)) / 2.6 * dt / 40
            x += speed * math.cos(heading) * dt / 40
            y += speed * math.sin(heading) * dt / 40
            speed = min(speed + 2.0 * dt / 40, 6.0)

    def relative(a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        c, s = math.cos(a[2]), math.sin(a[2])
        return np.array([c * dx + s * dy, -s * dx + c * dy, b[2] - a[2]])

    window = 8
    errors = [
        relative(truth[i], truth[i + window]) - relative(estimate[i], estimate[i + window])
        for i in range(len(truth) - window)
    ]
    assert np.abs(np.array(errors))[:, :2].max() < 0.02
