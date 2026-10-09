"""Dead reckoning from the signals every production vehicle reports.

The grid stays centred on the vehicle, so it needs only how far the vehicle
moved and turned between frames, never where it is on the map. Wheel-speed
sensors and the stability-control gyro report exactly that on any modern
vehicle's CAN bus, so they are integrated into a local odometry frame: smooth
and accurate over the grid's few seconds of memory, free to drift over
minutes, and tied to nothing in the world.

Each reading's speed and yaw rate are the means over the interval since the
previous reading, so they are integrated exactly along the arc they describe
(the velocity motion model, Thrun, Burgard and Fox, *Probabilistic Robotics*,
section 5.3). The integrated point is the rear axle: on a vehicle that does
not slide sideways it is the point that moves along the vehicle's heading.
The camera, 1.4 m ahead of it, swings sideways in every turn, so integrating
the camera as if it moved straight ahead would misplace the memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from offroad_autonomy.types import VehicleSignals


@dataclass(frozen=True)
class GroundPose:
    """A point on the ground in the odometry frame, and which way the vehicle
    faces there. The frame is right-handed with +Z up, like BeamNG's world."""

    x: float
    y: float
    forward_x: float
    forward_y: float

    @property
    def left_x(self) -> float:
        # Left is forward turned 90 deg anticlockwise in a right-handed frame with +Z up.
        return -self.forward_y

    @property
    def left_y(self) -> float:
        return self.forward_x

    def offset(self, forward_m: float, left_m: float) -> GroundPose:
        """The point ``forward_m`` ahead and ``left_m`` to the left, facing the same way."""
        return GroundPose(
            x=self.x + forward_m * self.forward_x + left_m * self.left_x,
            y=self.y + forward_m * self.forward_y + left_m * self.left_y,
            forward_x=self.forward_x,
            forward_y=self.forward_y,
        )


class DeadReckoning:
    def __init__(self, max_gap_s: float) -> None:
        self._max_gap_s = float(max_gap_s)
        self._x = 0.0
        self._y = 0.0
        self._heading = 0.0
        self._time: float | None = None
        #: Seconds between the last two readings; 0 when the last one restarted.
        self.interval_s = 0.0

    def reset(self) -> None:
        self._x = 0.0
        self._y = 0.0
        self._heading = 0.0
        self._time = None
        self.interval_s = 0.0

    def update(self, signals: VehicleSignals | None) -> GroundPose | None:
        """The rear axle's pose after this reading.

        ``None`` when this reading cannot be registered against the previous
        one: a missing or invalid reading, or too long a gap for constant
        speed and yaw rate to describe the motion. The grid then forgets
        rather than shifting its memory by a guess.
        """
        if signals is None or not signals.valid:
            self.reset()
            return None
        values = (signals.speed_mps, signals.yaw_rate_rps, signals.timestamp_s)
        if not all(math.isfinite(value) for value in values):
            self.reset()
            return None

        previous = self._time
        self._time = float(signals.timestamp_s)
        self.interval_s = 0.0
        if previous is not None:
            dt = self._time - previous
            if not 0.0 <= dt <= self._max_gap_s:
                return None
            self.interval_s = dt
            self._advance(float(signals.speed_mps) * dt, float(signals.yaw_rate_rps) * dt)
        return GroundPose(
            x=self._x,
            y=self._y,
            forward_x=math.cos(self._heading),
            forward_y=math.sin(self._heading),
        )

    def _advance(self, distance_m: float, turn_rad: float) -> None:
        # The chord of an arc of length d turning by t is d*sin(t/2)/(t/2),
        # along the heading halfway round. np.sinc keeps it exact as t -> 0.
        chord = distance_m * float(np.sinc(turn_rad / (2.0 * math.pi)))
        mid = self._heading + 0.5 * turn_rad
        self._x += chord * math.cos(mid)
        self._y += chord * math.sin(mid)
        self._heading = math.remainder(self._heading + turn_rad, 2.0 * math.pi)
