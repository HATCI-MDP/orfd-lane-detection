"""MPC tuning in SI units; vehicle geometry stays in PipelineConfig."""

import math
from dataclasses import dataclass, fields


@dataclass
class MPCConfig:
    horizon: int = 12
    dt_s: float = 0.25
    max_solve_time_s: float = 0.08
    max_iterations: int = 45
    ftol: float = 1e-4
    constraint_tolerance: float = 1e-5
    max_speed_mps: float = 4.0
    min_accel_mps2: float = -2.0
    max_accel_mps2: float = 1.0
    max_steering_rad: float = 0.50
    max_steering_rate_rad_s: float = 0.40
    q_position: float = 3.0
    q_heading: float = 4.0
    q_velocity: float = 2.0
    r_steering: float = 0.3
    r_accel: float = 0.15
    rd_steering: float = 100.0
    rd_accel: float = 0.5
    terminal_weight: float = 2.0
    reference_samples: int = 60
    camera_ahead_of_rear_axle_m: float = 1.4
    camera_right_of_rear_axle_m: float = 0.0
    throttle_accel_mps2: float = 9.0
    brake_decel_mps2: float = 8.0
    rolling_decel_mps2: float = 0.3

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"mpc.{f.name} must be finite and numeric")
        for name in ("horizon", "max_iterations", "reference_samples"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 2:
                raise ValueError(f"mpc.{name} must be an integer >= 2")
        for name in (
            "dt_s",
            "max_solve_time_s",
            "ftol",
            "constraint_tolerance",
            "max_speed_mps",
            "max_accel_mps2",
            "max_steering_rad",
            "max_steering_rate_rad_s",
            "terminal_weight",
            "throttle_accel_mps2",
            "brake_decel_mps2",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"mpc.{name} must be positive")
        if self.min_accel_mps2 >= 0 or self.max_steering_rad >= math.pi / 2:
            raise ValueError("MPC requires negative braking acceleration and steering < pi/2")
        for name in (
            "q_position",
            "q_heading",
            "q_velocity",
            "r_steering",
            "r_accel",
            "rd_steering",
            "rd_accel",
            "rolling_decel_mps2",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"mpc.{name} must be nonnegative")
