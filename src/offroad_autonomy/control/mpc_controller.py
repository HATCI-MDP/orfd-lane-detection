"""Constrained nonlinear MPC with a warm Stanley safety/fallback controller."""

import logging
import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from offroad_autonomy.control.mpc_reference import build_reference
from offroad_autonomy.control.stanley_controller import StanleyController
from offroad_autonomy.control.vehicle_model import bicycle_step, rollout_with_jacobian
from offroad_autonomy.types import ControlCommand, SteeringDebug

logger = logging.getLogger("offroad_autonomy.control")


class SolveTimeout(RuntimeError):
    pass


class MPCController:
    def __init__(self, config, camera=None):
        self.config, self.cfg = config, config.mpc
        if not math.isfinite(config.wheelbase_m) or config.wheelbase_m <= 0:
            raise ValueError("wheelbase_m must be positive")
        if not 0 < config.max_wheel_angle_deg < 90:
            raise ValueError("max_wheel_angle_deg must be in (0, 90)")
        self.fallback = StanleyController(config, camera)
        self.max_wheel = math.radians(config.max_wheel_angle_deg)
        self.reset()

    def reset(self):
        self.fallback.reset()
        self.previous = np.zeros(2)
        self.warm = None
        self.last_time = None
        self.last_reference = None
        self.last_prediction = None
        self.first_slew = self.cfg.max_steering_rate_rad_s * self.cfg.dt_s

    def _remember(self, command):
        cfg = self.cfg
        accel = (
            command.throttle * cfg.throttle_accel_mps2
            - command.brake * cfg.brake_decel_mps2
            - cfg.rolling_decel_mps2
        )
        self.previous = np.array([command.steering * self.max_wheel, accel])
        self.fallback.observe_applied_steering(command.steering)
        command.debug.final_steering = command.steering
        command.debug.acceleration_mps2 = accel
        return command

    def _fail(self, reason, command=None, elapsed=0.0, status=None):
        self.warm = None
        self.last_prediction = None
        if command is None:
            command = ControlCommand(
                steering=self.previous[0] / self.max_wheel,
                brake=self.config.max_brake,
                debug=SteeringDebug(),
            )
        debug = command.debug
        debug.controller = "mpc"
        debug.controller_fallback = True
        debug.controller_fallback_reason = reason
        debug.solver_status = status or reason
        debug.mpc_solve_ms = elapsed * 1000
        # Switching controllers must not bypass the MPC steering slew limit.
        wheel = np.clip(
            command.steering * self.max_wheel,
            self.previous[0] - self.first_slew,
            self.previous[0] + self.first_slew,
        )
        wheel = np.clip(wheel, -self.cfg.max_steering_rad, self.cfg.max_steering_rad)
        command.steering = float(wheel / self.max_wheel)
        logger.warning("MPC fallback: %s (%.1f ms)", reason, debug.mpc_solve_ms)
        return self._remember(command)

    def compute(self, plan, state):
        now = time.perf_counter()
        interval = self.cfg.dt_s if self.last_time is None else max(0.0, now - self.last_time)
        self.last_time = now
        # Never permit a delayed frame to authorize a large steering jump.
        first_dt = min(interval, self.cfg.dt_s)
        self.first_slew = min(
            self.cfg.max_steering_rate_rad_s * first_dt,
            self.config.max_steering_delta * self.max_wheel,
        )
        values = [
            state.speed_mps,
            state.heading_rad,
            *state.position,
            *state.velocity,
            *state.rotation,
        ]
        if not state.valid or not np.isfinite(values).all() or state.speed_mps < 0:
            return self._fail("invalid vehicle state")
        points = np.asarray(plan.centerline)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) < 2
            or not np.isfinite(points).all()
            or not np.isfinite(plan.speed_scale)
        ):
            return self._fail("missing or invalid trajectory")
        try:
            # Validate projection before handing a degenerate path to Stanley.
            path = self.fallback.tracking_reference(plan)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            return self._fail(str(exc))
        safety = self.fallback.compute(plan, state)
        vmax = min(self.cfg.max_speed_mps, self.config.speed_limit_mph * 0.44704)
        # The fallback inherits the MPC deployment's lower cruise ceiling.
        if safety.debug.target_speed_mps > vmax:
            safety.debug.target_speed_mps = vmax
            safety.debug.speed_reason = "MPC speed ceiling"
            speed_error = vmax - state.speed_mps
            if speed_error <= 0:
                safety.throttle = 0.0
                safety.brake = max(
                    safety.brake, min(self.config.max_brake, -speed_error * self.config.speed_kp)
                )
            else:
                safety.throttle = min(safety.throttle, speed_error * self.config.speed_kp)
        safety.throttle = min(
            safety.throttle,
            (self.cfg.max_accel_mps2 + self.cfg.rolling_decel_mps2) / self.cfg.throttle_accel_mps2,
        )
        if plan.fallback_active or plan.kalman_active or plan.speed_scale <= 0:
            return self._fail(plan.fallback_reason or "low-confidence trajectory", safety)
        if state.speed_mps > vmax:
            safety.throttle = 0.0
            safety.brake = max(
                safety.brake,
                min(self.config.max_brake, -self.cfg.min_accel_mps2 / self.cfg.brake_decel_mps2),
            )
            return self._fail("speed above MPC constraint", safety)
        initial = np.array([0.0, 0.0, 0.0, state.speed_mps])
        # Existing capture-to-actuation latency; integrate steering only because
        # the speed observation is newer than the camera exposure.
        delay = self.config.control_latency_s
        for _ in range(5):
            initial = bicycle_step(
                initial, [self.previous[0], 0.0], delay / 5, self.config.wheelbase_m
            )
        reference = build_reference(
            path, initial, self.config, min(vmax, safety.debug.target_speed_mps)
        )
        self.last_reference = reference
        started = time.perf_counter()
        try:
            solution, prediction = self._solve(initial, reference, first_dt, vmax, started)
            elapsed = time.perf_counter() - started
            if elapsed > self.cfg.max_solve_time_s:
                raise SolveTimeout("solver time budget exceeded")
            if not solution.success:
                return self._fail("solver failed", safety, elapsed, str(solution.message))
        except (
            SolveTimeout,
            ValueError,
            RuntimeError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as exc:
            return self._fail(str(exc), safety, time.perf_counter() - started)
        self.warm = solution.x.reshape(-1, 2)
        self.last_prediction = prediction
        delta, accel = self.warm[0]
        throttle, brake = self.acceleration_to_pedals(accel)
        # Retain the existing overspeed, path-end, clearance and comfort logic.
        # Safety can request more braking or less throttle than the optimizer.
        brake = max(brake, safety.brake)
        throttle = (
            0.0
            if brake > 0
            else min(
                throttle,
                self.config.max_throttle * self.fallback._saturation_scale(safety.debug.saturation),
            )
        )
        debug = safety.debug
        debug.controller = "mpc"
        debug.solver_success = True
        debug.solver_status = str(solution.message)
        debug.mpc_solve_ms = elapsed * 1000
        debug.mpc_cost = float(solution.fun)
        debug.target_speed_mps = float(reference[0, 3])
        debug.desired_steering = float(delta / self.max_wheel)
        return self._remember(
            ControlCommand(
                steering=float(delta / self.max_wheel), throttle=throttle, brake=brake, debug=debug
            )
        )

    def acceleration_to_pedals(self, acceleration):
        # Positive/negative requests use mutually exclusive pedals. Rolling
        # resistance compensation is applied only to nonnegative requests.
        if acceleration >= 0:
            return min(
                self.config.max_throttle,
                (acceleration + self.cfg.rolling_decel_mps2) / self.cfg.throttle_accel_mps2,
            ), 0.0
        return 0.0, min(self.config.max_brake, -acceleration / self.cfg.brake_decel_mps2)

    def _solve(self, initial, reference, first_dt, vmax, started):
        cfg, n = self.cfg, self.cfg.horizon

        def deadline(*_):
            if time.perf_counter() - started > cfg.max_solve_time_s:
                raise SolveTimeout("solver time budget exceeded")

        cap = min(cfg.max_steering_rad, self.max_wheel)
        lower = np.tile([-cap, cfg.min_accel_mps2], n)
        upper = np.tile([cap, cfg.max_accel_mps2], n)
        difference = np.eye(n) - np.eye(n, k=-1)
        rate = np.zeros((n, 2 * n))
        rate[:, ::2] = difference
        slew = np.full(n, cfg.max_steering_rate_rad_s * cfg.dt_s)
        slew[0] = min(
            cfg.max_steering_rate_rad_s * first_dt, self.config.max_steering_delta * self.max_wheel
        )
        rate_lo, rate_hi = -slew, slew.copy()
        rate_lo[0] += self.previous[0]
        rate_hi[0] += self.previous[0]
        speed_matrix = np.zeros((n, 2 * n))
        speed_matrix[:, 1::2] = cfg.dt_s * np.tril(np.ones((n, n)))
        matrix = np.vstack((rate, speed_matrix))
        constraint_lo = np.r_[rate_lo, np.full(n, -initial[3])]
        constraint_hi = np.r_[rate_hi, np.full(n, vmax - initial[3])]
        # Preserve Stanley's speed-dependent steering cap at every prediction
        # step. The affine extension above high_speed is conservative; below
        # full_speed the absolute wheel bound remains the active constraint.
        full_speed = self.config.steer_cap_full_speed_mps
        high_speed = max(full_speed + 1e-3, self.config.steer_cap_high_speed_mps)
        cap_slope = (1 - self.config.steer_cap_at_high_speed) / (high_speed - full_speed)
        before_speed = np.vstack((np.zeros(2 * n), speed_matrix[:-1]))
        steering_matrix = np.zeros((n, 2 * n))
        steering_matrix[:, ::2] = np.eye(n)
        speed_term = self.max_wheel * cap_slope * before_speed
        cap_rows = np.vstack((steering_matrix + speed_term, -steering_matrix + speed_term))
        matrix = np.vstack((matrix, cap_rows))
        constraint_lo = np.r_[constraint_lo, np.full(2 * n, -np.inf)]
        cap_upper = self.max_wheel * (1 + cap_slope * (full_speed - initial[3]))
        constraint_hi = np.r_[constraint_hi, np.full(2 * n, cap_upper)]
        if self.warm is None:
            guess = np.tile([np.clip(self.previous[0], -cap, cap), 0.0], (n, 1))
        else:
            guess = np.vstack((self.warm[1:], self.warm[-1]))
        guess = np.clip(guess.ravel(), lower, upper)
        weights = np.array([cfg.q_position, cfg.q_position, cfg.q_heading, cfg.q_velocity])
        input_weights = np.array([cfg.r_steering, cfg.r_accel])
        change_weights = np.array([cfg.rd_steering, cfg.rd_accel])
        stage_weights = np.ones(n)
        stage_weights[-1] = cfg.terminal_weight

        def objective(flat):
            deadline()
            controls = flat.reshape(n, 2)
            states, jac = rollout_with_jacobian(
                initial, controls, cfg.dt_s, self.config.wheelbase_m
            )
            error = states - reference[1:]
            error[:, 2] = np.arctan2(np.sin(error[:, 2]), np.cos(error[:, 2]))
            weighted = error * weights * stage_weights[:, None]
            change = np.diff(np.vstack((self.previous, controls)), axis=0)
            cost = (
                np.sum(error * weighted)
                + np.sum(controls**2 * input_weights)
                + np.sum(change**2 * change_weights)
            )
            grad = 2 * np.einsum("ki,kij->j", weighted, jac)
            input_grad = 2 * controls * input_weights
            change_grad = 2 * change * change_weights
            input_grad += change_grad
            input_grad[:-1] -= change_grad[1:]
            return float(cost), grad + input_grad.ravel()

        result = minimize(
            objective,
            guess,
            jac=True,
            method="SLSQP",
            bounds=Bounds(lower, upper),
            constraints=LinearConstraint(matrix, constraint_lo, constraint_hi),
            callback=deadline,
            options={"maxiter": cfg.max_iterations, "ftol": cfg.ftol},
        )
        deadline()
        # Independently reject invalid or infeasible "successful" solutions.
        tol = cfg.constraint_tolerance
        x = result.x
        if (
            not np.isfinite(x).all()
            or not np.isfinite(result.fun)
            or np.any(x < lower - tol)
            or np.any(x > upper + tol)
            or np.any(matrix @ x < constraint_lo - tol)
            or np.any(matrix @ x > constraint_hi + tol)
        ):
            raise ValueError("solver returned infeasible controls")
        prediction, _ = rollout_with_jacobian(
            initial, x.reshape(n, 2), cfg.dt_s, self.config.wheelbase_m
        )
        return result, prediction
