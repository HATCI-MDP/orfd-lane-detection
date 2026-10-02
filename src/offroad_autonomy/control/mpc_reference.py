"""Adapt the existing metric control reference; no new path planner."""

import numpy as np


def build_reference(path, initial, config, speed_ceiling):
    """N+1 [forward, right, heading, speed] samples, spaced in time.

    The unseen ground below the hood is joined to the first visible point,
    never obtained by extrapolating the polynomial backwards.
    """
    cfg = config.mpc
    forward = np.linspace(path.start, path.end, cfg.reference_samples)
    xy = np.column_stack(
        (
            forward + cfg.camera_ahead_of_rear_axle_m,
            path.right(forward) + cfg.camera_right_of_rear_axle_m,
        )
    )
    heading = np.arctan(path.slope(forward))
    curvature = np.abs(path.raw_curvature(forward))
    speed = np.minimum(
        speed_ceiling, np.sqrt(config.max_lateral_accel_mps2 / np.maximum(curvature, 1e-4))
    )
    # A straight connection over unobserved ground is only an interface adapter.
    # It does not move any observed planner points.
    if xy[0, 0] > 0:
        join_heading = np.arctan2(xy[0, 1], xy[0, 0])
        xy = np.vstack(([0.0, 0.0], xy))
        heading = np.r_[join_heading, heading]
        speed = np.r_[speed[0], speed]
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    speed = np.minimum(
        speed,
        np.sqrt(
            2
            * config.path_end_decel_mps2
            * np.maximum(distance[-1] - distance - config.path_end_margin_m, 0)
        ),
    )
    decel = min(config.curve_decel_mps2, -cfg.min_accel_mps2)
    for k in range(len(speed) - 2, -1, -1):
        speed[k] = min(
            speed[k], np.sqrt(speed[k + 1] ** 2 + 2 * decel * (distance[k + 1] - distance[k]))
        )
    segments = np.diff(xy, axis=0)
    fractions = np.clip(
        np.sum((initial[:2] - xy[:-1]) * segments, axis=1)
        / np.maximum(np.sum(segments**2, axis=1), 1e-9),
        0,
        1,
    )
    projected = xy[:-1] + fractions[:, None] * segments
    nearest = np.argmin(np.linalg.norm(projected - initial[:2], axis=1))
    s = distance[nearest] + fractions[nearest] * (distance[nearest + 1] - distance[nearest])
    ref = np.empty((cfg.horizon + 1, 4))
    travel_speed = initial[3]
    for k in range(cfg.horizon + 1):
        target = np.interp(s, distance, speed)
        ref[k] = (
            np.interp(s, distance, xy[:, 0]),
            np.interp(s, distance, xy[:, 1]),
            np.interp(s, distance, np.unwrap(heading)),
            target,
        )
        travel_speed += np.clip(
            target - travel_speed, cfg.min_accel_mps2 * cfg.dt_s, cfg.max_accel_mps2 * cfg.dt_s
        )
        s = min(distance[-1], s + max(travel_speed, 0) * cfg.dt_s)
    return ref
