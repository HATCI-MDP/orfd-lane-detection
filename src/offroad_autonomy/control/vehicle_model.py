"""Rear-axle bicycle: X forward, Y right, yaw/steering positive right."""

import numpy as np


def bicycle_step(state, control, dt, wheelbase):
    x, y, psi, v = state
    delta, accel = control
    return np.array(
        [
            x + v * np.cos(psi) * dt,
            y + v * np.sin(psi) * dt,
            psi + v / wheelbase * np.tan(delta) * dt,
            v + accel * dt,
        ]
    )


def rollout_with_jacobian(initial, controls, dt, wheelbase):
    """States after each input and exact sensitivities to all inputs."""
    n = len(controls)
    states = np.empty((n, 4))
    jac = np.zeros((n, 4, 2 * n))
    state = np.asarray(initial, dtype=float)
    sensitivity = np.zeros((4, 2 * n))
    for k, control in enumerate(controls):
        _, _, psi, v = state
        delta = control[0]
        a = np.eye(4)
        a[0, 2], a[0, 3] = -v * np.sin(psi) * dt, np.cos(psi) * dt
        a[1, 2], a[1, 3] = v * np.cos(psi) * dt, np.sin(psi) * dt
        a[2, 3] = np.tan(delta) * dt / wheelbase
        sensitivity = a @ sensitivity
        sensitivity[2, 2 * k] += v * dt / (wheelbase * np.cos(delta) ** 2)
        sensitivity[3, 2 * k + 1] += dt
        state = bicycle_step(state, control, dt, wheelbase)
        states[k], jac[k] = state, sensitivity
    return states, jac
