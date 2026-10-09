# BEV Grid Planner: Production-Available Inputs

Status: findings and proposal. No code has been changed yet.

## Summary

- **Images:** the BEV grid planner uses only the dashcam. The orbit camera never reaches it.
- **Motion:** the grid shifts its memory using the simulator's exact world position and
  heading. A production vehicle does not have that signal.
- **Fix:** replace the world pose with dead reckoning from wheel speed and yaw rate. This is
  small and conventional, and should change little in simulation. That expectation has not been
  measured yet.

## What the Planner Reads Today

### Camera Input

One camera, named `dashcam`, feeds the whole pipeline. `BeamNGClient.capture_frame()` reads only
the dashcam buffer (`simulation/beamng_client.py`), and that frame is the only image reaching
segmentation, the grid and control.

The orbit camera does not reach the planner:

- It is attached only when the run is started with `--presentation`.
- Its image goes only to the presentation video renderer, never to `pipeline.step_result()`
  (`main.py`).
- `tests/test_dashcam_capture.py::test_connect_attaches_only_the_dashcam` checks that a normal
  run attaches only the dashcam.

### Vehicle State Input

Only the vehicle state is a problem:

- `ground_pose()` in `planning/bev_grid.py` builds the grid's pose from `state.position` and
  `state.direction`. Both are the simulator's exact, drift-free world-frame values.
- `CenterlinePlanner._check_direction()` in `planning/centerline_planner.py` also reads the
  simulator's world velocity and direction vectors.

The controllers downstream use only `speed_mps`, which wheel-speed sensors provide. The MPC
also requires `position`, `velocity` and `rotation` to be finite numbers, but never computes
with them.

## Where the Input Camera Is

Set in `beamng.camera` in `configs/default.yaml`. Positions are relative to the vehicle's
reference origin in BeamNG vehicle space: +X left, -Y forward, +Z up.

| Property     | Value                                                                        |
|--------------|------------------------------------------------------------------------------|
| Lateral      | 0.0 m, on the vehicle centreline                                             |
| Longitudinal | 0.30 m forward of the vehicle origin                                         |
| Rear axle    | About 1.4 m behind the camera (`control.mpc.camera_ahead_of_rear_axle_m`)    |
| Height       | 1.85 m above the vehicle origin                                              |
| Orientation  | Pitched 8 degrees down, no roll or yaw                                       |
| Sensor       | GMSL2, 120 degree horizontal FOV (about 96 degrees vertical, derived)        |
| Resolution   | 960 x 620 at 30 FPS, processed at 720 x 465                                  |

Notes:

- **The key `forward_offset_m: -0.30` is misleadingly named.** It is passed straight through as
  the Y coordinate, and -Y is forward, so the camera sits 0.30 m *ahead* of the origin, not
  behind it. `tests/test_grid_planner.py` confirms this.
- **It is not really on the dashboard.** 1.85 m is roof height on the Hopper, and the code and
  README call it a "roof-mounted dashcam". The hood shows at the bottom of the image and is
  excluded by the ego mask.
- **The ground projection treats the vehicle origin as ground level.**
  `CameraModel.image_to_ground()` intersects rays with vehicle z = 0, so 1.85 m is used as the
  camera's height above the ground. Where the Hopper's reference origin actually sits has not
  been checked. If it is above the ground, every projected distance is off.

## Questions and Answers

### Should the Exact World Position Be a Concern?

Somewhat, but mainly about how much the simulation results can be trusted, not about the car
driving badly. With the exact world position, each new frame lines up with the grid's memory
perfectly. A real car cannot do that, so simulation results are slightly better than a real car
would get.

The effect is limited. The grid only remembers about 2 seconds of road (`planning.grid.memory_s`)
and redraws everything the camera sees on every frame. Pose error matters only for ground the
camera cannot see, mainly the roughly 3 m hidden under the hood.

### Does the BEV Need World Position?

No. The grid stays centred on the car, so all it needs is how far the car moved and turned since
the last frame. The code reads absolute position only as an easy way to compute that difference.
It never uses where the car is on the map.

### What Is the Fix?

Estimate the car's movement from wheel speed and yaw rate (dead reckoning). On every frame:

```
heading += yaw_rate * dt
x += speed * dt * cos(heading)
y += speed * dt * sin(heading)
```

This is the standard way occupancy grids handle the vehicle's own movement in production driving
stacks. Every production car with stability control already reports wheel speeds and yaw rate on
its CAN bus.

The change is small:

1. `simulation/` reports signed forward speed and yaw rate, emulating a wheel-speed sensor and a
   gyro, instead of handing the planner its world position.
2. The planner adds those up between frames into a local pose, in about 30 lines. The grid code
   that moves the memory (`TraversabilityGrid._shift`) stays the same.
3. The world-frame direction check in `CenterlinePlanner` is removed.
4. Tests and the README's grid section are updated.

### Will the Fix Hurt Performance or Make a Crash More Likely?

None of the following has been measured. These are estimates from reasoning, not test results.

- **In simulation, almost no change is expected.** Speed and yaw rate taken from the simulator
  are near exact, and adding them up over a 2 second window loses very little.
- **On a real car, errors stay small.** Wheel speed and gyro errors grow slowly, and the 2 second
  fade throws old memory away before error can build up. For example, an assumed gyro bias of
  0.5 degrees per second gives about 1 degree of heading error after 2 s. That moves a point 8 m
  ahead about 0.14 m sideways, roughly one 0.1 m grid cell. The sensor figure is a typical-order
  assumption, not a measurement.
- **The biggest real-world risk is wheel slip.** On mud or loose dirt the wheels turn faster than
  the car moves, so the car overestimates how far it travelled. Yaw rate is not affected by slip.
  The error only touches the hidden ground near the car; the ground ahead is re-observed on every
  frame.
- **Losing the signals fails safe.** If speed or yaw rate is missing, the grid clears its memory,
  as it already does without a pose. The perception gate then holds the last path briefly and
  stops the car.

Larger real-world error sources already exist elsewhere in the stack: the flat-ground assumption,
camera pitch when the suspension moves, and segmentation errors.

## Proposed Validation

Compare grid planner runs on several maps in three setups:

1. **Baseline:** the exact world position, as today.
2. **Clean Odometry:** dead reckoning from noise-free simulator speed and yaw rate.
3. **Noisy Odometry:** dead reckoning with added noise, gyro bias and wheel slip, as a
   stress test for a real-world case.

If setup 2 drives like setup 1, the fix costs nothing in simulation. Setup 3 shows how much
margin remains on a real vehicle.
