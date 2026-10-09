# BEV Grid Planner: Production-Available Inputs

Status: implemented. Checked offline on synthetic trails (`scripts/grid_sim.py`); not yet
compared in BeamNG.

## Summary

- **Images:** the BEV grid planner uses only the dashcam. The orbit camera never reaches it.
- **Motion:** the grid used to shift its memory by the simulator's exact world position and
  heading, which a production vehicle does not have. It now dead-reckons from wheel speed and
  yaw rate, which every production vehicle with stability control reports on its CAN bus.
- **Enforced by the contract:** a stack's `Observation.vehicle` is a `VehicleSignals` holding
  only speed, yaw rate, a timestamp and a validity flag. The world pose stays in the
  harness's `VehicleState`, for the safety checks and the benchmark, so no stack can read it.

## What a Stack Reads

### Camera Input

One camera, named `dashcam`, feeds the whole pipeline. `BeamNGClient.capture()` reads only the
sensors the stack declares (`simulation/beamng_client.py`), and that frame is the only image
reaching segmentation, the grid and control.

The orbit camera does not reach the planner:

- It is attached only when the run is started with `--presentation`.
- Its image goes only to the presentation renderer, never to the stack (`main.py`).
- `tests/test_dashcam_capture.py::test_connect_attaches_only_the_dashcam` checks that a normal
  run attaches only the dashcam.

### Vehicle Input

`VehicleSignals` (`types.py`) is all a stack learns about the vehicle:

| Field | Meaning | Production source |
| --- | --- | --- |
| `speed_mps` | Along the forward axis, negative when reversing | Wheel-speed sensors |
| `yaw_rate_rps` | About the vertical axis, positive turning left (ISO 8855) | Stability-control gyro |
| `timestamp_s` | When the reading was taken | The vehicle bus clock |
| `valid` | False when the reading failed | |

Both rates are means over the interval since the previous reading, which is what an odometry
module integrating the sensors at their own rate reports once per frame.

The controller uses only `speed_mps`. The grid planner uses all four. Nothing in the stack
reads a position, heading or world-frame velocity, and the world-frame direction check that
`CenterlinePlanner` used to run is gone.

## How the Simulator Emulates the Sensors

`OdometryEmulator` in `simulation/beamng_client.py` derives the signals from consecutive
BeamNG state readings, and only the rates leave it:

- **Speed:** the reference node's displacement along the mean heading over the interval. For
  a rigid body every point on the vehicle's axis moves the same distance along it: the chord
  of the rear axle's arc, which is converted to arc length. This is an ideal wheel-speed
  sensor, with no slip.
- **Yaw rate:** the heading change over the interval. BeamNG's state has no angular velocity,
  so this stands in for the integral of an ideal gyro.
- **Time:** the simulation time BeamNG reports, falling back to the wall clock. Simulation time
  keeps the rates right when BeamNG runs slower than real time.

## How the Planner Uses Them

`DeadReckoning` (`planning/odometry.py`) integrates each reading over its interval exactly
along the arc it describes (the velocity motion model, Thrun, Burgard and Fox, *Probabilistic
Robotics*, 5.3). It integrates the rear axle, the point a vehicle that does not slide moves
along its own heading, in a local frame that starts wherever the vehicle is and is tied to
nothing on the map. The camera's ground point, 1.4 m ahead of the rear axle, is offset from it
for the grid. `TraversabilityGrid._shift` is unchanged: it only ever needed the motion between
two frames.

The grid forgets, as it did without a pose, when:

- a reading is missing or invalid, or
- readings are more than `planning.grid.max_odometry_gap_s` (1 s) apart, or the clock goes
  backwards. Constant rates no longer describe the motion over such a gap.

## Measured Effect

Offline, on `scripts/grid_sim.py`, not in BeamNG:

- **Dead reckoning against the true motion:** with the emulator and dead reckoning on an
  accelerating, weaving bicycle, the relative pose error over any 2 s window stayed under
  2 cm. That is less than one 0.1 m grid cell. The 2 s window matches `planning.grid.memory_s`.
- **Closed loop:** with clean signals the planner finished all 12 synthetic trails (six of
  them held out from tuning) with no departure frame. With 5% wheel slip, a 0.5 deg/s gyro
  bias, noisy masks and 0.5 deg of suspension pitch, it finished all 36 runs. Its only
  departures were on the tightest held-out trail (3.5 m wide, 7 m radius), where the body
  clipped the inside edge by up to 0.16 m.

The grid itself was also checked:

- A single frame places road edges within 2.4 cm RMS.
- After six frames of motion with no new evidence, remembered edges are still within 2.9 cm.

Neither weighting far evidence down nor resampling the mask bilinearly improved the closed
loop, so neither was added.

## Where the Input Camera Is

Set in `camera` in the stack's `default.yaml`. Positions are relative to the vehicle's
reference origin in BeamNG vehicle space: +X left, -Y forward, +Z up.

| Property     | Value                                                                     |
|--------------|---------------------------------------------------------------------------|
| Lateral      | 0.0 m, on the vehicle centreline                                          |
| Longitudinal | 0.30 m forward of the vehicle origin                                      |
| Rear axle    | About 1.4 m behind the camera (`control.camera_ahead_of_rear_axle_m`)     |
| Height       | 1.85 m above the vehicle origin                                           |
| Orientation  | Pitched 8 degrees down, no roll or yaw                                    |
| Sensor       | GMSL2, 120 degree horizontal FOV (about 96 degrees vertical, derived)     |
| Resolution   | 960 x 620 at 30 FPS, processed at 720 x 465                               |

Notes:

- **The key `forward_offset_m: -0.30` is misleadingly named.** It is passed straight through as
  the Y coordinate, and -Y is forward, so the camera sits 0.30 m *ahead* of the origin, not
  behind it.
- **It is not really on the dashboard.** 1.85 m is roof height on the Hopper, and the code calls
  it a "roof-mounted dashcam". The hood shows at the bottom of the image and is excluded by the
  ego mask.
- **The ground projection treats the vehicle origin as ground level.**
  `CameraModel.image_to_ground()` intersects rays with vehicle z = 0, so 1.85 m is used as the
  camera's height above the ground. Where the Hopper's reference origin actually sits has not
  been checked. If it is above the ground, every projected distance is off.
- **Dead reckoning depends on `control.camera_ahead_of_rear_axle_m`.** The rear axle is the
  point integrated, so an error in that offset misplaces memory while turning, by about the
  offset error times the heading change.

## Remaining Real-World Error Sources

- **Wheel slip:** on mud or loose dirt the wheels turn faster than the vehicle moves, so
  dead reckoning overstates distance. Yaw rate is not affected. The error only touches the
  hidden ground near the vehicle; the ground ahead is re-observed on every frame.
- **Gyro bias:** an assumed 0.5 deg/s gives about 1 degree of heading error over the 2 s
  memory, which moves a point 8 m ahead about 0.14 m sideways. That figure is a typical-order
  assumption, not a measurement.
- **Projection:** the flat-ground assumption, camera pitch when the suspension moves, and
  segmentation errors are larger than any of the above.

## Still To Validate

Compare grid planner runs in BeamNG on several maps against the baseline planner, as
`docs/stacks/mono_yoloe/README.md` requires before the grid planner can become the default.
