# Swappable Driving Stacks Design

Status: Implemented on branch `swappable-driving-stacks`. Built and planned stacks, and
everything specific to each, are in [docs/stacks/](../stacks/README.md).

The repo is split into a fixed **harness** (simulator I/O, loop, safety, dashboard, scoring)
and swappable **driving stacks** that take sensor data in and put a control command out, so
any perception and planning approach can be run and compared on one branch.

## Problem

Comparing pipelines meant switching between branches, because what differed between them was
code, not config values. Future experiments (occupancy grids, cost maps, end-to-end models,
new sensors) are unknown, so the design fixes only the boundaries every stack must have:
sensors in, command out, something to show the operator.

## Goals And Non-Goals

**Goals**

- Run any perception, planning and control approach on one branch by changing a config file.
- Add a stack by adding a package, with no change to the harness, dashboard or benchmark.
- Compare any two stacks fairly: same map, spawn, duration, loop, safety and scoring.
- Keep the rules that carry weight: only `simulation/` imports `beamngpy`, and the control
  loop never blocks on the dashboard.

**Non-Goals**

- Reproducing numbers an old branch measured; loop timing and safe stop now come from the
  shared harness.
- A plugin framework or dynamic discovery; a dotted import path in the config is enough.
- Multiple vehicles, traffic or another simulator.
- Training, data collection or dataset tooling.

## Architecture

```
┌──────────────────────── Harness: fixed, written once ────────────────────────┐
│  Simulation           Main loop            Dashboard            Benchmark    │
│  attaches sensors     timing, safe stop,   draws generic        scores from  │
│  the stack declares   manual override      visual primitives    sim truth    │
└───────────────┬──────────────────────────────────────▲───────────────────────┘
                │ Observation                          │ StackOutput
                │ sensors + vehicle signals            │ command, visuals, metrics,
                ▼                                      │ safe-stop request
┌──────────────── Driving stacks: swappable, chosen by config ─────────────────┐
│  stack A              stack B              stack C              next stack   │
│  any sensors, models, planners and controllers behind the same contract      │
└──────────────────────────────────────────────────────────────────────────────┘
```

| Path | Role |
| --- | --- |
| `contract.py` | The only types the harness and a stack share |
| `main.py` | Harness loop, safe stop, manual override, shutdown |
| `simulation/` | BeamNG I/O; attaches the sensors the stack declares |
| `visualization/` | Dashboard and presentation, drawn only from visual primitives |
| `runtime/` | Dashboard thread, timing, reference routes, ground-truth benchmark |
| `stacks/<name>/` | One package per stack, with a `build(stack_config)` function and its own defaults |
| `perception/`, `planning/`, `control/` | Parts stacks may share; nothing requires them to |

## The Stack Contract

A stack is any object with three methods, the same shape the CARLA Leaderboard uses to
compare unrelated agents.

```python
class DrivingStack(Protocol):
    def sensors(self) -> list[SensorSpec]: ...   # read once, before connecting
    def step(self, obs: Observation) -> StackOutput: ...
    def reset(self) -> None: ...                 # called when the operator resumes


@dataclass(frozen=True)
class SensorSpec:
    name: str      # key the reading arrives under in Observation.sensors
    kind: str      # "camera" today; a new kind needs one adapter in simulation/
    params: dict   # pose, resolution, FOV, rate; validated before a map loads


@dataclass
class Observation:
    sensors: dict[str, np.ndarray]
    vehicle: VehicleSignals   # wheel speed and yaw rate, never the world pose
    timestamp: float
    frame_id: int = 0


@dataclass
class StackOutput:
    command: ControlCommand
    visuals: list[Visual] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)  # reported, never compared
    safe_stop: str = ""   # non-empty asks the harness for its safe stop
```

- **Inside a stack, anything goes:** its own types, threads, workers and models.
- **A stack never imports `beamngpy`.** It sees only `Observation`, so it is testable
  offline.
- **A stack sees only what a production vehicle senses.** `Observation.vehicle` is a
  `VehicleSignals`: speed and yaw rate as the wheel-speed sensors and the stability-control
  gyro report them. The simulator's world pose stays in the harness's `VehicleState`, for
  safety and scoring, so no stack can steer by it.
- **The observation is all or nothing.** If any declared sensor has no frame, the harness
  skips the iteration rather than handing over a partial observation.
- **Safety stays in the harness.** The stuck detector, `E` safe stop and manual override
  apply to every stack. A stack can request a safe stop through `safe_stop` for conditions
  only it can see, but cannot disable one. The stack keeps stepping under safe stop, and resuming with `P` calls its `reset()`
  because the operator may have moved the vehicle.
- **`step` must return within the loop budget** (`visualization.dashboard.latency_budget_ms`).
  Slow work belongs on the stack's own worker, returning the latest result. The harness
  times every `step`, logs overruns and reports them as `step_over_budget`, but cannot
  prevent them.

## Configuration

A config names its stack by import path and hands it a section only the stack reads.

```yaml
extends: default.yaml
stack: offroad_autonomy.stacks.<name>:build
stack_config:
  # Read only by the stack.
```

| Section | Read By | Holds |
| --- | --- | --- |
| `beamng`, `safety`, `ui`, `recording`, `presentation`, `visualization`, `benchmark` | Harness | Simulator, map, spawn, stuck detector, dashboard, recording, scoring |
| `stack` | Harness | Import path of the stack's `build(stack_config)` function |
| `stack_config` | The stack only | Its perception, planning, control and sensor rig |

- **Both sides validate strictly.** The harness checks a schema in code and each stack checks
  `stack_config` itself; `utils/config.py` lets a stack use its shipped `default.yaml` as the
  schema (`schema_from`, `check_keys`). Harness keys from before the split fail with a
  message naming where they moved.
- **Every tuning value lives in a config with its reason beside it:** harness values in
  `configs/default.yaml`, stack values in the stack's own defaults.
- **`extends:`, `auto` platform resolution and `BEAMNG_*` overrides** work as before and
  apply to every stack.

## Dashboard

The dashboard never sees a stack's types. A stack describes itself with five primitives:

| Primitive | Fields | Drawn As |
| --- | --- | --- |
| `ImageLayer` | name, image, camera | A page; the first is the main view |
| `MaskOverlay` | layer, mask, role (`traversable`, `excluded`, `info`), label | A tint on that layer |
| `Polyline` | layer, points, frame (`image` pixels or `ground` metres), style (`path`, `held`, `candidate`), label | A path on that layer |
| `Gauge` | name, value, range, thresholds, caption, text | A bar on the Driving Stack card, coloured and ticked by its own thresholds |
| `Status` | text, level (`good`, `warn`, `bad`), detail | A chip when `warn` or `bad`; good ones only in the readout |

- **Harness panels are always present:** speed, steering, pedals, safe-stop state, loop FPS
  and latency.
- **Pages:** key `0` is the main view; keys `1`-`9` show the stack's other layers in emission
  order, beside a readout of its statuses, gauges and metrics.
- **Ground polylines** are projected by the harness with the camera from the stack's
  `SensorSpec`, so a planner in metres draws correctly on any camera.
- **A layer's image may be a function**, called on the dashboard thread only while that page
  is shown, so a costly debug image never runs in the control loop.
- **Invalid roles, styles or levels fail when the stack builds the visual**, on the loop
  thread, instead of blanking the dashboard.
- How informative the dashboard is depends on what the stack emits: a stack that emits only
  an image gets a bare page.

## Benchmarking From Ground Truth

The harness scores every run from simulator ground truth, never from what a stack reports.

| Metric | From |
| --- | --- |
| Cross-track mean and max, completion | True position against a recorded reference route |
| Departures | True position more than `benchmark.departure_cross_track_m` (2 m) off the route |
| Distance, safe stops | True positions and the harness's safe-stop state |
| Steering rate, sign changes per minute | Commands sent, ignoring changes inside `steer_sign_deadband` |
| FPS, capture-to-actuation latency, step time | Measured by the harness, after the warm-up |

- **Reference routes are recorded drives** (`--record-route`), one JSON file per map and
  spawn in `benchmark.routes_dir`. Off-road trails are mostly missing from BeamNG's road
  network. Without a route, cross-track, completion and departures report null.
- **Driving metrics stop at the first safe stop;** what the operator drives afterwards is
  not the stack's.
- **Route matching starts at the route's beginning** and searches a window around the last
  match, so a closed circuit does not score complete at the spawn.
- **Stack metrics** are averaged into a separate `stack_metrics` section.

```bash
python scripts/run_experiments.py configs/a.yaml configs/b.yaml \
    --seconds 60 --map automation_test_track --spawn 0
```

Each config runs headless in its own process on the same map, spawn and duration, writing
one report per run and one comparison table.

## Performance

Measured offline with the built stack and its segmentation model mocked, so inference is
excluded:

- The contract adds about 0.01 ms per frame over calling a pipeline directly (building
  visuals and metrics).
- Dashboard render time is unchanged within noise from before the split (main view 8.3 ms
  against 9.2 ms, debug page 19.0 ms against 18.6 ms).
- Route matching adds about 0.02 ms per frame, only when benchmarking.

No BeamNG run has been made on this branch, so simulator loop timing is not yet measured.
Benchmark reports from before the split use different metrics and do not compare with new
ones.

## Limits

- **A new sensor kind** (lidar, radar, IMU) needs one adapter in `simulation/` the first time.
- **Harness changes** (multiple vehicles, another safety policy, another simulator) are
  outside the contract.
- **Departures approximate "left the drivable area"** with a 2 m distance from the recorded
  route, a design choice not yet checked per map.
- **A ported branch keeps its own perception, planning and control,** but its loop, safe stop
  and timing come from the harness, so its results will differ from what it measured alone.
