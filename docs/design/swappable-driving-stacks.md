# Swappable Driving Stacks Design

Status: Proposal, not scheduled. Written 2026-10-01.

Split the repo into a fixed **harness** (simulator I/O, loop, safety, dashboard, scoring)
and swappable **driving stacks** that take sensor data in and put a control command out,
so any perception and planning approach can be run and compared on one branch.

## Summary And Problem

Comparing pipelines today means switching between three branches:
`platform-config-and-dashboard-redesign`, `camera-position-experiments` and
`march-2026-03-30-network-attach`. Config overlays cannot replace those branches, because
what differs between them is code, not values.

- **`camera-position-experiments`** forked at `7d7ce3c` and runs a two-camera stereo rig
  with depth and stitching. Commit `0e31ea2` removed that code from the current branch, so
  its configs have nothing to drive here.
- **`march-2026-03-30-network-attach`** forked at `3021745`, before the Jetson rewrite. It
  steers in pixel space with a different speed law and no perception gate.
- **The current loader** fills one flat `PipelineConfig` and ignores unknown keys, so an
  old config loads without error and silently runs the wrong algorithm.

The experiments to come are unknown: BEV occupancy grids, cost maps, end-to-end models and
new sensors are all plausible. A design that fixes the boundary between perception and
planning (for example `PathPlan`) would break on the first of those, so this design fixes
only the boundaries that every driving stack must have.

## Goals And Non-Goals

**Goals**

- Run any perception, planning and control approach on one branch by changing a config
  file, including one camera, two cameras, stereo depth or a learned end-to-end policy.
- Add a new stack by adding a folder, with no change to the harness, the dashboard or the
  benchmark.
- Compare any two stacks fairly: same map, spawn, duration, loop and safety, scored the
  same way.
- Keep the dashboard working for every stack, including ones not yet invented.
- Keep the rules that carry weight today: only `simulation/` imports `beamngpy`, and the
  control loop never blocks on the dashboard.

**Non-Goals**

- Reproducing the exact numbers an old branch measured. Stacks share one harness, so loop
  timing and safe stop come from the harness, not from the branch.
- A plugin framework, entry points or dynamic discovery. A dotted import path in the
  config is enough.
- Multiple vehicles, traffic, or swapping the simulator. These change the harness and are
  out of scope.
- Training, data collection or dataset tooling.

## Fixed Boundaries

Every driving stack, whatever it does inside, takes sensor data in, puts a control command
out, and has something to show the operator. Those three are the only fixed boundaries;
everything between them is one swappable stack.

```
┌──────────────────────── Harness: fixed, written once ────────────────────────┐
│  Simulation           Main loop            Dashboard            Benchmark    │
│  attaches sensors     timing, safe stop,   draws generic        scores from  │
│  the stack declares   manual override      Visual primitives    sim truth    │
└───────────────┬──────────────────────────────────────▲───────────────────────┘
                │ Observation                          │ StackOutput
                │ sensors + vehicle state              │ command + visuals + metrics
                ▼                                      │
┌──────────────── Driving stacks: swappable, chosen by config ─────────────────┐
│  mono_yoloe           stereo_depth         march_legacy         next stack   │
│  one dashcam, YOLOE,  two cameras,         pixel-space          BEV, end-to- │
│  Stanley or MPC       stereo depth worker  Stanley, 1280x720    end, new     │
└──────────────────────────────────────────────────────────────────────────────┘
```

The harness is written once and owns everything that must be the same for a fair
comparison. A stack owns everything that varies between experiments, including which
sensors it needs and where they are mounted.

## The Stack Contract

A stack is any object with three methods: declare its sensors, step once per frame, reset.
The harness knows nothing else about it. This is the same shape the CARLA Leaderboard uses
to compare unrelated driving agents (`sensors()` plus `run_step(input_data, timestamp)`
returning a control).

```python
class DrivingStack(Protocol):
    def sensors(self) -> list[SensorSpec]:
        """Which cameras or other sensors to attach and where; the simulator
        fulfils this so a stack can change its rig without harness code."""

    def step(self, obs: Observation) -> StackOutput: ...

    def reset(self) -> None: ...


@dataclass
class SensorSpec:
    name: str                # key the reading arrives under in Observation.sensors
    kind: str                # "camera" today; "lidar", "imu" when the simulator supports them
    params: dict             # pose, resolution, FOV, rate; validated by the simulator adapter


@dataclass
class Observation:
    sensors: dict[str, np.ndarray]   # keyed by the names the stack declared
    vehicle: VehicleState
    timestamp: float


@dataclass
class StackOutput:
    command: ControlCommand
    visuals: list[Visual] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
```

Rules for stacks:

- **Inside a stack, anything goes.** It may use its own types, threads, workers and
  models. The current perception, planner and controller classes become internal parts of
  one stack.
- **A stack must return within the loop budget.** Slow work (stereo depth, a large model)
  runs on the stack's own worker and the stack returns the latest result, as the old
  `stereo_worker.py` did.
- **A stack never imports `beamngpy`.** It sees only `Observation`, so every stack stays
  testable offline and could run on a real car whose harness produces the same
  `Observation`.
- **Stacks may share parts.** A `stacks/common/` package can hold reusable pieces such as
  the YOLOE segmenter or the Stanley controller, but sharing is optional and never
  required by the harness.

## Configuration And Stack Selection

A config names its stack by Python import path and hands it a free-form section that only
the stack reads. Adding a stack means adding a folder; nothing in the harness lists the
stacks.

```yaml
# configs/experiments/stereo-wide-baseline.yaml
extends: ../default.yaml
stack: offroad_autonomy.stacks.stereo_depth:build
stack_config:
  rig:
    baseline_m: 0.8
    height_m: 0.95
    toe_out_deg: 0.0
  planner: advanced
  controller: stanley
```

The config splits into two parts with different owners:

| Section | Read By | Holds |
| --- | --- | --- |
| `beamng`, `safety`, `ui`, `benchmark` | Harness | Simulator connection, map, vehicle, spawn, safe stop, dashboard, run length |
| `stack` | Harness | Import path of the stack's `build(stack_config)` function |
| `stack_config` | The stack only | Everything about perception, planning, control and its sensor rig |

- **`extends:` stays.** Experiments remain small overlays on a base config, and the
  existing deep merge works unchanged.
- **Each part validates strictly.** The harness and every stack load their section into a
  dataclass and raise on unknown keys, as `MPCConfig` already does. A misspelled or
  outdated key fails at startup instead of silently falling back to a default.
- **Every tuning value still lives in a config with its reason beside it.** A stack ships
  its own `default.yaml` next to its code, and experiment configs override it.
- **`auto` platform resolution and the `BEAMNG_*` environment overrides stay in the
  harness** and apply to every stack.

## Dashboard

The dashboard draws a small set of generic primitives and never sees a stack's internal
types, so a stack invented next year renders with no dashboard change. Today
`AutonomyDashboard.render()` takes `PipelineStepResult` and `PathPlan` directly, which
ties it to one pipeline.

| Primitive | Fields | Example Uses |
| --- | --- | --- |
| `ImageLayer` | name, image | Camera feed, depth map, disparity, BEV grid |
| `MaskOverlay` | camera name, mask, colour role | Road segmentation on any camera |
| `Polyline` | frame (`ground` in metres or `image` in pixels), points, style | Centreline, MPC prediction, any candidate path |
| `Gauge` | name, value, range, thresholds | Confidence, depth coverage, solver time |
| `Status` | text, level (good, warn, bad) | "Fallback Active", "Gate Holding" |

- **Harness panels are always present.** Speed, steering, throttle, brake, safe-stop
  state, FPS and loop latency come from the harness, so every stack shows them.
- **Ground-frame polylines are projected by the harness** using the camera geometry from
  the stack's `SensorSpec`, so a planner in metres draws correctly on any camera.
- **Layers the stack emits become pages in the existing debug view.** The operator cycles
  through them with the current debug-view keys.
- **The threading stays as it is.** The dashboard worker reads the latest `StackOutput`
  snapshot on its own thread; the control loop never waits for it.
- **Trade-off:** a generic layout is less tailored than today's hand-placed panels. A
  stack may optionally provide its own layout later, but none is required.

## Benchmarking From Ground Truth

The harness scores every run from simulator ground truth, never from what the stack
reports about itself, so a stereo stack, an end-to-end stack and the March stack get the
same scores. Today's `BenchmarkRecorder` reads stack internals such as segmentation
confidence and path jitter, which a different stack may not have.

**Core metrics (every stack):**

- Cross-track error, mean and max, from the vehicle's true position against a reference
  route for that map and spawn.
- Departures: count of times the vehicle leaves the drivable area.
- Completion and distance driven before the run ended or safe stop fired.
- Loop FPS and end-to-end latency, measured by the harness around `step()`.
- Smoothness: steering rate and sign changes, from the commands sent.

**Stack metrics (optional):** whatever the stack puts in `StackOutput.metrics`, such as
depth coverage or MPC solve time, reported in a separate section that never feeds the
comparison.

**Experiment runner:**

```bash
python scripts/run_experiments.py configs/experiments/*.yaml \
    --seconds 60 --map automation_test_track --spawn 0
```

It runs each config in turn against the same simulator, map, spawn and duration, writes
one JSON report per run, and prints a single comparison table (an extension of today's
`compare_benchmarks.py`). The offline `sbend_sim.py` plant gets the same treatment, so
controller comparisons need no simulator.

Open question: where the reference route for cross-track error comes from (recorded
drive, BeamNG road network, or hand-placed waypoints per spawn).

## Migrating The Existing Branches

All three branches become stacks on one branch. Because the contract is only "sensors in,
command out", old code moves in largely as it is rather than being rewritten.

| Branch | Becomes | Sensors It Declares | Work Involved |
| --- | --- | --- | --- |
| `platform-config-and-dashboard-redesign` | `stacks/mono_yoloe/` | One GMSL2 dashcam | Wrap `AutonomyPipeline`, which already takes a frame and state and returns a command |
| `camera-position-experiments` | `stacks/stereo_depth/` | Left and right cameras of the stereo rig | Restore `stereo_depth.py`, `stitching.py`, `stereo_worker.py` and the rig geometry from `7d7ce3c`; its experiments become `stack_config` overlays |
| `march-2026-03-30-network-attach` | `stacks/march_legacy/` | One 1280x720 camera at its old pose | Copy its pipeline in nearly unchanged; its network-attach code is not needed because the harness already handles it |

What carries over and what does not:

- **Carries over:** each stack's own perception, planning and control behaviour, and its
  camera rig.
- **Does not carry over:** each branch's main loop, safe stop and timing. Those come from
  the shared harness, so results will differ from what the branches measured on their
  own. That is intended, because it is what makes the comparison fair.
- **`--record-video` and the orbit-camera inset** exist on both older branches but not on
  this one. They belong in the harness so every stack can be recorded.

## Limits, Risks And Open Questions

The design covers any change inside a stack. It does not cover changes to the harness
itself.

**Limits**

- **A new sensor type** (lidar, radar, IMU) needs one adapter in `simulation/` the first
  time. After that, any stack can request it.
- **Harness changes** such as multiple vehicles, a different safety policy or a different
  simulator are outside the contract.
- **Safety stays in the harness on purpose.** No stack can disable the safe stop, the
  stuck detector or the manual override.

**Risks**

- **Loop budget:** a heavy stack can still starve the loop. The harness should measure
  `step()` time and log when a stack exceeds the budget.
- **Generic dashboard:** less polished than today's tailored panels until stacks emit good
  visuals.
- **Kept-alive old code:** `march_legacy` keeps an older pipeline in the tree. Worth it
  only if it stays a baseline that is re-run.

**Changes To Project Rules (CLAUDE.md)**

- "The single GMSL2 dashcam supplies every stage" and "Perception uses RGB only" become
  rules per stack, not per repo.
- The architecture section describes harness plus stacks; the `perception/`, `planning/`
  and `control/` packages move under `stacks/` or `stacks/common/`.

**Open Questions**

- Source of the reference route for cross-track error (see Benchmarking From Ground
  Truth).
- Whether shared parts live in `stacks/common/` or stay as top-level packages that stacks
  import.
- Whether `march_legacy` is worth porting, or stays a frozen tag compared by recorded
  video.

## Rollout Plan

Five steps, each mergeable and tested on its own. Step 1 is the real test of the design:
if the current pipeline fits the contract cleanly, the rest follows.

1. **Contract and first stack.** Define `DrivingStack`, `Observation`, `StackOutput` and
   `SensorSpec`; wrap the current pipeline as `stacks/mono_yoloe/`. Done when tests,
   `pytest --cov` and `sbend_sim.py` give the same results as before.
2. **Generic dashboard and ground-truth scoring.** The dashboard draws `Visual`
   primitives; the benchmark scores from simulator truth. Done when `mono_yoloe` looks and
   scores as it does today.
3. **Experiment runner and strict config.** Add `scripts/run_experiments.py`; make the
   harness and stack configs reject unknown keys. Done when N configs run back to back
   into one comparison table.
4. **Stereo stack.** Restore the stereo rig, depth and stitching from `7d7ce3c` as
   `stacks/stereo_depth/`, with depth on its own worker. Port the camera-position
   experiments as overlays.
5. **March stack (optional).** Port only if it is needed as a baseline that keeps being
   re-run.

Alongside step 1, update CLAUDE.md and the README to describe harness plus stacks.
