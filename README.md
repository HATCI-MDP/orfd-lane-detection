# Off-Road Autonomy

## Introduction

Off-Road Autonomy drives a vehicle along unmarked dirt trails in BeamNG.tech. A fixed **harness** owns the simulator, the control loop, safety, the dashboard and scoring; a swappable **driving stack**, named in the config, turns sensor data into a control command. Different perception and planning approaches therefore run, and are compared, on one branch (see [Harness And Driving Stacks](#harness-and-driving-stacks)).

Each stack's sensors, settings, dashboard pages, rules and tools are documented in [docs/stacks/](docs/stacks/README.md).

It runs on Windows, WSL2, macOS or a Jetson. BeamNG.tech itself runs only on Windows, so every other machine connects to it over the network.

## Setup

Requires Python 3.10+ and BeamNG.tech 0.38 (`beamngpy` 1.35).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

A stack may need model files in `models/` or other setup; its doc in [docs/stacks/](docs/stacks/README.md) lists them.

### Connect To BeamNG

BeamNG.tech runs only on Windows. The Windows machine running it is the BeamNG host. The stack runs either on the BeamNG host or on another machine that connects to it.

| Stack Runs On                | What To Set                                                  | How To Start BeamNG      |
| ---------------------------- | ------------------------------------------------------------ | ------------------------ |
| Windows, on the BeamNG host  | `BEAMNG_HOME` to launch automatically, or start the server below | BeamNGpy server enabled |
| WSL2, on the BeamNG host     | Nothing                                                      | Listening on the network |
| macOS                        | `BEAMNG_HOST` (the BeamNG host's address)                    | Listening on the network |
| Jetson                       | `BEAMNG_HOST` (the BeamNG host's address)                    | Listening on the network |

For automatic launch on Windows, set the install folder in the same PowerShell session:

```powershell
$env:BEAMNG_HOME = "<your BeamNG.tech folder>"
$env:BEAMNG_LAUNCH = "true"
offroad-autonomy
```

The folder must contain `Bin64\BeamNG.tech.x64.exe`. To retain the path for future
terminals, use `[Environment]::SetEnvironmentVariable("BEAMNG_HOME", $env:BEAMNG_HOME, "User")`.
For local Windows connections with `launch: auto`, the app also reads that saved
user setting when neither the current environment nor YAML supplies an install
folder. Existing terminals can therefore use it without restarting VS Code.

With no install folder configured, the app only attaches to an existing server.
Opening BeamNG normally does not enable that server. Start it manually with:

```powershell
cd "<your BeamNG.tech folder>"
.\Bin64\BeamNG.tech.x64.exe -nosteam -tcom -tport 64256 -tcom-listen-ip "127.0.0.1"
```

For WSL2, macOS and Jetson, BeamNG must listen on the network. Start it from PowerShell on the BeamNG host:

```powershell
cd "<your BeamNG.tech folder>"
.\Bin64\BeamNG.tech.x64.exe -nosteam -tcom -tport 64256 -tcom-listen-ip "*"
```

If the connection is refused, allow TCP port 64256 through the Windows firewall. For macOS and Jetson, find the BeamNG host's address with `ipconfig`.

The first log line shows the address the stack connects to. `BEAMNG_HOST` overrides it on any machine.
The app uses port **64256**; BeamNG's `-tcom` default is **25252**, so include
`-tport 64256` or set `BEAMNG_PORT` to the server's actual port. See the
[BeamNG connection arguments](https://documentation.beamng.com/beamng_tech/arguments_and_settings/).

## Harness And Driving Stacks

A stack is any object with three methods, defined in `src/offroad_autonomy/contract.py`:

- `sensors()` returns the `SensorSpec`s to attach (cameras today), read once before connecting.
- `step(observation)` takes every declared sensor's reading and the vehicle's own speed and yaw
  rate (never the simulator's world pose), and returns a `StackOutput`: the control command,
  visuals for the dashboard, optional metrics, and an optional safe-stop request.
- `reset()` is called when the operator resumes after a safe stop.

A run config names its stack by import path and hands it a section only the stack reads:

```yaml
extends: default.yaml
stack: offroad_autonomy.stacks.<name>:build
stack_config:
  # Only the stack reads this section; its doc lists the keys.
```

| Section                                                                             | Read By        | Holds                                                                   |
| ----------------------------------------------------------------------------------- | -------------- | ----------------------------------------------------------------------- |
| `beamng`, `safety`, `ui`, `recording`, `presentation`, `visualization`, `benchmark` | Harness        | Simulator, map, spawn, stuck detector, dashboard, recording and scoring |
| `stack`                                                                             | Harness        | Import path of the stack's `build(stack_config)` function               |
| `stack_config`                                                                      | The stack only | Everything about its perception, planning, control and sensor rig       |

- **Every key is checked.** The harness refuses keys it does not know, and keys from before the
  split name where they moved (for example `perception` is now `stack_config.perception`).
  Each stack validates `stack_config` itself.
- **Adding a stack is adding a package** under `src/offroad_autonomy/stacks/` with a `build`
  function; nothing in the harness, dashboard or benchmark lists stacks. Stacks may reuse
  `perception/`, `planning/` and `control/`, but nothing requires them to.
- **A stack never imports `beamngpy`** and must return from `step` within the loop budget. Slow
  work belongs on the stack's own worker. The harness logs every `step` that takes longer than
  `visualization.dashboard.latency_budget_ms`.
- **Safety stays in the harness.** The stuck detector, `E` safe stop and manual override apply
  to every stack. A stack can ask for a safe stop for a condition only it can see, but it
  cannot disable one. The stack keeps running
  under safe stop, so the operator can see the road come back before resuming.

## Run

```bash
offroad-autonomy
```

This runs the stack named in `configs/default.yaml`; `--config <file>` runs another config.

The BeamNG client selects realistic shifting and forward drive on startup and
resume, so holding the brake cannot engage arcade reverse. Manual gearboxes use
first gear for low-speed off-road operation; automatics select Drive. Run
`python scripts/check_brake_hold.py --reproduce-arcade` to reproduce the former reversal and
verify the brake-hold/forward-restart fix.

- Starts BeamNG.tech when `BEAMNG_HOME` is set and it is not already running, then spawns the vehicle and opens the dashboard
- `E` safe stop, `P` resume, `0` main view, `1`-`9` the stack's own layers, `T` timing overlay, `Q` quit
- `--headless` runs without a window
- `--presentation-view` shows the orbit camera, the stack's main view and stats together live
- `--map` and `--spawn` override `beamng.map` and `beamng.spawn_index`
- `--record-route` records the vehicle's true path as the reference route for this map and spawn (see [Benchmark And Compare Stacks](#benchmark-and-compare-stacks))
- `--record-video` saves the dashboard to an mp4 (see Record The Dashboard below)
- `--presentation` saves a 1920 x 1080 video for an audience outside the team (see Record A Presentation below)

### Live Presentation View

```powershell
offroad-autonomy --presentation-view
```

Opens the split view with the orbit camera on the left and the stack's main view and
telemetry on the right. No video is recorded and ffmpeg is not required. Use
`E` to stop, `P` to resume and `Q` to quit. Add `--presentation` to record this
view too. `--headless` suppresses the live window.

### Record The Dashboard

```bash
offroad-autonomy --record-video
offroad-autonomy --record-video --headless --benchmark-seconds 60 --label baseline
```

`--record-video` saves the dashboard, exactly as drawn, to `output/videos/<label>_<timestamp>.mp4`
(or the path given with `--record-video-out`). The file is H.264 in yuv420p with the index at the
front, so it plays in the VS Code media preview and in browsers.

- Needs `ffmpeg` with libx264 on `PATH` (`sudo apt install ffmpeg`). Without it the run stops before
  connecting to BeamNG. A snap-packaged ffmpeg cannot write under `/tmp`, so keep the output in
  your home directory.
- Encoding runs in a separate ffmpeg process fed from its own thread, so neither the control loop
  nor the dashboard waits for it. If the encoder falls behind, frames are dropped from the video
  and counted in the log when the run ends.
- Frames are paced onto a constant `recording.fps`, so the video plays back at real speed even
  when the dashboard stalls.
- With `--headless` the dashboard is drawn off screen for the recording only.
- Quality, preset and queue length are in the `recording` section of `configs/default.yaml`.

### Record A Presentation

```bash
offroad-autonomy --presentation
offroad-autonomy --presentation --headless --presentation-out ~/videos/demo.mp4
offroad-autonomy --presentation --record-video
```

`--presentation` records a 1920 x 1080 "Split Story" video to
`output/presentations/<label>_<timestamp>.mp4` (or the path given with `--presentation-out`), apart
from the dashboard recordings in `output/videos/`. It uses the same ffmpeg recorder and `recording`
settings as `--record-video`. Add `--record-video` to record the dashboard as well, to its own
file. The two paths must differ. The window, if there is one, still shows the operator dashboard,
unchanged.

- **Left:** a chase camera behind and above the vehicle, labelled Orbit Camera. Until its first
  frame arrives the view reads No Orbit Signal.
- **Top right:** the stack's main view with its masks, paths and exclusions, drawn by the same code
  as the dashboard's view.
- **Bottom right:** speed, steering, Autonomy FPS, latency mean / p95 and the stack's first two gauges,
  coloured by the dashboard's rules and the gauges' own thresholds.
- **Safe stop:** the header chip turns solid red, the orbit view gets a red wash and border, and no
  path is drawn on the main view.

The orbit camera is attached when recording or showing a presentation, and no stack reads it. Its pose,
resolution and field of view are in the `presentation` section of `configs/default.yaml`. The main
loop polls it right after each frame of the stack's sensors, because the socket transport cannot be read from the
dashboard thread. That poll is logged as the `orbit_capture` stage and is part of the measured
primary loop, so the Autonomy FPS in the video is the real rate of the loop with the camera
attached. It has not yet been measured against BeamNG, so expect a lower FPS than without the flag
until it has.

Recording-only presentation frames are drawn on their own thread. The live presentation
uses the configured display threading mode; Tkinter requires drawing on the main thread.
Its render time is in the periodic runtime log line.

### Dashboard

The 1600 x 900 dashboard has a header with the autonomy status and the stack's name, the stack's main view, a Vehicle card, a Driving Stack card and a Runtime strip. Speed is the largest number on screen.

The dashboard never sees a stack's own types. A stack describes itself with five primitives, so a new stack renders with no dashboard change:

| Primitive     | Drawn As                                                                                          |
| ------------- | ------------------------------------------------------------------------------------------------- |
| `ImageLayer`  | A page. The first is the main view; keys `1`-`9` show the rest beside a readout of every metric   |
| `MaskOverlay` | A tint on a layer: traversable (teal with an edge), excluded (hatched magenta) or info            |
| `Polyline`    | A path on a layer, in image pixels or in ground metres projected with the declared camera         |
| `Gauge`       | A bar on the Driving Stack card, coloured by its own thresholds and ticked at the first           |
| `Status`      | A chip on the card when it is a warning or fault; good statuses appear only in the readout        |

- **Safe stop:** a red wash, a red border and a banner cover the main view, and no path is drawn.
- **Autonomy FPS:** green at `visualization.dashboard.target_fps` or above, amber from `fps_warn_fraction` of it up to the target, red below that. Latency is red when its p95 passes `latency_budget_ms`.
- **Dashboard FPS:** shown in its own tile and never colored, because a slow window says nothing about the vehicle.
- **Held path:** a `held` polyline is drawn dashed in amber.
- **Costly pages:** a layer's image may be a function. The dashboard calls it on its own thread, and only while that page is shown, so a costly debug image never runs in the control loop.

Text is drawn with Pillow using the bundled IBM Plex Sans and Mono fonts in `src/offroad_autonomy/visualization/fonts/` (SIL Open Font License, included). A missing font file stops startup.

## Benchmark And Compare Stacks

The harness scores every run from simulator ground truth, never from what the stack reports, so
any two stacks compare fairly.

1. Record a reference route once per map and spawn by driving it cleanly, either with the stack
   or by hand after `E`. It is saved on exit to
   `<benchmark.routes_dir>/<map>_spawn<index>.json`:

```bash
offroad-autonomy --map automation_test_track --spawn 0 --record-route
```

2. Run several configs back to back on the same map, spawn and duration. Each config runs headless
   in its own process; one JSON report per run goes to `output/experiments/`, followed by one
   comparison table:

```bash
python scripts/run_experiments.py configs/a.yaml configs/b.yaml \
    --seconds 60 --map automation_test_track --spawn 0
```

| Metric                                        | From                                                                                        |
| --------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Cross-track mean and max, completion          | The vehicle's true position against the reference route                                     |
| Departures                                    | Times the true position went more than `benchmark.departure_cross_track_m` off the route    |
| Distance, safe stops                          | True positions, and the harness's own safe-stop state                                       |
| Steering rate, sign changes per minute        | The commands sent; changes inside `benchmark.steer_sign_deadband` are ignored               |
| FPS, capture-to-actuation latency, step time  | Measured by the harness around `step`, after `--benchmark-warmup` seconds                   |

- Driving metrics stop at the first safe stop, because what the operator drives afterwards is not
  the stack's. A run without a route reports cross-track, completion and departures as null.
- Whatever the stack puts in `StackOutput.metrics` is averaged into a separate `stack_metrics`
  section that never feeds the comparison.

## Run On Jetson (Docker)

Requires JetPack 6 with the NVIDIA container runtime. Start BeamNG listening on the network first (see [Connect To BeamNG](#connect-to-beamng)).

1. Build the image, then do any device-specific preparation the stack's doc lists (such as exporting a TensorRT engine):

```bash
docker compose build
```

2. Run:

```bash
BEAMNG_HOST=192.168.1.50 docker compose up
```

- The container runs `configs/jetson.yaml`
- `models/`, `output/` and `configs/` are mounted from the host, so changes to run configs (including their `stack_config`) need no rebuild; a stack's own `default.yaml` is part of the package and does
- `docker compose stop` parks the vehicle before disconnecting

## Testing

| Command                                       | What it runs                                                 | Needs BeamNG |
| --------------------------------------------- | ------------------------------------------------------------ | ------------ |
| `pytest`                                      | Unit and integration tests (models and simulator are mocked) | No           |
| `pytest --cov`                                | Same suite plus the 70% coverage gate CI enforces            | No           |
| `docker compose --profile test run --rm test` | The same suite inside the Jetson image                       | No           |
| `offroad-autonomy --benchmark-seconds 90`     | Timed run scored from ground truth, to `output/benchmarks/`  | Yes          |
| `python scripts/run_experiments.py <configs>` | Several configs back to back, then one comparison table      | Yes          |

## Common Commands

| Command                                                         | What it does                                            |
| --------------------------------------------------------------- | ------------------------------------------------------- |
| `offroad-autonomy --config <file>`                              | Run the stack with a config                             |
| `offroad-autonomy --record-video`                               | Run and save the dashboard to `output/videos/` as mp4   |
| `offroad-autonomy --presentation`                               | Run and save a presentation to `output/presentations/`  |
| `ruff check src tests scripts`                                  | Lint                                                    |
| `ruff format src tests scripts`                                 | Format (CI runs it with `--check`)                      |
| `python -m build`                                               | Build the wheel and sdist                               |
| `pytest`                                                        | Run the tests                                           |
| `python scripts/compare_benchmarks.py output/benchmarks/*.json` | Compare benchmark runs side by side                     |
| `python scripts/run_experiments.py <configs> --seconds 60`      | Run configs back to back and compare them               |
| `offroad-autonomy --record-route`                               | Record the reference route for a map and spawn          |

## Development Notes

- Package code is in `src/offroad_autonomy/`. The harness loop is `main.py` and the contract is `contract.py`; each stack is a package under `stacks/`, documented in `docs/stacks/<stack>/`
- Only `simulation/beamng_client.py` imports `beamngpy`
- `configs/default.yaml` holds every harness value and each stack's `default.yaml` every stack value, with the reason for each next to its key; other files in `configs/` override them through `extends:` and `stack_config`
- `beamng.host`, `beamng.launch`, `beamng.camera_transport` and `ui.display_async` default to `auto` and are derived from the platform; `BEAMNG_HOST`, `BEAMNG_PORT`, `BEAMNG_HOME` and `BEAMNG_LAUNCH` override the `beamng:` block
- The dashboard runs on its own thread, and the control loop never waits for it
- Manual driving (W/A/S/D under safe stop) reads the OS key state and works only on Windows
- CI (`.github/workflows/ci.yml`) runs lint, format, the stacks' offline checks, tests with coverage and the build on Python 3.10; CodeQL scans weekly and on every pull request; Dependabot watches pip, the Docker base image and the Actions

## Contribution Rules

- Create a new branch from main for every change.
- Do not commit directly to main.
- Open a pull request into main when the change is ready.
- Keep pull requests small, focused, and easy to review.
- Run `ruff check src tests scripts`, `ruff format --check src tests scripts` and `pytest --cov` before opening a pull request.
- After changing a stack, run the checks its doc in `docs/stacks/` lists.
- No ternary expressions, and no single-line `if` bodies.
- Comments explain why, not what.
- Do not create commits unless explicitly asked.
- Before finishing, summarize what changed, what commands were run, what commands could not be run, and any remaining risks.
