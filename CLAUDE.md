# Off-Road Autonomy

Autonomous off-road driving in BeamNG.tech. A fixed harness (simulator I/O, loop, safety,
dashboard, scoring) runs a swappable driving stack chosen by the config. Each stack is
documented in `docs/stacks/<stack>/`. The single idea behind the design is that the control
loop never waits for anything slow: the dashboard runs on its own thread.

## Architecture

Keep these boundaries clear:

- `contract.py`: the only types the harness and a stack share (`DrivingStack`, `SensorSpec`,
  `Observation`, `StackOutput` and the visual primitives).
- `main.py`: the harness loop, safe stop, manual override and shutdown.
- `simulation/`: all BeamNG I/O. The only package that imports `beamngpy`. Attaches the
  sensors the stack declares.
- `runtime/`: the dashboard worker thread, timing, reference routes and ground-truth benchmarks.
- `visualization/`: the operator dashboard and window, drawn only from visual primitives.
- `stacks/`: one package per driving stack, each with a `build(stack_config)` function and
  its own defaults.
- `perception/`, `planning/`, `control/`: parts stacks may share; nothing requires them to.

Rules that carry weight:

- Nothing outside `simulation/` may import `beamngpy`, so every stack stays testable without
  a simulator.
- The harness never imports a stack's internal types, and the dashboard and benchmark never
  read them: a stack talks to the harness only through `contract.py`. Adding a stack must not
  change the harness, dashboard or benchmark.
- The benchmark scores from simulator ground truth, never from what a stack reports.
- Safety stays in the harness. A stack may ask for a safe stop but can never disable one.
- A stack's `step` must return within the loop budget; slow work belongs on its own worker,
  and a costly debug image is an `ImageLayer` whose image is a function.
- Each stack's own rules and checks are in `docs/stacks/<stack>/README.md`. Read it before
  changing that stack, and keep it current when the stack changes.
- `perception/__init__.py` must never import `RoadSegmenter`, because that pulls in Torch
  and Ultralytics for code that only needs geometry.
- The control loop must never block on the dashboard.
- Every tuning value lives in a config with its reason next to it, not as a literal in code:
  harness values in `configs/default.yaml`, stack values in that stack's own defaults.
  Both are validated strictly, so a new key must be added there before it can be set.
- Machine-specific simulator settings stay `auto` and resolve in `utils/environment.py`;
  never write a host, install path or transport for one machine into a config.

Prefer simple, well-defined boundaries over unnecessary abstractions or infrastructure.

## Non-Negotiable Style Rules

- Never use em dashes.
- Always use Title Case for user-facing titles and labels, such as README headings and
  workflow step names.
- Never use ternary expressions.
- Always use block-form `if` statements, even for single statements.

```python
# Correct
if ready:
    return

# Incorrect
if ready: return

# Incorrect
status = "Ready" if ready else "Pending"

# Correct
status = "Pending"
if ready:
    status = "Ready"
```

- Comments must explain **why**, not **what**.

```python
# Correct: BeamNG keeps applying the last command, so a client that vanishes leaves the vehicle driving.
self.park()

# Incorrect: Park the vehicle.
self.park()
```

## Engineering Principles

- Prefer readable control flow over clever or compact code.
- Keep simulator-specific logic isolated in `simulation/`.
- Keep secrets, credentials, tokens, and sensitive data out of source code, logs, stored
  files, and tests.
- Do not add infrastructure, dependencies, interfaces, or abstractions without a concrete
  need.
- Add tests for meaningful behavior and failure cases.
- Keep changes scoped to the task being implemented.
- Update documentation when behavior or architecture changes.
- Do not claim unfinished functionality is implemented, and do not quote performance
  numbers that were not measured.

## Before Finishing

Run the relevant repository checks and fix all failures:

```bash
ruff check src tests scripts
ruff format --check src tests scripts
pytest --cov
python -m build
```

Also run the checks listed in the doc of any stack you changed.

Review the final diff for unnecessary code, unused dependencies, style violations, secrets,
and accidental scope expansion.
