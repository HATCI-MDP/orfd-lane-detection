# mono_yoloe

The default stack: one dashcam, YOLOE-26 road segmentation, a ground-plane planner and
Stanley steering. Code is in `src/offroad_autonomy/stacks/mono_yoloe/`; the run config entry
is `stack: offroad_autonomy.stacks.mono_yoloe:build`.

## Summary

Each frame runs preprocess, segment, stabilise, plan and control in that order
(`pipeline.py`). Perception uses RGB segmentation only: there is no depth estimation, terrain
fusion or depth-based obstacle veto. Ground distances assume flat ground.

- **Segmentation:** YOLOE-26 with text prompts, or a custom semantic checkpoint.
- **Gate:** a frame fails if its road component is too small, disconnected or below
  `planning.gate.confidence_threshold` (0.18). A failed frame holds the last good path at
  reduced speed for `hold_frames`, then stops.
- **Planners** (`planning.mode`): `baseline` fits one quadratic through the centre of the
  largest road component touching the vehicle; `advanced` adds scoring, clearance and a
  Kalman filter; `grid` plans on a bird's-eye grid (see Grid Planner).
- **Control:** Stanley with curvature feedforward and a speed law, computed in metres on the
  ground. When the hood hides the nearest path, steering joins a visible lookahead point with
  a geometric pursuit arc instead of extrapolating the path under the hood, which could
  reverse the turn.

## Sensors

One GMSL2 dashcam supplies every stage, including the dashboard's main view: 120° horizontal
FOV, height 1.85 m, forward offset -0.30 m, pitch -8°, captured at 960x620 and 30 FPS and
processed at 720x465. The hood is excluded with a polygon (`camera.ego_mask`); pixels inside
it are invalid, not non-road.

## Config

Defaults, each with its reason, are in `src/offroad_autonomy/stacks/mono_yoloe/default.yaml`,
which is also the schema: `stack_config` may only set keys that file sets.

| Overlay | Changes |
| --- | --- |
| `configs/grid.yaml` | `planning.mode: grid` |
| `configs/jetson.yaml` | `perception.model_weights: models/yoloe-26s-seg.engine` (plus headless, a harness setting) |

### Model Weights

Put these in `models/`: `yoloe-26x-seg.pt` (default), `yoloe-26s-seg.pt` (Jetson) and
`mobileclip2_b.ts` (the YOLOE text encoder).

Custom semantic checkpoints such as `models/weights.pt` need Ultralytics 8.4.55 or newer. Set
`perception.model_weights` to the checkpoint and include its road class name in
`perception.prompts`; for semantic models the prompts select fixed classes and others are
excluded. The confidence readout is then the mean predicted probability over accepted road
pixels outside the ego mask.

On a Jetson, export a TensorRT engine on the device itself, since engines are tied to the GPU
that built them:

```bash
docker compose run --rm --entrypoint python autonomy scripts/export_engine.py --weights models/yoloe-26s-seg.pt
```

## Grid Planner

`planning.mode: grid` projects the road mask onto flat ground, fuses it over time using the
vehicle's motion, and picks the best of a fan of steerable arcs (`planning/bev_grid.py`,
`planning/arc_planner.py`). Settings are in `planning.grid`. The baseline planner stays the
default until the grid planner has been compared on several maps.

- Ground the camera cannot see is unknown, never "not road", so a trail running off the image
  side does not pull the path toward the visible part.
- Memory carries the road the hood now hides. It moves with dead reckoning from the wheel
  speed and yaw rate in `VehicleSignals` (`planning/odometry.py`), integrated at the rear
  axle in a local frame. The simulator's world pose never reaches the stack (see
  [bev-grid-production-inputs.md](bev-grid-production-inputs.md)). A missing reading, or a
  gap over `max_odometry_gap_s`, makes the grid forget rather than smear.
- Frames the gate rejects move the grid with the vehicle but add no evidence.
- Arcs end where the body would leave the road, and the path extends only as far as the grid
  has seen road, so the path-end speed law still applies.
- Each arc scores its length on road up to `reach_m` (6 m), its mean clearance over that
  first `reach_m`, and penalties for bending, for changing curvature and for straying from
  the motion-compensated previous path. Clearance and straying are measured against the best
  arc's clearance that frame, so they weigh the same on a 3 m trail as on a 14 m road.
- At a split it locks onto the branch closest to the current trajectory until only one
  remains or that branch is no longer safe.
- The controller gets the whole road mask the grid planned on (ROI rows not cut by the
  hood), not just the gate's largest component, which at a fork can be the other branch.

Limits:

- Flat-ground projection misplaces far cells on crests and in dips, and the grid has no
  height, so obstacles the segmenter labels as road are not seen.
- A mistaken road mask can accumulate in memory. Unknown cells count optimistically for
  clearance so unseen ground under the hood does not block all motion, which means clearance
  does not guarantee safe terrain.
- Road evidence comes only from the learned mask; BeamNG annotation, instance and depth
  buffers are disabled.
- Clearance is checked along the rear axle's arc with half the vehicle width plus
  `clearance_margin_m`; the swept corners of the body are not modelled. On the offline bench's
  tightest trail (3.5 m wide, 7 m radius) with noisy masks, the body still clipped the inside
  edge by up to 0.2 m.

## Dashboard

| Key | Page |
| --- | --- |
| `0` | Dashcam with the traversable mask, planned or held path and hood exclusion |
| `1` | Raw dashcam |
| `2` | Model mask and stabilised mask |
| `3` | Pipeline grid (raw, model segmentation, planner input, trajectory and control), drawn only while shown |

- **Gauges:** Road / Valid Px, ticked at `safety.min_road_fraction`; segmentation confidence,
  red below the gate threshold, amber up to `dashboard.confidence_good`, green above; valid
  ROI against ego coverage. The first two also appear in the presentation video.
- **Statuses:** `GATE HOLD` (warn) or `GATE STOP` (bad) with the gate's reason, `KALMAN`
  (warn) when the planner coasts, and the target speed and its reason.
- **Metrics:** segmentation, gate and controller internals (cross-track, trail edges,
  curvature, steering terms, target speed) and per-stage timings, shown in the readout and
  averaged into the benchmark's `stack_metrics`.

## Safety

The stack asks the harness for a safe stop when the road covers less than
`safety.min_road_fraction` of valid pixels for `safety.no_road_time_s`. The fraction is
measured over valid pixels only, so bodywork in frame cannot trip it.

## Rules

- RGB only, and the single dashcam supplies every stage. The dashboard consumes the same
  inference snapshot and never captures a separate image.
- `stacks/mono_yoloe/__init__.py` must never import the stack module at import time: it pulls
  in Torch and Ultralytics, and tools that only read this stack's config must not.
- After moving the camera or changing its lens, re-run `scripts/derive_ego_mask.py` and
  re-check `planning.roi_height`.
- After any change to the controller or its config, re-run `scripts/sbend_sim.py`.
- After any change to the grid planner or `planning.grid`, re-run `scripts/grid_sim.py`, clean
  and with `--odometry noisy --mask-noise --pitch-deg 0.5 --seeds 3`.

## Checks And Tools

| Command | What It Does | Needs BeamNG |
| --- | --- | --- |
| `python scripts/sbend_sim.py [--config a.yaml b.yaml]` | Closed-loop controller check on a synthetic S-bend; several configs print one comparison table. CI runs it | No |
| `python scripts/grid_sim.py [--odometry noisy --mask-noise --pitch-deg 0.5 --seeds 3]` | Closed-loop grid planner check on 12 synthetic trails, six held out from tuning, with production signals only | No |
| `python scripts/derive_ego_mask.py` | Regenerate the hood polygon after moving the camera | No |
| `python scripts/diagnose_perception.py capture` / `analyze` / `full` | Save simulator frames, then dump every perception stage | `capture` only |
| `python scripts/closed_loop_log.py --seconds 120 --summary` | Headless run that logs every control frame to CSV | Yes |
| `python scripts/export_engine.py` | Export the segmentation model to TensorRT | No |

With the shipped configs, the S-bend check gives mean off-centre 0.14 m, p95 0.46 m and max
0.56 m, the same as before the harness split.

The grid check, with clean signals, finishes all 12 trails with no departure frame, a minimum
body margin of 0.17 m and a mean offset of 0.14 m. With noise added it finishes all 36 runs;
the only departures are on `held_out_tight_r7_w3.5`, by up to 0.16 m. The synthetic S-bend,
grid check and planner tests use idealised masks: they test mechanics, not real perception
accuracy.
