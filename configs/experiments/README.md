# Camera Position Experiments

Each file here extends `../local-network.yaml` and changes exactly what its
name says, so switching between them is just `--config`:

```bash
offroad-autonomy --config configs/experiments/<name>.yaml
```

All three keep everything else - map, spawn, model, control gains - identical
to `local-network.yaml`, so any difference in behaviour comes from the camera
change alone (see the per-file caveats below for the one place that wasn't
kept perfectly isolated on the first attempt).

## wide-baseline.yaml

**Changes:** `stereo_rig.baseline_m` 0.6 m &rarr; 0.8 m. Nothing else.

**Tests:** whether the wider end of `default.yaml`'s stated "0.4-0.8 m
sensible range" actually buys better far-range depth, or just costs
near-field stereo overlap. `depth.num_disparities: 0` re-derives the search
range automatically (96 &rarr; 112 px), so no other retuning was needed.

**Status:** runs cleanly, matches the bumper rig's stereo geometry logging
with the new baseline. Not yet benchmarked against the baseline config for
actual depth coverage / `valid_disparity_fraction` - that comparison is the
open next step, via `scripts/diagnose_perception.py` or a benchmark run.

## toe-out.yaml

**Changes:** `stereo_rig.toe_out_deg` 0 &rarr; 8 (per camera). Nothing else.

**Tests:** `default.yaml`'s claim that toe-out "widens the stitched view but
eats stereo overlap" - a trade-off the code asserts but never measures.

**Important:** `toe_out_deg` is applied *per camera*
(`StereoRigSpec.mount()` yaws left `+toe_out_deg`, right `-toe_out_deg`), so
the two optical axes end up `2 * toe_out_deg` apart. `StereoRig`'s own
validator refuses anything past 20&deg; combined. A first attempt at 15&deg;
per camera (30&deg; combined) tripped that check and silently disabled
stereo depth for the whole run. 8&deg; per camera (16&deg; combined,
confirmed via `StereoRig(left, right).axis_angle_deg`) is comfortably under
the limit.

**Status:** runs and initializes correctly (confirmed rectification/stereo
rig logging shows 16.0&deg; apart, as expected). Driving-quality impact not
yet isolated on its own - see `windshield-mount.yaml` below for why that
matters more than it first looked.

## windshield-mount.yaml

**Changes:** `stereo_rig.height_m` 0.95 m &rarr; 1.85 m and
`forward_offset_m` -1.95 m &rarr; -0.30 m (the same position
`default.yaml`'s separate, non-computational display camera already uses).
`pitch_deg` is deliberately left at the bumper rig's -2.0, not the display
camera's -8.0, so only the mount *moves* rather than also being re-aimed.
Also carries two re-derived (not guessed) consequences of the move:
- `left`/`right` `ego_mask` - the bumper mount clears all bodywork by
  design, so it needs no exclusion mask. This mount sits behind the hood,
  and `scripts/derive_ego_mask.py` confirms it: 9.9% of each camera's frame
  is now hood.
- `planning.roi_height` 0.49 &rarr; 0.46 - re-measured with
  `CameraModel.ground_to_image()` to keep the same ~15 m lookahead at the
  new height. Counterintuitively the higher mount pushes a given distance to
  a row *closer* to the bottom of frame, so it needed a smaller value, not a
  larger one.

**Tests:** the windshield/roofline placement most 2-camera guidance
recommends, versus this vehicle's low bumper mount.

**A real bug this caught:** the first version of this file *also* added
8&deg; toe-out (matching toe-out.yaml), combining two variables in one
experiment. Because `segmentation_mode: left` means the entire perception
pipeline runs on one camera, toeing that camera away from dead-ahead on top
of the height change made the perception gate reject 145 of 162 frames
(`centerline too short`, then `no road at vehicle`, then zero confidence).
Starved of active control that long, with no parking brake applied during a
gate stop, the vehicle rolled backward down the spawn point's slope under
gravity - confirmed by comparing telemetry (`scripts/closed_loop_log.py`)
against a clean bumper-rig reference run, and by ruling out stale-vehicle/
session state as the cause first. Removing the toe-out (this file now has
none) dropped gate rejects to 1 of 142 and restored correct forward travel.

**Status:** drives in the correct direction and no longer stalls the gate,
but is not well-tuned - it saturates steering (19 frames at full lock),
posts a 1.33 rad p95 heading error, and the edge-clearance metric reports
it outside the trail on 102 of 142 frames in a 20 s run. Height/offset alone
is drivable; it is not yet a placement worth preferring over the bumper
mount. `roi_height` and the segmentation view are the next things to
re-tune if this is worth pursuing further.

## Comparing runs

```bash
python scripts/closed_loop_log.py --config configs/experiments/<name>.yaml \
    --seconds 20 --out output/diagnostics/<name>.csv --summary
```

prints per-run gate-reject counts, cross-track error, and edge-clearance
stats to the terminal and writes a full per-frame CSV - the fastest way to
compare a variant against `local-network.yaml`'s baseline numbers before
trusting anything by eye.
