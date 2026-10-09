# Driving Stacks

One folder per stack, named after its package in `src/offroad_autonomy/stacks/`. Everything
specific to a stack lives here; the repo README and `docs/design/` cover only the harness and
the contract every stack shares.

| Stack | Status | Docs |
| --- | --- | --- |
| `mono_yoloe` | Built; the default in `configs/default.yaml` | [mono_yoloe/](mono_yoloe/README.md) |
| `stereo_depth` | Planned | Not yet |
| `march_legacy` | Optional | Not yet |

**Planned ports:**

- **`stereo_depth`** from branch `camera-position-experiments` (`7d7ce3c`): left and right
  stereo cameras. Port `stereo_depth.py`, `stitching.py`, `stereo_worker.py` and the rig
  geometry to the current types, with depth on the stack's own worker; the branch's
  experiments become `stack_config` overlays.
- **`march_legacy`** from branch `march-2026-03-30-network-attach` (`3021745`): one 1280x720
  camera, pixel-space steering, no perception gate. Port only if it will be re-run as a
  baseline; otherwise keep it as a frozen tag compared by recorded video.

## What A Stack's README Covers

- **Summary:** what it senses, how it plans and controls, and what it does not do.
- **Sensors:** the rig it declares, and where its mount and lens are configured.
- **Config:** where its `default.yaml` lives, how it validates `stack_config`, and the
  overlays in `configs/` that select it.
- **Dashboard:** the pages, gauges and statuses it emits.
- **Safety:** any safe stop it requests through `StackOutput.safe_stop`.
- **Rules:** constraints that apply only inside this stack.
- **Checks And Tools:** offline checks to re-run after changing it, and its scripts.

Stack-specific design notes and findings go in the same folder.
