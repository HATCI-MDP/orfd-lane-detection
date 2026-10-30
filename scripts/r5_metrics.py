#!/usr/bin/env python3
"""Score R5 lighting-robustness runs recorded with ``offroad-autonomy --eval-log``.

Each run folder sits under its condition, for example::

    output/r5/noon/run1  output/r5/noon/run2  output/r5/evening/run1

    python scripts/r5_metrics.py output/r5/*/run* --baseline noon --out output/r5/summary

Writes ``results.csv`` (one row per run) and ``verdicts.json`` (PASS/FAIL per
metric per condition). M1 road IoU is not measured: BeamNG's annotation camera
labels the dirt trail as NATURE like the trees, and no hand labels were made.
Nothing is estimated: a value that cannot be measured is left
empty with the reason in ``notes``.

Pass criteria (proposed, not mentor-approved):

* M2: gate rejection rate < 5% and longest rejected streak < 10 frames.
* M3: mean |lateral offset| no more than 0.10 m above the baseline condition's
  mean over its runs.
* M4: zero road departures and zero interventions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from offroad_autonomy.runtime.route_metrics import (
    build_centerline,
    count_departures,
    lateral_offsets,
    rejection_stats,
    travel_headings,
)

M2_MAX_RATE_PCT = 5.0
M2_MAX_STREAK = 10
M3_MAX_WORSE_M = 0.10

COLUMNS = (
    "condition",
    "run_id",
    "time_of_day",
    "n_frames",
    "rejection_rate_pct",
    "longest_reject_streak",
    "mean_abs_offset_m",
    "departures",
    "interventions",
    "n_labeled_frames",
    "road_IoU",
    "run_completed",
    "notes",
)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def score_run(run: Path) -> dict:
    metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    frames = _read_csv(run / "frames.csv")
    notes = []

    accepted = []
    for row in frames:
        if row["gate_evaluated"] != "1":
            accepted.append(None)
            continue
        accepted.append(row["gate_ok"] == "1")
    gate = rejection_stats(accepted)
    if gate.evaluated < len(frames):
        notes.append(f"{len(frames) - gate.evaluated} frames not gate-evaluated (stale)")

    pose = _read_csv(run / "pose.csv")
    times = np.array([float(row["log_time_s"]) for row in pose])
    xy = np.array([[float(row["x"]), float(row["y"])] for row in pose])
    mean_offset = None
    departures = None
    trimmed = np.zeros(0, dtype=bool)
    if len(xy) >= 2:
        # Offsets, but not departures, skip the launch: leaving the road while
        # accelerating still counts.
        trimmed = times >= times[0] + float(metadata["trim_start_s"])
        roads = json.loads((run / "road_network.json").read_text(encoding="utf-8"))
        offsets, half_widths = lateral_offsets(xy, build_centerline(roads), travel_headings(xy))
        departures = count_departures(offsets, half_widths)
        if trimmed.any():
            mean_offset = float(np.mean(np.abs(offsets[trimmed])))
        else:
            notes.append("no position samples after the start trim")
    else:
        notes.append("fewer than 2 position samples; offset and departures not measurable")

    if metadata.get("outcome") != "completed":
        notes.append(
            f"outcome {metadata.get('outcome')} {metadata.get('outcome_detail', '')}".strip()
        )
    if metadata.get("images_dropped"):
        notes.append(f"{metadata['images_dropped']} images dropped by the log writer")
    notes.append(f"{int(trimmed.sum())} offset samples, {metadata.get('travelled_m')} m travelled")

    completed = "N"
    if metadata.get("run_completed") and departures == 0:
        completed = "Y"
    return {
        "condition": run.parent.name,
        "run_id": run.name,
        "time_of_day": metadata.get("daylight_pct"),
        "n_frames": len(frames),
        "rejection_rate_pct": _round(gate.rate_pct, 2),
        "longest_reject_streak": gate.longest_streak,
        "mean_abs_offset_m": _round(mean_offset, 4),
        "departures": departures,
        "interventions": metadata.get("interventions"),
        "n_labeled_frames": 0,
        "road_IoU": "",
        "run_completed": completed,
        "notes": "; ".join(notes + ["M1 not measured: no ground-truth labels"]),
    }


def _round(value: float | None, digits: int):
    if value is None:
        return ""
    return round(value, digits)


def verdicts(rows: list[dict], baseline: str) -> dict:
    by_condition: dict[str, list[dict]] = {}
    for row in rows:
        by_condition.setdefault(row["condition"], []).append(row)

    base_offsets = [
        r["mean_abs_offset_m"]
        for r in by_condition.get(baseline, [])
        if r["mean_abs_offset_m"] != ""
    ]
    base_mean = None
    noise = None
    if base_offsets:
        base_mean = float(np.mean(base_offsets))
    if len(base_offsets) >= 2:
        noise = float(max(base_offsets) - min(base_offsets))

    result = {
        "baseline": baseline,
        "baseline_mean_abs_offset_m": base_mean,
        "baseline_noise_m": noise,
    }
    for condition, runs in by_condition.items():
        m2 = "PASS"
        for run in runs:
            if run["rejection_rate_pct"] == "":
                m2 = "NOT MEASURED"
                break
            if (
                run["rejection_rate_pct"] >= M2_MAX_RATE_PCT
                or run["longest_reject_streak"] >= M2_MAX_STREAK
            ):
                m2 = "FAIL"

        offsets = [r["mean_abs_offset_m"] for r in runs if r["mean_abs_offset_m"] != ""]
        m3 = "NOT MEASURED"
        delta = None
        within_noise = None
        if offsets and base_mean is not None:
            delta = float(np.mean(offsets)) - base_mean
            m3 = "PASS"
            if delta > M3_MAX_WORSE_M:
                m3 = "FAIL"
            if noise is not None:
                within_noise = abs(delta) < noise
        if condition == baseline:
            m3 = "BASELINE"

        m4 = "PASS"
        for run in runs:
            if run["departures"] is None or run["interventions"] is None:
                m4 = "NOT MEASURED"
                break
            if run["departures"] > 0 or run["interventions"] > 0:
                m4 = "FAIL"

        result[condition] = {
            "M1": "NOT MEASURED (no ground-truth labels)",
            "M2": m2,
            "M3": m3,
            "M3_delta_vs_baseline_m": delta,
            "M3_within_baseline_noise": within_noise,
            "M4": m4,
            "runs": [r["run_id"] for r in runs],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("runs", nargs="+", type=Path, help="Run folders written by --eval-log.")
    parser.add_argument(
        "--baseline", default="noon", help="Condition folder name used as the M3 baseline."
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Summary folder for results.csv and verdicts.json."
    )
    args = parser.parse_args()

    rows = [score_run(run) for run in args.runs]
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = verdicts(rows, args.baseline)
    (args.out / "verdicts.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for row in rows:
        print(
            f"{row['condition']}/{row['run_id']}: frames {row['n_frames']}  reject "
            f"{row['rejection_rate_pct']}%  streak {row['longest_reject_streak']}  "
            f"|offset| {row['mean_abs_offset_m']} m  departures {row['departures']}  "
            f"interventions {row['interventions']}  completed {row['run_completed']}"
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
