"""Reproducible offline A/B comparison using the existing delayed S-bend plant."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from sbend_sim import ROOT, simulate

from offroad_autonomy.utils.config import load_config


def metrics(result, fps):
    rows = result.log
    steer = np.array([r["steer"] for r in rows])
    significant = steer[np.abs(steer) > 0.03]
    heading = np.array([r["heading_truth"] for r in rows])
    return {
        "completed": result.completed,
        "frames": len(rows),
        "mean_cte_m": float(np.mean([r["err"] for r in rows])),
        "max_cte_m": result.max_lateral_error,
        "mean_heading_error_rad": float(
            np.mean(np.abs(np.arctan2(np.sin(heading), np.cos(heading))))
        ),
        "steering_sign_changes": int(np.sum(np.diff(np.sign(significant)) != 0)),
        "mean_steering_rate_per_s": float(np.mean(np.abs(np.diff(steer))) * fps),
        "average_speed_mps": float(np.mean([r["v"] for r in rows])),
        "mean_control_ms": float(np.mean([r["control_ms"] for r in rows])),
        "p95_control_ms": float(np.percentile([r["control_ms"] for r in rows], 95)),
        "mean_solve_ms": float(np.mean([r["solve_ms"] for r in rows])),
        "fallback_frames": sum(r["fallback"] for r in rows),
        "solver_success_frames": sum(r["solver_success"] for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/mpc.yaml"))
    parser.add_argument("--radii", type=float, nargs="+", default=[10.0, 7.0, 5.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--out", default="output/diagnostics/controller_comparison.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    results = []
    for radius in args.radii:
        for seed in args.seeds:
            for name in ("stanley", "mpc"):
                result = simulate(
                    replace(cfg, controller=name),
                    radius=radius,
                    seed=seed,
                    fps=args.fps,
                    completion_margin_m=15.0,
                )
                row = dict(controller=name, radius_m=radius, seed=seed, **metrics(result, args.fps))
                results.append(row)
                print(json.dumps(row), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "kind": "synthetic bicycle; no perception or BeamNG",
                "fps": args.fps,
                "completion_margin_m": 15.0,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
