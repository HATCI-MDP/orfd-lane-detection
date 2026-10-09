#!/usr/bin/env python3
"""Run several configs back to back on the same map, spawn and duration.

    python scripts/run_experiments.py configs/default.yaml configs/grid.yaml \
        --seconds 60 --map automation_test_track --spawn 0

Each config runs headless as its own ``offroad-autonomy`` process against the
same simulator, so a crash or a stuck stack cannot leak into the next run.
One JSON report per run is written to ``--out-dir`` and a single comparison
table is printed at the end. Configs may name different stacks: the harness,
its safety and its scoring are the same for all of them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offroad_autonomy.runtime.benchmark import comparison_table


def run_command(config: Path, args: argparse.Namespace, report: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "offroad_autonomy.main",
        "--config",
        str(config),
        "--headless",
        "--benchmark-seconds",
        str(args.seconds),
        "--benchmark-warmup",
        str(args.warmup),
        "--benchmark-out",
        str(report),
        "--label",
        config.stem,
        "--spawn",
        str(args.spawn),
    ]
    if args.map:
        command += ["--map", args.map]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--warmup", type=float, default=5.0)
    parser.add_argument("--map", default="", help="Map for every run (default: each config's).")
    parser.add_argument("--spawn", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "output/experiments")
    parser.add_argument("--markdown", action="store_true", help="Print a Markdown table.")
    args = parser.parse_args()

    stems = [config.stem for config in args.configs]
    if len(set(stems)) != len(stems):
        sys.exit("Config file names must be unique: each names its report")

    reports = []
    failed = []
    for config in args.configs:
        report_path = args.out_dir / f"{config.stem}.json"
        print(f"=== {config} ===", flush=True)
        # A report left by an earlier run must not pass for this one.
        report_path.unlink(missing_ok=True)
        result = subprocess.run(run_command(config, args, report_path), check=False)
        if result.returncode != 0 or not report_path.exists():
            failed.append(config.name)
            continue
        reports.append(json.loads(report_path.read_text(encoding="utf-8")))

    if reports:
        print()
        for line in comparison_table(reports, markdown=args.markdown):
            print(line)
    if failed:
        sys.exit(f"Runs that produced no report: {', '.join(failed)}")


if __name__ == "__main__":
    main()
