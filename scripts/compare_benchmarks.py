#!/usr/bin/env python3
"""Tabulate benchmark reports from several runs side by side.

    python scripts/compare_benchmarks.py output/benchmarks/*.json

Each report comes from one ``--benchmark-seconds`` run of
``offroad-autonomy``. Every column is scored by the harness from simulator
ground truth, so runs of different stacks compare fairly; values a run could
not measure (no reference route, say) print as "-".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offroad_autonomy.runtime.benchmark import comparison_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--markdown", action="store_true", help="Emit a Markdown table.")
    args = parser.parse_args()

    reports = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.reports)]
    for line in comparison_table(reports, markdown=args.markdown):
        print(line)


if __name__ == "__main__":
    main()
