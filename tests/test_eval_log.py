"""The evaluation log must record what ran and never overwrite or stall."""

import csv
import json

import numpy as np
import pytest

from offroad_autonomy.planning.perception_gate import GateDecision
from offroad_autonomy.runtime.eval_log import COMPLETED, EvalLogger, EvalRun
from offroad_autonomy.types import PathPlan
from scripts import r5_metrics


def _gate(ok=True):
    # An accepted decision can carry a planner's failure, as the grid writes it.
    reason = "no drivable arc in the grid"
    if not ok:
        reason = "road area 1.0%<5%"
    return GateDecision(ok, reason, 0.6, 5000, 0.3, np.zeros((2, 2), bool), 0)


def _plan(status="valid"):
    return PathPlan(centerline=np.empty((0, 2)), status=status)


def _read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_refuses_a_folder_with_earlier_results(tmp_path):
    (tmp_path / "old.txt").write_text("x")
    with pytest.raises(FileExistsError):
        EvalLogger(tmp_path, {})


def test_frames_masks_gate_and_metadata_are_written(tmp_path):
    log = EvalLogger(tmp_path / "run", {"model_file": "m.pt"})
    image = np.full((6, 8, 3), 90, np.uint8)
    mask = np.zeros((4, 5), bool)
    mask[2:] = True
    log.frame(0, 0.0, image, mask, _gate(True), 0.3, 0.6, _plan())
    log.frame(1, 0.0, image, mask, _gate(False), 0.01, 0.2, _plan("held"))
    log.frame(2, 0.0, image, None, _gate(False), 0.0, 0.0, _plan("stale"))
    run = EvalRun(log, route_length_m=100.0, max_duration_s=60.0)
    run.step(10.0, (0.0, 0.0, 0.0))
    run.step(10.1, (3.0, 4.0, 0.0))
    run.finish("timeout")

    rows = _read(tmp_path / "run" / "frames.csv")
    assert [r["gate_evaluated"] for r in rows] == ["1", "1", "0"]
    assert [r["gate_ok"] for r in rows] == ["1", "0", ""]
    assert rows[0]["gate_reason"] == ""
    assert rows[1]["gate_reason"] == "road area 1.0%<5%"
    assert (tmp_path / "run" / "frames" / "000000.png").exists()
    assert (tmp_path / "run" / "masks" / "000001.png").exists()
    assert not (tmp_path / "run" / "masks" / "000002.png").exists()
    pose = _read(tmp_path / "run" / "pose.csv")
    assert float(pose[-1]["travelled_m"]) == pytest.approx(5.0)
    meta = json.loads((tmp_path / "run" / "run_metadata.json").read_text())
    assert meta["model_file"] == "m.pt"
    assert meta["outcome"] == "timeout"
    assert meta["run_completed"] is False
    assert meta["frames_logged"] == 3


def test_run_ends_at_route_length_and_counts_as_completed(tmp_path):
    run = EvalRun(EvalLogger(tmp_path / "run", {}), route_length_m=5.0, max_duration_s=60.0)
    assert run.step(0.0, (0.0, 0.0, 0.0)) is None
    assert run.step(0.1, (3.0, 4.0, 0.0)) == COMPLETED
    run.finish(COMPLETED)
    meta = json.loads((tmp_path / "run" / "run_metadata.json").read_text())
    assert meta["run_completed"] is True


def test_an_intervention_means_the_run_was_not_completed(tmp_path):
    run = EvalRun(EvalLogger(tmp_path / "run", {}), route_length_m=5.0, max_duration_s=60.0)
    run.intervention(0.0, "operator_safe_stop")
    run.finish(COMPLETED)
    meta = json.loads((tmp_path / "run" / "run_metadata.json").read_text())
    assert meta["interventions"] == 1
    assert meta["run_completed"] is False


def test_run_times_out_without_progress(tmp_path):
    run = EvalRun(EvalLogger(tmp_path / "run", {}), route_length_m=5.0, max_duration_s=2.0)
    run.step(0.0, None)
    assert run.step(2.5, None) == "timeout"
    run.finish("timeout")


def test_a_full_writer_queue_drops_images_instead_of_blocking(tmp_path):
    log = EvalLogger(tmp_path / "run", {}, queue_size=1)
    log._queue.put(("blocker", None, None))
    log.frame(0, 0.0, np.zeros((2, 2, 3), np.uint8), None, _gate(), 0.3, 0.6, _plan())
    assert log.dropped_images == 1
    log._queue.get()
    log.close({})


def _logged_run(root, condition, name, offset_m, rejected):
    """A synthetic straight drive at a fixed offset from the road centre."""
    log = EvalLogger(root / condition / name, {"trim_start_s": 1.0, "daylight_pct": 100})
    straight = {
        "r": {
            "edges": [
                {"left": [x, 3, 0], "middle": [x, 0, 0], "right": [x, -3, 0]}
                for x in range(0, 60, 5)
            ]
        }
    }
    log.write_json("road_network.json", straight)
    run = EvalRun(log, route_length_m=40.0, max_duration_s=100.0)
    for i in range(20):
        log.frame(
            i, 0.0, np.zeros((2, 2, 3), np.uint8), None, _gate(i not in rejected), 0.3, 0.6, _plan()
        )
        run.step(float(i) * 0.5, (2.0 * i, offset_m, 0.0))
    run.finish(COMPLETED)
    return root / condition / name


def test_metrics_score_m2_m3_m4_and_compare_to_the_baseline(tmp_path):
    runs = [
        _logged_run(tmp_path, "noon", "run1", 0.20, set()),
        _logged_run(tmp_path, "noon", "run2", 0.30, set()),
        _logged_run(tmp_path, "evening", "run1", -0.50, {3, 4}),
    ]
    rows = [r5_metrics.score_run(run) for run in runs]
    assert rows[0]["mean_abs_offset_m"] == pytest.approx(0.20)
    assert rows[2]["mean_abs_offset_m"] == pytest.approx(0.50)
    assert rows[2]["rejection_rate_pct"] == pytest.approx(10.0)
    assert rows[2]["longest_reject_streak"] == 2
    assert rows[0]["departures"] == 0
    assert rows[0]["run_completed"] == "Y"

    summary = r5_metrics.verdicts(rows, "noon")
    assert summary["baseline_mean_abs_offset_m"] == pytest.approx(0.25)
    assert summary["baseline_noise_m"] == pytest.approx(0.10)
    assert summary["noon"]["M2"] == "PASS"
    assert summary["evening"]["M2"] == "FAIL"
    assert summary["evening"]["M3"] == "FAIL"
    assert summary["evening"]["M3_delta_vs_baseline_m"] == pytest.approx(0.25)
    assert summary["evening"]["M4"] == "PASS"
    assert summary["evening"]["M1"].startswith("NOT MEASURED")
