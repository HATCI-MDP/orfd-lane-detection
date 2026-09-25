"""Tests for the mp4 dashboard recorder."""

import numpy as np

from offroad_autonomy.runtime.video_recorder import DashboardVideoRecorder


def test_writes_a_playable_file_and_creates_missing_directories(tmp_path):
    path = tmp_path / "videos" / "run.mp4"
    recorder = DashboardVideoRecorder(path, width=32, height=24, fps=10.0)

    for _ in range(5):
        recorder.write(np.zeros((24, 32, 3), dtype=np.uint8))
    recorder.close()

    assert path.exists()
    assert path.stat().st_size > 0
    assert recorder.frames_written == 5
