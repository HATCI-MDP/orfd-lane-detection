"""Writes the rendered dashboard canvas to an mp4 file, frame by frame.

Recording piggybacks on the dashboard's own render cadence rather than
timestamping frames itself: a frame the dashboard drops is simply never
written, exactly as it is simply never shown, so the video is "what the
operator saw," not a wall-clock-accurate capture.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("offroad_autonomy.runtime.video")


class DashboardVideoRecorder:
    def __init__(self, path: str | Path, width: int, height: int, fps: float) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self.path), fourcc, max(1.0, fps), (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {self.path}")
        self.frames_written = 0

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)
        self.frames_written += 1

    def close(self) -> None:
        self._writer.release()
