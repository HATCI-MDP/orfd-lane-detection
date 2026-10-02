"""Record the rendered dashboard to an H.264 mp4 without slowing anything down.

OpenCV's bundled FFmpeg can only write MPEG-4 Part 2 (``mp4v``), which
Chromium-based players such as the VS Code media preview cannot decode. The
frames are therefore piped to an ``ffmpeg`` process that encodes H.264 in
yuv420p, the combination every browser plays. Encoding runs in that separate
process, and the pipe is fed from a dedicated thread through a short queue, so
a slow encoder drops frames from the video instead of stalling the dashboard.
"""

from __future__ import annotations

import logging
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("offroad_autonomy.runtime.video")

X264_PRESETS = (
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
)

_STOP = object()


def default_video_path(label: str, folder: str = "output/videos") -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return Path(folder) / f"{label}_{timestamp}.mp4"


def build_ffmpeg_command(
    ffmpeg: str,
    path: Path,
    width: int,
    height: int,
    fps: float,
    crf: int,
    preset: str,
) -> list[str]:
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:g}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        # Browsers decode H.264 only in 4:2:0; libx264 would otherwise keep
        # 4:4:4 from the BGR input and the file would not play in VS Code.
        "-pix_fmt",
        "yuv420p",
        # Moves the index to the front so a player can start before reading
        # the whole file.
        "-movflags",
        "+faststart",
        str(path),
    ]


class FramePacer:
    """Maps dashboard frames, which arrive at an uneven rate, onto a constant
    frame rate so the video plays back at the speed the run happened."""

    def __init__(self, fps: float) -> None:
        self._fps = float(fps)
        self._t0: float | None = None
        self._emitted = 0

    def repeats(self, timestamp: float) -> int:
        """How many times to write the frame stamped ``timestamp``: 0 when it
        arrives before the next slot, more than 1 to fill a gap."""
        if self._t0 is None:
            self._t0 = timestamp
        slot = int((timestamp - self._t0) * self._fps + 0.5)
        count = slot - self._emitted + 1
        if count <= 0:
            return 0
        self._emitted += count
        return count


class VideoRecorder:
    def __init__(
        self,
        path: str | Path,
        width: int,
        height: int,
        fps: float,
        crf: int = 23,
        preset: str = "veryfast",
        queue_frames: int = 8,
    ) -> None:
        # yuv420p stores colour at half resolution in both directions.
        if width % 2 or height % 2:
            raise ValueError(f"H.264 yuv420p needs even dimensions, got {width}x{height}")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                "--record-video needs ffmpeg with libx264 on PATH "
                "(for example: sudo apt install ffmpeg)"
            )

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._shape = (height, width, 3)
        self._pacer = FramePacer(fps)
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(queue_frames)))
        self.frames_written = 0
        self.frames_dropped = 0
        self.frames_repeated = 0
        self.failed = False

        popen_kwargs: dict = {}
        # Ctrl+C reaches the whole process group; ffmpeg must not stop on it
        # before close() has flushed the last frames and written the index.
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True
        command = build_ffmpeg_command(ffmpeg, self.path, width, height, fps, crf, preset)
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            **popen_kwargs,
        )
        self._thread = threading.Thread(target=self._run, name="video-recorder", daemon=True)
        self._thread.start()

    def write(self, frame: np.ndarray, timestamp: float) -> None:
        """Never blocks: the caller is the dashboard thread, or the control
        loop when the dashboard runs inline."""
        if self.failed:
            return
        if frame.shape != self._shape or frame.dtype != np.uint8:
            raise ValueError(f"Expected a {self._shape} uint8 frame, got {frame.shape}")
        try:
            self._queue.put_nowait((frame, timestamp))
        except queue.Full:
            self.frames_dropped += 1

    def close(self, timeout: float = 10.0) -> None:
        # The queue may be full of frames the encoder has not reached; the
        # stop marker has to wait its turn so those frames still get written.
        # A writer that already died will never make room, so it is not waited on.
        if self._thread.is_alive():
            try:
                self._queue.put(_STOP, timeout=timeout)
            except queue.Full:
                self.failed = True
            self._thread.join(timeout)
        try:
            self._process.stdin.close()
        except OSError:
            pass
        try:
            self._process.wait(timeout)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
            self.failed = True
            logger.error(
                "ffmpeg did not finish in %.0f s; %s may be unplayable", timeout, self.path
            )
        if self._process.returncode != 0 and not self.failed:
            self.failed = True
            logger.error("ffmpeg exited with %s: %s", self._process.returncode, self._stderr())

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            frame, timestamp = item
            count = self._pacer.repeats(timestamp)
            if count == 0:
                continue
            # A memoryview hands ffmpeg the canvas without a 4 MB copy per frame.
            data = memoryview(np.ascontiguousarray(frame)).cast("B")
            try:
                for _ in range(count):
                    self._process.stdin.write(data)
            except (BrokenPipeError, OSError):
                self.failed = True
                logger.error("Video encoder stopped; recording disabled: %s", self._stderr())
                return
            self.frames_written += count
            self.frames_repeated += count - 1

    def _stderr(self) -> str:
        # The broken pipe is seen a moment before ffmpeg exits, and reading a
        # live process's stderr would block until it does.
        try:
            self._process.wait(1.0)
        except subprocess.TimeoutExpired:
            return ""
        if self._process.stderr is None:
            return ""
        try:
            return self._process.stderr.read().decode(errors="replace").strip()
        except (OSError, ValueError):
            return ""
