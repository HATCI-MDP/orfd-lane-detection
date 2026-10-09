#!/usr/bin/env python3
"""Offline closed-loop check of the grid planner on synthetic trails.

A kinematic bicycle drives trails drawn on a world raster at the real loop
rate (~4 FPS), with two frames of capture-to-actuation delay and BeamNG's
measured steering slew (~0.9/s), as in ``sbend_sim.py``. Every frame the
dashcam's road mask is rendered through the stack's own camera model and
handed to the real grid planner and ``StanleyController``. Speed and yaw
rate reach the planner through ``OdometryEmulator``, the same emulation the
BeamNG client uses, so the planner sees only what a production vehicle
would. No BeamNG and no segmentation model: this isolates the planner.

    python scripts/grid_sim.py
    python scripts/grid_sim.py --odometry noisy --mask-noise --pitch-deg 0.5 --seeds 3

``--odometry noisy`` adds 5% wheel slip, a 0.5 deg/s gyro bias and white
noise on both signals. ``--mask-noise`` jitters the trail edge, punches holes,
adds off-trail false positives and fails 5% of frames on confidence.
``--pitch-deg`` pitches the camera on its suspension while the planner
assumes the static mount, which is what misplaces far ground on a real car.

The first six trails chose the ``planning.grid`` settings; the ``held_out_``
ones never did. The body counts as off the trail when any point of a
1.9 m x 4.4 m footprint (0.9 m overhangs, assumed) leaves it, so a margin
below zero is a departure. Every number is from idealised synthetic masks:
it checks the planner's geometry and logic, not real perception.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offroad_autonomy.control import stanley_controller
from offroad_autonomy.control.stanley_controller import StanleyController
from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.perception.perception_view import PerceptionView
from offroad_autonomy.planning.centerline_planner import CenterlinePlanner
from offroad_autonomy.simulation.beamng_client import OdometryEmulator
from offroad_autonomy.stacks.mono_yoloe.config import load_pipeline_config_file
from offroad_autonomy.types import (
    PerceptionResult,
    PipelineConfig,
    StabilizedResult,
    VehicleSignals,
    mount_pose,
)

_RASTER_M = 0.05
_REAR_OVERHANG_M = 0.9
_FRONT_OVERHANG_M = 0.9
_SECONDS = 150.0
_SUBSTEPS = 20
_WARMUP_FRAMES = 20
#: The vehicle's reference origin, which BeamNG reports, sits this far behind the camera.
_ORIGIN_BEHIND_CAMERA_M = 0.30


def _trail(segments, start=(0.0, 0.0), heading=0.0, step=0.05) -> np.ndarray:
    """Centreline from ``(length_m, curvature)`` pieces; positive curvature turns left."""
    points = [np.array(start, dtype=float)]
    position = np.array(start, dtype=float)
    for length, curvature in segments:
        for _ in range(int(length / step)):
            heading += curvature * step
            position = position + step * np.array([math.cos(heading), math.sin(heading)])
            points.append(position.copy())
    return np.array(points)


def _fork(trunk_m: float, branches) -> list[np.ndarray]:
    trunk = _trail([(trunk_m, 0.0)])
    end = tuple(trunk[-1])
    return [np.vstack([trunk, _trail(segments, start=end)]) for segments in branches]


def scenarios() -> dict[str, tuple[list[np.ndarray], float]]:
    """``name -> (centrelines, trail width)``."""
    quarter = 0.5 * math.pi
    narrow = [(20, 0), (0.6 * math.pi * 12, -1 / 12), (15, 0), (quarter * 12, 1 / 12), (40, 0)]
    return {
        "sbend_r9_w6": (
            [_trail([(30, 0), (quarter * 9, 1 / 9), (quarter * 9, -1 / 9), (40, 0)])],
            6.0,
        ),
        "sbend_r6_w4.5": (
            [_trail([(25, 0), (quarter * 6, 1 / 6), (quarter * 6, -1 / 6), (40, 0)])],
            4.5,
        ),
        "narrow_w3_r12": (
            [_trail(narrow)],
            3.0,
        ),
        "wide_w14_r30": ([_trail([(20, 0), (quarter * 30, 1 / 30), (40, 0)])], 14.0),
        "hairpin_r5.5_w5": ([_trail([(20, 0), (0.75 * math.pi * 5.5, -1 / 5.5), (40, 0)])], 5.0),
        "fork_w4": (_fork(25, [[(10, 1 / 25), (40, 0)], [(10, -1 / 25), (40, 0)]]), 4.0),
        "held_out_sbend_r15_w5": (
            [_trail([(20, 0), (quarter * 15, -1 / 15), (quarter * 15, 1 / 15), (30, 0)])],
            5.0,
        ),
        "held_out_chicane_r8_w4": (
            [_trail([(20, 0)] + [(6, sign / 8) for sign in (1, -1, -1, 1, 1, -1)] + [(30, 0)])],
            4.0,
        ),
        "held_out_tight_r7_w3.5": (
            [_trail([(15, 0), (quarter * 7, 1 / 7), (10, 0), (quarter * 7, -1 / 7), (30, 0)])],
            3.5,
        ),
        "held_out_wide_w10_r50": ([_trail([(15, 0), (40, -1 / 50), (40, 1 / 50), (20, 0)])], 10.0),
        "held_out_hairpin_r6_w6": ([_trail([(20, 0), (math.pi * 6, 1 / 6), (40, 0)])], 6.0),
        "held_out_junction_w4.5": (_fork(20, [[(45, 0)], [(12, -1 / 15), (30, 0)]]), 4.5),
    }


class World:
    """Distance from any point to the nearest centreline, from a raster."""

    def __init__(self, centrelines: list[np.ndarray], width: float) -> None:
        points = np.vstack(centrelines)
        self.origin = points.min(axis=0) - 25.0
        extent = points.max(axis=0) + 25.0 - self.origin
        self.shape = (int(extent[1] / _RASTER_M) + 1, int(extent[0] / _RASTER_M) + 1)
        lines = np.full(self.shape, 255, np.uint8)
        for line in centrelines:
            cells = np.round((line - self.origin) / _RASTER_M).astype(np.int32)
            cv2.polylines(lines, [cells.reshape(-1, 1, 2)], False, 0, 1)
        self.distance = cv2.distanceTransform(lines, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        self.distance *= _RASTER_M
        self.half_width = 0.5 * width
        self.ends = [line[-1] for line in centrelines]

    def distance_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        col = np.clip(((x - self.origin[0]) / _RASTER_M).astype(int), 0, self.shape[1] - 1)
        row = np.clip(((y - self.origin[1]) / _RASTER_M).astype(int), 0, self.shape[0] - 1)
        return self.distance[row, col]


class GroundRays:
    """Where each planner-ROI pixel meets flat ground, per camera pitch offset."""

    def __init__(self, config: PipelineConfig, camera: CameraModel, pixels: np.ndarray) -> None:
        self._config = config
        self._camera = camera
        self._pixels = pixels
        self._cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def at_pitch(self, offset_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Cached per 0.1 deg: projecting every pixel again each frame is slow.
        key = int(round(offset_deg * 10.0))
        if key not in self._cache:
            spec = self._config.camera
            direction, up = mount_pose(-8.0 + key / 10.0)
            pitched = CameraModel(
                replace(spec, dir=direction, up=up), self._camera.width, self._camera.height
            )
            forward, right, keep = pitched.image_to_ground(self._pixels)
            self._cache[key] = (np.where(keep, forward, 0.0), np.where(keep, right, 0.0), keep)
        return self._cache[key]


@dataclass
class RunResult:
    name: str
    seed: int
    finished: bool
    departure_frames: int
    min_margin_m: float
    mean_offset_m: float
    p95_offset_m: float
    stop_frames: int
    seconds: float


def _noisy_mask(mask: np.ndarray, valid: np.ndarray, roi_top: int, rng) -> np.ndarray:
    height, width = mask.shape
    # Segmentation misses: thin holes inside the trail.
    for _ in range(40):
        row = int(rng.integers(roi_top, height))
        col = int(rng.integers(0, width))
        mask[row : row + 3, col : col + int(rng.integers(3, 14))] = False
    # Grass that looks like trail, off to the side.
    canvas = mask.view(np.uint8)
    for _ in range(int(rng.integers(0, 3))):
        centre = (int(rng.integers(0, width)), int(rng.integers(roi_top, height)))
        axes = (int(rng.integers(5, 30)), int(rng.integers(3, 10)))
        cv2.ellipse(canvas, centre, axes, 0, 0, 360, 1, -1)
    return mask & valid


def run(name: str, config: PipelineConfig, args: argparse.Namespace, seed: int) -> RunResult:
    centrelines, width = scenarios()[name]
    world = World(centrelines, width)
    rng = np.random.default_rng(seed)
    view = PerceptionView(config)
    camera = view.camera
    valid = view.valid_roi
    roi_top = int(round(camera.height * (1.0 - config.planner_roi_height)))
    u, v = camera.pixel_grid()
    used = np.flatnonzero((valid & (v >= roi_top)).ravel())
    rays = GroundRays(config, camera, np.stack([u.ravel()[used], v.ravel()[used]], axis=1))

    planner = CenterlinePlanner(config, camera=camera)
    controller = StanleyController(config, camera=camera)
    odometry = OdometryEmulator()
    wheelbase = config.wheelbase_m
    camera_ahead = config.camera_ahead_of_rear_axle_m
    max_wheel = math.radians(config.max_wheel_angle_deg)
    dt = 1.0 / args.fps

    # Rear axle 2 m along the first trail, facing along it.
    first = centrelines[0]
    heading = math.atan2(first[5][1] - first[0][1], first[5][0] - first[0][0])
    x = first[0][0] + 2.0 * math.cos(heading)
    y = first[0][1] + 2.0 * math.sin(heading)
    speed = 0.0
    wheel_command = 0.0
    pending = [0.0, 0.0]
    pitch_deg = 0.0
    gyro_bias = math.radians(0.5) * float(rng.choice([-1.0, 1.0]))
    now = 0.0
    offsets: list[float] = []
    departures = 0
    stops = 0
    min_margin = math.inf
    finished = False
    clock = SimpleNamespace(perf_counter=lambda: now)
    # The controller times its cross-track rate on the wall clock; replaying
    # faster than real time would inflate that rate.
    with mock.patch.object(stanley_controller, "time", clock):
        for _ in range(int(_SECONDS * args.fps)):
            cos_h, sin_h = math.cos(heading), math.sin(heading)
            camera_x = x + camera_ahead * cos_h
            camera_y = y + camera_ahead * sin_h
            origin = (
                camera_x - _ORIGIN_BEHIND_CAMERA_M * cos_h,
                camera_y - _ORIGIN_BEHIND_CAMERA_M * sin_h,
                0.0,
            )
            velocity = (speed * cos_h, speed * sin_h, 0.0)
            signals = odometry.read(origin, velocity, (cos_h, sin_h, 0.0), now)
            if args.odometry == "noisy":
                signals = VehicleSignals(
                    speed_mps=signals.speed_mps * 1.05 + rng.normal(0.0, 0.05),
                    yaw_rate_rps=signals.yaw_rate_rps + gyro_bias + rng.normal(0.0, 0.005),
                    timestamp_s=signals.timestamp_s,
                )

            if args.pitch_deg > 0.0:
                # Suspension pitch is slow, so it is correlated frame to frame.
                pitch_deg = 0.7 * pitch_deg + rng.normal(0.0, args.pitch_deg * math.sqrt(0.51))
            forward, right, hit = rays.at_pitch(pitch_deg)
            ground_x = camera_x + forward * cos_h + right * sin_h
            ground_y = camera_y + forward * sin_h - right * cos_h
            half_width = world.half_width
            confidence = 0.6
            if args.mask_noise:
                half_width += rng.normal(0.0, 0.15)
                if rng.random() < 0.05:
                    confidence = 0.1
            road = hit & (world.distance_at(ground_x, ground_y) < half_width)
            mask = np.zeros(camera.height * camera.width, dtype=bool)
            mask[used] = road
            mask = mask.reshape(camera.height, camera.width)
            if args.mask_noise:
                mask = _noisy_mask(mask, valid, roi_top, rng)
            stabilized = StabilizedResult(
                mask=mask,
                valid_roi=valid,
                raw_result=PerceptionResult(mask=mask, confidences=[confidence]),
            )
            plan = planner.plan(stabilized, signals)
            command = controller.compute(plan, signals)
            if plan.speed_scale <= 0.0:
                stops += 1

            longitudinal, lateral = np.meshgrid(
                np.linspace(-_REAR_OVERHANG_M, wheelbase + _FRONT_OVERHANG_M, 8),
                np.linspace(-config.vehicle_half_width_m, config.vehicle_half_width_m, 5),
            )
            body_x = x + longitudinal * cos_h + lateral * sin_h
            body_y = y + longitudinal * sin_h - lateral * cos_h
            margin = world.half_width - float(world.distance_at(body_x, body_y).max())
            min_margin = min(min_margin, margin)
            if margin < 0.0:
                departures += 1
            middle = (
                np.array([x + 0.5 * wheelbase * cos_h]),
                np.array([y + 0.5 * wheelbase * sin_h]),
            )
            offsets.append(float(world.distance_at(*middle)[0]))
            if any(math.hypot(x - end[0], y - end[1]) < 6.0 for end in world.ends):
                finished = True
                break

            # Bicycle at the rear axle, + steering = right, with the previous
            # commands (latency) and BeamNG's steering slew.
            h = dt / _SUBSTEPS
            for _ in range(_SUBSTEPS):
                step = 0.9 * h
                wheel_command += float(np.clip(pending[0] - wheel_command, -step, step))
                heading -= speed * math.tan(wheel_command * max_wheel) / wheelbase * h
                x += speed * math.cos(heading) * h
                y += speed * math.sin(heading) * h
                # Hopper in BeamNG: 0 -> 5.7 m/s in ~1.4 s at 0.45 throttle.
                speed = max(0.0, speed + (9.0 * command.throttle - 8.0 * command.brake - 0.3) * h)
            pending = pending[1:] + [command.steering]
            now += dt

    settled = np.array(offsets[_WARMUP_FRAMES:] or [0.0])
    return RunResult(
        name=name,
        seed=seed,
        finished=finished,
        departure_frames=departures,
        min_margin_m=min_margin,
        mean_offset_m=float(settled.mean()),
        p95_offset_m=float(np.percentile(settled, 95)),
        stop_frames=stops,
        seconds=len(offsets) / args.fps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(ROOT / "configs/grid.yaml"))
    parser.add_argument("--odometry", choices=("clean", "noisy"), default="clean")
    parser.add_argument("--mask-noise", action="store_true")
    parser.add_argument("--pitch-deg", type=float, default=0.0, help="suspension pitch std")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--trail", nargs="*", choices=sorted(scenarios()), default=None)
    args = parser.parse_args()

    config = load_pipeline_config_file(args.config)
    if config.planner_mode != "grid":
        sys.exit(f"{args.config} does not select planning.mode: grid")
    results = []
    for name in args.trail or list(scenarios()):
        for seed in range(args.seeds):
            result = run(name, config, args, seed)
            results.append(result)
            print(
                f"{name:24s} seed {seed}: finished={result.finished!s:5s} "
                f"departure frames {result.departure_frames:3d}  "
                f"min margin {result.min_margin_m:+.2f} m  "
                f"offset mean {result.mean_offset_m:.2f} p95 {result.p95_offset_m:.2f} m  "
                f"stop frames {result.stop_frames:3d}  {result.seconds:.0f} s",
                flush=True,
            )
    print(
        f"\n{len(results)} runs: {sum(not r.finished for r in results)} unfinished, "
        f"{sum(r.departure_frames for r in results)} departure frames, "
        f"min margin {min(r.min_margin_m for r in results):+.2f} m, "
        f"mean offset {np.mean([r.mean_offset_m for r in results]):.2f} m"
    )
    if any(not r.finished or r.departure_frames for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
