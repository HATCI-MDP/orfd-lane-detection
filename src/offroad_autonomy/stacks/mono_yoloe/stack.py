"""The mono_yoloe driving stack.

One GMSL2 dashcam supplies every stage: YOLOE-26 segments the drivable
trail, a planner fits a centreline on the ground and Stanley steers. Only the
dashcam's RGB image is used.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from offroad_autonomy.contract import (
    Gauge,
    ImageLayer,
    MaskOverlay,
    Observation,
    Polyline,
    SensorSpec,
    StackOutput,
    Status,
    Visual,
    camera_sensor,
)
from offroad_autonomy.stacks.mono_yoloe.config import load_pipeline_config
from offroad_autonomy.stacks.mono_yoloe.debug_view import pipeline_grid
from offroad_autonomy.stacks.mono_yoloe.pipeline import AutonomyPipeline
from offroad_autonomy.types import CameraFrame, PipelineConfig, PipelineStepResult

logger = logging.getLogger("offroad_autonomy.stacks.mono_yoloe")

MAIN_LAYER = "Dashcam"
RAW_LAYER = "Raw Dashcam"
MASK_LAYER = "Segmentation Mask"
PIPELINE_LAYER = "Pipeline Debug"


class NoRoadMonitor:
    """Asks for a safe stop once no road has been seen for a while.

    It watches the share of *valid* pixels that came back traversable rather
    than a mean detection score, because a score drops as soon as the
    segmenter spends detections on bodywork and would stop the vehicle on a
    perfectly clear trail.
    """

    def __init__(self, min_road_fraction: float, no_road_time_s: float) -> None:
        self._min_road_fraction = float(min_road_fraction)
        self._no_road_time_s = float(no_road_time_s)
        self._no_road_since: float | None = None

    def reset(self) -> None:
        self._no_road_since = None

    def update(self, now: float, road_fraction: float) -> str:
        if road_fraction >= self._min_road_fraction:
            self._no_road_since = None
            return ""
        if self._no_road_since is None:
            self._no_road_since = now
            return ""
        if now - self._no_road_since < self._no_road_time_s:
            return ""
        self.reset()
        return f"no traversable road ({road_fraction:.1%} of valid pixels)"


class MonoYoloeStack:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.pipeline = AutonomyPipeline(config)
        self._camera = config.camera.name
        self._no_road = NoRoadMonitor(config.safety_min_road_fraction, config.safety_no_road_time_s)
        valid_roi = self.pipeline.valid_roi
        # Constant for the run, so built once rather than every frame.
        self._excluded = ~valid_roi
        self._ego_coverage = self.pipeline.ego_coverage
        self._blank = np.zeros(valid_roi.shape + (3,), dtype=np.uint8)
        logger.info(
            "Perception on the '%s' camera; ego-vehicle exclusion %.1f%% of that view",
            self._camera,
            self._ego_coverage * 100.0,
        )

    def sensors(self) -> list[SensorSpec]:
        return [camera_sensor(self.config.camera)]

    def step(self, obs: Observation) -> StackOutput:
        capture = CameraFrame(
            image=obs.sensors.get(self._camera),
            timestamp=obs.timestamp,
            frame_id=obs.frame_id,
        )
        result = self.pipeline.step_result(capture, obs.vehicle)
        return StackOutput(
            command=result.command,
            visuals=self._visuals(result),
            metrics=self._metrics(result),
            safe_stop=self._no_road.update(obs.timestamp, result.stabilized.road_fraction),
        )

    def reset(self) -> None:
        self.pipeline.reset()
        self._no_road.reset()

    # ------------------------------------------------------------- dashboard

    def _visuals(self, result: PipelineStepResult) -> list[Visual]:
        config = self.config
        plan = result.plan
        visuals: list[Visual] = [
            ImageLayer(MAIN_LAYER, result.frame.preprocessed, camera=self._camera),
            MaskOverlay(MAIN_LAYER, result.stabilized.mask, "traversable", "Traversable Mask"),
        ]
        if len(plan.centerline) >= 2:
            style = "path"
            label = "Planned Path"
            if plan.fallback_active:
                style = "held"
                label = "Held Path"
            visuals.append(Polyline(MAIN_LAYER, plan.centerline, "image", style, label))
        if self._ego_coverage > 0.0:
            visuals.append(
                MaskOverlay(MAIN_LAYER, self._excluded, "excluded", "Ego Vehicle (Excluded)")
            )

        visuals += [
            ImageLayer(RAW_LAYER, result.frame.raw, camera=self._camera),
            ImageLayer(MASK_LAYER, self._blank),
            MaskOverlay(MASK_LAYER, result.perception.mask, "info", "Model Mask"),
            MaskOverlay(MASK_LAYER, result.stabilized.mask, "traversable", "Stabilised Mask"),
            ImageLayer(PIPELINE_LAYER, lambda: pipeline_grid(result)),
        ]

        road = result.stabilized.road_fraction
        road_floor = config.safety_min_road_fraction
        confidence = _mean(result.perception.confidences)
        gate = config.gate_min_confidence
        valid_pct = (1.0 - self._ego_coverage) * 100.0
        visuals += [
            Gauge(
                "Road / Valid Px",
                road,
                (0.0, config.gauge_road_full_scale),
                (road_floor,),
                caption=f"Tick: Safe Stop Threshold {road_floor:g}",
                text=f"{road:.3f}",
            ),
            Gauge(
                "Segmentation Confidence",
                confidence,
                (0.0, 1.0),
                (gate, config.gauge_confidence_good),
                caption=f"Tick: Gate Minimum {gate:g}",
                text=f"{confidence:.2f}",
            ),
            Gauge(
                "Valid ROI / Ego",
                1.0 - self._ego_coverage,
                text=f"{valid_pct:.0f}% / {self._ego_coverage * 100:.0f}%",
            ),
        ]
        visuals += self._statuses(result)
        return visuals

    @staticmethod
    def _statuses(result: PipelineStepResult) -> list[Status]:
        plan = result.plan
        statuses = []
        if plan.fallback_active:
            if plan.speed_scale > 0.0:
                statuses.append(Status("GATE HOLD", "warn", plan.fallback_reason))
            else:
                statuses.append(Status("GATE STOP", "bad", plan.fallback_reason))
        else:
            statuses.append(Status("Perceived Path", "good"))
        if plan.kalman_active:
            statuses.append(Status("KALMAN", "warn", "Planner is coasting on its Kalman filter"))
        dbg = result.command.debug
        if dbg is not None and dbg.speed_reason:
            statuses.append(
                Status(
                    "Target Speed", "good", f"{dbg.target_speed_mps:.1f} m/s ({dbg.speed_reason})"
                )
            )
        return statuses

    @staticmethod
    def _metrics(result: PipelineStepResult) -> dict[str, float]:
        confidences = result.perception.confidences
        metrics = {
            "segmentation_confidence": _mean(confidences),
            "max_confidence": float(max(confidences, default=0.0)),
            "detections": float(len(confidences)),
            "road_fraction": float(result.stabilized.road_fraction),
            "stability_score": float(result.stabilized.stability_score),
            "gate_speed_scale": float(result.plan.speed_scale),
        }
        dbg = result.command.debug
        if dbg is not None:
            metrics.update(
                {
                    "cross_track_m": dbg.cross_track_m,
                    "cross_track_rate_mps": dbg.cross_track_rate_mps,
                    "trail_edge_left_m": dbg.left_boundary_m,
                    "trail_edge_right_m": dbg.right_boundary_m,
                    "edge_risk": dbg.edge_risk,
                    "edge_clearance_m": dbg.edge_clearance_m,
                    "lateral_accel_mps2": dbg.lateral_accel_mps2,
                    "heading_error_deg": math.degrees(dbg.heading_error_rad),
                    "curvature_per_m": dbg.curvature,
                    "sharpest_ahead_per_m": dbg.max_curvature_ahead,
                    "steer_desired": dbg.desired_steering,
                    "steer_feedforward": dbg.feedforward_steering,
                    "steer_final": dbg.final_steering,
                    "saturation": dbg.saturation,
                    "target_speed_mps": dbg.target_speed_mps,
                    "lookahead_m": dbg.lookahead_m,
                    "rejoin_m": dbg.rejoin_m,
                }
            )
        for stage, elapsed in result.timings_ms.items():
            metrics[f"{stage}_ms"] = float(elapsed)
        return metrics


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def build(stack_config: dict) -> MonoYoloeStack:
    return MonoYoloeStack(load_pipeline_config(stack_config))
