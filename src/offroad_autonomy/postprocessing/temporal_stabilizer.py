"""Temporal mask stabilisation.

A raw per-frame mask flickers at its edges, and the planner would turn that
flicker into steering jitter.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.perception.ego_mask import road_fraction
from offroad_autonomy.types import PerceptionResult, PipelineConfig, StabilizedResult

logger = logging.getLogger("offroad_autonomy.postprocessing")


class TemporalStabilizer:
    def __init__(self, config: PipelineConfig) -> None:
        self._alpha = config.ema_alpha
        self._min_area_frac = config.min_mask_area_fraction
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.morphology_kernel_size, config.morphology_kernel_size),
        )
        # Separate switches so each stage can be verified on its own against
        # the raw mask.
        self._use_morphology = bool(config.enable_morphology)
        self._use_ema = bool(config.enable_ema)
        self._min_island_m2 = float(config.min_island_area_m2)
        self._hysteresis_high = float(config.hysteresis_high)
        self._camera_spec = config.camera
        self._max_range_m = float(config.projection_max_range_m)
        self._pixel_area: np.ndarray | None = None
        self._accum: np.ndarray | None = None
        self._prev_mask: np.ndarray | None = None

    def stabilize(self, result: PerceptionResult) -> StabilizedResult:
        smoothed = result.mask.astype(np.uint8)
        if self._use_ema:
            mask_f = result.mask.astype(np.float32)
            if self._accum is None or self._accum.shape != mask_f.shape:
                self._accum = mask_f
            else:
                # With alpha > 0.5 a single empty frame drops every pixel below
                # 0.5, so EMA does not bridge a one-frame dropout; the perception
                # gate's path hold does that instead.
                self._accum *= 1.0 - self._alpha
                self._accum += self._alpha * mask_f
            smoothed = (self._accum > 0.5).astype(np.uint8)

        if self._use_morphology:
            smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, self._kernel)
            smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, self._kernel)

        if self._hysteresis_high > 0.0 and result.road_probability is not None:
            smoothed = self._hysteresis(smoothed, result.road_probability)

        if self._min_island_m2 > 0.0:
            smoothed = self._remove_islands(smoothed)

        h, w = smoothed.shape
        min_area = int(h * w * self._min_area_frac)
        if int(smoothed.sum()) < min_area:
            smoothed = np.zeros_like(smoothed)

        stable_mask = smoothed.astype(bool)

        # Morphology can push the mask a few pixels outward, so re-apply the
        # ego exclusion afterwards: closing a gap must never grow the road
        # onto the vehicle's own bodywork.
        roi = result.valid_roi
        if roi is not None and roi.shape == stable_mask.shape:
            stable_mask &= roi

        stability = self._temporal_iou(stable_mask)
        self._prev_mask = stable_mask

        return StabilizedResult(
            mask=stable_mask,
            stability_score=stability,
            raw_result=result,
            valid_roi=roi,
            road_fraction=road_fraction(stable_mask, roi),
        )

    def _hysteresis(self, mask: np.ndarray, road_probability: np.ndarray) -> np.ndarray:
        """The model's own mask is the weak set, so this can only take road
        away: a region survives when part of it is confidently road, and a
        region the model was never sure of anywhere is dropped whole."""
        if road_probability.shape != mask.shape:
            raise ValueError("Road probability is not aligned with the mask")
        count, labels = cv2.connectedComponents(mask, connectivity=8)
        if count <= 1:
            return mask
        strong = (road_probability >= self._hysteresis_high) & mask.astype(bool)
        keep = np.zeros(count, dtype=np.uint8)
        keep[labels[strong]] = 1
        keep[0] = 0
        return np.take(keep, labels)

    def _remove_islands(self, mask: np.ndarray) -> np.ndarray:
        """A speck of false road would otherwise reach every planner as a
        destination, and the grid would remember it for seconds. Islands are
        judged by ground area, not pixel count, because a far pixel covers
        hundreds of times more ground than a near one: a pixel threshold would
        delete distant real road first."""
        count, labels = cv2.connectedComponents(mask, connectivity=8)
        if count <= 1:
            return mask
        area = self._pixel_area_for(mask.shape)
        # Summing over road pixels only halves the cost on a mostly empty frame.
        road = mask.astype(bool)
        island_m2 = np.bincount(labels[road], weights=area[road], minlength=count)
        keep = (island_m2 >= self._min_island_m2).astype(np.uint8)
        keep[0] = 0
        return np.take(keep, labels)

    def _pixel_area_for(self, shape: tuple[int, int]) -> np.ndarray:
        if self._pixel_area is None or self._pixel_area.shape != shape:
            camera = CameraModel(self._camera_spec, shape[1], shape[0])
            self._pixel_area = camera.ground_area_px(self._max_range_m)
        return self._pixel_area

    def _temporal_iou(self, mask: np.ndarray) -> float:
        if self._prev_mask is None:
            return 1.0
        if self._prev_mask.shape != mask.shape:
            return 1.0
        inter = int((mask & self._prev_mask).sum())
        union = int((mask | self._prev_mask).sum())
        if union <= 0:
            return 1.0
        return inter / union

    def reset(self) -> None:
        self._accum = None
        self._prev_mask = None
