"""Traversable-road segmentation.

Accepts PyTorch weights (open-vocabulary, prompted at load time) or an
exported TensorRT engine, which is what makes the model fast enough on a
Jetson; see ``scripts/export_engine.py``.

Two model output shapes are supported, because a fine-tuned checkpoint does
not necessarily produce the same thing YOLOE does:

``task="segment"``
    One mask per detection in ``result.masks``, each with a score in
    ``result.boxes.conf``. This is YOLOE.

``task="semantic"``
    A single full-resolution class-index map in ``result.semantic_mask``, with
    no detections and **no per-detection score at all**. A model fine-tuned
    for semantic segmentation produces this.

Both are reduced to the same ``(instances, confidences)`` pair before anything
downstream sees them, so the pipeline does not branch on model type.

A semantic model has no confidence to report, so ``_extract_masks`` returns
1.0, which is already this class's convention for a model without boxes. That
makes ``gate_min_confidence`` meaningless for such a model: set it to 0.0 and
let the gate's mask-based checks do the work. ``__init__`` warns if it is left
above zero, because a confidence of 1.0 silently passing a safety check is
worse than failing it.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from offroad_autonomy.perception.ego_mask import (
    apply_roi,
    road_fraction,
    weighted_confidence,
)
from offroad_autonomy.types import FramePacket, PerceptionResult, PipelineConfig

logger = logging.getLogger("offroad_autonomy.perception")


class RoadSegmenter:
    def __init__(self, config: PipelineConfig) -> None:
        weights = config.model_weights
        self._conf = config.confidence_threshold
        self._imgsz = int(config.perception_input_size)

        logger.info("Loading model weights: %s", weights)
        self._model = self._load(weights)
        self._task = getattr(self._model, "task", "segment")
        logger.info("Model task: %s", self._task)

        if self._task == "semantic" and float(config.gate_min_confidence) > 0.0:
            # The gate compares against a number this model cannot produce.
            logger.warning(
                "Model task is 'semantic', which has no per-detection "
                "confidence, but planning.gate.confidence_threshold is %.2f. "
                "Every frame will report a confidence of 1.0 and pass that "
                "check regardless of the mask. Set it to 0.0 and rely on "
                "min_road_pixels and min_mask_area instead.",
                float(config.gate_min_confidence),
            )

        # Exported engines have their classes compiled in; re-prompting them
        # is impossible, and fine-tuned checkpoints have fixed classes too.
        prompts = config.perception_prompts
        if prompts and Path(weights).suffix == ".pt":
            try:
                self._model.set_classes(list(prompts))
                logger.info("Open-vocab classes: %s", prompts)
            except (TypeError, AssertionError, AttributeError):
                logger.info("Model has fixed classes - skipping open-vocab prompts")

    @staticmethod
    def _load(weights: str) -> YOLO:
        """Load the weights without dictating what task they are for.

        ``task="segment"`` is right for YOLOE and wrong for a semantic
        checkpoint: ultralytics then runs the instance post-processor over a
        class-index map and raises ``IndexError: index 1 is out of bounds``.
        A checkpoint carries its own task, so prefer that and keep the explicit
        one as a fallback for weights that carry none.
        """
        model = YOLO(weights)
        if getattr(model, "task", None) in ("segment", "semantic"):
            return model
        logger.info("Weights declare no usable task; loading as 'segment'")
        return YOLO(weights, task="segment")

    def predict(
        self,
        frame: FramePacket,
        valid_roi: np.ndarray | None = None,
    ) -> PerceptionResult:
        """The model sees the whole image because it needs the context, and
        blanking a region out can itself confuse an open-vocabulary segmenter.
        Detections are clipped to ``valid_roi`` afterwards instead, so bodywork
        in frame cannot move the confidence or the traversable fraction.
        """
        img = frame.preprocessed
        h, w = frame.height, frame.width

        t0 = time.perf_counter()
        # Without retina_masks the masks come back at the padded network
        # resolution, and a plain resize shifts them against the frame.
        results = self._model.predict(
            img, conf=self._conf, imgsz=self._imgsz, verbose=False, retina_masks=True
        )
        t_ms = (time.perf_counter() - t0) * 1000.0

        result = None
        if results:
            result = results[0]
        instances, raw_confs = self._extract_masks(result, h, w)

        if valid_roi is not None and valid_roi.shape != (h, w):
            logger.debug("Ignoring ROI of shape %s for a %s frame", valid_roi.shape, (h, w))
            valid_roi = None

        combined = np.zeros((h, w), dtype=bool)
        for instance in instances:
            combined |= instance

        mask = apply_roi(combined, valid_roi)
        confs = weighted_confidence(instances, raw_confs, valid_roi)

        return PerceptionResult(
            mask=mask,
            confidences=confs,
            num_detections=len(confs),
            inference_time_ms=t_ms,
            valid_roi=valid_roi,
            road_fraction=road_fraction(mask, valid_roi),
        )

    @staticmethod
    def _semantic_mask(result) -> np.ndarray | None:
        """Road pixels from a ``task="semantic"`` result, or None if it is not one.

        ``semantic_mask`` is a result wrapper whose ``.data`` is a CUDA tensor,
        so it has to come off the device before numpy will touch it.

        Which class indices count as road is read from the model's own
        ``names``, so a checkpoint whose traversable class is not index 1 still
        works. If none of the names look like road, every non-background index
        is taken, which is the right reading of a two-class model that happens
        to name its classes something else.
        """
        semantic = getattr(result, "semantic_mask", None)
        if semantic is None:
            return None

        data = getattr(semantic, "data", semantic)
        for step in ("detach", "cpu"):
            if hasattr(data, step):
                data = getattr(data, step)()
        if hasattr(data, "numpy"):
            data = data.numpy()
        indices = np.squeeze(np.asarray(data))

        names = getattr(result, "names", None) or {}
        road = [i for i, name in names.items()
                if "road" in str(name).lower() or "travers" in str(name).lower()]
        if road:
            return np.isin(indices, road)
        logger.debug("No road-like class in %s; taking every non-background index",
                     dict(names))
        return indices > 0

    @staticmethod
    def _extract_masks(result, h: int, w: int) -> tuple[list[np.ndarray], list[float]]:
        """Reduce either model output shape to (instances, confidences).

        Per instance rather than merged, so a blob lying entirely on the ego
        body can be dropped from the confidence statistics. A semantic model
        has no instances, so it contributes one.
        """
        instances: list[np.ndarray] = []
        confs: list[float] = []

        if result is None:
            return instances, confs

        semantic = RoadSegmenter._semantic_mask(result)
        if semantic is not None:
            if semantic.shape != (h, w):
                semantic = cv2.resize(semantic.astype(np.uint8), (w, h),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
            # One instance, confidence 1.0: the same value this method already
            # uses for a detection without a box, because a semantic model has
            # no score to report. See the note in the module docstring about
            # gate_min_confidence.
            return [semantic], [1.0]

        if result.masks is None:
            return instances, confs

        for i, m in enumerate(result.masks.data):
            m_np = m.cpu().numpy().astype(np.uint8)
            if m_np.shape != (h, w):
                m_np = cv2.resize(m_np, (w, h), interpolation=cv2.INTER_NEAREST)
            instances.append(m_np.astype(bool))

            conf = 1.0
            if result.boxes is not None and i < len(result.boxes.conf):
                conf = float(result.boxes.conf[i].cpu())
            confs.append(conf)

        return instances, confs
