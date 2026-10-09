"""The mono_yoloe stages in order: preprocess, segment, stabilise, plan and control."""

from __future__ import annotations

import logging
import time

from offroad_autonomy.control.stanley_controller import StanleyController
from offroad_autonomy.perception.perception_view import PerceptionView
from offroad_autonomy.perception.road_segmenter import RoadSegmenter
from offroad_autonomy.planning.centerline_planner import CenterlinePlanner
from offroad_autonomy.postprocessing.temporal_stabilizer import TemporalStabilizer
from offroad_autonomy.preprocessing.image_preprocessor import ImagePreprocessor
from offroad_autonomy.types import CameraFrame, PipelineConfig, PipelineStepResult, VehicleSignals

logger = logging.getLogger("offroad_autonomy.stacks.mono_yoloe")


class AutonomyPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        logger.info("Initialising pipeline stages")
        self.view = PerceptionView(config)
        self.valid_roi = self.view.valid_roi
        self.preprocessor = ImagePreprocessor(config, target_size=self.view.size)
        self.segmenter = RoadSegmenter(config)
        self.stabilizer = TemporalStabilizer(config)
        self.planner = CenterlinePlanner(config, camera=self.view.camera)
        self.controller = StanleyController(config, camera=self.view.camera)

    @property
    def ego_coverage(self) -> float:
        return self.view.ego_coverage

    def step_result(self, capture: CameraFrame, signals: VehicleSignals) -> PipelineStepResult:
        if capture.image is None:
            raise ValueError("No frame available to segment")
        timings: dict[str, float] = {}
        t1 = time.perf_counter()

        frame = self.preprocessor.process(capture.image)
        t2 = time.perf_counter()
        timings["preprocess"] = (t2 - t1) * 1000.0

        perception = self.segmenter.predict(frame, self.valid_roi)
        t3 = time.perf_counter()
        timings["segmentation"] = (t3 - t2) * 1000.0

        stabilized = self.stabilizer.stabilize(perception)
        t4 = time.perf_counter()
        timings["postprocess"] = (t4 - t3) * 1000.0

        plan = self.planner.plan(stabilized, signals=signals)
        t5 = time.perf_counter()
        timings["planning"] = (t5 - t4) * 1000.0

        command = self.controller.compute(plan, signals)
        timings["control"] = (time.perf_counter() - t5) * 1000.0

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "step %d: seg=%.0fms road=%.1f%% stability=%.3f steer=%.3f",
                capture.frame_id,
                timings["segmentation"],
                perception.road_fraction * 100.0,
                stabilized.stability_score,
                command.steering,
            )

        return PipelineStepResult(
            frame=frame,
            perception=perception,
            stabilized=stabilized,
            plan=plan,
            command=command,
            capture=capture,
            timings_ms=timings,
        )

    def reset(self) -> None:
        self.stabilizer.reset()
        self.planner.reset()
        self.controller.reset()
