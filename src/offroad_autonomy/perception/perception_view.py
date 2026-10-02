"""The dashcam image geometry and ego exclusion shared by every stage."""

import numpy as np

from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.perception.ego_mask import EgoMask
from offroad_autonomy.types import CameraFrame, PipelineConfig


class PerceptionView:
    mode = "dashcam"

    def __init__(self, config: PipelineConfig) -> None:
        self.camera = CameraModel(config.camera, config.preprocess_width, config.preprocess_height)
        self.valid_roi = EgoMask(config.camera.ego_mask).valid_roi(
            (self.camera.height, self.camera.width)
        )

    @property
    def size(self) -> tuple[int, int]:
        return self.camera.width, self.camera.height

    @property
    def ego_coverage(self) -> float:
        return float(1.0 - self.valid_roi.mean())

    def image(self, capture: CameraFrame) -> np.ndarray | None:
        return capture.image
