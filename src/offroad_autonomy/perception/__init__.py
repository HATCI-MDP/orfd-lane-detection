"""Perception sub-package.

``RoadSegmenter`` is deliberately not re-exported here: importing it pulls in
Ultralytics and Torch, and the geometry stages below are useful (and tested)
without that cost. Import it from its own module when you need it.
"""

from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.perception.ego_mask import EgoMask
from offroad_autonomy.perception.perception_view import PerceptionView

__all__ = [
    "CameraModel",
    "EgoMask",
    "PerceptionView",
]
