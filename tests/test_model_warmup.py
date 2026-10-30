"""The first live frame must not pay the model's cold-start cost.

A cold first inference took 1.6 s on a laptop RTX GPU, which stretched the gap
between the first two wheel/gyro samples past the odometry limit, so the car
safe-stopped before it moved.
"""

from unittest.mock import Mock, patch

import numpy as np

from offroad_autonomy.perception.road_segmenter import RoadSegmenter
from offroad_autonomy.pipeline import RGBPlanningPipeline
from offroad_autonomy.types import PipelineConfig


def test_pipeline_warms_the_model_on_its_working_grid():
    with patch("offroad_autonomy.pipeline.RoadSegmenter") as segmenter:
        pipeline = RGBPlanningPipeline(PipelineConfig())
    segmenter.return_value.warmup.assert_called_once_with(pipeline.view.size)
    segmenter.return_value.predict.assert_not_called()


def test_warmup_runs_the_live_inference_call_on_a_blank_frame():
    model = Mock(task="segment")
    config = PipelineConfig(confidence_threshold=0.15, perception_input_size=1024)
    config.perception_prompts = []
    with patch("offroad_autonomy.perception.road_segmenter.YOLO", return_value=model):
        segmenter = RoadSegmenter(config)
    segmenter.warmup((720, 465))

    image = model.predict.call_args.args[0]
    assert image.shape == (465, 720, 3)
    assert image.dtype == np.uint8
    kwargs = model.predict.call_args.kwargs
    assert kwargs["imgsz"] == 1024
    assert kwargs["conf"] == 0.15
    assert kwargs["retina_masks"] is True
