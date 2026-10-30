"""A YOLOE checkpoint must take the road prompt whatever its file is called."""

from unittest.mock import Mock, patch

from ultralytics.nn.tasks import DetectionModel, YOLOEModel

from offroad_autonomy.perception.road_segmenter import RoadSegmenter
from offroad_autonomy.types import PipelineConfig


class _FakeYOLOE:
    def __init__(self, weights, task):
        self.weights = weights
        self.task = task
        self.model = Mock(spec=YOLOEModel)
        self.set_classes = Mock()


def _load(network):
    plain = Mock(task="segment")
    plain.model = network
    config = PipelineConfig(model_weights="models/weights.pt")
    config.perception_prompts = ["traversable road"]
    with (
        patch("offroad_autonomy.perception.road_segmenter.YOLO", return_value=plain),
        patch("offroad_autonomy.perception.road_segmenter.YOLOE", _FakeYOLOE),
    ):
        return RoadSegmenter(config), plain


def test_renamed_yoloe_checkpoint_is_reloaded_so_the_prompt_applies():
    segmenter, plain = _load(Mock(spec=YOLOEModel))
    assert isinstance(segmenter._model, _FakeYOLOE)
    assert segmenter._model.weights == "models/weights.pt"
    segmenter._model.set_classes.assert_called_once_with(["traversable road"])
    plain.set_classes.assert_not_called()


def test_non_yoloe_checkpoint_keeps_the_plain_loader():
    segmenter, plain = _load(Mock(spec=DetectionModel))
    assert segmenter._model is plain
