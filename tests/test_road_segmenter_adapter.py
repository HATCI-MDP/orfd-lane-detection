"""Both model output shapes reduce to the same (instances, confidences) pair.

The regression these guard against: ``RoadSegmenter`` used to assume one
output shape, so a semantic checkpoint either crashed in ultralytics' instance
post-processor or produced an empty mask that the gate then rejected on every
frame. A pipeline that publishes an empty path at full rate looks healthy in a
frame-rate metric, so this failure is quiet and worth testing directly.

No model is loaded here. ``_extract_masks`` is the only place that reads the
model's output shape, so stand-ins for the two result shapes are enough, and
the tests stay fast and runnable without a GPU.
"""

import numpy as np

from offroad_autonomy.perception.road_segmenter import RoadSegmenter

H, W = 40, 60


class _Boxes:
    def __init__(self, confs):
        self.conf = [_Scalar(c) for c in confs]

    def __len__(self):
        return len(self.conf)


class _Scalar:
    """Stands in for a torch scalar: float() goes through .cpu()."""

    def __init__(self, value):
        self._value = value

    def cpu(self):
        return self._value


class _MaskData:
    def __init__(self, arrays):
        self.data = [_Tensor(a) for a in arrays]


class _Tensor:
    def __init__(self, array):
        self._array = np.asarray(array)

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class _InstanceResult:
    """A task="segment" result: masks plus per-detection scores."""

    def __init__(self, masks, confs):
        self.masks = _MaskData(masks)
        self.boxes = _Boxes(confs)
        self.names = {0: "traversable road"}


class _SemanticResult:
    """A task="semantic" result: a class-index map, no masks and no boxes."""

    def __init__(self, indices, names=None):
        self.masks = None
        self.boxes = None
        self.semantic_mask = _Tensor(indices)
        self.names = names if names is not None else {0: "background",
                                                      1: "traversable-roads"}


def _road_band(value=1):
    """A class-index map with road across the bottom quarter."""
    indices = np.zeros((H, W), dtype=np.uint8)
    indices[25:] = value
    return indices


def test_instance_result_is_unchanged():
    """The YOLOE path must behave exactly as before."""
    a = np.zeros((H, W), dtype=np.uint8)
    a[30:] = 1
    b = np.zeros((H, W), dtype=np.uint8)
    b[:5] = 1
    instances, confs = RoadSegmenter._extract_masks(
        _InstanceResult([a, b], [0.9, 0.3]), H, W)
    assert len(instances) == 2
    assert confs == [0.9, 0.3]
    assert instances[0][35, 10] and not instances[0][10, 10]


def test_semantic_result_becomes_one_instance():
    instances, confs = RoadSegmenter._extract_masks(
        _SemanticResult(_road_band()), H, W)
    assert len(instances) == 1
    # 1.0 is this class's existing convention for a detection with no box.
    assert confs == [1.0]
    mask = instances[0]
    assert mask.dtype == np.bool_
    assert mask[30, 10] and not mask[10, 10]


def test_semantic_mask_is_resized_to_the_frame():
    """The class-index map can come back at a different size than the frame.

    Built at twice the frame size with the bottom half road, so the result is
    only correct if it was actually resized rather than returned as-is.
    """
    big = np.zeros((H * 2, W * 2), dtype=np.uint8)
    big[H:] = 1
    instances, _ = RoadSegmenter._extract_masks(_SemanticResult(big), H, W)
    assert instances[0].shape == (H, W)
    # Bottom half road, top half not, at the frame's own resolution.
    assert instances[0][H - 1, 0]
    assert not instances[0][0, 0]
    assert instances[0][H // 2 + 2:].all()
    assert not instances[0][: H // 2 - 2].any()


def test_road_class_is_found_by_name_not_index():
    """A checkpoint whose traversable class is not index 1 still works."""
    indices = np.zeros((H, W), dtype=np.uint8)
    indices[25:] = 3
    result = _SemanticResult(indices, names={0: "background", 1: "sky",
                                             2: "vegetation", 3: "drivable road"})
    instances, _ = RoadSegmenter._extract_masks(result, H, W)
    assert instances[0][30, 10]
    assert not instances[0][10, 10]


def test_unnamed_classes_fall_back_to_non_background():
    """A two-class model that names its classes something unexpected."""
    result = _SemanticResult(_road_band(), names={0: "a", 1: "b"})
    instances, _ = RoadSegmenter._extract_masks(result, H, W)
    assert instances[0][30, 10]
    assert not instances[0][10, 10]


def test_empty_semantic_mask_yields_an_empty_instance():
    """No road found is not the same as no mask: the gate must still see a
    mask it can reject, rather than an empty instance list."""
    instances, confs = RoadSegmenter._extract_masks(
        _SemanticResult(np.zeros((H, W), dtype=np.uint8)), H, W)
    assert len(instances) == 1
    assert not instances[0].any()
    assert confs == [1.0]


def test_missing_result_is_still_handled():
    assert RoadSegmenter._extract_masks(None, H, W) == ([], [])


def test_instance_result_without_masks_is_still_handled():
    class _Empty:
        masks = None
        boxes = None
        names = {}

    assert RoadSegmenter._extract_masks(_Empty(), H, W) == ([], [])
