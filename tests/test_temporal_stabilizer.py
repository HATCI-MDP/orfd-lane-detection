"""Unit tests for mask cleanup in the temporal stabiliser."""

import numpy as np

from offroad_autonomy.postprocessing.temporal_stabilizer import TemporalStabilizer
from offroad_autonomy.types import PerceptionResult, PipelineConfig

SHAPE = (465, 720)


def _stabilizer(**overrides) -> TemporalStabilizer:
    settings = {
        "model_weights": "dummy.pt",
        "enable_morphology": False,
        "enable_ema": False,
        "min_mask_area_fraction": 0.0,
        "min_island_area_m2": 1.0,
    }
    settings.update(overrides)
    return TemporalStabilizer(PipelineConfig(**settings))


def _stabilize(stabilizer: TemporalStabilizer, mask: np.ndarray) -> np.ndarray:
    return stabilizer.stabilize(PerceptionResult(mask=mask)).mask


def _blob(mask: np.ndarray, row: int, col: int, size: int = 6) -> None:
    mask[row : row + size, col : col + size] = True


def test_small_road_island_near_the_vehicle_is_removed():
    mask = np.zeros(SHAPE, dtype=bool)
    _blob(mask, 420, 100)

    cleaned = _stabilize(_stabilizer(), mask)

    assert not cleaned.any()


def test_same_pixel_island_far_away_is_kept():
    """A far pixel covers far more ground, so the same speck far away is a
    real patch of road and must survive."""
    mask = np.zeros(SHAPE, dtype=bool)
    _blob(mask, 225, 100)

    cleaned = _stabilize(_stabilizer(), mask)

    assert cleaned[225:231, 100:106].all()


def test_large_road_region_is_kept_beside_a_removed_island():
    mask = np.zeros(SHAPE, dtype=bool)
    mask[300:, 250:470] = True
    _blob(mask, 420, 100)

    cleaned = _stabilize(_stabilizer(), mask)

    assert cleaned[300:, 250:470].all()
    assert not cleaned[420:426, 100:106].any()


def test_island_above_the_horizon_counts_as_distant_road():
    mask = np.zeros(SHAPE, dtype=bool)
    _blob(mask, 50, 100)

    cleaned = _stabilize(_stabilizer(), mask)

    assert cleaned[50:56, 100:106].all()


def test_island_removal_is_off_at_zero_area():
    mask = np.zeros(SHAPE, dtype=bool)
    _blob(mask, 420, 100)

    cleaned = _stabilize(_stabilizer(min_island_area_m2=0.0), mask)

    assert cleaned[420:426, 100:106].all()


def test_island_removal_follows_the_mask_size():
    """Offline evaluation can run at another working size; the footprint must
    be computed for that grid, not the configured one."""
    mask = np.zeros((620, 960), dtype=bool)
    _blob(mask, 560, 100)

    cleaned = _stabilize(_stabilizer(), mask)

    assert not cleaned.any()


def _with_probability(mask: np.ndarray, probability: np.ndarray) -> PerceptionResult:
    return PerceptionResult(mask=mask, road_probability=probability)


def _hysteresis(**overrides) -> TemporalStabilizer:
    settings = {"min_island_area_m2": 0.0, "hysteresis_high": 0.8}
    settings.update(overrides)
    return _stabilizer(**settings)


def test_road_region_without_a_confident_pixel_is_dropped():
    mask = np.zeros(SHAPE, dtype=bool)
    mask[300:, 100:200] = True
    probability = np.where(mask, 0.6, 0.1).astype(np.float32)

    cleaned = _hysteresis().stabilize(_with_probability(mask, probability)).mask

    assert not cleaned.any()


def test_one_confident_pixel_keeps_its_whole_region():
    mask = np.zeros(SHAPE, dtype=bool)
    mask[300:, 100:200] = True
    probability = np.where(mask, 0.6, 0.1).astype(np.float32)
    probability[400, 150] = 0.95

    cleaned = _hysteresis().stabilize(_with_probability(mask, probability)).mask

    assert cleaned[300:, 100:200].all()


def test_hysteresis_never_adds_road_the_model_rejected():
    mask = np.zeros(SHAPE, dtype=bool)
    mask[300:, 100:200] = True
    probability = np.full(SHAPE, 0.95, dtype=np.float32)

    cleaned = _hysteresis().stabilize(_with_probability(mask, probability)).mask

    np.testing.assert_array_equal(cleaned, mask)


def test_hysteresis_is_off_at_zero():
    mask = np.zeros(SHAPE, dtype=bool)
    mask[300:, 100:200] = True
    probability = np.where(mask, 0.6, 0.1).astype(np.float32)

    cleaned = _hysteresis(hysteresis_high=0.0).stabilize(_with_probability(mask, probability))

    np.testing.assert_array_equal(cleaned.mask, mask)


def test_hysteresis_leaves_the_mask_alone_without_probabilities():
    """YOLOE instance heads have no per-pixel road probability."""
    mask = np.zeros(SHAPE, dtype=bool)
    mask[300:, 100:200] = True

    cleaned = _hysteresis().stabilize(PerceptionResult(mask=mask)).mask

    np.testing.assert_array_equal(cleaned, mask)
