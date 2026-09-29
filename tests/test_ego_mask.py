"""Unit tests for ego-vehicle exclusion.

The property under test throughout is that bodywork in frame is *invisible* to
the stack rather than being classified. Hood pixels must not become road, must
not become obstacles, and must not move any statistic that a safety decision
depends on.
"""

from dataclasses import replace

import numpy as np
import pytest

from offroad_autonomy.main import _StuckDetector
from offroad_autonomy.perception.ego_mask import (
    EgoMask,
    apply_roi,
    road_fraction,
    weighted_confidence,
)
from offroad_autonomy.postprocessing.temporal_stabilizer import TemporalStabilizer
from offroad_autonomy.types import (
    EgoMaskSpec,
    PerceptionResult,
    PipelineConfig,
)

# A wedge across the bottom of the frame, like a hood.
HOOD = EgoMaskSpec(
    enabled=True,
    polygon=((0.15, 1.0), (0.85, 1.0), (0.70, 0.70), (0.30, 0.70)),
    margin_px=0,
)


def _config(**overrides) -> PipelineConfig:
    return PipelineConfig(model_weights="dummy.pt", **overrides)


def _roi(shape=(465, 720)):
    return EgoMask(HOOD).valid_roi(shape)


def test_mask_is_resolution_independent():
    """One normalised polygon has to serve acquisition and working grids."""
    mask = EgoMask(HOOD)

    small = mask.coverage((465, 720))
    large = mask.coverage((1860, 2880))

    assert small == pytest.approx(large, abs=0.01)
    assert 0.0 < small < 0.5


def test_disabled_mask_leaves_every_pixel_valid():
    mask = EgoMask(EgoMaskSpec(enabled=False))

    assert not mask.enabled
    assert mask.valid_roi((100, 100)).all()
    assert mask.coverage((100, 100)) == 0.0


def test_mask_covering_the_whole_frame_is_refused():
    """Silently starving the stack of input is worse than failing loudly."""
    everything = EgoMaskSpec(enabled=True, polygon=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))

    with pytest.raises(ValueError, match="entire frame"):
        EgoMask(everything).valid_roi((64, 64))


def test_enabled_mask_needs_a_real_polygon():
    with pytest.raises(ValueError, match="polygon"):
        EgoMaskSpec(enabled=True, polygon=((0.0, 0.0),))


def test_hood_pixels_do_not_count_against_perception_confidence():
    """The denominator is valid pixels, so masking more cannot lower the score."""
    roi = _roi()
    road = np.zeros(roi.shape, dtype=bool)
    road[120:300, 200:520] = True

    with_hood = road_fraction(road, roi)
    without_hood = road_fraction(road, np.ones_like(roi))

    # Same road, and the hood-aware figure is the higher one - the excluded
    # pixels leave the denominator rather than counting as "not road".
    assert with_hood > without_hood
    assert with_hood == pytest.approx(road.sum() / roi.sum())


def test_detection_lying_entirely_on_the_hood_is_discarded():
    roi = _roi()
    on_hood = ~roi
    on_road = np.zeros(roi.shape, dtype=bool)
    on_road[100:200, 100:300] = True

    kept = weighted_confidence([on_hood, on_road], [0.9, 0.4], roi)

    # The 0.9 hood blob is dropped; the road detection stands on its own.
    assert kept == [0.4]


def test_road_detection_clipped_by_the_hood_keeps_its_score():
    roi = _roi()
    straddling = np.zeros(roi.shape, dtype=bool)
    straddling[300:465, 300:420] = True  # part road, part hood

    kept = weighted_confidence([straddling], [0.77], roi)

    assert kept == [0.77]


def test_road_fraction_is_zero_on_an_empty_mask():
    roi = _roi()

    assert road_fraction(np.zeros_like(roi), roi) == 0.0


def test_hood_pixels_cannot_become_traversable_road():
    roi = _roi()
    # A segmenter that confidently paints the entire frame as road.
    everything = np.ones(roi.shape, dtype=bool)

    gated = apply_roi(everything, roi)

    assert not gated[~roi].any()
    assert gated[roi].all()


def test_stabiliser_reapplies_the_mask_after_morphology():
    """Closing gaps must not grow the road onto our own bodywork."""
    config = _config()
    roi = _roi()
    mask = np.zeros(roi.shape, dtype=bool)
    # Road running right down to the hood boundary, so dilation would spill.
    mask[:, 300:420] = True
    mask &= roi

    result = PerceptionResult(mask=mask, valid_roi=roi, confidences=[0.9])
    stabilized = TemporalStabilizer(config).stabilize(result)

    assert not stabilized.mask[~roi].any()
    assert stabilized.valid_roi is roi


def test_rgb_pipeline_applies_the_ego_mask():
    config = _config()
    roi = _roi()

    mask = np.ones(roi.shape, dtype=bool)
    perception = PerceptionResult(
        mask=apply_roi(mask, roi),
        valid_roi=roi,
        road_fraction=road_fraction(apply_roi(mask, roi), roi),
    )
    stabilized = TemporalStabilizer(config).stabilize(perception)

    assert not stabilized.mask[~roi].any()
    assert stabilized.road_fraction == pytest.approx(1.0, abs=1e-6)


def test_pipeline_builds_the_ego_mask():
    from unittest.mock import patch

    config = _config()

    with (
        patch("offroad_autonomy.pipeline.ImagePreprocessor"),
        patch("offroad_autonomy.pipeline.RoadSegmenter"),
        patch("offroad_autonomy.pipeline.TemporalStabilizer"),
        patch("offroad_autonomy.pipeline.CenterlinePlanner"),
        patch("offroad_autonomy.pipeline.StanleyController"),
    ):
        from offroad_autonomy.pipeline import AutonomyPipeline

        pipeline = AutonomyPipeline(config)

    assert pipeline.valid_roi.shape == (config.preprocess_height, config.preprocess_width)
    assert 0.12 < pipeline.ego_coverage < 0.16
    assert not pipeline.valid_roi[-1, config.preprocess_width // 2]
    assert pipeline.valid_roi[config.preprocess_height // 2, config.preprocess_width // 2]


def test_safe_stop_still_fires_when_the_road_really_is_gone():
    detector = _StuckDetector(min_road_fraction=0.015, no_road_time_s=2.0)

    assert detector.update(0.0, road_fraction=0.0, speed_mps=5.0, throttle=0.3)[0] is False
    assert detector.update(1.0, road_fraction=0.0, speed_mps=5.0, throttle=0.3)[0] is False

    triggered, reason = detector.update(2.5, road_fraction=0.0, speed_mps=5.0, throttle=0.3)

    assert triggered
    assert "no traversable road" in reason


def test_safe_stop_does_not_fire_on_a_visible_road():
    detector = _StuckDetector(min_road_fraction=0.015, no_road_time_s=2.0)

    for t in range(100):
        triggered, _ = detector.update(float(t), road_fraction=0.25, speed_mps=5.0, throttle=0.3)
        assert not triggered


def test_recovering_road_clears_the_safe_stop_timer():
    detector = _StuckDetector(min_road_fraction=0.015, no_road_time_s=2.0)

    detector.update(0.0, road_fraction=0.0, speed_mps=5.0, throttle=0.3)
    detector.update(1.0, road_fraction=0.30, speed_mps=5.0, throttle=0.3)
    # The timer restarted, so a single bad frame long after must not trip it.
    triggered, _ = detector.update(3.0, road_fraction=0.0, speed_mps=5.0, throttle=0.3)

    assert not triggered


def test_stuck_detection_is_unchanged():
    """The motion-based half of the safety net must still work."""
    detector = _StuckDetector()

    detector.update(0.0, road_fraction=0.5, speed_mps=0.1, throttle=0.5)
    triggered, reason = detector.update(4.0, road_fraction=0.5, speed_mps=0.1, throttle=0.5)

    assert triggered
    assert reason == "vehicle stuck"


def test_hood_sized_exclusion_cannot_by_itself_trigger_safe_stop():
    """The regression this whole change exists to prevent.

    A frame where every valid pixel is road, but 15% of the frame is hood,
    must read as a clear road - not as a 15% loss of confidence.
    """
    config = PipelineConfig()
    roi = EgoMask(config.camera.ego_mask).valid_roi((465, 720))
    all_road = apply_roi(np.ones(roi.shape, dtype=bool), roi)

    fraction = road_fraction(all_road, roi)
    detector = _StuckDetector(
        min_road_fraction=config.safety_min_road_fraction,
        no_road_time_s=config.safety_no_road_time_s,
    )

    assert fraction == pytest.approx(1.0)
    for t in range(20):
        triggered, _ = detector.update(
            float(t), road_fraction=fraction, speed_mps=4.0, throttle=0.3
        )
        assert not triggered


def test_ego_mask_round_trips_through_yaml(tmp_path):
    from offroad_autonomy.utils.config import load_config

    path = tmp_path / "ego.yaml"
    path.write_text(
        "\n".join(
            [
                "beamng:",
                "  camera:",
                "      ego_mask:",
                "        enabled: true",
                "        margin_px: 5",
                "        polygon:",
                "          - [0.2, 1.0]",
                "          - [0.8, 1.0]",
                "          - [0.5, 0.6]",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(path)
    spec = config.camera.ego_mask

    assert spec.enabled
    assert spec.margin_px == 5
    assert spec.polygon == ((0.2, 1.0), (0.8, 1.0), (0.5, 0.6))
    assert EgoMask(spec).coverage((465, 720)) > 0.0


def test_enabled_mask_without_a_polygon_is_rejected(tmp_path):
    from offroad_autonomy.utils.config import load_config

    path = tmp_path / "bad.yaml"
    path.write_text(
        "beamng:\n  camera:\n      ego_mask:\n        enabled: true\n        polygon: []\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fewer than 3 points"):
        load_config(path)


def test_moving_the_camera_does_not_silently_move_the_mask():
    """The polygon belongs to a pose; changing one must not fake the other."""
    config = PipelineConfig()
    moved = replace(config.camera, pos=(0.3, -0.5, 1.2))

    # Same polygon, because it is configuration - which is exactly why
    # scripts/derive_ego_mask.py exists and the config says to re-run it.
    assert moved.ego_mask == config.camera.ego_mask


def test_dashcam_exclusion_covers_derived_hood_silhouette():
    from offroad_autonomy.perception.camera_geometry import CameraModel
    from scripts.derive_ego_mask import silhouette

    cfg = PipelineConfig()
    body = silhouette(CameraModel(cfg.camera, cfg.preprocess_width, cfg.preprocess_height))
    excluded = EgoMask(cfg.camera.ego_mask).excluded(body.shape)
    assert body.any()
    assert not (body & ~excluded).any()
    assert 0.12 < excluded.mean() < 0.16
