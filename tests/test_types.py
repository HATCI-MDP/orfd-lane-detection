"""Unit tests for shared types."""

import pytest

from offroad_autonomy.types import (
    GMSL2_SENSOR,
    HarnessConfig,
    PipelineConfig,
)


def test_pipeline_config_defaults():
    cfg = PipelineConfig()
    assert not hasattr(cfg, "beamng_port")
    assert cfg.confidence_threshold == 0.25
    assert cfg.perception_prompts == [
        "traversable road",
        "dirt road",
        "off-road trail",
        "drivable terrain",
        "gravel path",
    ]
    assert cfg.ema_alpha == 0.7


def test_gmsl2_sensor_defaults():
    """The rig's one camera model, as specified."""
    assert GMSL2_SENSOR.model == "GMSL2"
    assert (GMSL2_SENSOR.width, GMSL2_SENSOR.height) == (2880, 1860)
    assert GMSL2_SENSOR.fov_x_deg == 120.0
    assert GMSL2_SENSOR.target_fps >= 28.0


def test_camera_specs_expose_sensor_properties():
    """A mount reads its intrinsics through the shared sensor."""
    cfg = PipelineConfig()
    sensor = cfg.camera.sensor

    for spec in (cfg.camera,):
        assert spec.width == sensor.width
        assert spec.height == sensor.height
        assert spec.fov_x_deg == GMSL2_SENSOR.fov_x_deg
        assert spec.fov_y_deg == pytest.approx(GMSL2_SENSOR.fov_y_deg)
        assert spec.target_fps == sensor.target_fps


def test_working_resolution_defaults_are_a_clean_fraction():
    cfg = PipelineConfig()
    capture = cfg.camera.sensor

    assert (cfg.preprocess_width, cfg.preprocess_height) == (720, 465)
    # 3/4 of the capture, 1/4 of the imager - uniform in both axes.
    assert cfg.preprocess_width * 4 == capture.width * 3
    assert cfg.preprocess_height * 4 == capture.height * 3
    assert GMSL2_SENSOR.width // cfg.preprocess_width == 4


def test_harness_config_defaults_name_the_mono_yoloe_stack():
    cfg = HarnessConfig()
    assert cfg.beamng_port == 64256
    assert cfg.stack == "offroad_autonomy.stacks.mono_yoloe:build"
    assert cfg.stack_config == {}
