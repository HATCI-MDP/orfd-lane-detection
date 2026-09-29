"""Unit tests for configuration loading."""

from pathlib import Path

import pytest

from offroad_autonomy.types import DEFAULT_CAMERA, PipelineConfig
from offroad_autonomy.utils.config import load_config

REPO = Path(__file__).resolve().parents[1]
DEFAULT_YAML = REPO / "configs" / "default.yaml"


def test_load_config_reads_perception_prompts(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "perception:",
                '  model_weights: "dummy.pt"',
                "  prompts:",
                '    - "trail"',
                '    - "road"',
                "visualization:",
                "  dashboard:",
                "    colors:",
                "      BG: [1, 2, 3]",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.model_weights == "dummy.pt"
    assert config.perception_prompts == ["trail", "road"]
    assert config.dashboard_colors["BG"] == (1, 2, 3)


def test_legacy_depth_settings_cannot_enable_depth(tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text("depth:\n  enabled: true\nterrain:\n  fusion_weight: 1.0\n", encoding="utf-8")
    config = load_config(path)
    assert not hasattr(config, "depth_enabled")
    assert not hasattr(config, "depth_fusion_weight")


def test_vehicle_width_is_configured_with_control(tmp_path):
    path = tmp_path / "width.yaml"
    path.write_text("control:\n  vehicle_half_width_m: 1.2\n", encoding="utf-8")
    assert load_config(path).vehicle_half_width_m == pytest.approx(1.2)


def test_invalid_segmentation_mode_is_refused(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("perception:\n  segmentation_mode: center\n", encoding="utf-8")

    with pytest.raises(ValueError, match="segmentation_mode"):
        load_config(config_path)


def test_jetson_config_extends_the_default():
    config = load_config(REPO / "configs" / "jetson.yaml")
    base = load_config(DEFAULT_YAML)

    assert config.beamng_launch is False
    assert config.beamng_camera_transport == "socket"
    assert config.ui_headless is True
    # Everything the overlay does not mention is inherited.
    assert config.camera == base.camera
    assert config.map_spawns == base.map_spawns


def test_extends_cycle_is_refused(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Circular"):
        load_config(tmp_path / "a.yaml")


def test_dashcam_defaults_match_shipped_config():
    cfg = load_config(DEFAULT_YAML)
    assert cfg.camera == DEFAULT_CAMERA == PipelineConfig().camera
    assert cfg.camera.name == "dashcam"
    assert cfg.camera.pos == (0.0, -0.30, 1.85)
    assert cfg.camera.sensor.model == "GMSL2"
    assert cfg.camera.fov_x_deg == 120.0
    assert not hasattr(cfg, "left_camera")
    assert not hasattr(cfg, "right_camera")
    assert not hasattr(cfg, "display_camera")


def test_dashcam_mount_and_sensor_are_configurable(tmp_path):
    path = tmp_path / "camera.yaml"
    path.write_text(
        "beamng:\n  camera:\n    height_m: 2.1\n    pitch_deg: -12\n"
        "    sensor:\n      width: 1440\n      height: 930\n"
    )
    cfg = load_config(path)
    assert cfg.camera.pos == (0.0, -0.30, 2.1)
    assert cfg.camera.dir[2] == pytest.approx(-0.20791169)
    assert cfg.camera.width == 1440
    assert cfg.camera.fov_x_deg == 120.0


def test_old_multi_camera_config_requires_migration(tmp_path):
    path = tmp_path / "old.yaml"
    path.write_text("beamng:\n  cameras:\n    left: {}\n")
    with pytest.raises(ValueError, match="beamng.camera"):
        load_config(path)
