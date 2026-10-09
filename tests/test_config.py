"""Unit tests for configuration loading: the harness's sections and the stack's own."""

import math
from pathlib import Path
from unittest.mock import patch

import pytest

from offroad_autonomy.stacks.mono_yoloe.config import (
    load_pipeline_config,
    load_pipeline_config_file,
)
from offroad_autonomy.types import DEFAULT_CAMERA, DashboardThresholds, HarnessConfig
from offroad_autonomy.utils import environment
from offroad_autonomy.utils.config import load_config
from offroad_autonomy.utils.environment import PlatformFacts

REPO = Path(__file__).resolve().parents[1]
DEFAULT_YAML = REPO / "configs" / "default.yaml"


def _write(tmp_path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# ----------------------------------------------------------------- harness


def test_harness_reads_dashboard_colours_and_the_stack(tmp_path):
    path = _write(
        tmp_path,
        "stack: some.module:build\n"
        "stack_config:\n  anything: 1\n"
        "visualization:\n  dashboard:\n    colors:\n      BG: [1, 2, 3]\n",
    )

    config = load_config(path)

    assert config.dashboard_colors["BG"] == (1, 2, 3)
    assert config.stack == "some.module:build"
    # The harness never reads inside stack_config.
    assert config.stack_config == {"anything": 1}


def test_shipped_default_runs_the_mono_yoloe_stack():
    config = load_config(DEFAULT_YAML)

    assert config.stack == "offroad_autonomy.stacks.mono_yoloe:build"
    assert config.stack_config == {}


@pytest.mark.parametrize(
    ("block", "match"),
    [
        ("beamng:\n  mapp: x\n", "Unknown config key beamng.mapp"),
        ("depth:\n  enabled: true\n", "Unknown config key depth"),
        ("visualization:\n  dashboard:\n    colors:\n      PINK: [1, 2, 3]\n", "PINK"),
        ("ui: 3\n", "ui must be a mapping"),
    ],
)
def test_unknown_harness_keys_are_refused(tmp_path, block, match):
    with pytest.raises((ValueError, TypeError), match=match):
        load_config(_write(tmp_path, block))


@pytest.mark.parametrize(
    ("block", "moved_to"),
    [
        ("perception:\n  model_weights: x.pt\n", "stack_config.perception"),
        ("control:\n  stanley_gain_k: 1.0\n", "stack_config.control"),
        ("beamng:\n  camera:\n    height_m: 2.0\n", "stack_config.camera"),
        ("safety:\n  min_road_fraction: 0.02\n", "stack_config.safety.min_road_fraction"),
        ("ui:\n  debug_view: raw\n", "ui.debug_page"),
        (
            "visualization:\n  dashboard:\n    confidence_good: 0.6\n",
            "stack_config.dashboard.confidence_good",
        ),
    ],
)
def test_keys_from_before_the_split_name_where_they_moved(tmp_path, block, moved_to):
    with pytest.raises(ValueError, match="has moved") as caught:
        load_config(_write(tmp_path, block))
    assert moved_to in str(caught.value)


def test_jetson_config_extends_the_default(monkeypatch):
    monkeypatch.setenv("BEAMNG_HOST", "192.168.1.50")
    jetson = PlatformFacts(os_name="linux", default_gateway="192.168.1.1")
    with patch.object(environment, "detect_platform", return_value=jetson):
        config = load_config(REPO / "configs" / "jetson.yaml")
        base = load_config(DEFAULT_YAML)

    assert config.beamng_host == "192.168.1.50"
    assert config.beamng_launch is False
    assert config.beamng_camera_transport == "socket"
    assert config.ui_headless is True
    # Everything the overlay does not mention is inherited.
    assert config.map_spawns == base.map_spawns
    assert config.stack == base.stack
    stack = load_pipeline_config(config.stack_config)
    assert stack.model_weights == "models/yoloe-26s-seg.engine"
    assert stack.camera == load_pipeline_config(base.stack_config).camera


def test_extends_cycle_is_refused(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Circular"):
        load_config(tmp_path / "a.yaml")


def test_dashboard_thresholds_default_to_the_shipped_config():
    assert load_config(DEFAULT_YAML).dashboard_thresholds == DashboardThresholds()


@pytest.mark.parametrize(
    "block",
    [
        "target_fps: 0",
        "latency_budget_ms: -1",
        "fps_warn_fraction: 1.0",
        "fps_bar_scale: 1.0",
    ],
)
def test_invalid_dashboard_thresholds_are_refused(tmp_path, block):
    path = _write(tmp_path, f"visualization:\n  dashboard:\n    {block}\n")

    with pytest.raises(ValueError, match="visualization.dashboard"):
        load_config(path)


def test_harness_defaults_match_the_shipped_config():
    config = load_config(DEFAULT_YAML)
    defaults = HarnessConfig()

    for name in (
        "recording_fps",
        "recording_crf",
        "recording_preset",
        "recording_queue_frames",
        "safety_stuck_speed_mps",
        "safety_stuck_throttle_min",
        "safety_stuck_time_s",
        "benchmark_routes_dir",
        "benchmark_route_spacing_m",
        "benchmark_departure_cross_track_m",
        "benchmark_steer_sign_deadband",
        "ui_debug_page",
    ):
        assert getattr(config, name) == getattr(defaults, name), name


@pytest.mark.parametrize(
    "block",
    [
        "recording:\n  preset: turbo",
        "recording:\n  crf: 60",
        "recording:\n  fps: 0",
        "recording:\n  queue_frames: 0",
        "safety:\n  stuck_time_s: 0",
        "benchmark:\n  departure_cross_track_m: 0",
        "benchmark:\n  steer_sign_deadband: 1.5",
        "ui:\n  debug_page: 10",
    ],
)
def test_invalid_harness_settings_are_refused(tmp_path, block):
    section = block.split(":")[0]
    with pytest.raises(ValueError, match=section):
        load_config(_write(tmp_path, block + "\n"))


def test_orbit_camera_defaults_match_the_shipped_config():
    config = load_config(DEFAULT_YAML)
    defaults = HarnessConfig().orbit_camera

    assert config.orbit_camera.pos == pytest.approx(defaults.pos)
    assert config.orbit_camera.dir == pytest.approx(defaults.dir)
    assert config.orbit_camera.sensor == defaults.sensor


def test_orbit_camera_pose_and_sensor_are_configurable(tmp_path):
    path = _write(
        tmp_path,
        "presentation:\n"
        "  orbit_camera:\n"
        "    pos: [0.0, 8.0, 3.0]\n"
        "    pitch_deg: -20.0\n"
        "    sensor:\n"
        "      width: 640\n"
        "      height: 480\n"
        "      fov_h: 80.0\n",
    )

    orbit = load_config(path).orbit_camera

    assert orbit.pos == (0.0, 8.0, 3.0)
    assert orbit.dir[2] == pytest.approx(-math.sin(math.radians(20.0)))
    assert (orbit.width, orbit.height, orbit.fov_x_deg) == (640, 480, 80.0)


@pytest.mark.parametrize(
    "block",
    [
        "    pos: [0.0, 6.0]",
        "    pitch_deg: 95",
        "    sensor:\n      width: 0",
        "    sensor:\n      fov_h: 180",
        "    sensor:\n      target_fps: 0",
    ],
)
def test_invalid_orbit_camera_settings_are_refused(tmp_path, block):
    path = _write(tmp_path, "presentation:\n  orbit_camera:\n" + block + "\n")

    with pytest.raises(ValueError, match="presentation.orbit_camera"):
        load_config(path)


# ------------------------------------------------------------- mono_yoloe


def test_stack_reads_perception_prompts():
    config = load_pipeline_config(
        {"perception": {"model_weights": "dummy.pt", "prompts": ["trail", "road"]}}
    )

    assert config.model_weights == "dummy.pt"
    assert config.perception_prompts == ["trail", "road"]


@pytest.mark.parametrize(
    ("stack_config", "match"),
    [
        ({"control": {"stanley_gian_k": 1.0}}, "stack_config.control.stanley_gian_k"),
        ({"lidar": {}}, "stack_config.lidar"),
    ],
)
def test_unknown_stack_keys_are_refused(stack_config, match):
    with pytest.raises(ValueError, match=match):
        load_pipeline_config(stack_config)


def test_vehicle_geometry_is_configured_with_control():
    config = load_pipeline_config(
        {"control": {"vehicle_half_width_m": 1.2, "camera_ahead_of_rear_axle_m": 1.8}}
    )

    assert config.vehicle_half_width_m == pytest.approx(1.2)
    assert config.camera_ahead_of_rear_axle_m == pytest.approx(1.8)


def test_dashcam_defaults_match_shipped_config():
    cfg = load_pipeline_config_file(DEFAULT_YAML)
    assert cfg.camera == DEFAULT_CAMERA
    assert cfg.camera.name == "dashcam"
    assert cfg.camera.pos == (0.0, -0.30, 1.85)
    assert cfg.camera.sensor.model == "GMSL2"
    assert cfg.camera.fov_x_deg == 120.0


def test_dashcam_mount_and_sensor_are_configurable():
    cfg = load_pipeline_config(
        {"camera": {"height_m": 2.1, "pitch_deg": -12, "sensor": {"width": 1440, "height": 930}}}
    )
    assert cfg.camera.pos == (0.0, -0.30, 2.1)
    assert cfg.camera.dir[2] == pytest.approx(-0.20791169)
    assert cfg.camera.width == 1440
    assert cfg.camera.fov_x_deg == 120.0


def test_gauge_floors_are_the_gate_and_safe_stop_values():
    cfg = load_pipeline_config(
        {"planning": {"gate": {"confidence_threshold": 0.3}}, "safety": {"min_road_fraction": 0.05}}
    )

    assert cfg.gate_min_confidence == pytest.approx(0.3)
    assert cfg.safety_min_road_fraction == pytest.approx(0.05)


@pytest.mark.parametrize(
    "stack_config",
    [{"dashboard": {"confidence_good": 0.1}}, {"dashboard": {"road_bar_full_scale": 0.01}}],
)
def test_invalid_gauges_are_refused(stack_config):
    with pytest.raises(ValueError, match="stack_config.dashboard"):
        load_pipeline_config(stack_config)


def test_grid_profile_selects_the_grid_planner():
    config = load_pipeline_config_file(REPO / "configs" / "grid.yaml")
    default = load_pipeline_config_file(DEFAULT_YAML)

    assert config.planner_mode == "grid"
    assert config.grid == default.grid


@pytest.mark.parametrize(
    ("grid", "match"),
    [({"arc_size": 3}, "arc_size"), ({"arc_count": 40}, "odd"), ({"cell_m": 0}, "cell_m")],
)
def test_invalid_grid_settings_are_refused(grid, match):
    with pytest.raises(ValueError, match=match):
        load_pipeline_config({"planning": {"grid": grid}})


def test_a_config_running_another_stack_is_not_read_as_mono_yoloe(tmp_path):
    path = _write(tmp_path, "stack: some.other.stack:build\n")

    with pytest.raises(ValueError, match="not offroad_autonomy.stacks.mono_yoloe"):
        load_pipeline_config_file(path)
