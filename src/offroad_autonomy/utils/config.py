"""YAML -> ``PipelineConfig``."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

from offroad_autonomy.control.controller_config import MPCConfig
from offroad_autonomy.types import (
    CAMERA_TRANSPORTS,
    DEBUG_VIEWS,
    DEFAULT_CAMERA,
    DEFAULT_DASHBOARD_COLORS,
    DEFAULT_PERCEPTION_PROMPTS,
    GMSL2_CAPTURE_SENSOR,
    CameraSensor,
    CameraSpec,
    EgoMaskSpec,
    PipelineConfig,
    mount_pose,
)

logger = logging.getLogger("offroad_autonomy.config")

_TRUE_STRINGS = ("1", "true", "yes", "on")


def _parse_color(raw: object, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        return default
    try:
        return tuple(int(value) for value in raw)
    except (TypeError, ValueError):
        return default


def _load_dashboard_colors(raw: object) -> dict[str, tuple[int, int, int]]:
    colors = DEFAULT_DASHBOARD_COLORS.copy()
    if not isinstance(raw, dict):
        return colors
    for key, default in DEFAULT_DASHBOARD_COLORS.items():
        colors[key] = _parse_color(raw.get(key), default)
    return colors


def _parse_vec3(
    raw: object,
    default: tuple[float, float, float],
) -> tuple[float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        return default
    try:
        return tuple(float(value) for value in raw)
    except (TypeError, ValueError):
        return default


def _load_sensor(raw: object, default: CameraSensor = GMSL2_CAPTURE_SENSOR) -> CameraSensor:
    if not isinstance(raw, dict):
        return default
    return CameraSensor(
        model=str(raw.get("model", default.model)),
        width=int(raw.get("width", default.width)),
        height=int(raw.get("height", default.height)),
        fov_x_deg=float(raw.get("fov_h", default.fov_x_deg)),
        target_fps=float(raw.get("target_fps", default.target_fps)),
    )


def _load_ego_mask(raw: object, default: EgoMaskSpec) -> EgoMaskSpec:
    """A malformed polygon fails loudly: silently excluding the wrong region
    would hide real road or expose bodywork as terrain."""
    if not isinstance(raw, dict):
        return default

    enabled = bool(raw.get("enabled", default.enabled))
    margin = int(raw.get("margin_px", default.margin_px))

    raw_polygon = raw.get("polygon")
    if raw_polygon is None:
        polygon = default.polygon
    else:
        points: list[tuple[float, float]] = []
        if isinstance(raw_polygon, list):
            for point in raw_polygon:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    continue
                try:
                    points.append((float(point[0]), float(point[1])))
                except (TypeError, ValueError):
                    continue
        polygon = tuple(points)

    if enabled and len(polygon) < 3:
        raise ValueError(
            "beamng.camera.ego_mask is enabled but its polygon has fewer than 3 points"
        )
    return EgoMaskSpec(enabled=enabled, polygon=polygon, margin_px=margin)


def _load_camera(raw: dict) -> CameraSpec:
    default = DEFAULT_CAMERA
    direction, up = mount_pose(
        float(raw.get("pitch_deg", -8.0)),
        float(raw.get("roll_deg", 0.0)),
        float(raw.get("yaw_deg", 0.0)),
    )
    pos = (
        float(raw.get("lateral_offset_m", default.pos[0])),
        float(raw.get("forward_offset_m", default.pos[1])),
        float(raw.get("height_m", default.pos[2])),
    )
    return CameraSpec(
        name=str(raw.get("name", default.name)),
        pos=_parse_vec3(raw.get("pos"), pos),
        dir=_parse_vec3(raw.get("dir"), direction),
        up=_parse_vec3(raw.get("up"), up),
        sensor=_load_sensor(raw.get("sensor")),
        ego_mask=_load_ego_mask(raw.get("ego_mask"), default.ego_mask),
    )


def _deep_merge(base: dict, override: dict) -> dict:
    """Lists are replaced, not merged, so an overlay can shorten a polygon."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    """Resolves ``extends:`` so experiment and deployment configs only state
    what they change."""
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Circular 'extends' chain: {' -> '.join(map(str, seen + (path,)))}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    base_ref = raw.pop("extends", None)
    if base_ref:
        base = _read_yaml((path.parent / str(base_ref)), seen + (path,))
        raw = _deep_merge(base, raw)
    return raw


def _section(raw: dict, key: str) -> dict:
    value = raw.get(key)
    if isinstance(value, dict):
        return value
    return {}


def _planner_mode(raw: object) -> str:
    mode = str(raw).lower()
    if mode not in ("baseline", "advanced"):
        raise ValueError(f"planning.mode must be 'baseline' or 'advanced', got {mode!r}")
    return mode


def _camera_transport(raw: object) -> str:
    transport = str(raw).lower()
    if transport not in CAMERA_TRANSPORTS:
        raise ValueError(
            f"beamng.camera_transport must be one of {CAMERA_TRANSPORTS}, got {transport!r}"
        )
    return transport


def _apply_env_overrides(bng: dict) -> dict:
    """The simulator's address differs per deployment, so it can be set
    without editing a YAML file baked into a container image."""
    bng = dict(bng)
    if os.environ.get("BEAMNG_HOST"):
        bng["host"] = os.environ["BEAMNG_HOST"]
    if os.environ.get("BEAMNG_PORT"):
        bng["port"] = int(os.environ["BEAMNG_PORT"])
    if os.environ.get("BEAMNG_HOME"):
        bng["home"] = os.environ["BEAMNG_HOME"]
    if os.environ.get("BEAMNG_LAUNCH"):
        bng["launch"] = os.environ["BEAMNG_LAUNCH"].lower() in _TRUE_STRINGS
    return bng


def load_config(path: str | Path) -> PipelineConfig:
    raw = _read_yaml(Path(path))

    bng = _apply_env_overrides(_section(raw, "beamng"))
    if "cameras" in bng:
        raise ValueError(
            "Replace beamng.cameras with the single beamng.camera dashcam configuration"
        )
    camera = _load_camera(_section(bng, "camera"))
    perc = _section(raw, "perception")
    ui = _section(raw, "ui")
    safety = _section(raw, "safety")
    pre = _section(raw, "preprocessing")
    post = _section(raw, "postprocessing")
    plan = _section(raw, "planning")
    gate = _section(plan, "gate")
    ctrl = _section(raw, "control")
    controller = str(ctrl.get("controller", "stanley")).lower()
    if controller not in ("stanley", "mpc"):
        raise ValueError("control.controller must be stanley or mpc")
    dashboard = _section(_section(raw, "visualization"), "dashboard")

    if "segmentation_mode" in perc or "stitching" in raw:
        raise ValueError("Remove segmentation_mode and stitching: the dashcam is the only input")
    debug_view = str(ui.get("debug_view", "default")).lower()
    if debug_view not in DEBUG_VIEWS:
        raise ValueError(f"ui.debug_view must be one of {DEBUG_VIEWS}, got {debug_view!r}")

    return PipelineConfig(
        beamng_home=str(bng.get("home", "")),
        beamng_host=str(bng.get("host", "localhost")),
        beamng_port=int(bng.get("port", 64256)),
        beamng_launch=bool(bng.get("launch", True)),
        beamng_camera_transport=_camera_transport(bng.get("camera_transport", "shared_memory")),
        beamng_map=bng.get("map", "automation_test_track"),
        beamng_vehicle=bng.get("vehicle", "pickup"),
        beamng_spawn_index=bng.get("spawn_index", 0),
        camera=camera,
        map_spawns=bng.get("maps", {}),
        model_weights=perc.get("model_weights", "models/yoloe-26x-seg.pt"),
        confidence_threshold=perc.get("segmentation_threshold", 0.25),
        perception_input_size=perc.get("input_size", 640),
        perception_prompts=perc.get("prompts", DEFAULT_PERCEPTION_PROMPTS.copy()),
        preprocess_width=pre.get("target_width", 720),
        preprocess_height=pre.get("target_height", 465),
        enable_clahe=pre.get("enable_clahe", False),
        clahe_clip_limit=pre.get("clahe_clip_limit", 2.0),
        clahe_grid_size=pre.get("clahe_grid_size", 8),
        safety_min_road_fraction=safety.get("min_road_fraction", 0.015),
        safety_no_road_time_s=safety.get("no_road_time_s", 2.0),
        ema_alpha=post.get("ema_alpha", 0.7),
        min_mask_area_fraction=post.get("min_mask_area_fraction", 0.001),
        morphology_kernel_size=post.get("morphology_kernel_size", 5),
        enable_morphology=bool(post.get("enable_morphology", True)),
        enable_ema=bool(post.get("enable_ema", True)),
        planner_mode=_planner_mode(plan.get("mode", "baseline")),
        planner_roi_height=float(plan.get("roi_height", 0.50)),
        baseline_temporal_blend=float(plan.get("baseline_temporal_blend", 0.0)),
        baseline_max_shift_m=float(plan.get("baseline_max_shift_m", 0.5)),
        gate_min_confidence=float(gate.get("confidence_threshold", 0.18)),
        gate_min_mask_area=float(gate.get("min_mask_area", 0.05)),
        gate_hold_frames=int(gate.get("hold_frames", 10)),
        gate_hold_speed_scale=float(gate.get("hold_speed_scale", 0.4)),
        centerline_samples=plan.get("centerline_samples", 20),
        planner_backend=plan.get("backend", "heuristic"),
        planner_horizon_fraction=plan.get("horizon_fraction", 0.82),
        planner_smoothing_window=plan.get("smoothing_window", 7),
        planner_clearance_weight=plan.get("clearance_weight", 0.65),
        planner_prior_std_fraction=plan.get("prior_std_fraction", 0.10),
        planner_min_confidence=plan.get("min_confidence", 0.18),
        planner_segment_center_weight=plan.get("segment_center_weight", 0.68),
        planner_temporal_blend=plan.get("temporal_blend", 0.58),
        planner_max_lateral_step_px=plan.get("max_lateral_step_px", 32.0),
        planner_straight_blend=plan.get("straight_blend", 0.72),
        planner_straight_residual_px=plan.get("straight_residual_px", 4.0),
        planner_straight_heading_threshold=plan.get("straight_heading_threshold", 0.08),
        kalman_process_noise=plan.get("kalman_process_noise", 1e-3),
        kalman_measurement_noise=plan.get("kalman_measurement_noise", 1e-1),
        fallback_after_n_misses=plan.get("fallback_after_n_misses", 3),
        min_road_pixels=plan.get("min_road_pixels", 500),
        controller=controller,
        mpc=MPCConfig(**_section(ctrl, "mpc")),
        stanley_gain_k=ctrl.get("stanley_gain_k", 1.5),
        stanley_softening=ctrl.get("stanley_softening", 2.4),
        stanley_heading_gain=ctrl.get("stanley_heading_gain", 0.85),
        steering_ema_alpha=ctrl.get("steering_ema_alpha", 0.45),
        max_steering_delta=ctrl.get("max_steering_delta", 0.3),
        lookahead_base_m=float(ctrl.get("lookahead_base_m", 3.0)),
        lookahead_time_s=float(ctrl.get("lookahead_time_s", 0.6)),
        lookahead_min_m=float(ctrl.get("lookahead_min_m", 2.0)),
        lookahead_max_m=float(ctrl.get("lookahead_max_m", 8.0)),
        cross_track_tolerance_m=float(ctrl.get("cross_track_tolerance_m", 0.3)),
        steer_full_authority_speed_mps=float(ctrl.get("steer_full_authority_speed_mps", 3.0)),
        steer_speed_falloff=float(ctrl.get("steer_speed_falloff", 0.08)),
        wheelbase_m=float(ctrl.get("wheelbase_m", 2.6)),
        max_wheel_angle_deg=float(ctrl.get("max_wheel_angle_deg", 32.0)),
        path_end_margin_m=float(ctrl.get("path_end_margin_m", 1.5)),
        path_end_decel_mps2=float(ctrl.get("path_end_decel_mps2", 2.0)),
        max_lateral_accel_mps2=float(ctrl.get("max_lateral_accel_mps2", 0.5)),
        curve_decel_mps2=float(ctrl.get("curve_decel_mps2", 0.8)),
        control_latency_s=float(ctrl.get("control_latency_s", 0.35)),
        lookahead_curvature_gain=float(ctrl.get("lookahead_curvature_gain", 2.0)),
        curvature_feedforward_gain=float(ctrl.get("curvature_feedforward_gain", 1.0)),
        saturation_start=float(ctrl.get("saturation_start", 0.5)),
        saturation_min_speed_scale=float(ctrl.get("saturation_min_speed_scale", 0.35)),
        steer_cap_full_speed_mps=float(ctrl.get("steer_cap_full_speed_mps", 2.0)),
        steer_cap_high_speed_mps=float(ctrl.get("steer_cap_high_speed_mps", 5.0)),
        steer_cap_at_high_speed=float(ctrl.get("steer_cap_at_high_speed", 0.6)),
        max_target_speed_increase_mps=float(ctrl.get("max_target_speed_increase_mps", 0.1)),
        comfort_brake=float(ctrl.get("comfort_brake", 0.35)),
        min_curvature_span_m=float(ctrl.get("min_curvature_span_m", 2.0)),
        cross_track_soft_deadzone=bool(ctrl.get("cross_track_soft_deadzone", True)),
        latency_pose_prediction=bool(ctrl.get("latency_pose_prediction", True)),
        cross_track_lookahead_gain=float(ctrl.get("cross_track_lookahead_gain", 1.0)),
        cross_track_min_lookahead_m=float(ctrl.get("cross_track_min_lookahead_m", 1.5)),
        cross_track_recovery_m=float(ctrl.get("cross_track_recovery_m", 0.5)),
        vehicle_half_width_m=float(ctrl.get("vehicle_half_width_m", 0.95)),
        edge_margin_m=float(ctrl.get("edge_margin_m", 0.75)),
        edge_centering_gain=float(ctrl.get("edge_centering_gain", 0.7)),
        edge_speed_reduction=float(ctrl.get("edge_speed_reduction", 0.6)),
        drift_speed_gain=float(ctrl.get("drift_speed_gain", 2.0)),
        max_measured_lateral_accel_mps2=float(ctrl.get("max_measured_lateral_accel_mps2", 0.8)),
        target_speed_mph=float(ctrl.get("target_speed_mph", 12.0)),
        speed_limit_mph=float(ctrl.get("speed_limit_mph", 15.0)),
        min_turn_speed_mph=float(ctrl.get("min_turn_speed_mph", 7.0)),
        max_throttle=ctrl.get("max_throttle", 0.45),
        max_brake=ctrl.get("max_brake", 0.8),
        speed_kp=ctrl.get("speed_kp", 0.22),
        clearance_slow_m=ctrl.get("clearance_slow_m", 6.0),
        clearance_stop_m=ctrl.get("clearance_stop_m", 2.5),
        dashboard_colors=_load_dashboard_colors(dashboard.get("colors")),
        ui_headless=bool(ui.get("headless", False)),
        ui_render_every_n=max(1, int(ui.get("render_every_n", 1))),
        ui_debug_view=debug_view,
        ui_timing_overlay=bool(ui.get("timing_overlay", False)),
        ui_display_async=bool(ui.get("display_async", True)),
        ui_display_rate_hz=float(ui.get("display_rate_hz", 20.0)),
        runtime_log_interval_s=float(ui.get("log_interval_s", 5.0)),
    )
