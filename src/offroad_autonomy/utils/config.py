"""YAML -> ``HarnessConfig``, plus the helpers stacks use for their own section.

A config file names its stack and hands it ``stack_config``; this module
reads only the harness's sections, and both sides refuse keys they do not
know, so an outdated config fails at startup instead of silently running
with defaults.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

from offroad_autonomy.runtime.video_recorder import X264_PRESETS
from offroad_autonomy.types import (
    CAMERA_TRANSPORTS,
    DEFAULT_DASHBOARD_COLORS,
    DEFAULT_ORBIT_CAMERA,
    GMSL2_CAPTURE_SENSOR,
    MAX_DEBUG_PAGE,
    CameraSensor,
    CameraSpec,
    DashboardThresholds,
    HarnessConfig,
    mount_pose,
)
from offroad_autonomy.utils import environment

logger = logging.getLogger("offroad_autonomy.config")

_TRUE_STRINGS = ("1", "true", "yes", "on")

#: Harness ``section -> key -> (HarnessConfig field, parser)``. It is also the
#: schema for these sections; a key left out keeps the dataclass default.
_FIELDS: dict[str, dict[str, tuple]] = {
    "beamng": {
        "home": ("beamng_home", str),
        "host": ("beamng_host", str),
        "port": ("beamng_port", int),
        "launch": ("beamng_launch", bool),
        "camera_transport": ("beamng_camera_transport", lambda v: str(v).lower()),
        "map": ("beamng_map", str),
        "vehicle": ("beamng_vehicle", str),
        "spawn_index": ("beamng_spawn_index", int),
        # Its keys are map names, so it is a leaf for key checking.
        "maps": ("map_spawns", dict),
    },
    "safety": {
        "stuck_speed_mps": ("safety_stuck_speed_mps", float),
        "stuck_throttle_min": ("safety_stuck_throttle_min", float),
        "stuck_time_s": ("safety_stuck_time_s", float),
    },
    "ui": {
        "headless": ("ui_headless", bool),
        "render_every_n": ("ui_render_every_n", lambda v: max(1, int(v))),
        "display_async": ("ui_display_async", bool),
        "display_rate_hz": ("ui_display_rate_hz", float),
        "debug_page": ("ui_debug_page", int),
        "timing_overlay": ("ui_timing_overlay", bool),
        "log_interval_s": ("runtime_log_interval_s", float),
    },
    "recording": {
        "fps": ("recording_fps", float),
        "crf": ("recording_crf", int),
        "preset": ("recording_preset", lambda v: str(v).lower()),
        "queue_frames": ("recording_queue_frames", int),
    },
    "benchmark": {
        "routes_dir": ("benchmark_routes_dir", str),
        "route_spacing_m": ("benchmark_route_spacing_m", float),
        "departure_cross_track_m": ("benchmark_departure_cross_track_m", float),
        "steer_sign_deadband": ("benchmark_steer_sign_deadband", float),
    },
}
_THRESHOLD_KEYS = ("target_fps", "fps_warn_fraction", "latency_budget_ms")
_THRESHOLD_KEYS += ("fps_bar_scale", "latency_bar_scale")

#: ``None`` marks a leaf; a dict is a section whose keys are checked in turn.
HARNESS_SCHEMA: dict = {section: dict.fromkeys(keys) for section, keys in _FIELDS.items()}
HARNESS_SCHEMA.update(
    {
        "presentation": {
            "orbit_camera": {
                "pos": None,
                "pitch_deg": None,
                "sensor": dict.fromkeys(("model", "width", "height", "fov_h", "target_fps")),
            },
        },
        "visualization": {
            "dashboard": {"colors": dict.fromkeys(DEFAULT_DASHBOARD_COLORS)}
            | dict.fromkeys(_THRESHOLD_KEYS),
        },
        "stack": None,
        "stack_config": None,
    }
)

#: Where keys from before the harness and stack split now live, so an old
#: config names its fix instead of only being refused.
_MOVED_KEYS = {
    "perception": "stack_config.perception",
    "preprocessing": "stack_config.preprocessing",
    "postprocessing": "stack_config.postprocessing",
    "planning": "stack_config.planning",
    "control": "stack_config.control",
    "beamng.camera": "stack_config.camera",
    "safety.min_road_fraction": "stack_config.safety.min_road_fraction",
    "safety.no_road_time_s": "stack_config.safety.no_road_time_s",
    "ui.debug_view": "ui.debug_page (0 is the main view, 1-9 the stack's layers)",
    "visualization.dashboard.confidence_good": "stack_config.dashboard.confidence_good",
    "visualization.dashboard.road_bar_full_scale": "stack_config.dashboard.road_bar_full_scale",
}


# ----------------------------------------------------------------- generic


def read_yaml(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    """Resolves ``extends:`` so experiment and deployment configs only state
    what they change."""
    path = Path(path).resolve()
    if path in seen:
        raise ValueError(f"Circular 'extends' chain: {' -> '.join(map(str, seen + (path,)))}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must hold a mapping at the top level")
    base_ref = raw.pop("extends", None)
    if base_ref:
        base = read_yaml((path.parent / str(base_ref)), seen + (path,))
        raw = deep_merge(base, raw)
    return raw


def deep_merge(base: dict, override: dict) -> dict:
    """Lists are replaced, not merged, so an overlay can shorten a polygon."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def section(raw: dict, key: str) -> dict:
    value = raw.get(key)
    if isinstance(value, dict):
        return value
    return {}


def schema_from(defaults: dict) -> dict:
    """A stack's shipped defaults are its schema: a key is valid if they set it."""
    schema = {}
    for key, value in defaults.items():
        schema[key] = None
        if isinstance(value, dict):
            schema[key] = schema_from(value)
    return schema


def check_keys(
    raw: dict, schema: dict, where: str, moved: dict | None = None, at: str = ""
) -> None:
    """Raise on the first key ``schema`` does not list.

    ``moved`` maps a dotted path (relative to ``where``) to where that key
    went, for a clearer message than "unknown".
    """
    moved = moved or {}
    for key, value in raw.items():
        dotted = f"{at}{key}"
        if key not in schema:
            if dotted in moved:
                raise ValueError(f"{where}{dotted} has moved: set it as {moved[dotted]}")
            known = ", ".join(sorted(schema))
            raise ValueError(f"Unknown config key {where}{dotted} (known here: {known})")
        if schema[key] is None:
            continue
        if not isinstance(value, dict):
            raise TypeError(f"{where}{dotted} must be a mapping")
        check_keys(value, schema[key], where, moved, f"{dotted}.")


def load_sensor(raw: object, default: CameraSensor = GMSL2_CAPTURE_SENSOR) -> CameraSensor:
    if not isinstance(raw, dict):
        return default
    return CameraSensor(
        model=str(raw.get("model", default.model)),
        width=int(raw.get("width", default.width)),
        height=int(raw.get("height", default.height)),
        fov_x_deg=float(raw.get("fov_h", default.fov_x_deg)),
        target_fps=float(raw.get("target_fps", default.target_fps)),
    )


# ----------------------------------------------------------------- harness


def _parse_color(raw: object, default: tuple[int, int, int]) -> tuple[int, int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        return default
    try:
        return tuple(int(value) for value in raw)
    except (TypeError, ValueError):
        return default


def _load_orbit_camera(raw: dict) -> CameraSpec:
    """A bad pose fails at load time, before a run is recorded with the
    vehicle out of frame."""
    default = DEFAULT_ORBIT_CAMERA
    raw_pos = raw.get("pos", list(default.pos))
    if not isinstance(raw_pos, (list, tuple)) or len(raw_pos) != 3:
        raise ValueError("presentation.orbit_camera.pos must be a list of 3 numbers")
    pitch = float(raw.get("pitch_deg", -12.0))
    if not -90.0 < pitch < 90.0:
        raise ValueError("presentation.orbit_camera.pitch_deg must be between -90 and 90")
    sensor = load_sensor(raw.get("sensor"), default.sensor)
    if sensor.width < 1 or sensor.height < 1:
        raise ValueError("presentation.orbit_camera.sensor width and height must be positive")
    if not 0.0 < sensor.fov_x_deg < 180.0:
        raise ValueError("presentation.orbit_camera.sensor.fov_h must be between 0 and 180")
    if sensor.target_fps <= 0.0:
        raise ValueError("presentation.orbit_camera.sensor.target_fps must be > 0")
    direction, up = mount_pose(pitch)
    pos = tuple(float(value) for value in raw_pos)
    return CameraSpec(name=default.name, pos=pos, dir=direction, up=up, sensor=sensor)


def _load_dashboard(raw: dict) -> dict:
    colors = DEFAULT_DASHBOARD_COLORS.copy()
    raw_colors = raw.get("colors")
    if isinstance(raw_colors, dict):
        for key, default in DEFAULT_DASHBOARD_COLORS.items():
            colors[key] = _parse_color(raw_colors.get(key), default)
    values = {key: float(raw[key]) for key in _THRESHOLD_KEYS if key in raw}
    return {"dashboard_colors": colors, "dashboard_thresholds": DashboardThresholds(**values)}


def _validate(config: HarnessConfig) -> None:
    thresholds = config.dashboard_thresholds
    checks = (
        (
            config.beamng_camera_transport in CAMERA_TRANSPORTS,
            (
                f"beamng.camera_transport must be one of {CAMERA_TRANSPORTS}, "
                f"got {config.beamng_camera_transport!r}"
            ),
        ),
        (
            min(config.safety_stuck_speed_mps, config.safety_stuck_throttle_min) > 0.0
            and config.safety_stuck_time_s > 0.0,
            "safety stuck_speed_mps, stuck_throttle_min and stuck_time_s must be > 0",
        ),
        (
            0 <= config.ui_debug_page <= MAX_DEBUG_PAGE,
            f"ui.debug_page must be between 0 and {MAX_DEBUG_PAGE}, got {config.ui_debug_page}",
        ),
        (config.recording_fps > 0.0, "recording.fps must be > 0"),
        (0 <= config.recording_crf <= 51, "recording.crf must be between 0 and 51"),
        (
            config.recording_preset in X264_PRESETS,
            f"recording.preset must be one of {X264_PRESETS}, got {config.recording_preset!r}",
        ),
        (config.recording_queue_frames >= 1, "recording.queue_frames must be at least 1"),
        (
            min(config.benchmark_departure_cross_track_m, config.benchmark_route_spacing_m) > 0,
            "benchmark.departure_cross_track_m and route_spacing_m must be > 0",
        ),
        (
            0.0 <= config.benchmark_steer_sign_deadband < 1.0,
            "benchmark.steer_sign_deadband must be between 0 and 1",
        ),
        (
            min(thresholds.target_fps, thresholds.latency_budget_ms) > 0.0,
            "visualization.dashboard target_fps and latency_budget_ms must be > 0",
        ),
        (
            0.0 < thresholds.fps_warn_fraction < 1.0,
            "visualization.dashboard.fps_warn_fraction must be between 0 and 1",
        ),
        # A full scale at or below the target would push the target tick off the bar.
        (
            min(thresholds.fps_bar_scale, thresholds.latency_bar_scale) > 1.0,
            "visualization.dashboard bar scales must be above 1",
        ),
    )
    for ok, message in checks:
        if not ok:
            raise ValueError(message)


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


def _resolve_platform_settings(bng: dict, ui: dict) -> tuple[dict, dict]:
    """A fully explicit config never inspects the machine, so deployments
    that pin every value behave the same everywhere."""
    if not environment.needs_detection(bng, ui):
        return bng, ui
    facts = environment.detect_platform()
    if (
        facts.os_name == "windows"
        and bng.get("host") in (environment.AUTO, *environment.LOCAL_HOSTS)
        and bng.get("launch") == environment.AUTO
        and not bng.get("home")
        and "BEAMNG_HOME" not in os.environ
    ):
        saved_home = environment.saved_windows_beamng_home()
        if saved_home:
            bng = dict(bng, home=saved_home)
            logger.info("Using saved Windows BEAMNG_HOME: %s", saved_home)
    bng, ui = environment.resolve_auto(bng, ui, facts)
    platform = facts.os_name
    if facts.is_wsl:
        platform = f"wsl2 ({facts.wsl_networking})"
    action = "attaching to"
    if bng.get("launch"):
        action = "launching or attaching to"
    host = bng.get("host") or "<unset, set BEAMNG_HOST>"
    logger.info(
        "Platform %s: %s BeamNG at %s:%s over %s",
        platform,
        action,
        host,
        bng.get("port", 64256),
        bng.get("camera_transport"),
    )
    return bng, ui


def load_config(path: str | Path) -> HarnessConfig:
    raw = read_yaml(Path(path))
    check_keys(raw, HARNESS_SCHEMA, "", _MOVED_KEYS)

    sections = {name: section(raw, name) for name in _FIELDS}
    sections["beamng"], sections["ui"] = _resolve_platform_settings(
        _apply_env_overrides(sections["beamng"]), sections["ui"]
    )
    values = {}
    for name, fields in _FIELDS.items():
        for key, value in sections[name].items():
            field, parse = fields[key]
            values[field] = parse(value)

    stack_config = raw.get("stack_config")
    if stack_config is None:
        stack_config = {}
    if not isinstance(stack_config, dict):
        raise TypeError("stack_config must be a mapping")
    if "stack" in raw:
        values["stack"] = str(raw["stack"])

    config = HarnessConfig(
        **values,
        **_load_dashboard(section(section(raw, "visualization"), "dashboard")),
        orbit_camera=_load_orbit_camera(section(section(raw, "presentation"), "orbit_camera")),
        stack_config=stack_config,
    )
    _validate(config)
    return config
