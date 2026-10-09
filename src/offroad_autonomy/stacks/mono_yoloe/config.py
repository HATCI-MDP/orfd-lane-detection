"""``stack_config`` -> ``PipelineConfig`` for the mono_yoloe stack.

The stack's ``default.yaml`` is both its defaults and its schema: a run
config can only override keys that file sets, so an outdated key stops the
run at startup.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from offroad_autonomy.planning.grid_config import GridPlannerConfig
from offroad_autonomy.types import (
    DEFAULT_CAMERA,
    CameraSpec,
    EgoMaskSpec,
    PipelineConfig,
    mount_pose,
)
from offroad_autonomy.utils.config import (
    check_keys,
    deep_merge,
    load_config,
    load_sensor,
    read_yaml,
    schema_from,
    section,
)

DEFAULTS_PATH = Path(__file__).with_name("default.yaml")
PLANNER_MODES = ("baseline", "advanced", "grid")

#: A YAML key fills the field named ``prefix + key`` if there is one, else the
#: field named ``key``; these are the keys that fit neither.
_PREFIXES = {
    "perception": "perception_",
    "safety": "safety_",
    "planning": "planner_",
    "gate": "gate_",
    "dashboard": "gauge_",
}
_RENAMED = {
    ("perception", "segmentation_threshold"): "confidence_threshold",
    ("preprocessing", "target_width"): "preprocess_width",
    ("preprocessing", "target_height"): "preprocess_height",
    ("gate", "confidence_threshold"): "gate_min_confidence",
    ("dashboard", "road_bar_full_scale"): "gauge_road_full_scale",
}
_FIELD_NAMES = {f.name for f in fields(PipelineConfig)}
_FLAT_SECTIONS = (
    "perception",
    "safety",
    "preprocessing",
    "postprocessing",
    "planning",
    "control",
    "dashboard",
)


def stack_defaults() -> dict:
    return read_yaml(DEFAULTS_PATH)


def _field(section_name: str, key: str) -> str:
    candidates = (
        _RENAMED.get((section_name, key)),
        _PREFIXES.get(section_name, "") + key,
        key,
    )
    for name in candidates:
        if name in _FIELD_NAMES:
            return name
    raise KeyError(f"default.yaml key {section_name}.{key} has no PipelineConfig field")


def _load_ego_mask(raw: object, default: EgoMaskSpec) -> EgoMaskSpec:
    """A malformed polygon fails loudly: silently excluding the wrong region
    would hide real road or expose bodywork as terrain."""
    if not isinstance(raw, dict):
        return default
    polygon = default.polygon
    if raw.get("polygon") is not None:
        points = []
        for point in raw["polygon"]:
            if isinstance(point, (list, tuple)) and len(point) == 2:
                points.append((float(point[0]), float(point[1])))
        polygon = tuple(points)
    enabled = bool(raw.get("enabled", default.enabled))
    if enabled and len(polygon) < 3:
        raise ValueError(
            "stack_config.camera.ego_mask is enabled but its polygon has fewer than 3 points"
        )
    margin = int(raw.get("margin_px", default.margin_px))
    return EgoMaskSpec(enabled=enabled, polygon=polygon, margin_px=margin)


def _load_camera(raw: dict) -> CameraSpec:
    """The mount from offsets and angles in BeamNG vehicle space."""
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
        pos=pos,
        dir=direction,
        up=up,
        sensor=load_sensor(raw.get("sensor"), default.sensor),
        ego_mask=_load_ego_mask(raw.get("ego_mask"), default.ego_mask),
    )


def _load_grid(raw: dict) -> GridPlannerConfig:
    try:
        return GridPlannerConfig(**raw)
    except TypeError as exc:
        # A misspelled key would otherwise surface as a bare dataclass error.
        raise ValueError(f"stack_config.planning.grid: {exc}") from exc


def load_pipeline_config(stack_config: dict) -> PipelineConfig:
    defaults = stack_defaults()
    check_keys(stack_config, schema_from(defaults), "stack_config.")
    raw = deep_merge(defaults, stack_config)

    plain = PipelineConfig()
    values = {}
    groups = [(name, section(raw, name)) for name in _FLAT_SECTIONS]
    groups.append(("gate", section(section(raw, "planning"), "gate")))
    for name, entries in groups:
        for key, value in entries.items():
            if isinstance(value, dict):
                continue
            field = _field(name, key)
            # Cast to the field's own type, so a YAML 1 for a float field
            # cannot leave an int where the stages expect a float.
            values[field] = type(getattr(plain, field))(value)

    values["planner_mode"] = values["planner_mode"].lower()
    if values["planner_mode"] not in PLANNER_MODES:
        raise ValueError(
            f"stack_config.planning.mode must be one of {PLANNER_MODES}, "
            f"got {values['planner_mode']!r}"
        )
    config = PipelineConfig(
        **values,
        camera=_load_camera(section(raw, "camera")),
        grid=_load_grid(section(section(raw, "planning"), "grid")),
    )
    if config.gauge_confidence_good <= config.gate_min_confidence:
        raise ValueError(
            "stack_config.dashboard.confidence_good must be above "
            "planning.gate.confidence_threshold"
        )
    if config.gauge_road_full_scale <= config.safety_min_road_fraction:
        raise ValueError(
            "stack_config.dashboard.road_bar_full_scale must exceed safety.min_road_fraction"
        )
    return config


def load_pipeline_config_file(path: str | Path) -> PipelineConfig:
    """The stack settings a run config file resolves to, for offline tools
    that drive parts of this stack without the harness."""
    harness = load_config(path)
    if harness.stack.partition(":")[0] != __package__:
        raise ValueError(f"{path} runs the stack {harness.stack!r}, not {__package__}")
    return load_pipeline_config(harness.stack_config)
