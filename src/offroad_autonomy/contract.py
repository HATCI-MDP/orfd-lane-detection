"""The only types the harness and a driving stack share.

A stack declares the sensors it needs, turns each ``Observation`` into a
``StackOutput`` and can be reset. The harness knows nothing else about it, so
any perception, planning and control approach runs under the same loop,
safety, dashboard and scoring.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from offroad_autonomy.types import CameraSensor, CameraSpec, ControlCommand, VehicleState

#: What the simulator adapter can attach today; a new kind needs one adapter there.
SENSOR_KINDS = ("camera",)

#: The orbit camera is the harness's own, attached only for presentations.
RESERVED_SENSOR_NAMES = ("orbit",)

_CAMERA_PARAMS = ("pos", "dir", "up", "model", "width", "height", "fov_x_deg", "target_fps")

POLYLINE_FRAMES = ("image", "ground")
POLYLINE_STYLES = ("path", "held", "candidate")
MASK_ROLES = ("traversable", "excluded", "info")
STATUS_LEVELS = ("good", "warn", "bad")


@dataclass(frozen=True)
class SensorSpec:
    """One sensor a stack asks the simulator to attach.

    ``params`` is plain data so the stack never builds simulator objects; the
    simulator adapter validates it and refuses a kind it cannot attach.
    """

    name: str
    kind: str
    params: dict = field(default_factory=dict)


def camera_sensor(spec: CameraSpec) -> SensorSpec:
    sensor = spec.sensor
    return SensorSpec(
        name=spec.name,
        kind="camera",
        params={
            "pos": tuple(spec.pos),
            "dir": tuple(spec.dir),
            "up": tuple(spec.up),
            "model": sensor.model,
            "width": sensor.width,
            "height": sensor.height,
            "fov_x_deg": sensor.fov_x_deg,
            "target_fps": sensor.target_fps,
        },
    )


def camera_spec(sensor: SensorSpec) -> CameraSpec:
    """The mount and lens of a camera ``SensorSpec``.

    A wrong or missing parameter fails here, before a map is loaded, rather
    than as a camera rendering the wrong view.
    """
    if sensor.kind != "camera":
        raise ValueError(f"Sensor '{sensor.name}' is a {sensor.kind!r}, not a camera")
    given = set(sensor.params)
    expected = set(_CAMERA_PARAMS)
    if given != expected:
        raise ValueError(
            f"Camera sensor '{sensor.name}': missing {sorted(expected - given) or 'nothing'}, "
            f"unknown {sorted(given - expected) or 'nothing'}"
        )
    params = sensor.params
    lens = CameraSensor(
        model=str(params["model"]),
        width=int(params["width"]),
        height=int(params["height"]),
        fov_x_deg=float(params["fov_x_deg"]),
        target_fps=float(params["target_fps"]),
    )
    problem = ""
    if any(len(params[key]) != 3 for key in ("pos", "dir", "up")):
        problem = "pos, dir and up must have 3 components"
    elif lens.width < 1 or lens.height < 1:
        problem = "width and height must be positive"
    elif not 0.0 < lens.fov_x_deg < 180.0:
        problem = "fov_x_deg must be between 0 and 180"
    elif lens.target_fps <= 0.0:
        problem = "target_fps must be > 0"
    if problem:
        raise ValueError(f"Camera sensor '{sensor.name}': {problem}")
    return CameraSpec(
        name=sensor.name,
        pos=tuple(float(value) for value in params["pos"]),
        dir=tuple(float(value) for value in params["dir"]),
        up=tuple(float(value) for value in params["up"]),
        sensor=lens,
    )


def validate_sensors(sensors: list[SensorSpec]) -> None:
    names = [sensor.name for sensor in sensors]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Sensor names must be unique, repeated: {duplicates}")
    for sensor in sensors:
        if sensor.name in RESERVED_SENSOR_NAMES:
            raise ValueError(f"Sensor name '{sensor.name}' is reserved for the harness")
        if sensor.kind not in SENSOR_KINDS:
            raise ValueError(
                f"Sensor '{sensor.name}' has kind {sensor.kind!r}; the simulator adapter "
                f"supports {SENSOR_KINDS}"
            )
        if sensor.kind == "camera":
            camera_spec(sensor)


@dataclass
class Observation:
    """One frame of input: every declared sensor, read in the same loop iteration."""

    sensors: dict[str, np.ndarray]
    vehicle: VehicleState
    timestamp: float
    frame_id: int = 0


# ----------------------------------------------------------------- visuals


def _check_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    """Checked where the stack builds the visual, on the loop thread, so a
    typo stops the run at its first frame instead of blanking the dashboard
    from a thread whose errors are only logged."""
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")


@dataclass(frozen=True)
class ImageLayer:
    """A page of the dashboard. The first layer a stack emits is the main view.

    ``camera`` names the sensor the image was taken by, so ground polylines
    can be projected onto it; leave it empty for images that are not a
    camera view, such as a grid or a depth map.

    ``image`` may be a function returning the image. The dashboard calls it
    on its own thread and only while that page is shown, so a debug image
    that is costly to draw costs the control loop nothing. It must read only
    data this step produced, never state the next step will change.
    """

    name: str
    image: np.ndarray | Callable[[], np.ndarray]
    camera: str = ""

    def pixels(self) -> np.ndarray:
        if callable(self.image):
            return self.image()
        return self.image


@dataclass(frozen=True)
class MaskOverlay:
    layer: str
    mask: np.ndarray
    role: str = "traversable"
    label: str = ""

    def __post_init__(self) -> None:
        _check_choice("MaskOverlay.role", self.role, MASK_ROLES)


@dataclass(frozen=True)
class Polyline:
    """A path drawn on a layer.

    ``image`` points are pixels of the layer's image. ``ground`` points are
    ``(forward, right)`` metres on flat ground from the layer's camera, and
    the harness projects them with that camera's geometry.
    """

    layer: str
    points: np.ndarray
    frame: str = "image"
    style: str = "path"
    label: str = ""

    def __post_init__(self) -> None:
        _check_choice("Polyline.frame", self.frame, POLYLINE_FRAMES)
        _check_choice("Polyline.style", self.style, POLYLINE_STYLES)


@dataclass(frozen=True)
class Gauge:
    """A bar on the stack card.

    ``thresholds`` holds one value (below it is bad) or two (below the first
    is bad, below the second is a warning); the bar's tick marks the first.
    """

    name: str
    value: float
    range: tuple[float, float] = (0.0, 1.0)
    thresholds: tuple[float, ...] = ()
    caption: str = ""
    text: str = ""


@dataclass(frozen=True)
class Status:
    """A named condition. Warnings and faults show as chips; good ones only
    in the readout, so the card is quiet when nothing is wrong."""

    text: str
    level: str = "good"
    detail: str = ""

    def __post_init__(self) -> None:
        _check_choice("Status.level", self.level, STATUS_LEVELS)


Visual = ImageLayer | MaskOverlay | Polyline | Gauge | Status


@dataclass
class StackOutput:
    command: ControlCommand
    visuals: list[Visual] = field(default_factory=list)
    #: Reported in a separate benchmark section that never feeds the comparison.
    metrics: dict[str, float] = field(default_factory=dict)
    #: Non-empty asks the harness for its safe stop, for a condition only the
    #: stack can see (no road in view). The harness parks the vehicle and
    #: waits for the operator, as it does for its own triggers.
    safe_stop: str = ""


@runtime_checkable
class DrivingStack(Protocol):
    def sensors(self) -> list[SensorSpec]:
        """Which sensors to attach and where; read once, before connecting."""
        ...

    def step(self, obs: Observation) -> StackOutput:
        """Must return within the loop budget: slow work belongs on the
        stack's own worker, with ``step`` returning its latest result."""
        ...

    def reset(self) -> None:
        """Called when the operator resumes after a safe stop: the vehicle
        may have been moved by hand, so temporal state is stale."""
        ...


def load_stack(path: str, stack_config: dict) -> DrivingStack:
    """Build the stack named by ``module:function`` from its own config section.

    A dotted path keeps the harness from listing stacks: adding one is adding
    a module, with no registry to update.
    """
    module_name, _, attr = path.partition(":")
    if not module_name or not attr:
        raise ValueError(f"stack must be 'package.module:function', got {path!r}")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ValueError(f"stack module {module_name!r} cannot be imported: {exc}") from exc
    factory: Callable[[dict], DrivingStack] | None = getattr(module, attr, None)
    if factory is None or not callable(factory):
        raise ValueError(f"stack module {module_name!r} has no callable {attr!r}")
    stack = factory(dict(stack_config))
    if not isinstance(stack, DrivingStack):
        raise TypeError(f"{path} returned {type(stack).__name__}, which lacks sensors/step/reset")
    sensors = stack.sensors()
    if not sensors:
        raise ValueError(f"{path} declared no sensors; a stack needs at least one input")
    validate_sensors(sensors)
    return stack


def finite_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Metrics a report can average; ``inf`` and ``nan`` mean "not measured"."""
    clean = {}
    for name, value in metrics.items():
        number = float(value)
        if math.isfinite(number):
            clean[name] = number
    return clean
