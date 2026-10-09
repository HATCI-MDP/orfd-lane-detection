"""The operator dashboard, drawn from generic visual primitives.

It never sees a stack's internal types: a stack describes itself with
``ImageLayer``, ``MaskOverlay``, ``Polyline``, ``Gauge`` and ``Status``, and
the harness adds the panels every stack shares (vehicle, safe stop, loop
rate and latency). A stack written next year renders with no change here.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import NamedTuple

import cv2
import numpy as np

from offroad_autonomy.contract import (
    Gauge,
    ImageLayer,
    MaskOverlay,
    Polyline,
    SensorSpec,
    StackOutput,
    Status,
    camera_spec,
)
from offroad_autonomy.perception.camera_geometry import CameraModel
from offroad_autonomy.types import DEFAULT_DASHBOARD_COLORS, DashboardThresholds
from offroad_autonomy.visualization import bands
from offroad_autonomy.visualization.layout import Rect, build_layout
from offroad_autonomy.visualization.text import (
    BODY,
    BODY_STRONG,
    CAPTION,
    CAPTION_MONO,
    DISPLAY,
    HERO,
    HERO_SANS,
    LABEL,
    LARGE,
    LARGE_SANS,
    TITLE,
    VALUE,
    TextRenderer,
    TextStyle,
)
from offroad_autonomy.visualization.widgets import (
    CoverTransform,
    as_bgr,
    as_mask,
    blend_rect,
    blit_cover,
    chip_height,
    chip_width,
    cover_point,
    draw_autonomy_chip,
    draw_bar,
    draw_centered_bar,
    draw_chip,
    fill_rect,
    fit_image,
    outline_rect,
    smooth_points,
    wrap_text,
)

logger = logging.getLogger("offroad_autonomy.visualization.dashboard")

_ALL_STYLES: tuple[TextStyle, ...] = (
    DISPLAY,
    HERO,
    HERO_SANS,
    LARGE,
    LARGE_SANS,
    TITLE,
    VALUE,
    BODY,
    BODY_STRONG,
    LABEL,
    CAPTION,
    CAPTION_MONO,
)

# Row positions inside the cards, measured down from the card's top edge. The
# static layer and the per-frame values read the same numbers, so a label and
# its value cannot drift apart.
_CARD_TITLE_BASELINE = 30
_CARD_INSET = 20
_SPEED_BASELINE = 128
_STEER_BASELINE = 172
_STEER_BAR_TOP = 184
_PEDAL_BASELINE = 246
_PEDAL_BAR_TOP = 258
_GAUGE_BASELINE = 64
_GAUGE_PITCH = 82
_GAUGE_BAR_GAP = 12
_GAUGE_CAPTION_GAP = 44
#: Gauges past this many are listed in the readout only, so the status
#: chips keep their room on the card.
_CARD_GAUGES = 3
_STATUS_GAP = 40
_REASON_LINE_HEIGHT = 22
_CARD_BOTTOM_MARGIN = 14

_TILE_TITLE_BASELINE = 26
_TILE_VALUE_BASELINE = 76
_TILE_BAR_TOP = 90
_TILE_BAR_H = 8
_DASHBOARD_TILE_VALUE_BASELINE = 68
_DASHBOARD_TILE_NOTE_BASELINE = 94

# Frame pixels below the top edge of an excluded region.
_EXCLUDED_LABEL_DROP_PX = 22

_READOUT_W = 330
_READOUT_ROW_H = 22

_STATUS_COLORS = {"good": "GOOD", "warn": "WARN", "bad": "BAD"}
_ROLE_COLORS = {"traversable": "MASK_FILL", "excluded": "EGO_EXCLUDED", "info": "TEXT_SECONDARY"}
_STYLE_COLORS = {"path": "PATH_CORE", "held": "WARN", "candidate": "TEXT_SECONDARY"}

#: Unit suffixes of metric names, shown after the value instead of in the label.
_UNIT_SUFFIXES = (
    ("_mps2", "m/s2"),
    ("_mps", "m/s"),
    ("_per_m", "/m"),
    ("_per_s", "/s"),
    ("_deg", "deg"),
    ("_ms", "ms"),
    ("_m", "m"),
)


class MainView(NamedTuple):
    """The main page's image with overlays drawn on, before scaling.

    ``labels`` are text and frame-pixel anchors left for whoever scales the
    image, since text drawn here would be scaled with it.
    """

    image: np.ndarray
    labels: list[tuple[str, tuple[int, int]]]


@dataclass
class DashboardTelemetry:
    """The harness's own readouts, the same for every stack."""

    speed_mph: float
    steering: float
    throttle: float
    brake: float
    fps: float
    latency_ms: float
    latency_p95_ms: float = 0.0
    autopilot_active: bool = True
    safe_stop_reason: str = ""
    #: Never an autonomy number: it can drop to single figures without the
    #: vehicle noticing.
    dashboard_fps: float = 0.0
    timing_lines: list[str] = field(default_factory=list)


@dataclass
class Scene:
    """A ``StackOutput``'s visuals sorted by kind, in the order emitted."""

    layers: list[ImageLayer]
    masks: dict[str, list[MaskOverlay]]
    polylines: dict[str, list[Polyline]]
    gauges: list[Gauge]
    statuses: list[Status]

    @classmethod
    def of(cls, output: StackOutput) -> Scene:
        scene = cls([], {}, {}, [], [])
        for visual in output.visuals:
            if isinstance(visual, ImageLayer):
                scene.layers.append(visual)
            elif isinstance(visual, MaskOverlay):
                scene.masks.setdefault(visual.layer, []).append(visual)
            elif isinstance(visual, Polyline):
                scene.polylines.setdefault(visual.layer, []).append(visual)
            elif isinstance(visual, Gauge):
                scene.gauges.append(visual)
            elif isinstance(visual, Status):
                scene.statuses.append(visual)
        return scene

    def alerts(self) -> list[Status]:
        return [status for status in self.statuses if status.level != "good"]


def harness_statuses(telemetry: DashboardTelemetry, scene: Scene) -> list[Status]:
    """Chips for the stack card: the harness's safe stop first, then the
    stack's warnings and faults."""
    chips = []
    if not telemetry.autopilot_active:
        chips.append(Status("SAFE STOP", "bad", telemetry.safe_stop_reason))
    return chips + scene.alerts()


def metric_row(name: str, value: float) -> tuple[str, str]:
    unit = ""
    label = name
    for suffix, text in _UNIT_SUFFIXES:
        if name.endswith(suffix):
            label = name[: -len(suffix)]
            unit = " " + text
            break
    label = label.replace("_", " ").title()
    number = float(value)
    if not math.isfinite(number):
        return label, "n/a"
    return label, f"{number:+.3f}{unit}"


class AutonomyDashboard:
    def __init__(
        self,
        width: int = 1600,
        height: int = 900,
        colors: dict[str, tuple[int, int, int]] | None = None,
        sensors: list[SensorSpec] | None = None,
        thresholds: DashboardThresholds | None = None,
        stack_name: str = "",
    ) -> None:
        self.width = width
        self.height = height
        if thresholds is None:
            thresholds = DashboardThresholds()
        self._thresholds = thresholds
        self._colors = DEFAULT_DASHBOARD_COLORS.copy()
        if colors is not None:
            self._colors.update(colors)
        self._cameras = {}
        for sensor in sensors or []:
            if sensor.kind == "camera":
                self._cameras[sensor.name] = camera_spec(sensor)
        self._camera_models: dict[tuple[str, int, int], CameraModel] = {}
        self._stack_name = stack_name
        self._layout = build_layout(width, height)
        # Loaded here so a missing font stops startup, not the first frame.
        self._text = TextRenderer()
        self._text.preload(_ALL_STYLES)
        self._status_label_w = self._text.width("Status", BODY)
        self._excluded_cache: dict[tuple, tuple[np.ndarray, list, tuple[int, int] | None]] = {}
        self._warned_layers: set[str] = set()
        # Filling a large array with a colour costs ~5 ms at 1600x900;
        # copying a cached blank costs ~1 ms, so everything that never changes
        # between frames is drawn once into one image and copied.
        self._static = self._build_static()

    # ------------------------------------------------------------------ frame

    def render(
        self,
        output: StackOutput,
        telemetry: DashboardTelemetry,
        page: int = 0,
        timing_overlay: bool = False,
    ) -> np.ndarray:
        scene = Scene.of(output)
        if page >= len(scene.layers):
            page = 0

        canvas = self._static.copy()
        draw_autonomy_chip(
            canvas, self._text, self._colors, self._layout.header, 20, telemetry.autopilot_active
        )

        viewport = self._layout.viewport
        if not scene.layers:
            self._draw_no_image(canvas, viewport)
        elif page == 0:
            view = self.main_view(scene, telemetry.autopilot_active)
            transform = blit_cover(canvas, view.image, viewport)
            self._draw_labels(canvas, transform, view.labels)
            self._draw_mask_inset(canvas, viewport, scene)
            self._draw_legend(canvas, viewport, self.legend(scene, telemetry.autopilot_active))
            alerts = scene.alerts()
            if alerts:
                self._draw_alert_chip(canvas, viewport, alerts[0])
        else:
            self._draw_debug_page(canvas, viewport, scene, output, page, telemetry)
        if scene.layers:
            self._draw_viewport_label(canvas, viewport, scene.layers[page], page)
        if timing_overlay and telemetry.timing_lines:
            self._draw_timing_overlay(canvas, viewport, telemetry.timing_lines)
        if not telemetry.autopilot_active:
            self._draw_safe_stop_overlay(canvas, viewport)

        self._draw_vehicle_card(canvas, telemetry)
        self._draw_stack_card(canvas, scene, telemetry)
        self._draw_runtime_strip(canvas, telemetry)
        return canvas

    def main_view(self, scene: Scene, autopilot_active: bool) -> MainView:
        """The main page's image, for other layouts to scale.

        Polylines are left off under safe stop: the stack keeps running so
        the operator can see the road come back, but nothing should suggest
        it is steering while it is not."""
        return self._compose(scene.layers[0], scene, draw_paths=autopilot_active)

    def legend(self, scene: Scene, autopilot_active: bool) -> list[tuple[str, tuple]]:
        """Swatches for the main page, in the order the stack emitted them,
        with no path entry when no path is drawn."""
        if not scene.layers:
            return []
        name = scene.layers[0].name
        entries = []
        for overlay in scene.masks.get(name, []):
            if overlay.label:
                entries.append((overlay.label, self._colors[_ROLE_COLORS[overlay.role]]))
        if autopilot_active:
            for line in scene.polylines.get(name, []):
                if line.label:
                    entries.append((line.label, self._colors[_STYLE_COLORS[line.style]]))
        return entries

    # ----------------------------------------------------------- static layer

    def _build_static(self) -> np.ndarray:
        colors = self._colors
        layout = self._layout
        canvas = np.full((self.height, self.width, 3), colors["BG"], dtype=np.uint8)

        header = layout.header
        fill_rect(canvas, header, colors["PANEL_BG"])
        outline_rect(canvas, header, colors["CARD_BORDER"])
        title_w = self._text.draw(
            canvas,
            "OFF-ROAD AUTONOMY",
            header.x + 20,
            header.y + 36,
            TITLE,
            colors["TEXT_PRIMARY"],
        )
        subtitle = "Front Path View"
        if self._stack_name:
            subtitle = f"Stack {self._stack_name}"
        self._text.draw(
            canvas,
            subtitle,
            header.x + 20 + title_w + 16,
            header.y + 35,
            BODY,
            colors["TEXT_SECONDARY"],
        )

        self._static_card(canvas, layout.vehicle, "Vehicle")
        vehicle = layout.vehicle
        self._text.draw(
            canvas,
            "Steering",
            vehicle.x + _CARD_INSET,
            vehicle.y + _STEER_BASELINE,
            BODY,
            colors["TEXT_SECONDARY"],
        )
        half_gap = 20
        column_w = (vehicle.w - 2 * _CARD_INSET - half_gap) // 2
        self._text.draw(
            canvas,
            "Throttle",
            vehicle.x + _CARD_INSET,
            vehicle.y + _PEDAL_BASELINE,
            BODY,
            colors["TEXT_SECONDARY"],
        )
        self._text.draw(
            canvas,
            "Brake",
            vehicle.x + _CARD_INSET + column_w + half_gap,
            vehicle.y + _PEDAL_BASELINE,
            BODY,
            colors["TEXT_SECONDARY"],
        )

        self._static_card(canvas, layout.stack, "Driving Stack")

        for rect, title in (
            (layout.autonomy_fps, "Autonomy FPS"),
            (layout.latency, "Autonomy Latency Mean / P95"),
            (layout.dashboard_fps, "Dashboard FPS"),
        ):
            fill_rect(canvas, rect, colors["CARD_BG"])
            outline_rect(canvas, rect, colors["CARD_BORDER"])
            self._text.draw(
                canvas,
                title.upper(),
                rect.x + _CARD_INSET,
                rect.y + _TILE_TITLE_BASELINE,
                LABEL,
                colors["TEXT_SECONDARY"],
            )
        # The note that keeps a slow window from posing as a slow vehicle.
        self._text.draw(
            canvas,
            "Display Only, Not The Vehicle",
            layout.dashboard_fps.x + _CARD_INSET,
            layout.dashboard_fps.y + _DASHBOARD_TILE_NOTE_BASELINE,
            CAPTION,
            colors["TEXT_SECONDARY"],
        )
        return canvas

    def _static_card(self, canvas: np.ndarray, rect: Rect, title: str) -> None:
        fill_rect(canvas, rect, self._colors["CARD_BG"])
        outline_rect(canvas, rect, self._colors["CARD_BORDER"])
        self._text.draw(
            canvas,
            title.upper(),
            rect.x + _CARD_INSET,
            rect.y + _CARD_TITLE_BASELINE,
            LABEL,
            self._colors["TEXT_SECONDARY"],
        )

    # ----------------------------------------------------------------- header

    # --------------------------------------------------------------- viewport

    def _draw_viewport_label(
        self, canvas: np.ndarray, rect: Rect, layer: ImageLayer, page: int
    ) -> None:
        if page == 0:
            text = layer.name
            spec = self._cameras.get(layer.camera)
            if spec is not None:
                text = f"{layer.name} {spec.sensor.model} {spec.sensor.fov_x_deg:.0f} Deg"
            color = self._colors["TEXT_PRIMARY"]
        else:
            text = f"[{page}] {layer.name} (0 = Back)"
            color = self._colors["PATH_CORE"]
        self._draw_plate_text(canvas, rect.x + 16, rect.y + 16, text, BODY_STRONG, color)

    def _draw_plate_text(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        text: str,
        style: TextStyle,
        color: tuple[int, int, int],
    ) -> None:
        """Text on a dark translucent plate, so it reads over any scene."""
        cap = self._text.cap_height(style)
        plate = Rect(x, y, self._text.width(text, style) + 24, cap + 20)
        blend_rect(canvas, plate, self._colors["BG"], 0.8)
        self._text.draw(canvas, text, x + 12, y + 10 + cap, style, color)

    def _draw_no_image(self, canvas: np.ndarray, viewport: Rect) -> None:
        fill_rect(canvas, viewport, self._colors["PANEL_BG"])
        self._text.draw(
            canvas,
            "No Image From The Stack",
            viewport.x + viewport.w // 2,
            viewport.y + viewport.h // 2,
            LARGE_SANS,
            self._colors["TEXT_SECONDARY"],
            align="center",
        )

    def _draw_mask_inset(self, canvas: np.ndarray, viewport: Rect, scene: Scene) -> None:
        masks = [
            overlay
            for overlay in scene.masks.get(scene.layers[0].name, [])
            if overlay.role == "traversable"
        ]
        if not masks:
            return
        inset_w, inset_h = 172, 104
        inset = Rect(viewport.x + 16, viewport.bottom - 16 - inset_h, inset_w, inset_h)
        self._text.draw(
            canvas,
            "MASK",
            inset.x,
            inset.y - 8,
            LABEL,
            self._colors["TEXT_PRIMARY"],
        )
        shape = masks[0].mask.shape[:2]
        image = np.full(shape + (3,), self._colors["CARD_BG"], dtype=np.uint8)
        for overlay in masks:
            mask = as_mask(overlay.mask, shape)
            image[mask] = self._colors["MASK_FILL"]
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                image, contours, -1, self._colors["MASK_EDGE"], 2, lineType=cv2.LINE_AA
            )
        fitted = fit_image(image, inset.w, inset.h, fill=self._colors["CARD_BG"])
        canvas[inset.y : inset.bottom, inset.x : inset.right] = fitted
        outline_rect(canvas, inset, self._colors["CARD_BORDER"])

    def _draw_legend(
        self, canvas: np.ndarray, viewport: Rect, entries: list[tuple[str, tuple]]
    ) -> None:
        if not entries:
            return
        swatch = 14
        swatch_gap = 8
        entry_gap = 20
        pad_x = 16
        cap = self._text.cap_height(BODY)
        widths = [swatch + swatch_gap + self._text.width(label, BODY) for label, _ in entries]
        plate_w = sum(widths) + entry_gap * (len(entries) - 1) + 2 * pad_x
        plate_h = cap + 24
        plate = Rect(
            viewport.right - 16 - plate_w,
            viewport.bottom - 16 - plate_h,
            plate_w,
            plate_h,
        )
        blend_rect(canvas, plate, self._colors["BG"], 0.8)
        x = plate.x + pad_x
        baseline = plate.y + 12 + cap
        for (label, color), width in zip(entries, widths, strict=True):
            fill_rect(canvas, Rect(x, baseline - swatch + 2, swatch, swatch), color)
            self._text.draw(
                canvas,
                label,
                x + swatch + swatch_gap,
                baseline,
                BODY,
                self._colors["TEXT_PRIMARY"],
            )
            x += width + entry_gap

    def _draw_alert_chip(self, canvas: np.ndarray, viewport: Rect, status: Status) -> None:
        text = status.text.upper()
        width = chip_width(self._text, text)
        draw_chip(
            canvas,
            self._text,
            viewport.x + (viewport.w - width) // 2,
            viewport.y + 20,
            text,
            self._colors[_STATUS_COLORS[status.level]],
            self._colors["BG"],
            solid=True,
        )

    def _draw_labels(
        self,
        canvas: np.ndarray,
        transform: CoverTransform,
        labels: list[tuple[str, tuple[int, int]]],
    ) -> None:
        for text, anchor in labels:
            x, y = cover_point(transform, anchor)
            self._text.draw(
                canvas,
                text,
                x,
                y,
                BODY_STRONG,
                self._colors["TEXT_PRIMARY"],
                align="center",
            )

    def _draw_timing_overlay(self, canvas: np.ndarray, viewport: Rect, lines: list[str]) -> None:
        row_h = 17
        plate_w = 360
        top = viewport.y + 60
        plate_h = min(viewport.h - 120, 16 + row_h * len(lines))
        plate = Rect(viewport.x + 16, top, plate_w, plate_h)
        blend_rect(canvas, plate, self._colors["BG"], 0.8)
        for index, line in enumerate(lines):
            baseline = top + 20 + row_h * index
            if baseline > plate.bottom - 4:
                break
            self._text.draw(
                canvas, line, plate.x + 10, baseline, CAPTION_MONO, self._colors["TEXT_PRIMARY"]
            )

    def _draw_safe_stop_overlay(self, canvas: np.ndarray, viewport: Rect) -> None:
        colors = self._colors
        blend_rect(canvas, viewport, colors["BAD"], 0.22)
        outline_rect(canvas, viewport, colors["BAD"], 6)

        banner_w, banner_h = 700, 190
        banner = Rect(
            viewport.x + (viewport.w - banner_w) // 2,
            viewport.y + (viewport.h - banner_h) // 2 - 40,
            banner_w,
            banner_h,
        )
        fill_rect(canvas, banner, colors["CARD_BG"])
        outline_rect(canvas, banner, colors["BAD"], 3)
        centre = banner.x + banner.w // 2
        self._text.draw(
            canvas, "SAFE STOP", centre, banner.y + 70, HERO_SANS, colors["BAD"], align="center"
        )
        self._text.draw(
            canvas,
            "Manual Control Required",
            centre,
            banner.y + 118,
            LARGE_SANS,
            colors["TEXT_PRIMARY"],
            align="center",
        )
        self._text.draw(
            canvas,
            "W/A/S/D Drive  |  Space Brake  |  P Resume Autopilot",
            centre,
            banner.y + 160,
            BODY,
            colors["TEXT_SECONDARY"],
            align="center",
        )

    # ------------------------------------------------------------- debug page

    def _draw_debug_page(
        self,
        canvas: np.ndarray,
        viewport: Rect,
        scene: Scene,
        output: StackOutput,
        page: int,
        telemetry: DashboardTelemetry,
    ) -> None:
        """One of the stack's own layers, beside a readout of everything it
        reported this frame."""
        colors = self._colors
        fill_rect(canvas, viewport, colors["PANEL_BG"])
        # Below the page chip, which sits over the top-left corner.
        chip_clearance = 64
        region = Rect(
            viewport.x,
            viewport.y + chip_clearance,
            viewport.w - _READOUT_W - 16,
            viewport.h - chip_clearance,
        )
        view = self._compose(scene.layers[page], scene, draw_paths=telemetry.autopilot_active)
        canvas[region.y : region.bottom, region.x : region.right] = fit_image(
            view.image, region.w, region.h, colors["PANEL_BG"], cv2.INTER_AREA
        )

        readout = Rect(viewport.right - _READOUT_W, viewport.y, _READOUT_W, viewport.h)
        fill_rect(canvas, readout, colors["CARD_BG"])
        outline_rect(canvas, readout, colors["CARD_BORDER"])
        left = readout.x + 16
        right = readout.right - 16
        baseline = readout.y + 28
        for label, value in self.readout(scene, output):
            if baseline > readout.bottom - 8:
                break
            if value is None:
                baseline += 8
                self._text.draw(
                    canvas, label.upper(), left, baseline, LABEL, colors["TEXT_SECONDARY"]
                )
            else:
                label_w = self._text.draw(
                    canvas, label, left, baseline, CAPTION, colors["TEXT_SECONDARY"]
                )
                value = self._fit_text(value, right - left - label_w - 12, CAPTION_MONO)
                self._text.draw(
                    canvas,
                    value,
                    right,
                    baseline,
                    CAPTION_MONO,
                    colors["TEXT_PRIMARY"],
                    align="right",
                )
            baseline += _READOUT_ROW_H

    @staticmethod
    def readout(scene: Scene, output: StackOutput) -> list[tuple[str, str | None]]:
        """Label and value rows. A ``None`` value is a heading.

        Timings (``*_ms``) are left to the timing overlay, which lists them
        beside the harness's own stages."""
        rows: list[tuple[str, str | None]] = []
        if scene.statuses:
            rows.append(("Status", None))
            for status in scene.statuses:
                rows.append((status.text, status.detail or status.level.title()))
        if scene.gauges:
            rows.append(("Gauges", None))
            for gauge in scene.gauges:
                rows.append((gauge.name, gauge.text or f"{gauge.value:.3f}"))
        metrics = [(n, v) for n, v in output.metrics.items() if not n.endswith("_ms")]
        if metrics:
            rows.append(("Metrics", None))
            rows += [metric_row(name, value) for name, value in metrics]
        return rows

    # ---------------------------------------------------------------- vehicle

    def _draw_vehicle_card(self, canvas: np.ndarray, telemetry: DashboardTelemetry) -> None:
        colors = self._colors
        card = self._layout.vehicle
        left = card.x + _CARD_INSET
        inner_w = card.w - 2 * _CARD_INSET

        speed = f"{telemetry.speed_mph:.1f}"
        speed_w = self._text.draw(
            canvas, speed, left, card.y + _SPEED_BASELINE, DISPLAY, colors["TEXT_PRIMARY"]
        )
        self._text.draw(
            canvas,
            "mph",
            left + speed_w + 12,
            card.y + _SPEED_BASELINE,
            TITLE,
            colors["TEXT_SECONDARY"],
        )

        direction = "Right"
        if telemetry.steering < 0.0:
            direction = "Left"
        right = card.right - _CARD_INSET
        word_w = self._text.draw(
            canvas,
            direction,
            right,
            card.y + _STEER_BASELINE,
            CAPTION,
            colors["TEXT_SECONDARY"],
            align="right",
        )
        self._text.draw(
            canvas,
            f"{telemetry.steering:+.2f}",
            right - word_w - 8,
            card.y + _STEER_BASELINE,
            VALUE,
            colors["TEXT_PRIMARY"],
            align="right",
        )
        draw_centered_bar(
            canvas,
            Rect(left, card.y + _STEER_BAR_TOP, inner_w, 10),
            telemetry.steering,
            colors["TEXT_PRIMARY"],
            colors["MUTED_LINE"],
            colors["TEXT_SECONDARY"],
        )

        half_gap = 20
        column_w = (inner_w - half_gap) // 2
        for index, (value, color) in enumerate(
            ((telemetry.throttle, colors["GOOD"]), (telemetry.brake, colors["BAD"]))
        ):
            x = left + index * (column_w + half_gap)
            self._text.draw(
                canvas,
                f"{value:.2f}",
                x + column_w,
                card.y + _PEDAL_BASELINE,
                VALUE,
                colors["TEXT_PRIMARY"],
                align="right",
            )
            draw_bar(
                canvas,
                Rect(x, card.y + _PEDAL_BAR_TOP, column_w, 10),
                value,
                color,
                colors["MUTED_LINE"],
            )

    # ------------------------------------------------------------------ stack

    def gauge_color(self, gauge: Gauge) -> tuple[int, int, int]:
        level = bands.gauge_level(gauge.value, gauge.thresholds)
        if level is None:
            return self._colors["TEXT_PRIMARY"]
        return self._colors[level]

    @staticmethod
    def gauge_fraction(gauge: Gauge, value: float) -> float:
        low, high = gauge.range
        span = high - low
        if span <= 0.0:
            return 0.0
        return (value - low) / span

    def _draw_stack_card(
        self, canvas: np.ndarray, scene: Scene, telemetry: DashboardTelemetry
    ) -> None:
        colors = self._colors
        card = self._layout.stack
        left = card.x + _CARD_INSET
        right = card.right - _CARD_INSET
        inner_w = card.w - 2 * _CARD_INSET

        gauges = scene.gauges[:_CARD_GAUGES]
        baseline = card.y + _GAUGE_BASELINE
        status_baseline = baseline
        for gauge in gauges:
            color = self.gauge_color(gauge)
            label_w = self._text.draw(
                canvas, gauge.name, left, baseline, BODY, colors["TEXT_SECONDARY"]
            )
            text = gauge.text or f"{gauge.value:.2f}"
            text = self._fit_text(text, inner_w - label_w - 12, VALUE)
            self._text.draw(canvas, text, right, baseline, VALUE, color, align="right")
            tick = None
            if gauge.thresholds:
                tick = self.gauge_fraction(gauge, gauge.thresholds[0])
            bar_color = color
            if not gauge.thresholds:
                bar_color = colors["TEXT_SECONDARY"]
            draw_bar(
                canvas,
                Rect(left, baseline + _GAUGE_BAR_GAP, inner_w, 10),
                self.gauge_fraction(gauge, gauge.value),
                bar_color,
                colors["MUTED_LINE"],
                tick=tick,
                tick_color=colors["TEXT_PRIMARY"],
            )
            if gauge.caption:
                self._text.draw(
                    canvas,
                    gauge.caption,
                    left,
                    baseline + _GAUGE_CAPTION_GAP,
                    CAPTION,
                    colors["TEXT_SECONDARY"],
                )
            status_baseline = baseline + _GAUGE_CAPTION_GAP + _STATUS_GAP
            baseline += _GAUGE_PITCH

        self._text.draw(canvas, "Status", left, status_baseline, BODY, colors["TEXT_SECONDARY"])
        chips = harness_statuses(telemetry, scene)
        chips_bottom = self._draw_status_chips(canvas, card, status_baseline, chips)
        details = "; ".join(status.detail.strip() for status in chips if status.detail.strip())
        self._draw_reason(canvas, card, chips_bottom, details)

    def _draw_status_chips(
        self, canvas: np.ndarray, card: Rect, baseline: int, chips: list[Status]
    ) -> int:
        """One chip per active condition, wrapping, since a run can collect
        more than fit on one row."""
        if not chips:
            chips = [Status("NONE", "good")]
        chip_h = chip_height(self._text)
        row_start = card.x + _CARD_INSET
        row_end = card.right - _CARD_INSET
        x = row_start + self._status_label_w + 12
        # The label's baseline is the chip text's baseline, so the chip sits
        # level with it.
        y = baseline - 10 - self._text.cap_height(BODY_STRONG)
        for status in chips:
            text = status.text.upper()
            width = chip_width(self._text, text)
            if x + width > row_end and x > row_start:
                x = row_start
                y += chip_h + 8
            color = self._colors[_STATUS_COLORS[status.level]]
            draw_chip(canvas, self._text, x, y, text, color, self._colors["BG"])
            x += width + 8
        return y + chip_h

    def _draw_reason(self, canvas: np.ndarray, card: Rect, chips_bottom: int, reason: str) -> None:
        left = card.x + _CARD_INSET
        if not reason:
            return
        color = self._colors["TEXT_PRIMARY"]
        lines = wrap_text(self._text, reason, BODY, card.w - 2 * _CARD_INSET)
        baseline = chips_bottom + 26
        room = (card.bottom - _CARD_BOTTOM_MARGIN - baseline) // _REASON_LINE_HEIGHT + 1
        room = max(room, 1)
        if len(lines) > room:
            lines = lines[:room]
            lines[-1] = self._ellipsize(lines[-1], card.w - 2 * _CARD_INSET, BODY)
        for index, line in enumerate(lines):
            self._text.draw(canvas, line, left, baseline + _REASON_LINE_HEIGHT * index, BODY, color)

    def _ellipsize(self, text: str, max_width: int, style: TextStyle) -> str:
        while text and self._text.width(text + "...", style) > max_width:
            text = text[:-1]
        return text.rstrip() + "..."

    def _fit_text(self, text: str, max_width: int, style: TextStyle) -> str:
        if self._text.width(text, style) <= max_width:
            return text
        return self._ellipsize(text, max_width, style)

    # ---------------------------------------------------------------- runtime

    def _draw_runtime_strip(self, canvas: np.ndarray, telemetry: DashboardTelemetry) -> None:
        colors = self._colors
        thresholds = self._thresholds
        layout = self._layout

        fps_level = bands.fps_level(
            telemetry.fps, thresholds.target_fps, thresholds.fps_warn_fraction
        )
        self._draw_health_tile(
            canvas,
            layout.autonomy_fps,
            f"{telemetry.fps:.1f}",
            "fps",
            fps_level,
            bands.fps_note(fps_level, thresholds.target_fps),
            telemetry.fps / (thresholds.target_fps * thresholds.fps_bar_scale),
            1.0 / thresholds.fps_bar_scale,
        )

        latency_level = bands.latency_level(telemetry.latency_p95_ms, thresholds.latency_budget_ms)
        self._draw_health_tile(
            canvas,
            layout.latency,
            f"{telemetry.latency_ms:.0f} / {telemetry.latency_p95_ms:.0f}",
            "ms",
            latency_level,
            bands.latency_note(latency_level, thresholds.latency_budget_ms),
            telemetry.latency_p95_ms
            / (thresholds.latency_budget_ms * thresholds.latency_bar_scale),
            1.0 / thresholds.latency_bar_scale,
        )

        tile = layout.dashboard_fps
        self._text.draw(
            canvas,
            f"{telemetry.dashboard_fps:.1f}",
            tile.x + _CARD_INSET,
            tile.y + _DASHBOARD_TILE_VALUE_BASELINE,
            LARGE,
            colors["TEXT_SECONDARY"],
        )

    def _draw_health_tile(
        self,
        canvas: np.ndarray,
        tile: Rect,
        value: str,
        unit: str,
        level: str,
        note: str,
        fraction: float,
        tick: float,
    ) -> None:
        color = self._colors[level]
        if level != bands.GOOD:
            outline_rect(canvas, tile, color, 2)
        left = tile.x + _CARD_INSET
        baseline = tile.y + _TILE_VALUE_BASELINE
        value_w = self._text.draw(canvas, value, left, baseline, HERO, color)
        self._text.draw(
            canvas, unit, left + value_w + 10, baseline, BODY, self._colors["TEXT_SECONDARY"]
        )
        self._text.draw(
            canvas,
            note,
            tile.right - _CARD_INSET,
            tile.y + _TILE_TITLE_BASELINE,
            LABEL,
            color,
            align="right",
        )
        draw_bar(
            canvas,
            Rect(left, tile.y + _TILE_BAR_TOP, tile.w - 2 * _CARD_INSET, _TILE_BAR_H),
            fraction,
            color,
            self._colors["MUTED_LINE"],
            tick=tick,
            tick_color=self._colors["TEXT_PRIMARY"],
        )

    # ---------------------------------------------------------------- overlay

    def _compose(self, layer: ImageLayer, scene: Scene, draw_paths: bool) -> MainView:
        display = as_bgr(layer.pixels()).copy()
        shape = display.shape[:2]
        labels: list[tuple[str, tuple[int, int]]] = []
        overlays = scene.masks.get(layer.name, [])
        # Exclusions first, so the road tint and path draw over the hatching.
        for overlay in overlays:
            if overlay.role == "excluded":
                anchor = self._draw_excluded(display, as_mask(overlay.mask, shape))
                if anchor is not None and overlay.label:
                    labels.append((overlay.label.upper(), anchor))
        for overlay in overlays:
            if overlay.role != "excluded":
                self._draw_mask(display, as_mask(overlay.mask, shape), overlay.role)
        if draw_paths:
            for line in scene.polylines.get(layer.name, []):
                points = self._layer_points(layer, line, shape)
                if points is not None and len(points) >= 2:
                    self._draw_polyline(display, points, line.style)
        return MainView(display, labels)

    def _draw_mask(self, display: np.ndarray, mask: np.ndarray, role: str) -> None:
        if not mask.any():
            return
        color = self._colors[_ROLE_COLORS[role]]
        tinted = display.copy()
        tinted[mask] = color
        if role == "info":
            cv2.addWeighted(tinted, 0.35, display, 0.65, 0, display)
            return
        cv2.addWeighted(tinted, 0.28, display, 0.72, 0, display)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(display, contours, -1, self._colors["MASK_EDGE"], 2, lineType=cv2.LINE_AA)

    def _layer_points(
        self, layer: ImageLayer, line: Polyline, shape: tuple[int, int]
    ) -> np.ndarray | None:
        points = np.asarray(line.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            return None
        if line.frame == "image":
            return points.astype(np.float32)
        spec = self._cameras.get(layer.camera)
        if spec is None:
            # Once per layer: a misconfigured stack would otherwise log every frame.
            if layer.name not in self._warned_layers:
                self._warned_layers.add(layer.name)
                logger.warning(
                    "Ground polyline on layer '%s' skipped: it names no declared camera",
                    layer.name,
                )
            return None
        key = (spec.name, shape[1], shape[0])
        model = self._camera_models.get(key)
        if model is None:
            model = CameraModel(spec, shape[1], shape[0])
            self._camera_models[key] = model
        ahead = points[:, 0] > 0.1
        if ahead.sum() < 2:
            return None
        return model.ground_to_image(points[ahead, 0], points[ahead, 1]).astype(np.float32)

    def _draw_polyline(self, display: np.ndarray, points: np.ndarray, style: str) -> None:
        pts = smooth_points(points)
        pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        core = self._colors[_STYLE_COLORS[style]]
        if style == "candidate":
            cv2.polylines(display, [pts_i], False, core, 1, lineType=cv2.LINE_AA)
            return
        if style == "held":
            # Dashed and amber: the path is being held, not freshly perceived.
            dash, gap = 8, 6
            for start in range(0, len(pts_i), dash + gap):
                cv2.polylines(
                    display, [pts_i[start : start + dash]], False, core, 4, lineType=cv2.LINE_AA
                )
        else:
            glow = display.copy()
            cv2.polylines(glow, [pts_i], False, self._colors["PATH_GLOW"], 12, lineType=cv2.LINE_AA)
            cv2.addWeighted(glow, 0.22, display, 0.78, 0, display)
            cv2.polylines(display, [pts_i], False, core, 4, lineType=cv2.LINE_AA)
            cv2.polylines(
                display, [pts_i], False, self._colors["PATH_HIGHLIGHT"], 1, lineType=cv2.LINE_AA
            )
        target = tuple(np.round(pts[max(0, len(pts) // 3)]).astype(int))
        cv2.circle(display, target, 6, core, -1, lineType=cv2.LINE_AA)
        cv2.circle(display, target, 6, self._colors["PATH_HIGHLIGHT"], 1, lineType=cv2.LINE_AA)

    def _excluded_layer(self, mask: np.ndarray) -> tuple[np.ndarray, list, tuple[int, int] | None]:
        """Cached because an exclusion such as the hood rarely changes."""
        key = (mask.shape, int(mask.sum()))
        cached = self._excluded_cache.get(key)
        if cached is None:
            h, w = mask.shape
            hatch = np.zeros((h, w), dtype=np.uint8)
            for offset in range(-h, w, 14):
                cv2.line(hatch, (offset, 0), (offset + h, h), 1, 1)
            stripes = mask & hatch.astype(bool)
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            centre = None
            if contours:
                biggest = max(contours, key=cv2.contourArea)
                moments = cv2.moments(biggest)
                if moments["m00"] > 0:
                    # Near the top edge, not the centroid: the bottom of the
                    # frame is where the legend sits and would cover the label.
                    top = int(biggest[:, 0, 1].min())
                    centre = (int(moments["m10"] / moments["m00"]), top + _EXCLUDED_LABEL_DROP_PX)
            cached = (stripes, list(contours), centre)
            self._excluded_cache = {key: cached}
        return cached

    def _draw_excluded(self, display: np.ndarray, mask: np.ndarray) -> tuple[int, int] | None:
        """Hatched so it reads as excluded, not as terrain."""
        if not mask.any():
            return None
        colour = self._colors["EGO_EXCLUDED"]
        stripes, contours, centre = self._excluded_layer(mask)
        tinted = display.copy()
        tinted[mask] = colour
        tinted[stripes] = colour
        cv2.addWeighted(tinted, 0.3, display, 0.7, 0, display)
        cv2.drawContours(display, contours, -1, colour, 1, lineType=cv2.LINE_AA)
        return centre
