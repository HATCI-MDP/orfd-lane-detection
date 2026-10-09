"""Tests for the generic dashboard.

Every fixture here is a hand-built ``StackOutput``: the dashboard must draw
any stack from the visual primitives alone, without knowing its types.
"""

import numpy as np
import pytest

from offroad_autonomy.contract import (
    Gauge,
    ImageLayer,
    MaskOverlay,
    Polyline,
    StackOutput,
    Status,
    camera_sensor,
)
from offroad_autonomy.types import DEFAULT_CAMERA, DEFAULT_DASHBOARD_COLORS, ControlCommand
from offroad_autonomy.visualization.dashboard import (
    AutonomyDashboard,
    DashboardTelemetry,
    Scene,
    metric_row,
)
from tests.conftest import count_color

COLORS = DEFAULT_DASHBOARD_COLORS
H, W = 465, 720


def _output(
    style: str = "path",
    statuses: tuple[Status, ...] = (),
    image: np.ndarray | None = None,
    extra: tuple = (),
) -> StackOutput:
    """A flat grey frame, so any pixel in a palette colour was drawn by the dashboard."""
    if image is None:
        image = np.full((H, W, 3), 90, dtype=np.uint8)
    road = np.zeros((H, W), dtype=bool)
    road[250:400, 250:470] = True
    hood = np.zeros((H, W), dtype=bool)
    hood[420:, 200:520] = True
    path = np.array([[360.0, 440.0], [360.0, 300.0], [370.0, 200.0]])
    visuals = [
        ImageLayer("Front", image, camera="dashcam"),
        MaskOverlay("Front", road, "traversable", "Traversable Mask"),
        Polyline("Front", path, "image", style, "Planned Path"),
        MaskOverlay("Front", hood, "excluded", "Hood"),
        ImageLayer("Second", np.zeros((100, 200, 3), dtype=np.uint8)),
        Gauge("Health", 0.8, (0.0, 1.0), (0.2, 0.5), caption="Tick: Floor 0.2"),
        Gauge("Plain", 0.5),
        *statuses,
        *extra,
    ]
    return StackOutput(
        command=ControlCommand(),
        visuals=visuals,
        metrics={"cross_track_m": 0.25, "speed_mps": 3.0, "edge_m": float("inf"), "seg_ms": 4.0},
    )


def _telemetry(**overrides) -> DashboardTelemetry:
    fields = {
        "speed_mph": 9.4,
        "steering": 0.14,
        "throttle": 0.3,
        "brake": 0.0,
        "fps": 8.0,
        "latency_ms": 90.0,
        "latency_p95_ms": 110.0,
    }
    fields.update(overrides)
    return DashboardTelemetry(**fields)


def _dashboard() -> AutonomyDashboard:
    return AutonomyDashboard(sensors=[camera_sensor(DEFAULT_CAMERA)], stack_name="test_stack")


@pytest.mark.parametrize("page", [0, 1, 5])
def test_every_page_renders_and_unknown_pages_fall_back_to_the_main_view(page):
    frame = _dashboard().render(
        _output(), _telemetry(timing_lines=["capture  1.0 ms"]), page=page, timing_overlay=True
    )

    assert frame.shape == (900, 1600, 3)


def test_a_stack_with_no_image_still_renders():
    output = StackOutput(command=ControlCommand(), visuals=[Status("WARMING UP", "warn")])

    for page in range(3):
        frame = _dashboard().render(output, _telemetry(autopilot_active=False), page=page)
        assert frame.shape == (900, 1600, 3)


def test_safe_stop_draws_no_path_and_shows_the_banner():
    dashboard = _dashboard()

    driving = dashboard.render(_output(), _telemetry())
    stopped = dashboard.render(_output(), _telemetry(autopilot_active=False))

    assert count_color(driving, COLORS["PATH_CORE"]) > 0
    assert count_color(stopped, COLORS["PATH_CORE"]) == 0
    # The banner border is drawn in the fault colour inside the viewport.
    assert count_color(stopped, COLORS["BAD"]) > 0


def test_held_path_is_dashed_in_the_warning_colour():
    frame = _dashboard().render(_output(style="held"), _telemetry())

    assert count_color(frame, COLORS["WARN"]) > 0
    assert count_color(frame, COLORS["PATH_CORE"]) == 0


def test_ground_polyline_is_projected_with_the_declared_camera():
    dashboard = _dashboard()
    ground = np.array([[3.0, 0.0], [6.0, 0.2], [10.0, 0.5]])
    output = StackOutput(
        command=ControlCommand(),
        visuals=[
            ImageLayer("Front", np.full((H, W, 3), 90, np.uint8), camera="dashcam"),
            Polyline("Front", ground, "ground", "path"),
        ],
    )

    view = dashboard.main_view(Scene.of(output), autopilot_active=True)

    drawn = (view.image == np.array(COLORS["PATH_CORE"], np.uint8)).all(axis=2)
    rows, cols = np.nonzero(drawn)
    assert len(rows) > 0
    # Straight ahead of a centred camera lands on the image's centre column.
    assert abs(np.median(cols) - W / 2) < 40


def test_ground_polyline_on_a_layer_without_a_camera_is_skipped():
    dashboard = _dashboard()
    output = StackOutput(
        command=ControlCommand(),
        visuals=[
            ImageLayer("Grid", np.zeros((100, 100, 3), dtype=np.uint8)),
            Polyline("Grid", np.array([[3.0, 0.0], [6.0, 0.0]]), "ground", "path"),
        ],
    )

    view = dashboard.main_view(Scene.of(output), autopilot_active=True)

    assert count_color(view.image, COLORS["PATH_CORE"]) == 0


def test_a_layer_drawn_on_demand_is_drawn_only_while_its_page_is_shown():
    calls = []

    def draw():
        calls.append(1)
        return np.zeros((50, 80, 3), dtype=np.uint8)

    output = _output(extra=(ImageLayer("Costly", draw),))
    dashboard = _dashboard()

    dashboard.render(output, _telemetry(), page=0)
    assert calls == []
    dashboard.render(output, _telemetry(), page=2)
    assert calls == [1]


def test_runtime_tile_border_follows_the_fps_band():
    dashboard = _dashboard()
    tile = dashboard._layout.autonomy_fps
    probe = (tile.y, tile.x + 40)

    on_target = dashboard.render(_output(), _telemetry(fps=7.0))
    amber = dashboard.render(_output(), _telemetry(fps=6.0))
    red = dashboard.render(_output(), _telemetry(fps=3.0))

    assert tuple(on_target[probe]) == COLORS["CARD_BORDER"]
    assert tuple(amber[probe]) == COLORS["WARN"]
    assert tuple(red[probe]) == COLORS["BAD"]


def test_dashboard_fps_never_takes_a_health_colour():
    dashboard = _dashboard()
    tile = dashboard._layout.dashboard_fps

    slow = dashboard.render(_output(), _telemetry(dashboard_fps=1.0))

    region = slow[tile.y : tile.bottom, tile.x : tile.right]
    for name in ("BAD", "WARN"):
        assert count_color(region, COLORS[name]) == 0


def test_gauge_colour_follows_its_thresholds_on_the_stack_card():
    dashboard = _dashboard()
    card = dashboard._layout.stack

    def card_pixels(value):
        output = _output()
        output.visuals = [
            v
            if not (isinstance(v, Gauge) and v.name == "Health")
            else Gauge("Health", value, (0.0, 1.0), (0.2, 0.5))
            for v in output.visuals
        ]
        frame = dashboard.render(output, _telemetry())
        return frame[card.y : card.bottom, card.x : card.right]

    assert count_color(card_pixels(0.1), COLORS["BAD"]) > 0
    assert count_color(card_pixels(0.3), COLORS["WARN"]) > 0


def test_long_details_and_many_statuses_stay_inside_the_stack_card():
    dashboard = _dashboard()
    card = dashboard._layout.stack
    quiet = dashboard.render(_output(), _telemetry())

    loud_statuses = (
        Status("GATE HOLD", "warn", "word " * 200),
        Status("KALMAN", "warn"),
        Status("SOMETHING ELSE", "bad", "more " * 50),
    )
    loud = dashboard.render(
        _output(statuses=loud_statuses),
        _telemetry(),
        page=0,
    )

    outside = np.ones(quiet.shape[:2], dtype=bool)
    outside[card.y : card.bottom, card.x : card.right] = False
    viewport = dashboard._layout.viewport
    # The first alert also gets a chip over the viewport, which is expected.
    outside[viewport.y : viewport.bottom, viewport.x : viewport.right] = False
    assert np.array_equal(quiet[outside], loud[outside])
    assert not np.array_equal(quiet, loud)


def test_safe_stop_chip_leads_the_stack_statuses():
    from offroad_autonomy.visualization.dashboard import harness_statuses

    scene = Scene.of(_output(statuses=(Status("GATE STOP", "bad", "no road"), Status("OK"))))

    chips = harness_statuses(_telemetry(autopilot_active=False, safe_stop_reason="stuck"), scene)

    assert [(c.text, c.detail) for c in chips] == [("SAFE STOP", "stuck"), ("GATE STOP", "no road")]


def test_render_does_not_modify_the_cached_static_layer():
    dashboard = _dashboard()
    before = dashboard._static.copy()
    first = dashboard._static

    dashboard.render(_output(), _telemetry(autopilot_active=False))
    dashboard.render(_output(), _telemetry(fps=2.0), page=1)

    assert dashboard._static is first
    assert np.array_equal(dashboard._static, before)


def test_excluded_colour_is_distinct_from_mask_and_fault_colours():
    ego = np.array(COLORS["EGO_EXCLUDED"], dtype=float)
    for name in ("MASK_FILL", "BAD", "GOOD", "WARN", "PATH_CORE"):
        other = np.array(COLORS[name], dtype=float)
        assert np.linalg.norm(ego - other) > 100.0, name


def test_legend_follows_the_stack_and_drops_paths_under_safe_stop():
    dashboard = _dashboard()
    scene = Scene.of(_output())

    driving = [label for label, _ in dashboard.legend(scene, autopilot_active=True)]
    stopped = [label for label, _ in dashboard.legend(scene, autopilot_active=False)]

    assert driving == ["Traversable Mask", "Hood", "Planned Path"]
    assert stopped == ["Traversable Mask", "Hood"]


def test_readout_lists_statuses_gauges_and_metrics_without_timings():
    output = _output(statuses=(Status("Target Speed", "good", "3.0 m/s (curve)"),))

    rows = dict(AutonomyDashboard.readout(Scene.of(output), output))

    assert rows["Status"] is None
    assert rows["Target Speed"] == "3.0 m/s (curve)"
    assert rows["Cross Track"] == "+0.250 m"
    assert rows["Speed"] == "+3.000 m/s"
    assert rows["Edge"] == "n/a"
    assert "Seg" not in rows


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("lateral_accel_mps2", ("Lateral Accel", "+1.000 m/s2")),
        ("curvature_per_m", ("Curvature", "+1.000 /m")),
        ("heading_error_deg", ("Heading Error", "+1.000 deg")),
        ("detections", ("Detections", "+1.000")),
    ],
)
def test_metric_units_come_from_the_name_suffix(name, expected):
    assert metric_row(name, 1.0) == expected
