"""The harness: connect to BeamNG and run the configured driving stack until stopped.

The loop, safe stop, manual override, dashboard and scoring live here and are
the same for every stack. The stack only turns observations into commands.
The dashboard runs on its own thread, so this loop never waits for it.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from offroad_autonomy.contract import DrivingStack, Observation, StackOutput, load_stack
from offroad_autonomy.runtime.benchmark import BenchmarkRecorder
from offroad_autonomy.runtime.display_worker import DisplayState, DisplayWorker
from offroad_autonomy.runtime.route import Route, RouteRecorder, route_path
from offroad_autonomy.runtime.timing import DISPLAY_STAGES, MAIN_STAGES, RuntimeStats
from offroad_autonomy.runtime.video_recorder import VideoRecorder, default_video_path
from offroad_autonomy.simulation.beamng_client import BeamNGClient, BeamNGConnectionError
from offroad_autonomy.types import (
    MAX_DEBUG_PAGE,
    ControlCommand,
    HarnessConfig,
    VehicleState,
)
from offroad_autonomy.utils.config import load_config
from offroad_autonomy.utils.logger import setup_logger
from offroad_autonomy.visualization import (
    AutonomyDashboard,
    DashboardTelemetry,
    DashboardWindow,
)
from offroad_autonomy.visualization.presentation import (
    PRESENTATION_HEIGHT,
    PRESENTATION_WIDTH,
    PresentationRenderer,
)

logger = logging.getLogger("offroad_autonomy.main")

_shutdown = False
_WINDOW_TITLE = "Off-Road Autonomy Dashboard"
_DASHBOARD_WIDTH = 1600
_DASHBOARD_HEIGHT = 900
# Apart from output/videos, so presentations are not lost among the many
# dashboard recordings made while tuning.
_PRESENTATION_DIR = "output/presentations"
_MPH_PER_MPS = 2.2369362920544

_MANUAL_STEER = 0.4
_MANUAL_THROTTLE = 0.25
_MANUAL_BRAKE = 0.35

_SAFE_STOP_COMMAND = ControlCommand(steering=0.0, throttle=0.0, brake=0.0, parkingbrake=1.0)


def _signal_handler(signum, frame) -> None:
    del signum, frame
    global _shutdown
    _shutdown = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="offroad_autonomy - autonomous off-road driving in BeamNG",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="YAML configuration file.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the dashboard window (overrides ui.headless).",
    )
    parser.add_argument("--map", default="", help="BeamNG map (overrides beamng.map).")
    parser.add_argument(
        "--spawn", type=int, default=None, help="Spawn index (overrides beamng.spawn_index)."
    )
    parser.add_argument(
        "--benchmark-seconds",
        type=float,
        default=0.0,
        help="Stop after this many seconds and write a benchmark report.",
    )
    parser.add_argument(
        "--benchmark-warmup",
        type=float,
        default=5.0,
        help="Seconds excluded from the timing figures while models and caches warm up.",
    )
    parser.add_argument(
        "--benchmark-out",
        default="",
        help="Benchmark JSON path (default: output/benchmarks/<label>.json).",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Name for the benchmark run (default: the config file name).",
    )
    parser.add_argument(
        "--record-route",
        nargs="?",
        const="",
        default=None,
        help=(
            "Record the vehicle's true path as the reference route for this map and spawn, "
            "saved on exit (default path: <benchmark.routes_dir>/<map>_spawn<index>.json). "
            "Drive it cleanly, by the stack or by hand after E."
        ),
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help=(
            "Record the dashboard to an H.264 mp4 (needs ffmpeg on PATH). "
            "With --headless the dashboard is drawn off screen for the recording only."
        ),
    )
    parser.add_argument(
        "--record-video-out",
        default="",
        help="Video path (default: output/videos/<label>_<timestamp>.mp4).",
    )
    parser.add_argument(
        "--presentation",
        action="store_true",
        help=(
            "Record a 1920x1080 presentation video (orbit camera, the stack's main view, key "
            "stats) to its own mp4 (needs ffmpeg on PATH). Combine with --record-video to also "
            "record the dashboard, to a separate file."
        ),
    )
    parser.add_argument(
        "--presentation-view",
        action="store_true",
        help=(
            "Show the orbit camera, the stack's main view and stats in the live window. "
            "Add --presentation to also record this view."
        ),
    )
    parser.add_argument(
        "--presentation-out",
        default="",
        help=f"Presentation path (default: {_PRESENTATION_DIR}/<label>_<timestamp>.mp4).",
    )
    return parser


def _open_video(path: str | Path, width: int, height: int, config: HarnessConfig) -> VideoRecorder:
    try:
        return VideoRecorder(
            path,
            width,
            height,
            fps=config.recording_fps,
            crf=config.recording_crf,
            preset=config.recording_preset,
            queue_frames=config.recording_queue_frames,
        )
    except (RuntimeError, OSError) as exc:
        sys.exit(f"Cannot record video: {exc}")


class _StuckDetector:
    """The harness's own safe-stop trigger, the same for every stack: the
    wheels are pushing and the vehicle is not moving."""

    def __init__(self, speed_mps: float, throttle_min: float, time_s: float) -> None:
        self._speed_mps = float(speed_mps)
        self._throttle_min = float(throttle_min)
        self._time_s = float(time_s)
        self._stuck_since: float | None = None

    def reset(self) -> None:
        self._stuck_since = None

    def update(self, now: float, speed_mps: float, throttle: float) -> tuple[bool, str]:
        if speed_mps >= self._speed_mps or throttle <= self._throttle_min:
            self._stuck_since = None
            return False, ""
        if self._stuck_since is None:
            self._stuck_since = now
            return False, ""
        if now - self._stuck_since < self._time_s:
            return False, ""
        self.reset()
        return True, "vehicle stuck"


@dataclass
class _Control:
    """What the operator and the safe stop have decided; keys and triggers change it."""

    autopilot_active: bool = True
    safe_stop_reason: str = ""
    page: int = 0
    timing_overlay: bool = False


def _timing_lines(stats: RuntimeStats, output: StackOutput, display: DisplayWorker) -> list[str]:
    """One heading per thread, so a slow dashboard can never be mistaken for
    a slow vehicle, and the stack's own stages under the loop that ran them."""
    lines = ["AUTONOMY LOOP"] + stats.format_lines(MAIN_STAGES)
    stack_lines = [
        f"{name[:-3]:<16} last {value:6.1f} ms"
        for name, value in output.metrics.items()
        if name.endswith("_ms")
    ]
    if stack_lines:
        lines += ["", "STACK"] + stack_lines
    return lines + ["", "DASHBOARD"] + display.stats.format_lines(DISPLAY_STAGES)


def _build_dashboard_telemetry(
    state: VehicleState,
    command: ControlCommand,
    output: StackOutput,
    stats: RuntimeStats,
    control: _Control,
    display: DisplayWorker,
) -> DashboardTelemetry:
    primary = stats.stage("primary_loop")
    telemetry = DashboardTelemetry(
        speed_mph=state.speed_mps * _MPH_PER_MPS,
        steering=command.steering,
        throttle=command.throttle,
        brake=command.brake,
        fps=stats.fps(),
        latency_ms=primary.mean_ms,
        latency_p95_ms=primary.p95_ms,
        autopilot_active=control.autopilot_active,
        safe_stop_reason=control.safe_stop_reason,
        dashboard_fps=display.fps(),
    )
    if control.timing_overlay:
        telemetry.timing_lines = _timing_lines(stats, output, display)
    return telemetry


def _log_runtime(
    stats: RuntimeStats,
    display: DisplayWorker | None = None,
    presentation: DisplayWorker | None = None,
) -> None:
    primary = stats.stage("primary_loop")
    step = stats.stage("stack_step")
    message = (
        f"autonomy {stats.fps():.1f} FPS  primary {primary.mean_ms:.1f}/"
        f"{primary.p95_ms:.1f} ms  stack step {step.mean_ms:.1f}/{step.p95_ms:.1f} ms (mean/p95)"
    )
    if display is not None:
        total = display.stats.stage("dashboard_total")
        message += (
            f" | dashboard {display.fps():.1f} Hz  {total.mean_ms:.1f} ms"
            f"  (drawn {display.rendered}, superseded {display.dropped})"
        )
    if presentation is not None:
        render = presentation.stats.stage("dashboard_render")
        message += (
            f" | presentation {presentation.fps():.1f} Hz  render {render.mean_ms:.1f}/"
            f"{render.p95_ms:.1f} ms (drawn {presentation.rendered}, "
            f"superseded {presentation.dropped})"
        )
    logger.info(message)
    for line in stats.format_lines(MAIN_STAGES):
        logger.debug("  %s", line)
    if display is not None:
        for line in display.stats.format_lines(DISPLAY_STAGES):
            logger.debug("  %s", line)


def _manual_command(held: set[str]) -> ControlCommand:
    if not held:
        return _SAFE_STOP_COMMAND
    steer = 0.0
    if "a" in held:
        steer -= _MANUAL_STEER
    if "d" in held:
        steer += _MANUAL_STEER
    if "space" in held:
        return ControlCommand(steering=steer, throttle=0.0, brake=1.0, parkingbrake=1.0)
    if "w" in held:
        return ControlCommand(steering=steer, throttle=_MANUAL_THROTTLE)
    if "s" in held:
        return ControlCommand(steering=steer, brake=_MANUAL_BRAKE)
    return ControlCommand(steering=steer)


def _open_window(width: int, height: int) -> DashboardWindow | None:
    """A missing display must not stop the vehicle, so a failed window
    degrades to headless instead of aborting the run."""
    try:
        return DashboardWindow(_WINDOW_TITLE, width, height)
    except RuntimeError as exc:
        logger.warning("No dashboard window (%s) - running headless", exc)
        return None


def _display_worker(
    render, window: DashboardWindow | None, record, config: HarnessConfig
) -> DisplayWorker:
    """Off-screen drawing always gets its own thread. A window gets one only
    on the OpenCV backend: Tkinter's event loop belongs to the thread that
    built the root window."""
    show = lambda canvas: True
    read_key = lambda: -1
    asynchronous = True
    if window is not None:
        show = window.show
        read_key = lambda: window.last_key
        asynchronous = config.ui_display_async and window.backend == "opencv"
    worker = DisplayWorker(
        render=render,
        show=show,
        read_key=read_key,
        record=record,
        rate_hz=config.ui_display_rate_hz,
        asynchronous=asynchronous,
    )
    worker.start()
    return worker


def _start_displays(
    args: argparse.Namespace,
    config: HarnessConfig,
    sensors: list,
    video: VideoRecorder | None,
    presentation_video: VideoRecorder | None,
) -> tuple[DashboardWindow | None, DisplayWorker | None, DisplayWorker | None]:
    """The operator dashboard and the presentation, each live in a window,
    recorded, or both. Returns the window whose keys drive the vehicle."""
    show_presentation = args.presentation_view and not config.ui_headless
    show_dashboard = not config.ui_headless and not show_presentation
    record = None
    if video is not None:
        record = video.write
    window = None
    display = None
    if show_dashboard or video is not None:
        dashboard = AutonomyDashboard(
            width=_DASHBOARD_WIDTH,
            height=_DASHBOARD_HEIGHT,
            colors=config.dashboard_colors,
            sensors=sensors,
            thresholds=config.dashboard_thresholds,
            stack_name=stack_name(config.stack),
        )
        if show_dashboard:
            window = _open_window(dashboard.width, dashboard.height)

        def render(state: DisplayState) -> np.ndarray:
            return dashboard.render(
                state.output, state.telemetry, page=state.page, timing_overlay=state.timing_overlay
            )

    if window is not None or video is not None:
        display = _display_worker(render, window, record, config)
        if window is not None:
            if config.ui_display_async and not display.asynchronous:
                logger.warning(
                    "Dashboard is running inline: the '%s' window backend cannot be "
                    "driven from another thread, so rendering is part of the loop period",
                    window.backend,
                )
            logger.info(
                "Keys: E safe stop, P resume, 0 main view, 1-%d the stack's layers, "
                "T timing overlay",
                MAX_DEBUG_PAGE,
            )
        else:
            logger.info("No window: dashboard drawn off screen for the recording only")
    elif not show_presentation:
        logger.info("Headless: no dashboard; manual control is unavailable")

    presentation_display = None
    if args.presentation or show_presentation:
        presentation = PresentationRenderer(
            colors=config.dashboard_colors,
            sensors=sensors,
            thresholds=config.dashboard_thresholds,
        )

        def render_presentation(state: DisplayState) -> np.ndarray:
            return presentation.render(state.output, state.telemetry, state.orbit)

        presentation_window = None
        if show_presentation:
            presentation_window = _open_window(PRESENTATION_WIDTH, PRESENTATION_HEIGHT)
            window = presentation_window
        record = None
        if presentation_video is not None:
            record = presentation_video.write
        presentation_display = _display_worker(
            render_presentation, presentation_window, record, config
        )
        if presentation_window:
            logger.info("Live presentation view: E safe stop, P resume, Q quit")
        elif presentation_video:
            logger.info("Presentation video drawn off screen for the recording only")
    return window, display, presentation_display


def _handle_keys(
    worker: DisplayWorker,
    control: _Control,
    client: BeamNGClient,
    stuck_detector: _StuckDetector,
    stack: DrivingStack,
) -> None:
    for key in worker.drain_keys():
        if key in (ord("t"), ord("T")):
            control.timing_overlay = not control.timing_overlay
        elif ord("0") <= key <= ord("0") + MAX_DEBUG_PAGE:
            control.page = key - ord("0")
            logger.info("Dashboard page %d", control.page)
        elif key in (ord("e"), ord("E")) and control.autopilot_active:
            client.park()
            control.autopilot_active = False
            control.safe_stop_reason = "operator pressed E"
            logger.info("Safe stop triggered - autopilot disabled, manual control required")
        elif key in (ord("p"), ord("P")) and not control.autopilot_active:
            client.release_park()
            stuck_detector.reset()
            # The operator may have moved the vehicle, so whatever the stack
            # remembered about the road behind it no longer applies.
            stack.reset()
            control.autopilot_active = True
            control.safe_stop_reason = ""
            logger.info("Autopilot resumed")


def video_paths(args: argparse.Namespace, label: str) -> tuple[Path, Path]:
    """Where the dashboard and the presentation recordings go.

    Two encoders writing one file would corrupt it, so a run that asks for
    both at the same path is stopped before anything is recorded.
    """
    dashboard = Path(args.record_video_out or default_video_path(label))
    presentation = Path(args.presentation_out or default_video_path(label, _PRESENTATION_DIR))
    if args.record_video and args.presentation and dashboard.resolve() == presentation.resolve():
        sys.exit("--record-video-out and --presentation-out must name different files")
    return dashboard, presentation


def _log_video(video: VideoRecorder) -> None:
    if video.failed:
        logger.error("Video recording failed; %s may be incomplete", video.path)
        return
    logger.info(
        "Video saved to %s: %d frames (%d repeated to keep real time, %d dropped by a busy encoder)",
        video.path,
        video.frames_written,
        video.frames_repeated,
        video.frames_dropped,
    )


def stack_name(path: str) -> str:
    """The short name a run is labelled with: the stack's package."""
    return path.partition(":")[0].rsplit(".", 1)[-1]


def _observation(frames: dict, state: VehicleState) -> Observation:
    first = next(iter(frames.values()))
    return Observation(
        sensors={name: frame.image for name, frame in frames.items()},
        vehicle=state,
        timestamp=min(frame.timestamp for frame in frames.values()),
        frame_id=first.frame_id,
    )


def _benchmark_recorder(
    args: argparse.Namespace, config: HarnessConfig, label: str, config_path: Path
) -> BenchmarkRecorder:
    path = route_path(config.benchmark_routes_dir, config.beamng_map, config.beamng_spawn_index)
    route = None
    if path.exists():
        route = Route.load(path)
        logger.info("Scoring against route %s (%.0f m)", path, route.length_m)
    else:
        logger.warning(
            "No reference route at %s: cross-track, completion and departures will be null. "
            "Record one with --record-route.",
            path,
        )
    return BenchmarkRecorder(label, str(config_path), config, route, args.benchmark_warmup)


def _write_benchmark(args: argparse.Namespace, recorder: BenchmarkRecorder, stats) -> None:
    report = recorder.report(stats)
    path = recorder.save(args.benchmark_out or f"output/benchmarks/{recorder.label}.json", report)
    logger.info("Benchmark report written to %s", path)
    logger.info(
        "  %.1f FPS  primary p95 %s ms  distance %s m  cross-track mean %s m  "
        "departures %s  safe stops %d",
        report["main_fps"] or 0.0,
        report["primary_latency_p95_ms"],
        report["distance_m"],
        report["cross_track_mean_m"],
        report["departures"],
        report["safe_stops"],
    )


def _save_route(args: argparse.Namespace, config: HarnessConfig, recorder: RouteRecorder) -> None:
    out = args.record_route or route_path(
        config.benchmark_routes_dir, config.beamng_map, config.beamng_spawn_index
    )
    try:
        path = recorder.save(out, config.beamng_map, config.beamng_spawn_index)
    except ValueError as exc:
        logger.error("Route not saved: %s", exc)
        return
    logger.info("Reference route saved to %s (%d points)", path, len(recorder.points))


def main() -> None:
    global _shutdown
    _shutdown = False
    args = build_parser().parse_args()

    setup_logger(level=getattr(logging, args.log_level))

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Configuration file not found: {config_path}")

    config = load_config(config_path)
    if args.headless:
        config.ui_headless = True
    if args.map:
        config.beamng_map = args.map
    if args.spawn is not None:
        config.beamng_spawn_index = args.spawn
    logger.info("Configuration loaded from %s; stack %s", config_path, config.stack)

    # SIGTERM is what `docker stop` sends; without a handler the process dies
    # mid-loop and the vehicle is never parked.
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    stack = load_stack(config.stack, config.stack_config)
    sensors = stack.sensors()
    show_presentation = args.presentation_view and not config.ui_headless
    use_presentation = args.presentation or show_presentation
    client = BeamNGClient(config, sensors, orbit=use_presentation)
    stats = RuntimeStats()
    control = _Control(page=config.ui_debug_page, timing_overlay=config.ui_timing_overlay)
    stuck_detector = _StuckDetector(
        speed_mps=config.safety_stuck_speed_mps,
        throttle_min=config.safety_stuck_throttle_min,
        time_s=config.safety_stuck_time_s,
    )
    step_budget_ms = config.dashboard_thresholds.latency_budget_ms

    label = args.label or config_path.stem
    recorder = None
    if args.benchmark_seconds > 0 or args.benchmark_out:
        recorder = _benchmark_recorder(args, config, label, config_path)
    route_recorder = None
    if args.record_route is not None:
        route_recorder = RouteRecorder(config.benchmark_route_spacing_m)

    # Opened before connecting so a missing ffmpeg stops the run before a map
    # is loaded and a vehicle spawned for nothing.
    video = None
    presentation_video = None
    video_path, presentation_path = video_paths(args, label)
    if args.record_video:
        video = _open_video(video_path, _DASHBOARD_WIDTH, _DASHBOARD_HEIGHT, config)
        logger.info("Recording the dashboard to %s", video.path)
    if args.presentation:
        presentation_video = _open_video(
            presentation_path, PRESENTATION_WIDTH, PRESENTATION_HEIGHT, config
        )
        logger.info("Recording the presentation video to %s", presentation_video.path)

    window = None
    display = None
    presentation_display = None
    frame_count = 0
    overruns = 0
    t_start = time.perf_counter()
    t_last_log = t_start
    try:
        client.connect()
        window, display, presentation_display = _start_displays(
            args, config, sensors, video, presentation_video
        )
        workers = [worker for worker in (display, presentation_display) if worker is not None]
        # Keys come from whichever window the operator is looking at.
        key_source = display
        if show_presentation:
            key_source = presentation_display

        logger.info("Entering main loop - Ctrl+C or SIGTERM to stop")
        t_start = time.perf_counter()
        orbit_image: np.ndarray | None = None

        while not _shutdown:
            t_iter = time.perf_counter()
            with stats.time("capture"):
                frames = client.capture()
            if frames is None:
                time.sleep(0.005)
                continue

            # Polled here, not on the dashboard thread, because the socket
            # transport is not thread safe. A missed read keeps the last
            # image so the video does not flash the placeholder.
            if use_presentation:
                with stats.time("orbit_capture"):
                    latest_orbit = client.capture_orbit()
                if latest_orbit is not None:
                    orbit_image = latest_orbit

            with stats.time("vehicle_state"):
                state = client.get_vehicle_state()

            # The stack keeps running under safe stop so the operator can see
            # the road come back before handing control over again.
            t_step = time.perf_counter()
            output = stack.step(_observation(frames, state))
            step_ms = (time.perf_counter() - t_step) * 1000.0
            stats.record("stack_step", step_ms)
            if step_ms > step_budget_ms:
                overruns += 1

            if control.autopilot_active:
                command = output.command
                triggered, reason = stuck_detector.update(
                    now=t_iter, speed_mps=state.speed_mps, throttle=command.throttle
                )
                if not triggered and output.safe_stop:
                    triggered, reason = True, output.safe_stop
                if triggered:
                    control.autopilot_active = False
                    control.safe_stop_reason = reason
                    client.park()
                    command = _SAFE_STOP_COMMAND
                    logger.warning("Safe stop triggered automatically: %s", reason)
            else:
                held: set[str] = set()
                if window is not None:
                    held = window.held_keys
                command = _manual_command(held)

            with stats.time("actuation"):
                client.send_controls(command)
            primary_ms = (time.perf_counter() - t_iter) * 1000.0
            stats.record("primary_loop", primary_ms)

            # The command has already gone out and primary_loop is recorded,
            # so nothing below counts as autonomy latency.
            if workers and frame_count % config.ui_render_every_n == 0:
                telemetry = _build_dashboard_telemetry(
                    state, command, output, stats, control, workers[0]
                )
                # One snapshot for both: each worker only reads it.
                snapshot = DisplayState(
                    output=output,
                    telemetry=telemetry,
                    page=control.page,
                    timing_overlay=control.timing_overlay,
                    orbit=orbit_image,
                )
                for worker in workers:
                    worker.publish(snapshot)

            if key_source is not None:
                if key_source.closed:
                    _shutdown = True
                _handle_keys(key_source, control, client, stuck_detector, stack)

            full_ms = (time.perf_counter() - t_iter) * 1000.0
            stats.record("full_loop", full_ms)
            stats.tick()
            frame_count += 1

            if recorder is not None:
                recorder.add(
                    now=time.perf_counter(),
                    state=state,
                    command=command,
                    autopilot_active=control.autopilot_active,
                    primary_ms=primary_ms,
                    step_ms=step_ms,
                    full_ms=full_ms,
                    stack_metrics=output.metrics,
                )
            if route_recorder is not None:
                route_recorder.add(state)

            now = time.perf_counter()
            if now - t_last_log >= config.runtime_log_interval_s:
                _log_runtime(stats, display, presentation_display)
                if overruns:
                    logger.warning(
                        "Stack step exceeded the %.0f ms loop budget %d times since the last "
                        "report; move slow work onto the stack's own worker",
                        step_budget_ms,
                        overruns,
                    )
                    overruns = 0
                t_last_log = now
            if args.benchmark_seconds > 0 and now - t_start >= args.benchmark_seconds:
                logger.info("Benchmark duration reached")
                break

    except BeamNGConnectionError as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Stop drawing before the window is torn down and the video is closed.
        for worker in (display, presentation_display):
            if worker is not None:
                worker.stop()
        for recording in (video, presentation_video):
            if recording is not None:
                recording.close()
                _log_video(recording)
        if window is not None:
            window.close()
        client.disconnect()
        elapsed = time.perf_counter() - t_start
        logger.info("Session ended - %d frames in %.1f s", frame_count, elapsed)
        _log_runtime(stats, display, presentation_display)
        if route_recorder is not None:
            _save_route(args, config, route_recorder)
        if recorder is not None:
            _write_benchmark(args, recorder, stats)


if __name__ == "__main__":
    main()
