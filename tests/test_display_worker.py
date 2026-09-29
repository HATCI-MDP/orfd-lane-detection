"""Dashboard worker snapshot and threading behavior."""

import numpy as np

from offroad_autonomy.runtime.display_worker import DisplayState, DisplayWorker


def _worker(**kwargs) -> tuple[DisplayWorker, list]:
    drawn: list = []

    def render(state):
        drawn.append(state)
        return np.zeros((4, 4, 3), dtype=np.uint8)

    kwargs.setdefault("show", lambda frame: True)
    kwargs.setdefault("read_key", lambda: -1)
    return DisplayWorker(render=render, asynchronous=False, rate_hz=0.0, **kwargs), drawn


def test_a_failing_dashboard_does_not_raise_into_the_control_loop():
    """Rendering is best effort: a broken window must not stop the vehicle."""

    def explode(state):
        raise RuntimeError("no window")

    worker = DisplayWorker(
        render=explode,
        show=lambda f: True,
        read_key=lambda: -1,
        asynchronous=False,
        rate_hz=0.0,
    )

    worker.publish(DisplayState(result=object(), telemetry=object()))

    assert worker.failed == 1
    assert worker.rendered == 0


def test_keys_are_queued_for_the_control_loop_to_drain():
    keys = iter([ord("e"), ord("p"), -1])
    worker, _ = _worker(read_key=lambda: next(keys))

    for _ in range(3):
        worker.publish(DisplayState(result=object(), telemetry=object()))

    assert worker.drain_keys() == [ord("e"), ord("p")]
    assert worker.drain_keys() == []


def test_publishing_never_blocks_when_asynchronous():
    """Snapshots are overwritten, not queued, so the loop cannot back up."""
    worker, _ = _worker()
    worker._async = True  # without starting the thread, nothing consumes

    for _ in range(50):
        worker.publish(DisplayState(result=object(), telemetry=object()))

    assert worker.dropped == 49
    assert worker._pending is not None


def test_worker_draws_on_a_background_thread_without_the_caller_waiting():
    """The whole point: publish() returns, and drawing happens elsewhere."""
    import threading

    drew = threading.Event()
    seen: dict = {}

    def render(state):
        seen["thread"] = threading.current_thread().name
        drew.set()
        return np.zeros((4, 4, 3), dtype=np.uint8)

    worker = DisplayWorker(
        render=render,
        show=lambda f: True,
        read_key=lambda: -1,
        asynchronous=True,
        rate_hz=0.0,
    )
    worker.start()
    try:
        worker.publish(DisplayState(result=object(), telemetry=object()))
        assert drew.wait(timeout=5.0), "worker never drew"
    finally:
        worker.stop()

    assert seen["thread"] != threading.current_thread().name
    assert seen["thread"] == "display-worker"
    assert worker.rendered >= 1
    assert worker.stats.stage("dashboard_total").count >= 1


def test_worker_renders_the_exact_inference_snapshot():
    state = DisplayState(result=object(), telemetry=object())
    worker, drawn = _worker()
    worker.publish(state)
    assert drawn == [state]
