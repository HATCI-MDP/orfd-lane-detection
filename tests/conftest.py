"""Shared test helpers: running the harness without a simulator, and counting pixels."""

from unittest.mock import Mock

import numpy as np
import pytest

from offroad_autonomy.types import HarnessConfig


def count_color(image: np.ndarray, color: tuple[int, int, int]) -> int:
    """Pixels exactly in ``color``: on a flat test frame, the ones the renderer drew."""
    return int((image == np.array(color, dtype=np.uint8)).all(axis=2).sum())


@pytest.fixture
def run_harness(monkeypatch):
    """Runs ``main()`` with a fake stack and client in place of the simulator.

    Keyword arguments replace further names in ``offroad_autonomy.main``.
    """
    import offroad_autonomy.main as app

    def run(stack, client, argv=("--headless",), config=None, **patches):
        if config is None:
            config = HarnessConfig(ui_headless=True)
        replacements = {
            "setup_logger": lambda **kwargs: None,
            "load_config": lambda path: config,
            "load_stack": lambda path, stack_config: stack,
            "BeamNGClient": lambda *args, **kwargs: client,
        }
        replacements.update(patches)
        monkeypatch.setattr("sys.argv", ["offroad-autonomy", *argv])
        monkeypatch.setattr(app.signal, "signal", Mock())
        for name, value in replacements.items():
            monkeypatch.setattr(app, name, value)
        app.main()
        return app

    return run
