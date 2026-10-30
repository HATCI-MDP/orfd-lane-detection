"""Ground-truth vehicle position for evaluation logs, off the control loop.

BeamNG answers requests once per rendered frame, so any extra request in the
control loop usually waits a whole game frame (25-33 ms). Reading the true
position that way slowed a logged loop from 10 Hz to 7.7 Hz, and the car
drove differently because it was being measured. This sampler instead opens
its own connection and polls the game engine on its own thread, which left
the control loop's request time unchanged (64.8 vs 64.9 ms) and samples at
about 30 Hz.

Nothing that steers may read it: the stack must stay camera plus wheel/gyro
only, so the truth exists solely to score a run.
"""

from __future__ import annotations

import json
import logging
import threading
import time

logger = logging.getLogger("offroad_autonomy.simulation")

# The game engine's object position for the vehicle; measured equal to the
# vehicle's reference node (BeamNG's state "pos") to within 1 mm.
_POSITION_QUERY = """
local o = scenetree.findObject('{name}')
if not o then return nil end
local p = o:getPosition()
return jsonEncode({{x = p.x, y = p.y, z = p.z}})
"""


class PositionSampler:
    def __init__(self, host: str, port: int, vehicle_name: str, rate_hz: float) -> None:
        self._host = host
        self._port = port
        self._query = _POSITION_QUERY.format(name=vehicle_name)
        self._period = 1.0 / float(rate_hz)
        self._bng = None
        self._samples: list[tuple[float, float, float, float]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.failures = 0

    def start(self) -> None:
        from beamngpy import BeamNGpy

        self._bng = BeamNGpy(self._host, self._port)
        self._bng.open(launch=False)
        # A misnamed vehicle would otherwise log nothing for a whole run.
        if self._read() is None:
            self._bng.disconnect()
            raise RuntimeError("Position sampler cannot find the vehicle in the scene")
        self._thread = threading.Thread(target=self._run, name="position-sampler", daemon=True)
        self._thread.start()

    def drain(self) -> list[tuple[float, float, float, float]]:
        """Samples since the last call, as (host monotonic time, x, y, z)."""
        with self._lock:
            samples, self._samples = self._samples, []
        return samples

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._bng is not None:
            # disconnect, never close: close() would quit the simulator the
            # control connection is still parking the vehicle on.
            self._bng.disconnect()
            self._bng = None

    def _read(self) -> tuple[float, float, float, float] | None:
        reply = self._bng.control.queue_lua_command(self._query, response=True)
        if isinstance(reply, str):
            reply = json.loads(reply)
        if not isinstance(reply, dict):
            return None
        return (time.perf_counter(), float(reply["x"]), float(reply["y"]), float(reply["z"]))

    def _run(self) -> None:
        next_time = time.perf_counter()
        while not self._stop.is_set():
            try:
                sample = self._read()
            except Exception as exc:
                # A missed sample is a gap in the log; it must not end the run.
                self.failures += 1
                logger.debug("Position sample failed: %s", exc)
                sample = None
            if sample is not None:
                with self._lock:
                    self._samples.append(sample)
            next_time += self._period
            self._stop.wait(max(0.0, next_time - time.perf_counter()))
