"""Live regression: a sustained brake must stop, then forward throttle must drive.

Optionally reproduce arcade brake-to-reverse first on a fresh scenario. Parks
and disconnects in all cases. Does not load perception or change user settings.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offroad_autonomy.simulation.beamng_client import BeamNGClient
from offroad_autonomy.types import ControlCommand
from offroad_autonomy.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--reproduce-arcade", action="store_true")
    parser.add_argument("--out", default="output/diagnostics/brake_hold.json")
    args = parser.parse_args()
    client = BeamNGClient(load_config(args.config))
    rows = []

    def sample(phase, command, seconds):
        start = time.perf_counter()
        while time.perf_counter() - start < seconds:
            client.send_controls(command)
            time.sleep(0.2)
            state = client.get_vehicle_state()
            if not state.valid:
                raise RuntimeError("invalid vehicle state during brake test")
            direction = np.asarray(client._vehicle.state["dir"], dtype=float)
            signed_speed = float(np.dot(state.velocity, direction) / np.linalg.norm(direction))
            row = {
                "phase": phase,
                "time_s": time.perf_counter() - start,
                "signed_speed_mps": signed_speed,
                "gear_index": client._vehicle.sensors["electrics"].data.get("gear_index"),
            }
            rows.append(row)
            if phase == "arcade" and signed_speed < -2.0:
                break

    try:
        from beamngpy.sensors import Electrics

        client.connect()
        client._vehicle.sensors.attach("electrics", Electrics())
        if args.reproduce_arcade:
            client._vehicle.set_shift_mode("arcade")
            sample("arcade", ControlCommand(brake=0.8), 5.0)
            # Stop the existing reverse motion before testing a brake hold at
            # standstill. In arcade even park()'s service brake drives reverse.
            client._vehicle.set_shift_mode("realistic_automatic")
            client.park()
            deadline = time.perf_counter() + 8.0
            while client.get_vehicle_state().speed_mps > 0.1:
                if time.perf_counter() > deadline:
                    raise RuntimeError("vehicle did not stop before the brake-hold test")
                time.sleep(0.1)
            client.release_park()
        sample("realistic_brake_hold", ControlCommand(brake=0.8), 6.0)
        sample("forward_restart", ControlCommand(throttle=0.2), 3.0)
    finally:
        client.disconnect()
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    for phase in dict.fromkeys(row["phase"] for row in rows):
        speeds = [r["signed_speed_mps"] for r in rows if r["phase"] == phase]
        gears = sorted({r["gear_index"] for r in rows if r["phase"] == phase}, key=str)
        print(f"{phase}: speed {min(speeds):.3f}..{max(speeds):.3f} m/s, gears {gears}")
    stopped = [r for r in rows if r["phase"] == "realistic_brake_hold"]
    assert stopped and all(r["signed_speed_mps"] > -0.15 for r in stopped), "brake caused reverse"
    assert all(r["gear_index"] is not None and r["gear_index"] >= 0 for r in stopped)
    forward = [r["signed_speed_mps"] for r in rows if r["phase"] == "forward_restart"]
    assert forward and max(forward) > 1.0, "forward drive did not resume"
    assert any(r["gear_index"] > 0 for r in rows if r["phase"] == "forward_restart")


if __name__ == "__main__":
    main()
