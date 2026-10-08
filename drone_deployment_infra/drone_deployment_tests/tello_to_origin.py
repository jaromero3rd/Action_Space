#!/home/jaimeromero/action-space/.venv/bin/python
"""Fly a single drone to the origin (tag 10), using the MAP, and land at 1.5 m.

This drives off the mapping/localization pipeline (core/positional.py + config/map.yaml),
not a single tag. Each frame it computes the drone's position in the tag-10 frame from
ANY mapped tag(s) in view (5, 6, 7, 10, ...), then yaws to face the origin and flies
forward toward it, climbing/descending to keep it level. Because it fuses whatever tags
are visible, covering any one tag (even tag 10) does not stop it. Lands within 1.5 m of
the origin.

Single quick-connect drone over djitellopy. Lands on Ctrl-C or after --seconds.
KEEP THE AREA CLEAR AND BE READY TO CATCH IT.

  ./tello_to_origin.py                    # first fleet.yaml quick_connect drone, 1.5 m
  ./tello_to_origin.py --drone TELLO-3 --range 1.5 --seconds 40
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import yaml
from djitellopy import Tello

DD_ROOT = Path(__file__).resolve().parent.parent
CONFIG = DD_ROOT / "config"
sys.path.insert(0, str(DD_ROOT / "core"))
from positional import Positioner  # noqa: E402

# Control gains / limits (rc units, -100..100). Gentle on purpose.
K_YAW_DEG, MAX_YAW = 1.6, 40     # yaw per degree of bearing to the origin
K_UD, MAX_UD = 60.0, 25          # up/down per unit of vertical direction
K_FWD, MAX_FWD, MIN_FWD = 35.0, 25, 10
FACE_TOL_DEG = 12                # must be roughly facing the origin to land


def quick_connect_drone(override: str | None) -> str:
    if override:
        return override
    qc = (yaml.safe_load((CONFIG / "fleet.yaml").read_text()) or {}).get("quick_connect") or []
    if not qc:
        raise SystemExit("no quick_connect drones in config/fleet.yaml")
    return str(qc[0])


def dongle_iface(drone: str) -> str | None:
    reg = json.loads((CONFIG / "dongles.json").read_text())
    for d in reg.get("dongles", []):
        if d.get("drone") == drone:
            return d.get("iface")
    return None


def clamp(value: float, limit: int) -> int:
    return int(max(-limit, min(limit, value)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drone", help="ssid like TELLO-3; default = first fleet.yaml quick_connect")
    parser.add_argument("--range", type=float, default=1.5, help="land this far from the origin, metres")
    parser.add_argument("--seconds", type=float, default=40.0, help="safety cap; land after this")
    args = parser.parse_args()

    ssid = quick_connect_drone(args.drone)
    key = ssid.lower().replace("-", "_")
    print(f"to-origin on {ssid} via the map; stop at {args.range:.1f} m. Be ready to catch it.", flush=True)
    iface = dongle_iface(ssid)
    if iface:
        subprocess.run(["nmcli", "con", "up", ssid, "ifname", iface], capture_output=True, text=True, timeout=30)

    positioner = Positioner(key)   # loads config/map.yaml + config/camera/<key>.yaml
    print(f"map tags: {positioner.tag_map.mapped()}", flush=True)

    tello = Tello()
    tello.connect()
    print(f"battery {tello.get_battery()}%", flush=True)
    tello.streamon()
    reader = tello.get_frame_read()

    time.sleep(2.0)
    tello.takeoff()
    start = time.time()
    try:
        while time.time() - start < args.seconds:
            frame = reader.frame
            if frame is None or frame.shape[1] < 600:
                time.sleep(0.03)
                continue
            loc = positioner.locate(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if loc is None:
                tello.send_rc_control(0, 0, 0, 0)   # no mapped tag in view -> hold, don't fly blind
                time.sleep(0.05)
                continue
            rng = loc["range"]
            vx, vy, vz = loc["to_origin_cam"]       # camera frame: x right, y down, z forward
            bearing_deg = math.degrees(math.atan2(vx, vz))   # + = origin is to the right
            if rng <= args.range and abs(bearing_deg) < FACE_TOL_DEG:
                print(f"within {rng:.2f} m of origin, facing it -> landing (tags {loc['tags']})", flush=True)
                break
            yaw = clamp(bearing_deg * K_YAW_DEG, MAX_YAW)
            up = clamp(-vy * K_UD, MAX_UD)          # origin above -> climb
            error = rng - args.range
            fwd = clamp(error * K_FWD, MAX_FWD) if error > 0 else 0
            if 0 < fwd < MIN_FWD:
                fwd = MIN_FWD
            if abs(bearing_deg) > 20:               # face the origin before committing forward
                fwd = min(fwd, MIN_FWD)
            tello.send_rc_control(0, fwd, up, yaw)
            print(f"range {rng:4.2f} m  bearing {bearing_deg:+5.1f}  tags {loc['tags']}  "
                  f"rc(f={fwd} u={up} yaw={yaw})", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Ctrl-C -> landing", flush=True)
    finally:
        try:
            tello.send_rc_control(0, 0, 0, 0)
            tello.land()
        except Exception as exc:
            print(f"land error: {exc}", flush=True)
        try:
            tello.streamoff()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
