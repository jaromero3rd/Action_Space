#!/usr/bin/env python3
"""drone_gui with Auto flying higher than its tag.

  ../../../.venv/bin/python -u drone_gui_high.py                  # 0.5 m above the tag
  ../../../.venv/bin/python -u drone_gui_high.py --above 0.3      # any height, metres
  ../../../.venv/bin/python -u drone_gui_high.py --above 0.5 --port 8781

Same page, buttons and safety limits as ../../drone_gui/drone_gui.py (that file is
not changed). Only Auto differs: stock Auto keeps the tag centred vertically in the
picture, i.e. flies at tag height; this keeps the tag ABOVE_M below centre instead.

The camera sees only ~21 deg above/below centre, so the aim is capped at MAX_DOWN_DEG
to keep the tag in the picture: the full height is reached from ~2.2 m out; at the
1.3 m standoff it settles ~0.3 m above. Raise the standoff in the page for the full
0.5 m at the landing point.
"""

from __future__ import annotations

import math
import runpy
import sys
import time
from pathlib import Path

GUI_DIR = Path(__file__).resolve().parents[2] / "drone_gui"
MAX_DOWN_DEG = 13.0   # tag at most this far below centre (half the vertical FOV is ~21 deg)
ELEV_DEADBAND_DEG = 3.0   # vertical aim error tolerated before climbing/sinking

# Search when the tag is lost, as in ../../swarm_ops/swarm_formation_flip.py --fast:
# hold SEARCH_HEIGHT_M, look, turn a 30 deg step, look again ... up to a full turn,
# then land. Stock Auto only swung +-20 deg (rc 20 turns ~10 deg/s).
SEARCH_HEIGHT_M = 1.55    # formation settled here, ~tag 10 centre seen from takeoff height
KP_HEIGHT = 50.0          # rc per metre of height error
MAX_RC_Z = 50             # measured: rc 20 climbed only ~7.5 cm/s
SEARCH_LOOK_S = 0.5       # pause after each turn (stationary frames decode, turning ones blur)
SEARCH_STEP_DEG = 30      # camera sees ~54 deg wide, steps still overlap
SEARCH_YAW_RC = 70
SEARCH_DEG_PER_S = 40.0   # turn rate assumed at SEARCH_YAW_RC (no IMU yaw in the GUI)
SEARCH_MAX_DEG = 380      # a full turn without the tag -> land
LOST_LAND_S = 30.0        # stock 10 s would land before the full turn ends


def search_sticks(drone, now: float) -> tuple[int, int, int, int]:
    s = drone.__dict__.setdefault("_search", {})
    if s.get("lost_since") != drone.lost_since:           # a new loss: start over
        s.update(lost_since=drone.lost_since, sub="look", sub_t=now, searched=0.0)
    h = drone.height
    up = 0 if h is None else int(max(-MAX_RC_Z, min(MAX_RC_Z, KP_HEIGHT * (SEARCH_HEIGHT_M - h))))
    yaw = 0
    if s["sub"] == "look":
        if now - s["sub_t"] > SEARCH_LOOK_S:
            s.update(sub="turn", sub_t=now)
    else:
        turned = (now - s["sub_t"]) * SEARCH_DEG_PER_S
        if turned >= SEARCH_STEP_DEG:
            s["searched"] += turned
            s.update(sub="look", sub_t=now)
        else:
            yaw = SEARCH_YAW_RC
    if s["searched"] >= SEARCH_MAX_DEG:
        drone.land(f"tag {drone.target_tag} not found in a full turn")
        return (0, 0, 0, 0)
    drone.msg = f"search: tag {drone.target_tag}, turned ~{s['searched']:.0f} deg, h {h} m"
    return (0, 0, up, yaw)


def main() -> None:
    above = 0.5
    if "--above" in sys.argv:
        i = sys.argv.index("--above")
        above = float(sys.argv[i + 1])
        del sys.argv[i:i + 2]

    sys.path.insert(0, str(GUI_DIR))
    import tello_unit

    stock = tello_unit.approach_rc

    def approach_rc_high(obs, standoff, pids, dt):
        aim = min(math.degrees(math.atan2(above, max(obs["range"], 0.1))), MAX_DOWN_DEG)
        return stock(dict(obs, elev=obs["elev"] - aim), standoff, pids, dt)   # elev: + = tag below centre

    import sim_drone

    def with_search(stock_send_rc):
        def send_rc(self, sticks):
            # The stock loop sends (0, 0, 0, +-20) in search mode; replace it with the stepped search.
            if self.mode == "search" and self.lost_since is not None:
                sticks = search_sticks(self, time.time())
            stock_send_rc(self, sticks)
        return send_rc

    tello_unit.approach_rc = approach_rc_high
    tello_unit.Drone.send_rc = with_search(tello_unit.Drone.send_rc)
    sim_drone.SimDrone.send_rc = with_search(sim_drone.SimDrone.send_rc)   # --sim runs the same search
    tello_unit.ELEV_DEADBAND_DEG = ELEV_DEADBAND_DEG   # stock 8 deg stops the climb ~0.3 m short
    tello_unit.LOST_LAND_S = LOST_LAND_S
    print(f"[high] Auto flies {above:.2f} m above its tag (aim capped at {MAX_DOWN_DEG:.0f} deg below centre, "
          f"deadband {ELEV_DEADBAND_DEG:.0f} deg); search holds {SEARCH_HEIGHT_M} m and turns "
          f"{SEARCH_STEP_DEG} deg steps up to {SEARCH_MAX_DEG} deg", flush=True)
    sys.argv[0] = str(GUI_DIR / "drone_gui.py")
    runpy.run_path(str(GUI_DIR / "drone_gui.py"), run_name="__main__")


if __name__ == "__main__":
    main()
