"""Run swarm_ops/swarm_formation.py unchanged, with two GUI adjustments applied at runtime.

  1. Command socket on UDP 8889 (some Tellos, e.g. drone 9, reply only to 8889; SO_BINDTODEVICE
     keeps each dongle's 8889 separate). The original binds 9000+drone number.
  2. cv2.imshow writes JPEGs into $GUI_FRAME_DIR (atomic rename, ~8 per second) instead of
     opening OpenCV windows, so the GUI can embed the views. cv2.waitKey reports no key.

Usage is identical to swarm_formation.py: python gui_formation.py --only 6,7 [--fly]
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

SWARM_OPS = Path(__file__).resolve().parents[1] / "swarm_ops"
sys.path.insert(0, str(SWARM_OPS))
os.chdir(SWARM_OPS)

import cv2  # noqa: E402

import swarm_formation as sf  # noqa: E402

FRAME_DIR = os.environ.get("GUI_FRAME_DIR", "")
_last: dict[str, float] = {}


def _imshow(name, img):
    now = time.time()
    if now - _last.get(name, 0.0) < 0.12:
        return
    _last[name] = now
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ok:
        path = os.path.join(FRAME_DIR, name.replace(" ", "_") + ".jpg")
        with open(path + ".tmp", "wb") as f:
            f.write(buf.tobytes())
        os.replace(path + ".tmp", path)


_orig_sock = sf.Drone._sock
sf.Drone._sock = lambda self, port: _orig_sock(self, 8889 if port >= 9000 else port)
if FRAME_DIR:
    cv2.imshow = _imshow
    cv2.waitKey = lambda *a: 255

if __name__ == "__main__":
    sf.main()
