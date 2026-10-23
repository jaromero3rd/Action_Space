"""Run ../swarm_ops/swarm_formation.py on the drones the GUI has connected.

The formation script owns its own sockets (8889/8890/11111 per dongle), so the GUI lets go of
the chosen drones first (they show up again once the script exits). Land = Ctrl+C to the script,
the same as pressing Ctrl+C in a terminal.
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque

from tello_unit import DD_ROOT, TELLO, dev_socket

SCRIPT = DD_ROOT / "swarm_ops" / "swarm_formation.py"
FRAME_DIR = DD_ROOT / "outputs" / "formation_frames"


class FormationRunner:
    def __init__(self, fleet) -> None:
        self.fleet = fleet
        self.proc: subprocess.Popen | None = None
        self.lines: deque[str] = deque(maxlen=80)
        self.numbers: list[int] = []
        self.ifaces: list[str] = []
        self.fly = False
        self.exit_code: int | None = None
        self.started = 0.0

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, numbers: list[int], fly: bool) -> str | None:
        """Returns an error string, or None once the script is launched."""
        if self.running:
            return "a formation is already running"
        drones = [d for d in self.fleet.all()
                  if d.number is not None and d.link and (not numbers or d.number in numbers)]
        if numbers:
            missing = sorted(set(numbers) - {d.number for d in drones})
            if missing:
                return f"drone(s) {missing} not connected (use Connect first)"
        if not drones:
            return "no connected drones with a number in config/dongles.json"
        busy = [d.name for d in drones if d.flying or d.busy]
        if busy:
            return f"{busy} flying or busy: land them first"
        self.numbers = sorted(d.number for d in drones)
        self.ifaces = [d.iface for d in drones]
        self.fly, self.exit_code, self.started = fly, None, time.time()
        self.lines.clear()
        shutil.rmtree(FRAME_DIR, ignore_errors=True)
        FRAME_DIR.mkdir(parents=True, exist_ok=True)
        self.fleet.release(self.ifaces)
        time.sleep(0.5)
        cmd = [sys.executable, "-u", str(SCRIPT), "--only", ",".join(map(str, self.numbers))]
        if fly:
            cmd.append("--fly")
        self.lines.append("$ " + " ".join(cmd[2:]))
        self.proc = subprocess.Popen(cmd, cwd=str(SCRIPT.parent), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                     env=dict(os.environ, FORMATION_FRAME_DIR=str(FRAME_DIR)))
        if fly:   # the GUI's confirm dialog already asked "space clear?"; the script wants FLY typed
            self.proc.stdin.write("FLY\n")
            self.proc.stdin.flush()
        threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()
        return None

    def _read(self, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            self.lines.append(line.rstrip()[:200])
        self.exit_code = proc.wait()
        self.lines.append(f"-- formation script exited ({self.exit_code}); drones return to the GUI in a few s")
        self.fleet.unreserve(self.ifaces)

    def stop(self) -> None:
        """Land everything (the script lands on SIGINT)."""
        if self.running:
            self.lines.append("-- LAND requested")
            self.proc.send_signal(signal.SIGINT)

    def emergency(self) -> None:
        """Cut the motors from a fresh socket on each dongle (the script owns the normal ports)."""
        if not self.running:
            return
        for iface in self.ifaces:
            try:
                s = dev_socket(iface, 0)
                s.sendto(b"emergency", (TELLO, 8889))
                s.close()
            except OSError as exc:
                self.lines.append(f"-- emergency on {iface} failed: {exc}")
        self.lines.append("-- EMERGENCY motor stop sent")
        self.proc.send_signal(signal.SIGINT)

    def close(self) -> None:
        if self.running:
            self.stop()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                pass

    def frames(self) -> list[str]:
        """Names of the live views the script is writing (drone cameras, then the top view)."""
        try:
            names = sorted(p.stem for p in FRAME_DIR.glob("*.jpg"))
        except OSError:
            return []
        return [n for n in names if n != "top_view"] + (["top_view"] if "top_view" in names else [])

    def status(self) -> dict:
        return {"frames": self.frames(), "running": self.running, "fly": self.fly, "drones": self.numbers,
                "exit": self.exit_code, "lines": list(self.lines)[-14:]}
