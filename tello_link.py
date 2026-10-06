"""Join TELLO-3 and TELLO-4 when their radios appear.

Each disconnected dongle is checked about every 8 seconds. A drone that is
already associated is left alone. Home WiFi on the built-in radio is never touched.
"""

import subprocess
import time
from pathlib import Path

ROOT = Path("/home/jaimeromero/action-space")
PYTHON = ROOT / ".venv" / "bin" / "python"
VIDEO = ROOT / "tello_dual_video.py"
PERIOD_S = 8

# SSID, USB interface, NetworkManager profile.
DRONES = (
    ("TELLO-3", "wlx58d8125eda77", "TELLO-3"),
    ("TELLO-4", "wlx6c4cbce344fc", "TELLO-4"),
)


def _nmcli(args, timeout=25):
    return subprocess.run(
        ["nmcli", *args],
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def device_links():
    result = _nmcli(["-t", "-f", "DEVICE,STATE,CONNECTION", "device", "status"])
    links = {}
    for line in result.stdout.splitlines():
        parts = line.split(":")
        if len(parts) >= 3:
            links[parts[0]] = (parts[1], parts[2])
    return links


def ssid_visible(iface, ssid):
    _nmcli(["device", "wifi", "rescan", "ifname", iface], timeout=15)
    time.sleep(3)
    result = _nmcli(["-t", "-f", "SSID", "device", "wifi", "list", "ifname", iface])
    if any(line.split(":")[0] == ssid for line in result.stdout.splitlines()):
        return True
    # The built-in radio often hears the drone when the USB stick's scan is stale.
    # Do not join on the built-in radio. A sighting there only means the AP is up.
    cached = _nmcli(["-t", "-f", "SSID", "device", "wifi", "list"])
    return any(line.split(":")[0] == ssid for line in cached.stdout.splitlines())


def connect_drone(links, spec):
    """Join one Tello on its USB dongle. True when that link is up."""
    ssid, iface, profile = spec
    state, connection = links.get(iface, ("missing", ""))
    if state.startswith("connected") and connection == profile:
        return True
    if state.startswith("connect"):
        print(f"{ssid} still joining", flush=True)
        return False
    if not ssid_visible(iface, ssid):
        return False
    print(f"{ssid} found, joining {iface}", flush=True)
    result = _nmcli(["connection", "up", profile, "ifname", iface], timeout=30)
    if result.returncode == 0:
        print(f"{ssid} connected", flush=True)
        return True
    err = (result.stderr or result.stdout).strip().splitlines()
    print(f"{ssid} join failed: {err[-1] if err else result.returncode}", flush=True)
    return False


def video_running():
    listing = subprocess.run(["ps", "-eo", "args"], text=True, capture_output=True, check=False)
    for line in listing.stdout.splitlines():
        if "tello_dual_video.py" in line and "python" in line:
            return True
    return False


def ensure_video():
    if video_running():
        return
    print("start video", flush=True)
    subprocess.Popen(
        [str(PYTHON), "-u", str(VIDEO), "--play"],
        cwd=ROOT,
        start_new_session=True,
    )


def main():
    print("link watch every", PERIOD_S, "s", flush=True)
    ensure_video()
    cycle = 0
    while True:
        started = time.time()
        connected = []
        for spec in DRONES:
            connected.append(connect_drone(device_links(), spec))
        cycle += 1
        if all(connected):
            if cycle == 1 or cycle % 5 == 0:
                print("both drones connected", flush=True)
        ensure_video()
        time.sleep(max(0, PERIOD_S - (time.time() - started)))


if __name__ == "__main__":
    main()
