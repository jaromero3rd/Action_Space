# tello_ops

Tooling for standard DJI/Ryze Tello drones (SDK 1.3, not EDU) controlled from this
laptop. Long-term goal: build a 3D map of AprilTags (tag36h11) from the drone camera,
then localize drones against it. Current scope: connection check, live video, camera calibration.

## Safety rule

Propellers are removed, and the code enforces it anyway.
`tello_ops/comms/tello_link.py` only ever sends these exact strings:

    command  streamon  streamoff  battery?  sdk?  sn?  wifi?  time?

Any other command (takeoff, land, rc, motoron, go, flip, emergency, ...) raises
`CommandNotAllowedError` before any socket call. `tests/test_allowlist.py` checks
this, including that a fake drone never receives a blocked command.
Do not widen `ALLOWED_COMMANDS` without a deliberate review.

## Setup (Ubuntu 24.04)

1. Plug in the TP-Link Archer T2U PLUS. The in-kernel `rtw88_8821au` driver picks it
   up as `wlx58d8125ecf0d`.
2. Find the drone's BSSID and record it in `config/drones.yaml` (see
   [Drone identity](#drone-identity)):

       nmcli -f SSID,BSSID,SIGNAL dev wifi list ifname wlx58d8125ecf0d --rescan yes

3. Create a NetworkManager profile pinned to that adapter and BSSID, which never takes
   the default route:

       nmcli con add type wifi con-name TELLO-4 ssid TELLO-4 \
           connection.interface-name wlx58d8125ecf0d 802-11-wireless.bssid <BSSID> \
           ipv4.method auto ipv4.never-default yes ipv6.method ignore \
           connection.autoconnect no
       nmcli con up TELLO-4

   Check that `ip -br -4 addr` shows a `192.168.10.x` address on `wlx58d8125ecf0d`.
4. Create the Python environment:

       python3 -m venv .venv
       .venv/bin/pip install -r requirements.txt setuptools
       .venv/bin/pip install -e .

ROS Jazzy's `PYTHONPATH` leaks into the venv. `pyproject.toml` disables the ROS pytest
plugins that would otherwise crash `pytest`.

## Networking

- The drone is always `192.168.10.1`. `tello_ops/utils/net.py` finds our address on
  `192.168.10.0/24`.
- Every socket binds to that address, never `0.0.0.0`, so each drone can later run
  on its own adapter:
  - commands on `:8889`
  - state on `:8890`
  - video on `:11111`
- `config/drones.yaml` maps a drone name to its interface (or a fixed local IP).

## Drone identity

Every Tello answers at `192.168.10.1`, so the IP alone doesn't tell you which drone
you've reached. `scripts/01_connect_and_view.py --drone NAME` checks the drone against
its `config/drones.yaml` entry and exits with code 2 before `streamon` if it doesn't
match:

1. **Wi-Fi:** the adapter's active SSID and BSSID (from `nmcli -t`) must match
   `ssid` / `bssid`. This runs before anything is sent to the drone.
2. **Serial:** after `command`, the `sn?` reply is compared with `serial`:
   - `null`: not recorded yet. The script prints the reply, says what to add to the
     config, and exits.
   - `unsupported`: SDK 1.3 firmware answers `unknown command: sn?`. The check passes
     only if the drone still says that, so identity rests on the BSSID (the drone's own
     AP MAC address).
   - `"0TQZ..."`: the reply must match exactly.

`tests/test_identity.py` runs the script's `run()` against a fake drone and checks
that every mismatch exits with code 2 and never sends `streamon`.

## Run

    .venv/bin/pytest
    .venv/bin/python scripts/01_connect_and_view.py --drone tello_4    # -v for debug logs

The script:

1. Logs PASS/FAIL for each step: config, local IP, identity (Wi-Fi), SDK mode,
   identity (serial), state packets, streamon, first frame.
2. Prints a connection report.
3. Opens a video window with FPS, battery (from state packets), frame count and
   frame age.

`battery?` is sent every 5 s as a keepalive. Press `q` in the window or Ctrl+C to
send `streamoff` and shut down cleanly.

## Camera calibration

This calibrates the 960x720 video stream (pinhole model, 5 distortion coefficients)
from a ChArUco board (DICT_5X5_100, 7x5 squares) shown on this laptop's screen.
Propellers stay off; you move the drone by hand.

1. **Make the board** (once per screen):

       .venv/bin/python scripts/02_make_charuco.py      # or --width W --height H

   This writes `data/boards/charuco_7x5_2560x1600.png` and a `.yaml` next to it. The
   yaml is the exact board definition that calibration reads.
2. **Show it:** `eog -f data/boards/charuco_7x5_2560x1600.png`. Don't zoom or resize
   afterwards. Set brightness to max and turn the screen so no lamp reflects in it.
3. **Measure** one square edge in mm with a ruler. For accuracy, measure across a
   whole row of 7 squares and divide by 7. Expect about 41 mm on this screen; a
   1% error here becomes a 1% error in every distance measured later.
4. **Capture and calibrate:**

       .venv/bin/python scripts/03_calibrate.py --drone tello_4 --square-mm 41.2

   This runs the same identity checks as `01` (exit 2 on mismatch) and checks the
   frames are 960x720. Then it auto-captures into `data/calib_tello_4_<timestamp>/`.
   There is no window by default (the board covers the screen; `--no-headless` shows
   a preview). Each saved frame plays a shutter sound, and finishing plays a
   "complete" sound.
5. **Move the drone slowly and hold still for each beep.** A frame is saved only if:
   - ≥ 60% of the 24 corners are found;
   - it is sharp (`--min-sharpness`, default 150, Laplacian variance inside the board);
   - it differs from every saved frame in position, size or tilt;
   - at least 0.5 s has passed since the last capture.

   Aim for:
   - the board in every cell of a 4x3 grid over the image, especially the corners
     and edges, where distortion is strongest;
   - tilts of 30–45° left, right, up and down;
   - near and far views.

   Each saved frame logs `saved 12/40 | ... | need: top-left, bottom-right`, and every
   2 s a status line gives the reason the last frame was rejected and its sharpness.
   If sharp-looking frames are rejected as `blurry`, lower `--min-sharpness` to just
   under the values you see. Capture stops at 40 frames, or on Ctrl+C (with ≥ 20
   frames it then calibrates).

Calibration always runs from the files in the session directory, so rerunning it
gives identical numbers:

    .venv/bin/python scripts/03_calibrate.py --from-dir data/calib_tello_4_<timestamp> --fix-k3

Several sessions of the same drone and board can be combined: pass `--from-dir A B`.
The report then goes to a new `data/calib_tello_4_combined_<time>/`, and the source
sessions are left untouched. `--fix-k3` holds k3 at 0; the Tello lens doesn't need it,
and a free k3 tends to trade off against k2.

### Judging the result

The script logs and saves `calib_report.yaml` with two passes. Pass 2 drops frames
whose error is > 3x the median. **PASS** requires all of:
- RMS < 1.0 px;
- coverage ≥ 80% (≥ 10 of 12 cells have ≥ 3 corners);
- ≥ 20 frames;
- a stable fx: each frame is left out once and the calibration rerun, and the fx
  values must span < 0.5%. A large spread means a few frames decide the focal length;
  the report names the most influential one. Add strongly tilted (40–50°) and near/far
  views.

Only a PASS writes `config/camera/tello_4.yaml` (K, dist, rms, frames, coverage, date,
source session). A FAIL writes nothing to `config/` and says what to capture more of.

Check these yourself as well:
- `coverage_map.png`: green cells are covered, red are missing. Sparse corner cells
  mean the edge distortion is poorly constrained, even on a PASS.
- `undistorted_sample.png`: straight edges of the board and the screen bezel should
  come out straight on the right.
- Plausible numbers for a Tello at 960x720: fx ≈ fy ≈ 900–950, cx ≈ 480, cy ≈ 360
  (within a few tens of px), k1 small and negative. An RMS of 0.3–0.7 px is typical
  for its H.264 stream.
- Per-frame errors should be similar. A few frames far above the rest usually mean
  motion blur or glare; pass 2 drops them.

## Layout

    config/drones.yaml     per-drone interface / IP / notes
    config/tags.yaml       placeholder: tag id -> black-square size (m), tag36h11
    config/camera/         per-drone calibration results (written on PASS)
    tello_ops/comms/       UDP command client, allowlist, state listener
    tello_ops/video/       H.264 receiver + PyAV decoder (latest frame only)
    tello_ops/calib/       board, capture gate, coverage, calibration, session files
    tello_ops/session.py   identity-checked connect + video start/stop (all scripts)
    tello_ops/utils/       network discovery, config loading, sounds
    scripts/               numbered entry-point scripts
    data/                  recorded sessions (gitignored)
    tests/                 pytest suite
