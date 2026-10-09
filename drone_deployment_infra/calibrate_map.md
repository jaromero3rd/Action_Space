# Calibrate & map (tello_calibration)

Commands to calibrate a Tello's camera (focal length) and build / use an AprilTag room
map in the **tag-10 origin frame**. Everything lives in the `tello_calibration/` package
and uses its no-takeoff comms allowlist — these scripts never fly the drone.

Examples use **drone_3** (`tello_3`). Swap `tello_3` for `tello_4` etc.

## 0. One-time setup

The package is editable-installed into the repo venv, so scripts import cleanly:

```bash
~/action-space/.venv/bin/pip install -e ~/action-space/tello_calibration
```

Run everything from the package directory with the repo venv:

```bash
cd ~/action-space/tello_calibration
PY=~/action-space/.venv/bin/python
```

Config lives in `tello_calibration/config/`:
- `fleet.yaml` `drones:` — one entry per drone (iface, ssid, bssid). `tello_3` → stick `wlx6c4cbce33375`.
- `camera/<drone>.yaml` — camera matrix + distortion (focal length = `camera_matrix[0][0]`).
- `tags.yaml` — `origin_tag: 10`; `sizes_mm` lists every tag to map (1-9 = 132.08 mm, 10 = 571.5 mm).
- `map.yaml` — the room map (written by step 4).

## 1. Connect to the drone

Power on TELLO-3, plug in its stick, bring up its profile, then confirm identity + video:

```bash
nmcli con up TELLO-3                                   # stick wlx6c4cbce33375
$PY scripts/01_connect_and_view.py --drone tello_3     # -v for debug logs; identity-gated live view
```

## 2. Calibrate the camera (focal length) — only if `camera/<drone>.yaml` is missing

```bash
$PY scripts/02_make_charuco.py                         # print the ChArUco board (fills the page)
# measure one printed black square in mm, then:
$PY scripts/03_calibrate.py --drone tello_3 --square-mm 41.2
```

Move the board slowly across the whole frame until coverage passes. It writes
`config/camera/tello_3.yaml` (the focal length and distortion). Re-run on saved frames:

```bash
$PY scripts/03_calibrate.py --from-dir data/calib_tello_3_<timestamp> --fix-k3
```

Check detections/thresholds live first if needed:

```bash
$PY scripts/04_tag_check.py --drone tello_3            # per-tag distance, angle, accept/reject
```

## 3. Choose which tags to map

Every tag you want in the map must have its black-square size in `config/tags.yaml`
`sizes_mm` (already set: 1-9 = 132.08 mm, 10 = 571.5 mm). Tag 10 is the origin — it must
be seen together with the others so they link into one frame.

## 4. Build the map (tag-10 frame)

```bash
$PY scripts/05_map.py --drone tello_3                  # writes config/map.yaml
```

Point the drone so the camera sees **tag 10 together with other tags** (two tags in one
frame links them; a chain 10↔6↔5 works too). The window shows `mapped N: [...]`; green =
placed in the map, yellow = seen but not yet linked to the origin. Press `q` to finish —
it prints each tag's x/y/z in the tag-10 frame and saves `config/map.yaml`.

Save to a specific file:

```bash
$PY scripts/05_map.py --drone tello_3 --out config/map_room.yaml
```

## 5. Localize the drone against the map

```bash
$PY scripts/06_localize.py --drone tello_3             # reads config/map.yaml
```

From any mapped tag in view it prints the drone's position relative to tag 10
(`drone @ origin x.. y.. z..`), using the nearest tag and reporting the spread when
several are visible. Use `--map config/map_room.yaml` for a non-default map.

## 6. Self-checks (no drone needed)

```bash
$PY -m tello_calibration.tags.mapping                  # mapping math self-check
```

## Notes

- Coordinates are in the **tag-10 frame**: x right, y up, z out of the tag face, in metres.
- A tag is only placed if it shares a frame (directly or via a chain) with tag 10.
- The map + any later image is all `06_localize.py` (or `TagMap.camera_in_origin` in
  `tello_calibration/tags/mapping.py`) needs to place the drone relative to origin.
