# config

Central configuration for real-world drone deployment. Read by the `cal_and_map`
package, the `core/` scripts, and the tests.

| File | What it holds |
|---|---|
| `tags.yaml` | Which AprilTags go in the map and their sizes. `origin_tag: 10` (the map origin). `sizes_mm`: black-square edge per tag id (1-9 = 132.08 mm / 5.2 in, 10 = 571.5 mm / 22.5 in). Also the detector options and the quality gate. A tag must have a size here to be mapped. |
| `camera/<drone>.yaml` | Camera calibration per drone, written by `cal_and_map/scripts/03_calibrate.py`. The **focal length** is `camera_matrix[0][0]` (fx); also fy, principal point, distortion, and the calibration RMS/coverage. |
| `drones.yaml` | Per-drone network identity for the calibration comms: USB `iface`, `ssid`, access-point `bssid`, serial. Used to verify we're talking to the right drone. |
| `dongles.json` | Which USB WiFi stick flies which drone: `number`, `iface` (`wlx<mac>`), `mac`, `drone`, `video_port`. Written/edited by `core/tello_dongle_setup.py`. |
| `fleet.yaml` | Which drones we fly and their roles: `fly_count`, `roles.attack`, `roles.defense`, and `quick_connect` (drones brought up first for calibration/mapping/single-drone and small-swarm tests). |
| `map.yaml` | The room map (written by `cal_and_map/scripts/05_map.py`): every mapped tag's pose in the tag-10 frame. Read by `core/positional.py` to localize the drone. |

Notes:
- Drone names (e.g. `TELLO-3`) must match across `dongles.json`, `fleet.yaml`, and
  `camera/<drone>.yaml`.
- `cal_and_map` reads this folder via `CONFIG_DIR` (`tello_calibration/utils/config.py`),
  and `core`/tests resolve it relative to `drone_deployment_infra/`.
