# drone_deployment_tests

The policies we're developing to drive the drone down the track, plus the link/flight
tests used to validate them. The flight engine (`core/tello_dual_video.py`) imports the
active policy from here.

Run with the repo venv; the test scripts are executable (venv shebang), so `./name.py`
works, or `../../.venv/bin/python <script>`.

| File | What | Run |
|---|---|---|
| `tello_policy.py` | The current track-driving policy: per-axis **PID** (yaw / lateral / distance), yaw-first alignment, approach to `LAND_RANGE_M` (1.3 m), and the search-on-tag-loss behavior. Tunable gains at the top. Imported by the engine. | (library — flown via `core/tello_link.py`) |
| `tello_range_test.py` | Walk-range test: carry the drone (no takeoff) away from the base at tag 10 and read the max range it stays connected, using the tag-10 distance as the ruler. | `./tello_range_test.py [TELLO-3]` |
| `tello_hover_test.py` | Link test under flight: takes off and holds (or flies a forward/back pattern), streaming rc steadily with passive state-based link detection; logs disconnect/reconnect timing. | `./tello_hover_test.py [--rate 20] [--speed 30 --leg 3] [--seconds 60]` |

New policies go here as separate modules; point the engine at the one you want to fly.
Flight recordings and per-run logs land in `../recordings/` and `../outputs/`.

Safety: these command real motors. Keep the area clear and be ready to catch the drone;
`tello_link.py` lands on signal loss and won't re-launch mid-flight.
