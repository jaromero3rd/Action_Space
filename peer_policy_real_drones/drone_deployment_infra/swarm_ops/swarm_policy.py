#!/usr/bin/env python3
"""Fly the attacking drones with the trained 4-drone room policy (AS-Tello-Room-Peer-v0).

Four Tellos start on the floor 6-7 m from tag 10, take off, and the policy flies each one to
its landing spot (1 m in front of the tag-10 wall, 0.5 m apart), then each lands on its own.
Everything the policy sees is built exactly like the simulator built it in training:

  camera frame -> core/positional.py Positioner (every mapped tag) -> body centre + yaw
  (room_policy.camera_pose_to_world) -> the laptop Kalman filter, updated in the simulator's
  order (predict, restart if back from lost, fuse, freeze when lost) -> core/room_policy.py
  RoomPolicy (observation, 8-frame history, normalizer, policy, waypoint + landing gate) ->
  one `rc a b c d` per drone every 74 ms, and `land` when the policy's landing gate says so.

DEFAULT IS DRY-RUN: no takeoff, no rc sent; it prints what it WOULD send.  --fly to fly.

  python swarm_policy.py                                   # dry-run, attack drones from fleet.yaml
  python swarm_policy.py --drones TELLO-3,TELLO-4,TELLO-5,TELLO-1 --fly
  python swarm_policy.py --drones ... --spots 2,0,3,1 --fly

Spots are numbered 0-3 from LEFT to RIGHT as you face tag 10 (Y = -0.75, -0.25, 0.25,
0.75 m). Default (--spots auto): from the floor fixes, the leftmost drone takes spot 0, the
next spot 1, ... so no two paths cross. The policy was trained on every assignment and
--spots 2,0,3,1 forces one, but crossing paths can pass ~0.3 m apart: the simulator only
counts < 0.25 m as a collision, and --min-sep then lands everyone.

Safety (any one lands every drone): q in the status window, Ctrl+C, an error, a drone's
link going silent, a filtered position leaving the room, two airborne drones closer than
--min-sep, --max-time, a drone without a tag for --lost-land seconds, a drone at the landing
battery. (One drone down mid-room is a state the policy never trained on: the others then
hovered short of their waypoints in a replay, so the team lands together.) SPACE in the
status window = EMERGENCY motor stop all.

Writes recordings/<NNN>_SWARM.policy.jsonl (one row per tick: feed, observation-ready
state, rc, land) and <NNN>_SWARM.policy_meta.json / .policy_report.json.

NEEDS: config/dongles.json (dongle <-> drone), config/map.yaml + tags.yaml (must match the
policy's training snapshot; checked), camera calibration config/camera/tello_<n>.yaml
(default.yaml is used for uncalibrated drones, with a warning), and numpy opencv-python
pupil-apriltags av pyyaml torch. A model.zip checkpoint also needs stable-baselines3; the
default is the TorchScript export policy.pt (export_policy_pt.py), which needs torch only.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DD_ROOT = HERE.parent
REPO = DD_ROOT.parent
for _p in (DD_ROOT / "core", DD_ROOT / "cal_and_map", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fleet import load_fleet_config  # noqa: E402
from positional import Positioner  # noqa: E402
from positional_kalman_filter import KalmanCfg, KalmanCV  # noqa: E402
from room_policy import RoomPolicy, camera_pose_to_world  # noqa: E402
from swarm_approach import Drone, iface_ip, parallel, preflight  # noqa: E402

DEFAULT_CHECKPOINT = REPO / "simulation_infra/outputs/qualified_four_drone_policy_pt/policy.pt"
DONGLES = DD_ROOT / "config" / "dongles.json"
RECORDINGS = DD_ROOT / "recordings"



class _TorchScriptModel:
    """The slice of SB3's PPO that RoomPolicy uses, backed by export_policy_pt.py's policy.pt."""

    def __init__(self, module):
        import torch

        self.torch, self.module = torch, module
        self.observation_space = SimpleNamespace(shape=(int(module.obs_dim),))
        self.action_space = SimpleNamespace(shape=(int(module.act_dim),))

    def predict(self, history, deterministic=True):
        x = self.torch.as_tensor(np.asarray(history, np.float32)).reshape(-1, self.observation_space.shape[0])
        with self.torch.no_grad():
            return self.module(x).numpy(), None


class _FrameNormalizer:
    """VecNormalize.normalize_obs with the stats stored in policy.pt."""

    def __init__(self, module):
        self.mean, self.var = module.obs_mean.numpy(), module.obs_var.numpy()
        self.clip, self.eps = float(module.clip_obs), float(module.epsilon)

    def normalize_obs(self, obs):
        return np.clip((obs - self.mean) / np.sqrt(self.var + self.eps), -self.clip, self.clip).astype(np.float32)


def load_policy(ckpt: Path) -> RoomPolicy:
    """RoomPolicy for a model.zip (needs stable-baselines3) or a policy.pt export (torch only).

    For policy.pt, RoomPolicy's constructor is bypassed (it calls PPO.load) and the same
    manifest + contract checks are repeated here; step()/reset() are RoomPolicy's own.
    """
    if ckpt.suffix != ".pt":
        return RoomPolicy(ckpt)
    import torch

    from room_policy import namespace, peer_feature_count

    policy = RoomPolicy.__new__(RoomPolicy)
    policy.manifest = json.loads((ckpt.parent / "params/run.json").read_text())
    policy.cfg = cfg = namespace(policy.manifest["env"])
    if not cfg.land_gate_enabled or not cfg.kalman.enabled:
        raise ValueError("Deployment adapter requires the estimated-state gate and Kalman filtering")
    if cfg.rc.use_envelope:
        raise ValueError("policy.pt loading does not support rc.use_envelope; use the model.zip")
    module = torch.jit.load(str(ckpt), map_location="cpu").eval()
    policy.model = _TorchScriptModel(module)
    policy.normalizer = _FrameNormalizer(module) if policy.manifest["agent"].get("normalize_input") else None
    policy.n_stack = policy.manifest["agent"].get("frame_stack", 1)
    policy.central = bool(policy.manifest["agent"].get("central", False))
    own_dim = (25 if cfg.start_on_floor else 24) + (peer_feature_count(cfg.num_drones) if getattr(cfg, "peer_observations", False) else 0)
    per = cfg.num_drones if policy.central else 1
    if policy.model.observation_space.shape != (own_dim * per * policy.n_stack,):
        raise ValueError("Checkpoint observation layout does not match the room contract")
    if policy.model.action_space.shape != (5 * per,):
        raise ValueError("Checkpoint action layout does not match the room contract")
    policy.lo, policy.hi = -np.asarray(cfg.rc.rc_limit), np.asarray(cfg.rc.rc_limit)
    policy.reset()
    return policy


MIN_BATTERY_FLY = 40         # % to take off
LAND_BATTERY = 15            # % -> that drone lands
LINK_SILENT_S = 2.0          # no state packet this long -> land all
ROOM_MARGIN_M = 0.1          # filtered position this far outside the room -> land all
FLOOR_FIX_S = 15.0           # preflight: each drone must see a tag on the floor within this


class PolicyDrone(Drone):
    """swarm_approach.Drone with the room localizer instead of the tag-10-only solver."""

    def __init__(self, name, num, iface, ssid, fly, cfg):
        super().__init__(num, iface, ssid, None, None, fly)
        self.name = name
        self.cfg = cfg
        self.positioner = Positioner(f"tello_{num}")
        self.fix = None              # (world position, yaw, frame time, frame id) of the newest located frame
        self.fix_lock = threading.Lock()
        self.taking_off = False
        self.landing = False         # land sent (or being sent)
        self.land_thread = None

    def perception_loop(self):
        cfg, last = self.cfg, 0
        ts = cfg.tag_sensor
        while not self.stop.is_set():
            if self.frame_id == last or self.frame is None:
                time.sleep(0.003)
                continue
            last, img, t = self.frame_id, self.frame, self.frame_t
            try:
                r = self.positioner.locate(img)
            except Exception as ex:  # a bad frame never kills the loop
                self.reason = f"locate error: {ex}"
                continue
            if r is None:
                self.reason = "no mapped tag"
                continue
            pos, yaw = camera_pose_to_world(r["xyz"], r["R"], cfg.tag10_height,
                                            tuple(ts.camera_offset), ts.camera_pitch)
            with self.fix_lock:
                self.fix = (np.asarray(pos, float), float(yaw), t, last)
            self.reason = f"tags {r['tags']}"

    def height_m(self):
        """Like the simulator's height feed: metres above the floor in 0.1 m steps."""
        if not self.airborne or self.landed:
            return 0.0
        cm = self.height_cm()                       # downward ToF, None if stale / out of range
        if cm is None:
            try:
                cm = int(self.st.get("h", 0))
            except ValueError:
                cm = 0
        return round(cm / 100.0, 1)

    def land_async(self):
        if self.landing or self.landed:
            return
        self.landing = True
        if not self.fly:
            self.landed = True
            return
        self.land_thread = threading.Thread(target=self.land, daemon=True)
        self.land_thread.start()


def load_drones(names, fly, cfg):
    data = json.loads(DONGLES.read_text())
    entries = {e["drone"]: e for e in data.get("dongles", data) if e.get("drone")}
    missing = [n for n in names if n not in entries]
    if missing:
        sys.exit(f"Not in {DONGLES}: {missing} (wire them with core/tello_dongle_setup.py)")
    drones = []
    for n in names:
        e = entries[n]
        num = int(re.sub(r"\D", "", n))
        cam = DD_ROOT / "config" / "camera" / f"tello_{num}.yaml"
        if not cam.exists():
            print(f"[{n}] WARNING: no camera calibration {cam.name}; using camera/default.yaml "
                  f"(positions will be less accurate)")
        drones.append(PolicyDrone(n, num, e["iface"], e["ssid"], fly, cfg))
    return drones


def spot_positions(cfg):
    k = np.arange(cfg.spot_count) - 0.5 * (cfg.spot_count - 1)
    return np.stack([np.full_like(k, cfg.spot_distance, dtype=float), k * cfg.spot_spacing, 0 * k], -1)


def check_room_snapshot(policy, ckpt):
    """The policy learned one map: refuse to fly with a different one."""
    assets = policy.manifest.get("assets", {})
    for key, live in (("room_map", DD_ROOT / "config/map.yaml"), ("room_tags", DD_ROOT / "config/tags.yaml")):
        if key not in assets:
            continue
        want = hashlib.sha256((ckpt.parent / assets[key]["path"]).read_bytes()).hexdigest()
        got = hashlib.sha256(live.read_bytes()).hexdigest()
        if want != got:
            sys.exit(f"{live} differs from the map the policy was trained on ({assets[key]['path']}). "
                     f"Re-train on the current map or restore it.")


def next_session():
    RECORDINGS.mkdir(exist_ok=True)
    nums = [int(m[1]) for p in RECORDINGS.iterdir() if (m := re.match(r"(\d{3})_", p.name))]
    return f"{max(nums, default=0) + 1:03d}"


def draw_status(drones, spots, feed, now, t0, fly):
    S, W, H = 70, 640, 520                          # px per metre: X (out of the wall) down, Y right
    img = np.full((H, W, 3), 30, np.uint8)
    ox, oy = W // 2, 30

    def px(p):
        return int(ox + S * p[1]), int(oy + S * p[0])

    cv2.line(img, px((0, -2.0)), px((0, 1.7)), (200, 200, 200), 2)  # tag-10 wall
    for i, s in enumerate(spots):
        cv2.circle(img, px(s), int(0.3 * S), (90, 90, 90), 1)
        cv2.putText(img, str(i), px(s), 0, 0.5, (150, 150, 150), 1)
    colors = [(0, 128, 255), (255, 200, 0), (200, 50, 255), (60, 255, 140)]
    for i, d in enumerate(drones):
        p = feed["position"][i]
        c = colors[i % 4]
        cv2.circle(img, px(p), 6, c, -1 if feed["fresh"][i] else 1)
        cv2.line(img, px(p), px(feed["target"][i]), c, 1)
        state = ("LANDED" if d.landed else "LANDING" if d.landing else "TAKEOFF" if d.taking_off
                 else "LOST" if feed["lost"][i] else "FLY")
        cv2.putText(img, f"{d.name} {state} bat {d.battery}% h {feed['height'][i]:.1f}", (10, H - 90 + 20 * i),
                    0, 0.5, c, 1)
    cv2.putText(img, f"{'FLY' if fly else 'DRY-RUN'}  t {now - t0:5.1f} s   q = land all, SPACE = EMERGENCY",
                (10, 18), 0, 0.5, (0, 255, 255), 1)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--drones", default="", help="4 names, e.g. TELLO-3,TELLO-4,TELLO-5,TELLO-1 "
                    "(default: roles.attack in config/fleet.yaml)")
    ap.add_argument("--spots", default="auto", help="spot index per drone in --drones order, e.g. 2,0,3,1 "
                    "(default auto: left-to-right by start position, no crossing paths)")
    ap.add_argument("--fly", action="store_true", help="actually fly (default dry-run)")
    ap.add_argument("--rc-scale", type=float, default=1.0,
                    help="multiply every rc stick (0-1]. The simulator's full-stick speed is ~0.73 m/s; "
                         "flights 032-034 measured ~2 m/s, so <1 brings the real drone closer to training")
    ap.add_argument("--max-time", type=float, default=75.0, help="seconds after takeoff -> land all (training episode: 75 s)")
    ap.add_argument("--lost-land", type=float, default=5.0, help="a drone without a tag this long lands")
    ap.add_argument("--min-sep", type=float, default=0.3, help="two airborne drones closer (filtered) -> land all")
    ap.add_argument("--stagger", type=float, default=0.3, help="seconds between takeoffs")
    args = ap.parse_args()
    if not 0.0 < args.rc_scale <= 1.0:
        sys.exit("--rc-scale must be in (0, 1]")
    fly = args.fly
    print(f"MODE: {'FLY' if fly else 'DRY-RUN (no takeoff, no rc sent)'}", flush=True)

    ckpt = Path(args.checkpoint).resolve()
    policy = load_policy(ckpt)
    cfg = policy.cfg
    D = cfg.num_drones
    if policy.central or not getattr(cfg, "peer_observations", False):
        print("note: this is not the shared-peer room policy; continuing with the adapter's contract")
    check_room_snapshot(policy, ckpt)
    names = [n.strip() for n in args.drones.split(",") if n.strip()] or load_fleet_config()["attack"]
    if len(names) != D:
        sys.exit(f"The policy flies exactly {D} drones; got {len(names)}: {names}. "
                 f"Pass --drones with {D} names (fleet.yaml roles.attack has {load_fleet_config()['attack']}).")
    auto_spots = args.spots == "auto"
    spot_idx = list(range(D)) if auto_spots else [int(s) for s in args.spots.split(",")]
    if sorted(spot_idx) != list(range(cfg.spot_count))[:D] or len(set(spot_idx)) != D:
        sys.exit(f"--spots must be auto or a permutation of 0..{cfg.spot_count - 1}, one per drone")
    spots = spot_positions(cfg)
    dt_policy = cfg.sim.dt * cfg.decimation          # 74 ms: one rc packet per drone per tick
    kcfg = KalmanCfg(**{f.name: getattr(cfg.kalman, f.name) for f in dataclasses.fields(KalmanCfg)
                        if hasattr(cfg.kalman, f.name)})
    room_lo = np.array([cfg.wall_x, cfg.room_y_min, -0.5]) - ROOM_MARGIN_M
    room_hi = np.array([cfg.room_x_max, cfg.side_wall_y, cfg.ceiling]) + ROOM_MARGIN_M
    print(f"policy {ckpt} ({policy.manifest.get('task', 'room')}), {D} drones, tick {dt_policy * 1000:.0f} ms, "
          f"rc scale {args.rc_scale}", flush=True)
    drones = load_drones(names, fly, cfg)
    for d in drones:
        if not iface_ip(d.iface):
            sys.exit(f"[{d.name}] {d.iface} has no 192.168.10.x address: connect it to {d.ssid} first")
    res = parallel(drones, lambda d: preflight(d, fly))
    if not all(res.get(d.num) for d in drones):
        for d in drones:
            d.close()
        sys.exit("Preflight FAILED -> nothing takes off.")
    if fly and any(d.battery is None or d.battery < MIN_BATTERY_FLY for d in drones):
        for d in drones:
            d.close()
        sys.exit(f"Battery below {MIN_BATTERY_FLY}% on {[d.name for d in drones if (d.battery or 0) < MIN_BATTERY_FLY]}")

    # The simulator starts each episode from the drone's position: here, a tag fix on the floor.
    print("waiting for every drone to see a mapped tag from the floor ...", flush=True)
    t_wait = time.time()
    while any(d.fix is None for d in drones) and time.time() - t_wait < FLOOR_FIX_S:
        time.sleep(0.1)
    blind = [d.name for d in drones if d.fix is None]
    if blind:
        for d in drones:
            d.close()
        sys.exit(f"No tag fix from the floor for {blind}: turn them toward tag 10 / the mapped tags.")
    start = np.stack([d.fix[0] for d in drones])
    start_yaw = np.array([d.fix[1] for d in drones])
    gap = np.linalg.norm(start[:, None, :2] - start[None, :, :2], axis=-1) + 9 * np.eye(D)
    if auto_spots:  # leftmost start (smallest Y) -> spot 0: no crossing paths
        spot_idx = [0] * D
        for rank, i in enumerate(np.argsort(start[:, 1])):
            spot_idx[i] = rank
    target = spots[spot_idx]
    for d, p, i, t in zip(drones, start, spot_idx, target):
        print(f"  {d.name} at ({p[0]:.2f}, {p[1]:+.2f}, {p[2]:.2f}) m, battery {d.battery}% -> spot {i} "
              f"({t[0]:.2f}, {t[1]:+.2f})", flush=True)
    if gap.min() < 0.5:
        print(f"WARNING: two drones start {gap.min():.2f} m apart (training kept >= 0.5 m)", flush=True)

    session = next_session()
    stem = RECORDINGS / f"{session}_SWARM"
    meta = {"task": policy.manifest.get("task"), "checkpoint": str(ckpt),
            "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest()[:16], "fly": fly,
            "drones": names, "spots": spot_idx, "targets": target.tolist(), "rc_scale": args.rc_scale,
            "step_s": dt_policy, "kalman": dataclasses.asdict(kcfg), "start": start.tolist(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    Path(f"{stem}.policy_meta.json").write_text(json.dumps(meta, indent=1))
    log = open(f"{stem}.policy.jsonl", "w")

    # feed state, exactly as the simulator keeps it (tello_room_env._update_feed)
    kf = KalmanCV(kcfg, (D,))
    pub_pos, pub_yaw = start.copy(), start_yaw.copy()
    fix_age, acquired = np.zeros(D), np.zeros(D, bool)
    fresh, lost = np.zeros(D, bool), np.zeros(D, bool)
    last_fix_id = [d.fix[3] for d in drones]
    blind_s = np.zeros(D)
    report = {"end": None, "steps": 0, "fresh": [0] * D, "land_sent_t": [None] * D, "landed_t": [None] * D,
              "min_sep_m": None}
    end = None

    def land_all(why):
        nonlocal end
        if end is None:
            end = why
            print(f">>> {why} -> landing all", flush=True)
        for d in drones:
            d.land_async()

    try:
        if fly:
            if input("All PASS. Space clear, drones on their start marks, facing tag 10? Type FLY: ").strip() != "FLY":
                sys.exit("Aborted, no takeoff.")

            def takeoff(d):
                time.sleep(args.stagger * drones.index(d))
                d.airborne = True
                for _ in range(2):
                    if d.send("takeoff", 20) == "ok":
                        break
                    time.sleep(1.0)
                    h = re.sub(r"[^0-9]", "", d.send("height?", 5))
                    if h and int(h) >= 3:             # reply lost but it is up
                        break
                else:
                    d.log("takeoff NOT confirmed")
                    d.takeoff_failed = True
                d.taking_off = False

            for d in drones:
                d.taking_off, d.takeoff_failed = True, False
                threading.Thread(target=takeoff, args=(d,), daemon=True).start()

        t0 = t_prev = time.time()
        cv2.namedWindow("swarm policy")
        while True:
            t_loop = time.time()
            now = t_loop
            dt = now - t_prev
            t_prev = now
            landed = np.array([d.landed for d in drones])
            taking_off = np.array([d.taking_off for d in drones])

            # this tick's fixes: the newest located frame of each drone, if new
            arrived = np.zeros(D, bool)
            z, yaw_fix = pub_pos.copy(), pub_yaw.copy()
            for i, d in enumerate(drones):
                with d.fix_lock:
                    f = d.fix
                if f is not None and f[3] != last_fix_id[i]:
                    arrived[i], last_fix_id[i] = True, f[3]
                    z[i], yaw_fix[i] = f[0], f[1]
            kf.predict(dt, active=~lost & ~landed)
            kf.reset(lost & arrived)                 # back from lost: restart from this fix
            kf.update(z, mask=arrived)
            pub_pos = np.where(kf.ready[:, None], kf.p, pub_pos)
            pub_yaw = np.where(arrived, yaw_fix, pub_yaw)
            fresh = arrived
            acquired |= arrived
            fix_age = np.where(arrived, 0.0, fix_age + dt)
            lost = (fix_age > cfg.lost_after) & ~landed
            kf.freeze(lost | landed)
            height = np.array([d.height_m() for d in drones])

            feed = {"position": pub_pos, "velocity": kf.v, "position_variance": kf.a, "velocity_variance": kf.c,
                    "yaw": pub_yaw, "fix_age": fix_age, "fresh": fresh, "acquired": acquired, "lost": lost,
                    "height": height, "taking_off": taking_off, "landed": landed, "target": target}
            out = policy.step(feed)
            rc = np.rint(np.asarray(out["rc"]) * args.rc_scale).astype(int)

            # landing: the policy's gate (auto-land at the waypoint), plus per-drone safety
            for i, d in enumerate(drones):
                if out["land_request"][i] and not d.landing:
                    d.log(f"policy: land (waypoint reached, gate ready) at t {now - t0:.1f} s")
                    report["land_sent_t"][i] = round(now - t0, 2)
                    d.land_async()
                if not d.landing and not taking_off[i] and fly:
                    blind_s[i] = blind_s[i] + dt if not fresh[i] else 0.0
                    # the policy never trained with a teammate forced down mid-room: in a
                    # replay with one drone blacked out, the other three hovered short of
                    # their waypoints until the time limit. So one safety landing lands all.
                    if blind_s[i] > args.lost_land:
                        land_all(f"{d.name} without a tag for {blind_s[i]:.1f} s")
                    elif d.battery is not None and d.battery <= LAND_BATTERY:
                        land_all(f"{d.name} battery {d.battery}%")
                if d.landed and report["landed_t"][i] is None:
                    report["landed_t"][i] = round(now - t0, 2)

            # team safety: link, room, separation, time
            if fly:
                if any(getattr(d, "takeoff_failed", False) for d in drones):
                    land_all("takeoff failed")
                silent = [d.name for d in drones if not d.landed and now - d.st_t > LINK_SILENT_S]
                if silent:
                    land_all(f"no state packets from {silent}")
                flying = ~np.array([d.landed or d.landing for d in drones]) & acquired
                out_room = flying & ((pub_pos < room_lo) | (pub_pos > room_hi)).any(-1)
                if out_room.any():
                    land_all(f"{[names[i] for i in np.nonzero(out_room)[0]]} left the room")
                both = flying[:, None] & flying[None, :] & ~np.eye(D, dtype=bool)
                sep = np.linalg.norm(pub_pos[:, None] - pub_pos[None], axis=-1)
                if both.any():
                    m = float(sep[both].min())
                    report["min_sep_m"] = m if report["min_sep_m"] is None else min(report["min_sep_m"], m)
                    if m < args.min_sep:
                        land_all(f"two drones {m:.2f} m apart")
                if now - t0 > args.max_time:
                    land_all(f"max time {args.max_time:.0f} s")

            # send: one rc per flying drone per tick (never during takeoff or landing)
            for i, d in enumerate(drones):
                if fly and d.airborne and not (d.taking_off or d.landing or d.landed):
                    d.send_noreply("rc %d %d %d %d" % tuple(rc[i]))
            report["steps"] += 1
            report["fresh"] = [a + int(b) for a, b in zip(report["fresh"], fresh)]
            log.write(json.dumps({
                "t": round(now - t0, 3), "dt": round(dt, 4), "rc": rc.tolist(), "land": out["land_request"],
                "land_ready": out["land_ready"], "act": np.round(out["action"], 4).tolist(),
                "pos": np.round(pub_pos, 3).tolist(), "vel": np.round(kf.v, 3).tolist(),
                "yaw": np.round(pub_yaw, 3).tolist(), "fresh": fresh.tolist(), "lost": lost.tolist(),
                "fix_age": np.round(fix_age, 3).tolist(), "h": height.tolist(), "taking_off": taking_off.tolist(),
                "landing": [d.landing for d in drones], "landed": landed.tolist(),
                "battery": [d.battery for d in drones], "why": [d.reason for d in drones]}) + "\n")

            if not fly and report["steps"] % 14 == 0:  # ~1 s
                print(f"t {now - t0:5.1f}  " + "  ".join(f"{d.name}: rc {tuple(rc[i])} {'fix' if fresh[i] else 'LOST' if lost[i] else '-'}"
                                                         for i, d in enumerate(drones)), flush=True)
            cv2.imshow("swarm policy", draw_status(drones, spots, feed, now, t0, fly))
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                land_all("q pressed")
            if k == ord(" ") and fly:
                print("!!! EMERGENCY: motors off all", flush=True)
                for d in drones:
                    d.send_noreply("emergency")
                    d.landed = d.landing = True
                end = "emergency"
                break
            if all(d.landed for d in drones):
                end = end or ("all landed by the policy" if all(report["land_sent_t"]) else "all landed")
                break
            if not fly and now - t0 > args.max_time:
                end = "dry-run time up"
                break
            time.sleep(max(0.0, dt_policy - (time.time() - t_loop)))
    except KeyboardInterrupt:
        end = end or "Ctrl+C"
        print("\nCtrl+C -> landing all", flush=True)
    except Exception as ex:
        end = end or f"error: {ex!r}"
        print(f"ERROR {ex!r} -> landing all", flush=True)
        raise
    finally:
        for d in drones:                             # lands already under way finish first
            if d.land_thread is not None:
                d.land_thread.join(timeout=20)
        if fly:
            parallel([d for d in drones if d.airborne and not d.landed], lambda d: d.land())
        for d in drones:
            d.close()
        report["end"] = end
        report["battery"] = [d.battery for d in drones]
        report["final_pos"] = np.round(pub_pos, 3).tolist()
        report["xy_err_to_spot"] = np.round(np.linalg.norm((pub_pos - target)[:, :2], axis=-1), 3).tolist()
        log.close()
        Path(f"{stem}.policy_report.json").write_text(json.dumps(report, indent=1))
        cv2.destroyAllWindows()
        print(f"Done: {end}. Log {stem}.policy.jsonl", flush=True)


if __name__ == "__main__":
    main()
