#!/usr/bin/env python3
"""Swarm approach to AprilTag 10: each drone flies to its own slot on an arc 1 m from
the tag, facing it, keeping safe spacing from the others; then all hold and land.

DEFAULT IS DRY-RUN: no takeoff, no rc sent. Shows what it WOULD send.  --fly to fly.

Needs (see NEEDS section at bottom of --help):
  dongles.json                      adapter <-> drone map (same file as multi_hover.py)
  <camera-dir>/tello_<n>.yaml       camera calibration per drone (K + dist)
  python libs: numpy opencv-python pupil-apriltags av pyyaml (tello_ops venv has them)

Usage:
  python swarm_approach.py --config dongles.json --only 4              # dry-run
  python swarm_approach.py --config dongles.json --only 4 --fly
  python swarm_approach.py --config dongles.json --only 1,3,4,5 --fly

Keys (camera windows focused): q = stop (land all), SPACE = EMERGENCY motor stop all.
Ctrl+C / any error / link loss / low battery -> land all.
"""
import argparse
import csv
import json
import math
import os
import re
import socket
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import yaml

# ======================= SETTINGS (edit here) =======================
TAG_ID = 10
TAG_SIZE_M = 0.5715          # tag 10 black-square edge (22.5 in)
ARC_RADIUS_M = 1.0           # distance from tag center to each slot
ARC_RADIUS_MAX_M = 2.0       # grow up to this if N drones don't fit
MIN_SPACING_M = 0.7          # slot-to-slot spacing
MAX_VIEW_ANGLE_DEG = 55      # slots stay within this angle from tag normal
HEIGHT_OFFSET_M = 0.0        # slot height relative to tag center (+ = higher)

KP_POS = 40.0                # rc units per meter of position error
KP_YAW = 1.0                 # rc units per degree of yaw error
MAX_RC_XY = 20               # speed caps (rc range is -100..100)
MAX_RC_Z = 20
MAX_RC_YAW = 25
DEADBAND_POS_M = 0.05
DEADBAND_YAW_DEG = 3.0
POSE_ALPHA = 0.6             # low-pass on pose (1 = no filtering)

D_SAFE_M = 0.6               # below this: slow approach + push apart (< MIN_SPACING)
D_STOP_M = 0.4               # below this: no approach, only push apart
K_REP = 60.0                 # rc units per meter inside D_SAFE
LOST_DRIFT_MPS = 0.2         # lost drone's no-go bubble grows this fast
MIN_TAG_DIST_M = 0.6         # never get closer to the tag than this

TOL_POS_M = 0.10             # "in slot" tolerance
TOL_YAW_DEG = 5.0
SETTLE_S = 2.0               # in tolerance this long -> HOLD
POSE_MAX_AGE_S = 0.5         # older pose = tag lost
LOST_LAND_S = 5.0            # lost longer -> that drone lands
SEARCH_HEIGHT_CM = 160       # height above floor while searching (climbs right after takeoff)
KP_HEIGHT = 0.5              # rc units per cm of height error while searching
SEARCH_YAW_RC = 25           # rotation speed during each search turn
LOOK_S = 2.0                 # after reaching search height, hover & look before turning
SEARCH_STEP_DEG = 20         # search: turn this much...
SEARCH_LOOK_S = 0.8          # ...then hold still this long to look (no motion blur)
GLIMPSE_HOLD_S = 2.0         # any glimpse of tag 10: freeze up to this long for a lock
CLIMB_TIMEOUT_S = 6.0        # stop waiting for search height after this
RECOVER_YAW_RC = 20          # lost tag: turn toward where it was last seen
SEARCH_MAX_TURN_DEG = 400    # turned this much without seeing the tag -> land

MIN_BATTERY_FLY = 40
LAND_BATTERY = 15
MIN_DECISION_MARGIN = 30
MIN_TAG_SIDE_PX = 30
LOOP_HZ = 20
RES = (960, 720)
# ====================================================================

DRONE_ADDR = ("192.168.10.1", 8889)
SO_BINDTODEVICE = getattr(socket, "SO_BINDTODEVICE", 25)
NOFLY = {"command", "streamon", "streamoff", "battery?", "height?"}
RC_RE = re.compile(r"^rc (-?\d+) (-?\d+) (-?\d+) (-?\d+)$")


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


# ======================= GEOMETRY (pure) =======================
# Tag frame (tag 10): x right, y down, z into the wall. "In front" = negative z.
# Assumes the tag is mounted upright (tag y ~ gravity) and the drone flies level.

def obj_points(size):
    h = size / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)


def rotz90(k):
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k % 4]
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)


def _rot_angle_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def solve_tag_pose(corners, K, D, size, prior=None, max_err_px=3.0):
    """Corner-order-independent tag pose -> (R, t, reason), camera<-tag.
    Tries all 4 start corners x 2 windings x 2 IPPE solutions; keeps poses that see
    the tag's FRONT face with low reprojection error; fixes the in-plane 90-deg
    ambiguity by choosing tag y closest to camera-down (tag upright, drone level).
    If two clearly different drone poses fit equally well (near-frontal views), pick
    the one closest to `prior` (p, psi); with no prior, the one whose tag normal is
    horizontal (tag upright)."""
    obj = obj_points(size)
    cands = []
    for wind in (corners, corners[::-1]):
        for k in range(4):
            pts = np.roll(wind, k, axis=0).reshape(-1, 1, 2).astype(np.float64)
            n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, pts, K, D,
                                                        flags=cv2.SOLVEPNP_IPPE)
            errs = np.array(errs).reshape(-1)
            for i in range(n):
                if not np.isfinite(errs[i]) or errs[i] > max_err_px:
                    continue
                R, _ = cv2.Rodrigues(rvecs[i])
                t = tvecs[i].reshape(3)
                if R[2, 2] <= 0 or t[2] <= 0:   # must see the front face, in front
                    continue
                best = max(range(4), key=lambda j: (R @ rotz90(j))[1, 1])
                R = R @ rotz90(best)
                p, psi = drone_in_tag_frame(R, t)
                cands.append((float(errs[i]), R, t, p, psi))
    if not cands:
        return None, None, "no valid pose"
    cands.sort(key=lambda c: c[0])
    e0 = cands[0][0]

    def same(a, b):
        return np.linalg.norm(a[3] - b[3]) < 0.15 and abs(wrap_deg(a[4] - b[4])) < 8.0
    rivals = [c for c in cands if c[0] < 2.0 * e0 + 0.5 and not same(c, cands[0])]
    if not rivals:
        return cands[0][1], cands[0][2], "ok"
    if prior is None:   # use gravity: tag upright -> its normal is horizontal
        pick = min([cands[0]] + rivals, key=lambda c: abs(c[1][1, 2]))
        return pick[1], pick[2], "ok (gravity)"
    pp, ppsi = prior
    pick = min([cands[0]] + rivals,
               key=lambda c: np.linalg.norm(c[3] - pp) + abs(wrap_deg(c[4] - ppsi)) / 30.0)
    return pick[1], pick[2], "ok (prior)"


def drone_in_tag_frame(R, t):
    """Camera position p (tag frame) and yaw psi (deg). psi=0 faces the tag;
    positive = turned clockwise seen from above."""
    p = -R.T @ t
    f = R.T @ np.array([0.0, 0.0, 1.0])   # camera forward in tag frame
    psi = math.degrees(math.atan2(f[0], f[2]))
    return p, psi


def make_slots(n):
    """Slots on an arc around the tag. Returns (list of (pos, yaw_deg), radius)."""
    r = ARC_RADIUS_M
    while r <= ARC_RADIUS_MAX_M + 1e-9:
        if n == 1:
            phis = [0.0]
        else:
            dphi = 2 * math.asin(min(1.0, MIN_SPACING_M / (2 * r)))
            span = (n - 1) * dphi
            if span / 2 > math.radians(MAX_VIEW_ANGLE_DEG):
                r += 0.1
                continue
            phis = [(i - (n - 1) / 2) * dphi for i in range(n)]
        slots = []
        for phi in phis:
            pos = np.array([r * math.sin(phi), -HEIGHT_OFFSET_M, -r * math.cos(phi)])
            slots.append((pos, math.degrees(-phi)))   # face the tag center
        return slots, r
    raise ValueError(f"{n} drones don't fit (spacing {MIN_SPACING_M} m, "
                     f"angle {MAX_VIEW_ANGLE_DEG} deg, radius <= {ARC_RADIUS_MAX_M} m)")


def bearing(p):
    return math.atan2(p[0], -p[2])   # 0 = straight in front, + = to the right


def assign_slots(positions, slots):
    """positions: {num: p}. Sorted left->right to sorted slots: no crossing paths."""
    order = sorted(positions, key=lambda k: bearing(positions[k]))
    slot_order = sorted(range(len(slots)), key=lambda i: bearing(slots[i][0]))
    return {num: slot_order[i] for i, num in enumerate(order)}


def match_slots(positions, slots):
    """Optimal drone->slot matching (min total horizontal distance) for the drones in
    `positions` {num: p}; a min-total-length matching never has crossing paths.
    Brute force over slot permutations (fine for <= 8 drones, run only on lock events)."""
    from itertools import permutations
    nums = list(positions)
    best, best_cost = None, float("inf")
    for perm in permutations(range(len(slots)), len(nums)):
        cost = sum(float(np.hypot(slots[k][0][0] - positions[n][0],
                                  slots[k][0][2] - positions[n][2]))
                   for n, k in zip(nums, perm))
        if cost < best_cost:
            best, best_cost = dict(zip(nums, perm)), cost
    return best


def control(p, psi, slot, others):
    """p, psi: drone pose (tag frame). slot: (pos, yaw). others: list of
    (p_other, extra_radius). Returns (rc tuple, pos_err_norm, yaw_err)."""
    target, _ = slot
    e = target - p
    yaw_face = math.degrees(math.atan2(-p[0], -p[2]))   # look at tag from here
    e_yaw = wrap_deg(yaw_face - psi)                     # (= slot yaw at the slot)
    if np.linalg.norm(e) < DEADBAND_POS_M:
        e = np.zeros(3)
    v = KP_POS * e                                   # desired velocity, tag frame
    # --- separation (horizontal x-z plane) ---
    scale, rep = 1.0, np.zeros(3)
    for q, extra in others:
        dvec = np.array([p[0] - q[0], 0.0, p[2] - q[2]])
        d = float(np.linalg.norm(dvec))
        d_safe, d_stop = D_SAFE_M + extra, D_STOP_M + extra
        if d < d_safe:
            u = dvec / d if d > 1e-6 else np.array([1.0, 0, 0])
            rep += K_REP * (d_safe - d) * u
            scale = min(scale, max(0.0, (d - d_stop) / (d_safe - d_stop)))
    v = v * scale + rep
    # --- never too close to the tag ---
    dist_h = math.hypot(p[0], p[2])
    if dist_h < MIN_TAG_DIST_M and dist_h > 1e-6:
        u = np.array([p[0], 0.0, p[2]]) / dist_h       # tag -> drone
        inward = -float(v @ u)
        if inward > 0:
            v = v + inward * u
        v = v + 10.0 * u
    # --- clamp ---
    vxz = np.array([v[0], v[2]])
    nxz = np.linalg.norm(vxz)
    if nxz > MAX_RC_XY:
        v[0], v[2] = v[0] * MAX_RC_XY / nxz, v[2] * MAX_RC_XY / nxz
    v[1] = float(np.clip(v[1], -MAX_RC_Z, MAX_RC_Z))
    # --- tag frame -> drone body (a=right, b=forward, c=up, d=yaw cw) ---
    s, c_ = math.sin(math.radians(psi)), math.cos(math.radians(psi))
    fwd = v[0] * s + v[2] * c_
    right = v[0] * c_ - v[2] * s
    up = -v[1]
    yaw = 0.0 if abs(e_yaw) < DEADBAND_YAW_DEG else KP_YAW * e_yaw
    yaw = float(np.clip(yaw, -MAX_RC_YAW, MAX_RC_YAW))
    rc = tuple(int(round(x)) for x in (right, fwd, up, yaw))
    return rc, float(np.linalg.norm(target - p)), e_yaw


# ======================= DRONE I/O =======================
def iface_ip(iface):
    out = subprocess.run(["ip", "-j", "-4", "addr", "show", iface],
                         capture_output=True, text=True).stdout or "[]"
    for dev in json.loads(out):
        for a in dev.get("addr_info", []):
            if a.get("local", "").startswith("192.168.10."):
                return a["local"]
    return None


def active_ssid(iface):
    out = subprocess.run(["nmcli", "-t", "-f", "ACTIVE,SSID", "dev", "wifi", "list",
                          "ifname", iface, "--rescan", "no"],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("yes:"):
            return line[4:].replace("\\:", ":")
    return None


def load_camera(path):
    with open(path) as f:
        c = yaml.safe_load(f)
    K = c.get("K", c.get("camera_matrix"))
    D = c.get("dist", c.get("dist_coeffs", c.get("D")))
    if isinstance(K, dict):
        K = K.get("data")
    if isinstance(D, dict):
        D = D.get("data")
    if K is None or D is None:
        raise ValueError(f"{path}: no K/dist found (keys: {list(c)})")
    K = np.array(K, np.float64).reshape(3, 3)
    D = np.array(D, np.float64).reshape(-1)
    res = c.get("resolution")
    if res is not None:
        if isinstance(res, str):
            res = [int(x) for x in re.split(r"[x, ]+", res) if x]
        elif isinstance(res, dict):
            res = [res.get("width"), res.get("height")]
        if list(res)[:2] != list(RES):
            raise ValueError(f"{path}: calibrated at {res}, stream is {RES}")
    return K, D


class Drone:
    def __init__(self, num, iface, ssid, K, D, fly):
        self.num, self.iface, self.ssid, self.K, self.D = num, iface, ssid, K, D
        self.name, self.fly = f"drone{num}", fly
        self.sock = self._sock(9000 + num)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.frame = None; self.frame_t = 0.0; self.frame_id = 0; self.frames = 0
        self.packets = 0
        self.overlay = None
        self.pose = None            # (p, psi, t) filtered
        self.seen_t = None          # last time tag was seen
        self.reason = "no frames"
        self.battery = None
        self.airborne = False
        self.landed = False
        self.st = {}                # latest Tello state packet (key -> str)
        self.st_t = 0.0
        self.last_yaw = None
        self.yaw_acc = 0.0          # degrees turned since search start
        self.dstate = "SEARCH"      # SEARCH -> LOCKED (wait for all) -> APPROACH
        self.sub = "CLIMB"          # search sub-phase: CLIMB -> LOOK -> ROTATE
        self.sub_t = None
        self.seen_side = 0          # +1 tag last seen right of center, -1 left
        self.glimpse_t = 0.0        # last time tag 10 appeared at all (even unusable)
        self.turn_start = 0.0       # yaw_acc at start of current search turn
        self.search_turned = 0.0    # degrees turned while searching

    def _sock(self, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.SOL_SOCKET, SO_BINDTODEVICE, self.iface.encode())
        s.bind(("", port))
        return s

    def log(self, msg):
        print(f"[{self.name}] {msg}", flush=True)

    def _check(self, cmd):
        if cmd in NOFLY:
            return
        if self.fly and cmd in ("takeoff", "land", "emergency"):
            return
        m = RC_RE.match(cmd)
        if self.fly and m and all(abs(int(x)) <= 100 for x in m.groups()):
            return
        raise PermissionError(f"blocked command {cmd!r} (fly={self.fly})")

    def send(self, cmd, timeout=10, quiet=False):
        self._check(cmd)
        with self.lock:
            self.sock.setblocking(False)
            try:
                while True:
                    self.sock.recv(1024)
            except (BlockingIOError, OSError):
                pass
            self.sock.settimeout(timeout)
            self.sock.sendto(cmd.encode(), DRONE_ADDR)
            try:
                r = self.sock.recv(1024).decode(errors="ignore").strip().lower()
            except socket.timeout:
                r = "timeout"
        if not quiet:
            self.log(f"{cmd} -> {r}")
        return r

    def send_noreply(self, cmd):
        self._check(cmd)
        self.sock.sendto(cmd.encode(), DRONE_ADDR)

    def land(self):
        if not self.airborne or self.landed:
            return
        self.send_noreply("rc 0 0 0 0")
        for _ in range(3):
            if self.lock.acquire(timeout=0.5):
                self.lock.release()
                if self.send("land", 15) == "ok":
                    self.landed = True
                    return
            else:
                self.send_noreply("land")
                time.sleep(1)
        self.log("WARNING: land NOT confirmed - catch this drone!")

    # ---- video + perception threads ----
    def video_loop(self):
        import av
        vs = self._sock(11111)
        vs.settimeout(1.0)
        codec = av.CodecContext.create("h264", "r")
        while not self.stop.is_set():
            try:
                data = vs.recv(65536)
            except socket.timeout:
                continue
            self.packets += 1
            try:
                for pkt in codec.parse(data):
                    for fr in codec.decode(pkt):
                        self.frame = fr.to_ndarray(format="bgr24")
                        self.frame_t = time.time()
                        self.frame_id += 1
                        self.frames += 1
            except Exception:
                pass
        vs.close()

    def perception_loop(self):
        from pupil_apriltags import Detector
        det = Detector(families="tag36h11", nthreads=1, quad_decimate=1.0,
                       refine_edges=True)
        last = 0
        while not self.stop.is_set():
            if self.frame_id == last or self.frame is None:
                time.sleep(0.005)
                continue
            last, img, t = self.frame_id, self.frame.copy(), self.frame_t
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found = None
            dets = det.detect(gray)
            for d in dets:
                if d.tag_id != TAG_ID:
                    continue
                c = np.array(d.corners, np.float64)
                self.glimpse_t = t
                self.seen_side = 1 if c[:, 0].mean() > self.K[0, 2] else -1
                side = min(np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4))
                if d.decision_margin < MIN_DECISION_MARGIN:
                    self.reason = f"margin {d.decision_margin:.0f}"
                elif side < MIN_TAG_SIDE_PX:
                    self.reason = f"small {side:.0f}px"
                else:
                    prior = None
                    if self.pose is not None and t - self.pose[2] < 1.0:
                        prior = (self.pose[0], self.pose[1])
                    R, tv, why = solve_tag_pose(c, self.K, self.D, TAG_SIZE_M, prior)
                    self.reason = why
                    if R is not None:
                        found = (c, R, tv)
                cv2.polylines(img, [c.astype(np.int32).reshape(-1, 1, 2)], True,
                              (0, 255, 0) if found else (0, 0, 255), 2)
            if found:
                p, psi = drone_in_tag_frame(found[1], found[2])
                if self.pose is None or t - self.pose[2] > POSE_MAX_AGE_S:
                    self.pose = (p, psi, t)
                else:
                    p0, s0, _ = self.pose
                    a = POSE_ALPHA
                    self.pose = (a * p + (1 - a) * p0,
                                 s0 + a * wrap_deg(psi - s0), t)
                self.seen_t = t
                self.seen_side = 1 if found[0][:, 0].mean() > self.K[0, 2] else -1
            elif not any(d.tag_id == TAG_ID for d in dets):
                self.reason = "tag not seen"
            self.overlay = img

    def state_loop(self):
        """Tello status stream (port 8890): height (tof), yaw, battery, ..."""
        ss = self._sock(8890)
        ss.settimeout(1.0)
        while not self.stop.is_set():
            try:
                data = ss.recv(2048).decode(errors="ignore")
            except socket.timeout:
                continue
            vals = dict(kv.split(":", 1) for kv in data.strip().split(";") if ":" in kv)
            self.st, self.st_t = vals, time.time()
            try:
                y = float(vals["yaw"])
                if self.last_yaw is not None:
                    self.yaw_acc += abs(wrap_deg(y - self.last_yaw))
                self.last_yaw = y
            except (KeyError, ValueError):
                pass
        ss.close()

    def height_cm(self):
        """Height above floor from the downward ToF sensor; None if unknown/stale."""
        if time.time() - self.st_t > 1.0:
            return None
        try:
            tof = int(self.st.get("tof", -1))
        except ValueError:
            return None
        return tof if 10 <= tof <= 400 else None

    def battery_loop(self):
        while not self.stop.wait(5):
            b = self.send("battery?", 3, quiet=True)
            if b.isdigit():
                self.battery = int(b)

    def visible(self, now):
        return self.pose is not None and now - self.pose[2] < POSE_MAX_AGE_S

    def close(self):
        self.stop.set()
        try:
            self.send("streamoff", 3, quiet=True)
        except Exception:
            pass
        self.sock.close()


# ======================= MAIN =======================
def load_drones(cfg, only, cam_dir, fly):
    with open(cfg) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("dongles", [])
    entries = [e for e in data if e.get("ssid")]
    if only:
        entries = [e for e in entries if int(e["number"]) in only]
        missing = only - {int(e["number"]) for e in entries}
        if missing:
            sys.exit(f"No config/ssid for drone(s) {sorted(missing)} in {cfg}")
    if not entries:
        sys.exit("No drones selected.")
    drones, bad = [], []
    for e in entries:
        n = int(e["number"])
        path = os.path.join(cam_dir, f"tello_{n}.yaml")
        if not os.path.exists(path):
            bad.append(f"drone{n}: missing calibration {path}")
            continue
        try:
            K, D = load_camera(path)
        except Exception as ex:
            bad.append(f"drone{n}: {ex}")
            continue
        drones.append(Drone(n, e["iface"], e["ssid"], K, D, fly))
    if bad:
        sys.exit("Cannot run:\n  " + "\n  ".join(bad))
    return drones


def parallel(drones, fn):
    res, ths = {}, []
    for d in drones:
        def w(d=d):
            try:
                res[d.num] = fn(d)
            except Exception as ex:
                d.log(f"error: {ex}")
                res[d.num] = False
        t = threading.Thread(target=w, daemon=True)
        t.start()
        ths.append(t)
    for t in ths:
        while t.is_alive():
            t.join(0.2)
    return res


def preflight(d, fly):
    if not iface_ip(d.iface):
        d.log(f"FAIL: {d.iface} has no 192.168.10.x (not connected?)")
        return False
    ssid = active_ssid(d.iface)
    if ssid != d.ssid:
        d.log(f"FAIL: {d.iface} on {ssid!r}, expected {d.ssid!r}")
        return False
    threading.Thread(target=d.state_loop, daemon=True).start()
    for _ in range(5):
        if d.send("command", 3) == "ok":
            break
        time.sleep(1)
    else:
        d.log("FAIL: no reply to command")
        return False
    b = d.send("battery?", 5)
    d.battery = int(b) if b.isdigit() else None
    if fly and (d.battery is None or d.battery < MIN_BATTERY_FLY):
        d.log(f"FAIL: battery {b}% < {MIN_BATTERY_FLY}%")
        return False
    threading.Thread(target=d.video_loop, daemon=True).start()
    for attempt in range(3):                    # UDP replies can get lost
        r = d.send("streamon", 5)
        if r == "ok":
            break
        if "unknown" in r or "unkown" in r:
            d.log("FAIL: drone doesn't know streamon -> update firmware in Tello app")
            return False
        time.sleep(1)
    else:
        d.log("FAIL: streamon not confirmed after 3 tries")
        return False
    t0, resent = time.time(), False
    while d.frames < 5 and time.time() - t0 < 15:
        if not resent and time.time() - t0 > 6:  # nudge a slow stream once
            d.send("streamon", 5)
            resent = True
        time.sleep(0.1)
    if d.frames < 5:
        if d.packets == 0:
            d.log("FAIL: no video packets at all (adapter/hub link? another program "
                  "using port 11111?)")
        else:
            d.log(f"FAIL: {d.packets} video packets but only {d.frames} decoded frames "
                  f"(weak Wi-Fi / packet loss?)")
        return False
    h, w = d.frame.shape[:2]
    if (w, h) != RES:
        d.log(f"FAIL: frame {w}x{h}, calibration is {RES}")
        return False
    threading.Thread(target=d.perception_loop, daemon=True).start()
    threading.Thread(target=d.battery_loop, daemon=True).start()
    t0 = time.time()
    while not d.st and time.time() - t0 < 3:
        time.sleep(0.1)
    if not d.st:
        d.log("WARNING: no status packets -> search height hold disabled")
    d.log(f"PASS (battery {d.battery}%, video ok, status {'ok' if d.st else 'missing'})")
    return True


def draw_topdown(drones, slots, assign, now):
    S, W, H = 150, 600, 500                     # px per meter, canvas
    img = np.full((H, W, 3), 30, np.uint8)
    ox, oy = W // 2, 40                          # tag at top center

    def px(p):
        return int(ox + p[0] * S), int(oy - p[2] * S)   # -z (front) goes down
    cv2.line(img, (ox - int(TAG_SIZE_M * S / 2), oy), (ox + int(TAG_SIZE_M * S / 2), oy),
             (255, 255, 255), 4)
    cv2.putText(img, "TAG 10", (ox - 30, oy - 10), 0, 0.5, (255, 255, 255), 1)
    for i, (sp, _) in enumerate(slots or []):
        cv2.circle(img, px(sp), 6, (0, 200, 255), 1)
        cv2.putText(img, f"s{i}", (px(sp)[0] + 8, px(sp)[1]), 0, 0.4, (0, 200, 255), 1)
    for d in drones:
        if d.pose is None:
            continue
        p, psi, t = d.pose
        col = (0, 255, 0) if d.visible(now) else (0, 0, 255)
        if not d.visible(now):
            r = D_SAFE_M + LOST_DRIFT_MPS * (now - t)
            cv2.circle(img, px(p), int(r * S), (0, 0, 120), 1)
        cv2.circle(img, px(p), 8, col, -1)
        hx = p + 0.2 * np.array([math.sin(math.radians(psi)), 0, math.cos(math.radians(psi))])
        cv2.line(img, px(p), px(hx), col, 2)
        cv2.putText(img, str(d.num), (px(p)[0] + 10, px(p)[1] - 10), 0, 0.5, col, 1)
        if assign and d.num in assign and slots:
            cv2.line(img, px(p), px(slots[assign[d.num]][0]), (90, 90, 90), 1)
    return img


def main():
    ap = argparse.ArgumentParser(epilog="NEEDS: dongles.json; tello_<n>.yaml camera "
                                 "calibration per drone in --camera-dir; libs numpy "
                                 "opencv-python pupil-apriltags av pyyaml.")
    ap.add_argument("--config", default="dongles.json")
    ap.add_argument("--only", default="", help="e.g. 4 or 1,3,4,5")
    here = os.path.dirname(os.path.abspath(__file__))
    cam_default = os.path.join(here, "camera")                 # swarm_ops/camera/
    if not os.path.isdir(cam_default):
        cam_default = os.path.join(here, "..", "config", "camera")  # tello_ops layout
    ap.add_argument("--camera-dir", default=cam_default,
                    help="folder with tello_<n>.yaml (default: ./camera, else ../config/camera)")
    ap.add_argument("--hold", type=float, default=5.0, help="seconds to hold in slots")
    ap.add_argument("--fly", action="store_true", help="actually fly (default dry-run)")
    ap.add_argument("--height", type=int, default=SEARCH_HEIGHT_CM,
                    help="search height above floor in cm (climbs right after takeoff)")
    ap.add_argument("--search-speed", type=int, default=SEARCH_YAW_RC,
                    help="rotation speed while searching (rc units, 10-60)")
    args = ap.parse_args()
    search_h = max(60, min(250, args.height))
    yaw_rc = max(10, min(60, args.search_speed))
    only = {int(x) for x in args.only.split(",") if x.strip()}
    fly = args.fly
    print(f"MODE: {'FLY' if fly else 'DRY-RUN (no takeoff, no rc sent)'}")

    drones = load_drones(args.config, only, args.camera_dir, fly)
    slots, radius = make_slots(len(drones))
    print(f"Drones: {[d.name for d in drones]}  slots radius {radius:.2f} m"
          + ("  (WARNING: > 1 m to fit spacing)" if radius > ARC_RADIUS_M + 1e-9 else ""))

    res = parallel(drones, lambda d: preflight(d, fly))
    if not all(res.get(d.num) for d in drones):
        for d in drones:
            d.close()
        sys.exit("Preflight FAILED -> nothing takes off.")

    os.makedirs("logs", exist_ok=True)
    logf = open(time.strftime("logs/approach_%Y%m%d_%H%M%S.csv"), "w", newline="")
    log = csv.writer(logf)
    log.writerow(["t", "drone", "state", "visible", "px", "py", "pz", "psi",
                  "pos_err", "yaw_err", "a", "b", "c", "d", "height_cm", "turned_deg",
                  "battery"])

    assign = {}                  # drone num -> slot index (claimed on first sighting)
    phase, settle_t, hold_t = "RUN", None, None
    try:
        if fly:
            if input("All PASS. Space clear, drones >= 1 m apart, facing the tag area? "
                     "Type FLY: ").strip() != "FLY":
                sys.exit("Aborted, no takeoff.")
            for d in drones:
                d.airborne = True

            def height_dm(d):
                h = re.sub(r"[^0-9]", "", d.send("height?", 5))
                return int(h) if h else -1

            def to(d):
                time.sleep(0.5 * drones.index(d))   # staggered takeoff
                for attempt in range(2):
                    if d.send("takeoff", 20) == "ok":
                        return True
                    time.sleep(1.0)
                    h = height_dm(d)                 # reply lost? check if airborne
                    if h >= 3:
                        d.log(f"takeoff reply lost but airborne (height {h} dm) -> ok")
                        return True
                    d.log(f"not airborne (height {h} dm), retrying takeoff")
                return False
            if not all(parallel(drones, to).values()):
                print("A takeoff failed -> landing all")
                return
        for d in drones:
            d.yaw_acc = 0.0                          # count search rotation from here

        period = 1.0 / LOOP_HZ
        while True:
            t_loop = time.time()
            now = t_loop
            active = [d for d in drones if not d.landed]
            if not active:
                print("All drones landed.")
                return
            for d in active:
                if d.battery is not None and d.battery < LAND_BATTERY:
                    print(f"{d.name} battery {d.battery}% -> landing all")
                    return
            cmds, all_in = {}, True

            for d in active:
                # ---------- SEARCH: climb + rotate, lock on the moment tag is seen ----
                if d.dstate == "SEARCH":
                    if d.visible(now):
                        d.dstate = "LOCKED"
                        d.log(f"LOCKED on tag 10 after searching {d.search_turned:.0f} deg")
                    else:
                        all_in = False
                        h = d.height_cm()
                        c = 0 if h is None else int(np.clip(KP_HEIGHT * (search_h - h),
                                                            -MAX_RC_Z, MAX_RC_Z))
                        if d.sub_t is None:
                            d.sub_t = now
                        yaw = 0
                        glimpse = now - d.glimpse_t < GLIMPSE_HOLD_S
                        if d.sub == "CLIMB":
                            if (h is not None and abs(search_h - h) < 15) \
                                    or now - d.sub_t > CLIMB_TIMEOUT_S:
                                d.sub, d.sub_t = "LOOK", now
                        elif d.sub == "LOOK":
                            look = LOOK_S if d.search_turned == 0 else SEARCH_LOOK_S
                            froze_long = now - d.sub_t > GLIMPSE_HOLD_S   # cap the freeze
                            if now - d.sub_t > look and (not glimpse or froze_long):
                                d.sub, d.sub_t, d.turn_start = "TURN", now, d.yaw_acc
                        elif d.sub == "TURN":
                            turned = (d.yaw_acc - d.turn_start) if d.st else \
                                (now - d.sub_t) * 40.0          # no status: estimate
                            new_glimpse = d.glimpse_t > d.sub_t + 0.3     # seen during turn
                            if new_glimpse or turned >= SEARCH_STEP_DEG:
                                d.search_turned += max(0.0, turned)   # stop and look
                                d.sub, d.sub_t = "LOOK", now
                            else:
                                yaw = yaw_rc
                        cmds[d.num] = ((0, 0, c, yaw), None, None)
                        if fly and d.search_turned > SEARCH_MAX_TURN_DEG:
                            print(f"{d.name} searched {d.search_turned:.0f} deg without "
                                  f"seeing tag 10 -> landing all")
                            return
                        continue
                # ---------- LOCKED: stop, face the tag, hold height, wait for all ------
                if d.dstate == "LOCKED":
                    all_in = False
                    h = d.height_cm()
                    c = 0 if h is None else int(np.clip(KP_HEIGHT * (search_h - h),
                                                        -MAX_RC_Z, MAX_RC_Z))
                    if d.visible(now):
                        p, psi, _ = d.pose
                        e = wrap_deg(math.degrees(math.atan2(-p[0], -p[2])) - psi)
                        yaw = 0 if abs(e) < DEADBAND_YAW_DEG else int(
                            np.clip(KP_YAW * e, -MAX_RC_YAW, MAX_RC_YAW))
                    else:
                        lost = now - (d.seen_t or now)
                        yaw = RECOVER_YAW_RC * d.seen_side if lost > 0.5 else 0
                        if lost > LOST_LAND_S:
                            d.log("lost tag while waiting -> searching again")
                            d.dstate, d.sub, d.sub_t = "SEARCH", "LOOK", now
                    cmds[d.num] = ((0, 0, c, yaw), None, None)
                    continue
                # ---------- APPROACH / HOLD ----------
                if not d.visible(now):
                    lost = now - (d.seen_t or now)
                    yaw = RECOVER_YAW_RC * d.seen_side if lost > 0.5 else 0
                    cmds[d.num] = ((0, 0, 0, yaw), None, None)
                    all_in = False
                    if lost > LOST_LAND_S and fly:
                        d.log(f"tag lost {lost:.1f}s -> landing this drone")
                        d.land()
                    continue
                p, psi, _ = d.pose
                others = []
                for o in active:
                    if o is d or o.pose is None:
                        continue
                    extra = 0.0 if o.visible(now) else LOST_DRIFT_MPS * (now - o.pose[2])
                    others.append((o.pose[0], extra))
                rc, pe, ye = control(p, psi, slots[assign[d.num]], others)
                cmds[d.num] = (rc, pe, ye)
                if pe > TOL_POS_M or abs(ye) > TOL_YAW_DEG:
                    all_in = False

            waiting = [d for d in active if d.dstate == "LOCKED"]
            if waiting and all(d.dstate in ("LOCKED", "APPROACH") for d in active) \
                    and all(d.pose is not None for d in active):
                assign.clear()
                assign.update(match_slots({d.num: d.pose[0] for d in active}, slots))
                for d in waiting:
                    d.dstate = "APPROACH"
                print(f">>> all drones locked on -> approaching slots {assign}")

            if phase == "RUN":
                settle_t = (settle_t or now) if all_in else None
                if settle_t and now - settle_t > SETTLE_S:
                    phase, hold_t = "HOLD", now
                    print(f">>> all in slots, holding {args.hold:.0f}s")
            elif now - hold_t > args.hold:
                print(">>> hold done -> landing all")
                return

            # ---- send / show ----
            for d in active:
                rc, pe, ye = cmds.get(d.num, ((0, 0, 0, 0), None, None))
                if fly:
                    d.send_noreply("rc %d %d %d %d" % rc)
                p = d.pose[0] if d.pose else (np.nan,) * 3
                psi = d.pose[1] if d.pose else np.nan
                h = d.height_cm()
                log.writerow([f"{now:.3f}", d.num, d.dstate, d.visible(now),
                              *[f"{x:.3f}" for x in p], f"{psi:.1f}",
                              "" if pe is None else f"{pe:.3f}",
                              "" if ye is None else f"{ye:.1f}", *rc,
                              "" if h is None else h, f"{d.yaw_acc:.0f}", d.battery])
                img = d.overlay if d.overlay is not None else np.zeros((RES[1], RES[0], 3), np.uint8)
                img = img.copy()
                tag = "SEE" if d.visible(now) else "LOST: " + d.reason
                st_txt = d.dstate + (f"/{d.sub}" if d.dstate == "SEARCH" else "")
                lines = [f"{d.name} {st_txt}{' HOLD' if phase == 'HOLD' else ''}  {tag}",
                         f"rc a{rc[0]:+d} b{rc[1]:+d} c{rc[2]:+d} d{rc[3]:+d}"
                         + ("" if fly else "  (DRY-RUN)"),
                         f"bat {d.battery}%  height {h if h is not None else '?'} cm"
                         + (f"  searched {d.search_turned:.0f} deg" if d.dstate == "SEARCH" else "")]
                if d.pose is not None:
                    lines.append(f"pos x{p[0]:+.2f} y{p[1]:+.2f} z{p[2]:+.2f} m  yaw {psi:+.0f}")
                if pe is not None:
                    lines.append(f"err {pe:.2f} m  {ye:+.0f} deg  slot s{assign.get(d.num)}")
                for i, s_ in enumerate(lines):
                    cv2.putText(img, s_, (10, 30 + 28 * i), 0, 0.7, (0, 255, 255), 2)
                cv2.imshow(d.name, cv2.resize(img, (480, 360)))
            cv2.imshow("top view", draw_topdown(drones, slots, assign, now))
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                print("q -> landing all")
                return
            if k == ord(" ") and fly:
                print("!!! EMERGENCY: motors off all")
                for d in drones:
                    d.send_noreply("emergency")
                    d.landed = True
                return
            time.sleep(max(0.0, period - (time.time() - t_loop)))
    except KeyboardInterrupt:
        print("\nCtrl+C -> landing all")
    finally:
        if fly:
            parallel([d for d in drones if d.airborne and not d.landed], lambda d: d.land())
        for d in drones:
            d.close()
        logf.close()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()