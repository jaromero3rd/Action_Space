"""Fly TELLO-3 and TELLO-4 on parallel lanes in front of the AprilTag.

The tag normal is the line straight out of the tag center. Each drone holds
a line 0.5 m off that normal, so the two lines are 1 m apart, and flies
along the normal. TELLO-3 stays on the left. TELLO-4 stays on the right.
After the first sighting the climb stick stays at zero and the pose comes
from odometry. Inside 1 m of the tag plane, the drone stops and lands.
"""

import math

MIN_HEIGHT_M = 0.25
SEARCH_CEILING_M = 2.0
# Image vertical error: -1 is the top edge, 0 is center.
# A tag above this stays in climb. -0.55 is the upper third of the frame.
LOCK_VERTICAL = -0.55
# A tag that stays visible this long locks even if it is still above LOCK_VERTICAL.
LOCK_AFTER_S = 1.5
MAX_FORWARD = 25
MAX_RIGHT = 25
MAX_YAW = 35
CLIMB = 22
LANE_M = 0.5
LAND_M = 1.0
LANES = {"TELLO-3": -LANE_M, "TELLO-4": LANE_M}


def _clamp(value, low, high):
    return max(low, min(high, value))


def apply_yaw(rotation, dyaw_deg):
    """Rotate a tag-to-camera matrix by a right yaw, degrees."""
    if dyaw_deg == 0:
        return rotation
    theta = math.radians(dyaw_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    yawed = (
        (cos_t, 0.0, -sin_t),
        (0.0, 1.0, 0.0),
        (sin_t, 0.0, cos_t),
    )
    return tuple(
        tuple(sum(yawed[i][k] * rotation[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def cam_in_tag(rotation, pose_t):
    """Camera position in the tag frame. x right, y down, z negative in front."""
    translated = [
        sum(rotation[j][i] * pose_t[j] for j in range(3)) for i in range(3)
    ]
    return [-value for value in translated]


def yaw_error_deg(rotation):
    """Degrees the nose must yaw right to look along the tag normal."""
    forward_x = rotation[0][2]
    forward_z = rotation[2][2]
    return math.degrees(math.atan2(forward_x, forward_z))


def track_rc(cam, yaw_err_deg, height_m, acquired, lane_m):
    """Return ((right, forward, up, yaw), mode).

    Climb only before the tag has been seen. After that, up is zero.
    Forward closes the distance while yaw and strafe fix the lane.
    Land when the drone is on that lane and within 1 m of the tag.
    """
    hover = (0, 0, 0, 0)
    if height_m is None:
        return hover, "wait"
    if height_m < MIN_HEIGHT_M:
        return hover, "ground"
    if not acquired or cam is None:
        if height_m < SEARCH_CEILING_M:
            return (0, 0, CLIMB, 0), "climb"
        return hover, "no tag"

    x, _y, z = (float(v) for v in cam)
    ahead = -z
    dist = math.sqrt(sum(float(v) * float(v) for v in cam))
    yaw = int(_clamp(yaw_err_deg * 2.0, -MAX_YAW, MAX_YAW))
    lateral = x - lane_m
    right = int(_clamp(-lateral * 40.0, -MAX_RIGHT, MAX_RIGHT))
    aimed = abs(yaw_err_deg) < 8
    on_lane = abs(lateral) < 0.15
    if on_lane and aimed and (dist <= LAND_M or ahead <= LAND_M):
        return hover, "land"
    forward = 0
    if ahead > LAND_M:
        forward = int(_clamp((ahead - LAND_M) * 8, 8, MAX_FORWARD))
        if abs(yaw_err_deg) >= 25:
            forward = min(forward, 10)
    if not aimed:
        return (right, forward, 0, yaw), "aim"
    if not on_lane:
        return (right, forward, 0, yaw), "lane"
    return (right, forward, 0, yaw), "track"


def predict_odom(pose_t, vel, dyaw_deg, dt):
    """Move the tag pose using the drone's own speed.

    pose_t is the tag in the camera frame: x right, y down, z forward, meters.
    vel is (right, forward, up) in m/s from the state packet. dyaw_deg is the
    change in heading, positive to the right.
    """
    if dt <= 0:
        return list(pose_t)
    x, y, z = (float(v) for v in pose_t)
    theta = math.radians(dyaw_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    x, z = x * cos_t - z * sin_t, x * sin_t + z * cos_t
    right, forward, up = vel
    x -= right * dt
    z = max(0.3, z - forward * dt)
    y += up * dt
    return [x, y, z]


def predict_tag(pose_t, sticks, dt):
    """Move the last tag pose by the sticks just sent.

    sticks are (right, forward, up, yaw). Yaw right and climb are positive.
    Forward speed at rc 100 is about 0.8 m/s. Yaw at rc 100 is about 60 deg/s.
    """
    if dt <= 0:
        return list(pose_t)
    x, y, z = (float(v) for v in pose_t)
    right, forward, up, yaw = sticks
    theta = math.radians((yaw / 100.0) * 60.0 * dt)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    x, z = x * cos_t - z * sin_t, x * sin_t + z * cos_t
    z = max(0.3, z - (forward / 100.0) * 0.8 * dt)
    x -= (right / 100.0) * 0.8 * dt
    y = y + (up / 100.0) * 0.6 * dt
    return [x, y, z]
