"""Build a map of AprilTags in the origin tag's (tag 10) frame and save it.

Point the drone around the room so the camera sees the origin tag together with the
other tags (two or more tags in a frame links them). Every accepted detection feeds a
pose graph; tags chain back to tag 10. 'q' or Ctrl+C stops and writes the map to
config/map.yaml. A later script (06_localize.py) uses it to place the drone.

This never flies: comms is the calibration allowlist (no takeoff/rc).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from tello_calibration.calib.io import camera_from_dict, load_yaml
from tello_calibration.session import IdentityError, SessionError, resolve_drone, \
    verified_link, video_stream
from tello_calibration.tags.detect import TagDetector, load_tag_config
from tello_calibration.tags.mapping import MapBuilder, TagMap
from tello_calibration.utils.config import CONFIG_DIR
from tello_calibration.video.stream import VideoReceiver

log = logging.getLogger("map")
WINDOW = "map"
GREEN, YELLOW, RED = (0, 220, 0), (0, 220, 220), (0, 0, 255)


def draw(image: np.ndarray, observations, detector: TagDetector, tag_map: TagMap, fps: float) -> None:
    for obs in observations:
        placed = obs.tag_id in tag_map.poses
        color = GREEN if (obs.accepted and placed) else YELLOW if obs.accepted else RED
        pts = obs.corners.round().astype(np.int32)
        cv2.polylines(image, [pts], True, color, 2)
        where = "mapped" if placed else "seen" if obs.accepted else obs.reasons[0]
        x, y = pts.min(axis=0)
        text = f"id {obs.tag_id} {where}"
        cv2.putText(image, text, (int(x), max(int(y) - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 4)
        cv2.putText(image, text, (int(x), max(int(y) - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 2)
    mapped = tag_map.mapped()
    cv2.putText(image, f"{fps:4.1f} fps  mapped {len(mapped)}: {mapped}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def map_loop(video: VideoReceiver, detector: TagDetector, origin_tag: int) -> TagMap:
    builder = MapBuilder(origin_tag)
    tag_map = builder.build()
    last_id, frames, t0, fps, last_build = 0, 0, time.monotonic(), 0.0, 0.0
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    while True:
        frame, frame_id, _ = video.get_latest()
        if frame is not None and frame_id != last_id:
            last_id, frames = frame_id, frames + 1
            observations = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            builder.add_frame(observations)
            now = time.monotonic()
            if now - last_build > 0.5:        # rebuild periodically for live feedback
                tag_map = builder.build()
                last_build = now
            image = frame.copy()
            draw(image, observations, detector, tag_map, fps)
            cv2.imshow(WINDOW, image)
        now = time.monotonic()
        if now - t0 >= 1.0:
            fps, frames, t0 = frames / (now - t0), 0, now
        if (cv2.waitKey(1) & 0xFF == ord("q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return builder.build()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drone", required=True, help="drone name in config/fleet.yaml")
    parser.add_argument("--tags", type=Path, default=CONFIG_DIR / "tags.yaml")
    parser.add_argument("--camera", type=Path, help="default: config/camera/<drone>.yaml")
    parser.add_argument("--out", type=Path, default=CONFIG_DIR / "map.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, datefmt="%H:%M:%S",
                        format="%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s: %(message)s")

    camera_path = args.camera or CONFIG_DIR / "camera" / f"{args.drone}.yaml"
    if not camera_path.exists():
        log.error("No calibration at %s; run 03_calibrate.py first", camera_path)
        return 1
    camera = load_yaml(camera_path)
    K, dist = camera_from_dict(camera)
    tags = load_tag_config(args.tags)
    detector = TagDetector(tags, K, dist)
    log.info("Origin tag %d. Show it with other tags to link them. 'q' to finish.", tags.origin_tag)

    try:
        drone, local_ip = resolve_drone(args.drone)
        with verified_link(drone, local_ip) as link, video_stream(link, local_ip) as video:
            try:
                tag_map = map_loop(video, detector, tags.origin_tag)
            finally:
                cv2.destroyAllWindows()
    except IdentityError:
        return 2
    except SessionError:
        return 1
    except KeyboardInterrupt:
        log.info("Ctrl+C received")
        return 0

    if tags.origin_tag not in tag_map.poses or len(tag_map.mapped()) < 1:
        log.error("Origin tag %d never seen with others; no map written.", tags.origin_tag)
        return 1
    tag_map.save(args.out)
    for tag_id in tag_map.mapped():
        t = tag_map.poses[tag_id][:3, 3]
        log.info("tag %2d  x %+.3f  y %+.3f  z %+.3f m  (seen %d, edges %d)", tag_id,
                 t[0], t[1], t[2], tag_map.seen.get(tag_id, 0), tag_map.samples.get(tag_id, 0))
    log.info("wrote %s (%d tags in tag-%d frame)", args.out, len(tag_map.mapped()), tags.origin_tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
