"""Record a clip of the defend scenario by capturing the viewport.

Replicator's render-variable path is broken on this machine, which takes out camera
sensors and Isaac Lab's built-in video recording. Kit's viewport rendering works fine --
it is what feeds the WebRTC livestream -- so this captures the viewport frame by frame
and stitches the PNGs into an mp4.

    python record_clip.py [frames] [out.mp4]
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="Record a defend-scenario clip.")
parser.add_argument("--frames", type=int, default=150, help="Frames to capture.")
parser.add_argument("--capture_every", type=int, default=2, help="Sim steps between captures.")
parser.add_argument("--out", type=str, default="/mnt/data/isaac/videos/defend_chase.mp4")
parser.add_argument("--fps", type=int, default=20)
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher({"headless": True, "enable_cameras": True})
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import action_space_kit.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport  # noqa: E402

FRAME_DIR = "/mnt/data/isaac/tmp/clip_frames"
os.makedirs(FRAME_DIR, exist_ok=True)
for stale in os.listdir(FRAME_DIR):
    os.remove(os.path.join(FRAME_DIR, stale))

env_cfg = parse_env_cfg("AS-Defend-v0", device="cuda:0", num_envs=1)
# point the viewport at the protected site so the interception is visible
env_cfg.viewer.eye = (9.0, 9.0, 6.0)
env_cfg.viewer.lookat = (0.0, 0.0, 1.0)

env = gym.make("AS-Defend-v0", cfg=env_cfg).unwrapped
obs, _ = env.reset()

viewport = get_active_viewport()
num_attackers = env.cfg.num_attackers
captured = 0
step = 0

print(f"[clip] capturing {args.frames} frames", flush=True)
while captured < args.frames:
    # greedy pursuit: each defender flies at the nearest attacker
    actions = {}
    for agent in env.cfg.possible_agents:
        rel = obs[agent][:, 9 : 9 + 3 * num_attackers].reshape(env.num_envs, num_attackers, 3)
        distance = torch.linalg.norm(rel, dim=-1)
        target = rel[torch.arange(env.num_envs, device=env.device), distance.argmin(dim=-1)]
        direction = target / torch.clamp(torch.linalg.norm(target, dim=-1, keepdim=True), min=1e-6)
        actions[agent] = torch.cat([direction, torch.zeros(env.num_envs, 1, device=env.device)], dim=-1)

    obs, _, _, _, _ = env.step(actions)
    env.sim.render()
    step += 1

    if step % args.capture_every == 0:
        capture_viewport_to_file(viewport, os.path.join(FRAME_DIR, f"frame_{captured:05d}.png"))
        captured += 1
        # the capture is asynchronous; give Kit a tick to flush it
        simulation_app.update()

# let the last captures land
for _ in range(60):
    simulation_app.update()

frames = sorted(f for f in os.listdir(FRAME_DIR) if f.endswith(".png"))
print(f"[clip] wrote {len(frames)} frames", flush=True)

env.close()
simulation_app.close()

# stitch with the ffmpeg binary that ships with imageio-ffmpeg
import subprocess  # noqa: E402

import imageio_ffmpeg  # noqa: E402

os.makedirs(os.path.dirname(args.out), exist_ok=True)
cmd = [
    imageio_ffmpeg.get_ffmpeg_exe(),
    "-y",
    "-framerate",
    str(args.fps),
    "-i",
    os.path.join(FRAME_DIR, "frame_%05d.png"),
    "-c:v",
    "libx264",
    "-pix_fmt",
    "yuv420p",
    "-vf",
    "scale=trunc(iw/2)*2:trunc(ih/2)*2",
    args.out,
]
proc = subprocess.run(cmd, capture_output=True, text=True)
if proc.returncode == 0 and os.path.exists(args.out):
    print(f"[clip] wrote {args.out} ({os.path.getsize(args.out)} bytes)", flush=True)
else:
    print(f"[clip] ffmpeg failed: {proc.stderr[-600:]}", flush=True)
    sys.exit(1)
