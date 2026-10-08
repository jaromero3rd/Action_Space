"""Play (and score) a Stable-Baselines3 checkpoint of an Action Space task.

Adapted from Isaac Lab's scripts/reinforcement_learning/sb3/play.py, using the same
shared-policy wrapper as training. Prints episode statistics as episodes finish.

    python scripts/sb3/play.py --task AS-Tello-Approach-v0 --num_envs 16 --checkpoint logs/sb3/AS-Tello-Approach-v0/<run>/model.zip
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="sb3_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (default: latest run).")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--episodes", type=int, default=0, help="Stop after this many finished episodes (0 = run forever).")
parser.add_argument("--no_dr", action="store_true", default=False, help="Fly with nominal dynamics (no randomization).")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import re
import time

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict

from isaaclab_rl.sb3 import Sb3VecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path

import action_space_kit.tasks  # noqa: F401
from action_space_kit.rl.sb3_shared import Sb3SharedPolicyVecEnv


def vecnormalize_path(checkpoint_path: str) -> Path:
    """model.zip -> model_vecnormalize.pkl; model_N_steps.zip -> model_vecnormalize_N_steps.pkl."""
    p = Path(checkpoint_path)
    m = re.match(r"model_(\d+)_steps$", p.stem)
    name = f"model_vecnormalize_{m.group(1)}_steps.pkl" if m else "model_vecnormalize.pkl"
    return p.with_name(name)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with a Stable-Baselines3 agent."""
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if hasattr(env_cfg, "debug_vis"):
        env_cfg.debug_vis = True  # scene markers, for the livestream
    if args_cli.no_dr and hasattr(env_cfg, "dynamics"):
        env_cfg.dynamics.domain_randomization = False

    if args_cli.checkpoint is None:
        log_root_path = os.path.abspath(os.path.join("logs", "sb3", args_cli.task))
        checkpoint_path = get_checkpoint_path(log_root_path, ".*", "model.zip", sort_alpha=False)
    else:
        checkpoint_path = os.path.abspath(args_cli.checkpoint)
    log_dir = os.path.dirname(checkpoint_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    shared = isinstance(env.unwrapped, DirectMARLEnv)
    env = Sb3SharedPolicyVecEnv(env) if shared else Sb3VecEnvWrapper(env)
    base_env = env

    norm_path = vecnormalize_path(checkpoint_path)
    if norm_path.exists():
        print(f"Loading saved normalization: {norm_path}")
        env = VecNormalize.load(str(norm_path), env)
        env.training = False
        env.norm_reward = False
    else:
        print(f"[WARN] No normalization file at {norm_path}; running with raw observations.")
    frame_stack = agent_cfg.get("frame_stack", 1)
    if frame_stack > 1:
        env = VecFrameStack(env, n_stack=frame_stack)

    print(f"Loading checkpoint from: {checkpoint_path}")
    agent = PPO.load(checkpoint_path, env, print_system_info=True)
    dt = base_env.unwrapped_env.step_dt if shared else base_env.unwrapped.step_dt

    totals: dict[str, float] = {}
    count = 0.0
    obs = env.reset()
    if shared:
        # a full reset staggers episode clocks for training; scored episodes must get the whole
        # episode, or the first one in every env is cut short and counted as a failure
        base_env.unwrapped_env.episode_length_buf[:] = 0
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions, _ = agent.predict(obs, deterministic=True)
            obs, _, _, _ = env.step(actions)
        if shared and base_env.log_buffer:
            for entry in base_env.log_buffer:
                n = entry.get("Episode/count", 0.0)
                count += n
                for k, v in entry.items():
                    if k != "Episode/count":
                        totals[k] = totals.get(k, 0.0) + v * n
            base_env.log_buffer.clear()
            print(f"[{int(count)} episodes] " + ", ".join(f"{k.split('/')[-1]}={v / count:.3f}" for k, v in sorted(totals.items())))
            if args_cli.episodes and count >= args_cli.episodes:
                break
        timestep += 1
        if args_cli.video and timestep == args_cli.video_length:
            break
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
