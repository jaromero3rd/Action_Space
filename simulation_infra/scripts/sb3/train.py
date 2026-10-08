"""Train an Action Space task with Stable-Baselines3 PPO.

Adapted from Isaac Lab's scripts/reinforcement_learning/sb3/train.py. The difference: a
multi-agent task is wrapped with ``Sb3SharedPolicyVecEnv``, so all drones train one shared
policy that each real drone can run on its own, instead of Isaac Lab's centralized
``multi_agent_to_single_agent`` conversion.

    python scripts/sb3/train.py --task AS-Tello-Approach-v0 --headless --num_envs 512 --max_iterations 500
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import signal
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="sb3_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--checkpoint", type=str, default=None, help="Continue the training from checkpoint.")
parser.add_argument("--max_iterations", type=int, default=None, help="PPO iterations (rollout + update cycles).")
parser.add_argument("--save_interval", type=int, default=50, help="Save a checkpoint every n iterations.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def cleanup_pbar(*args):
    """Stop training and close the progress bar cleanly on ctrl+c."""
    import gc

    for obj in gc.get_objects():
        if "tqdm_rich" in type(obj).__name__:
            obj.close()
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, cleanup_pbar)

"""Rest everything follows."""

import os
import random
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import action_space_kit.tasks  # noqa: F401
from action_space_kit.rl.sb3_shared import LogExtrasCallback, Sb3SharedPolicyVecEnv


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with a Stable-Baselines3 agent."""
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # every agent of a MARL env is one SB3 row, so rollouts are that many times larger
    num_agents = len(env_cfg.possible_agents) if isinstance(env_cfg, DirectMARLEnvCfg) else 1
    num_rows = env_cfg.scene.num_envs * num_agents
    if args_cli.max_iterations is not None:
        agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * num_rows

    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root_path = os.path.abspath(os.path.join("logs", "sb3", args_cli.task))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = os.path.join(log_root_path, run_info)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    (Path(log_dir) / "command.txt").write_text(" ".join(sys.orig_argv))

    agent_cfg = process_sb3_cfg(agent_cfg, num_rows)
    policy_arch = agent_cfg.pop("policy")
    n_timesteps = agent_cfg.pop("n_timesteps")
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = Sb3SharedPolicyVecEnv(env)
        callbacks = [LogExtrasCallback(env)]
    else:
        env = Sb3VecEnvWrapper(env)
        callbacks = []

    frame_stack = agent_cfg.pop("frame_stack", 1)
    norm_args = {key: agent_cfg.pop(key) for key in ("normalize_input", "normalize_value", "clip_obs") if key in agent_cfg}
    if norm_args.get("normalize_input"):
        print(f"Normalizing input, {norm_args=}")
        env = VecNormalize(
            env,
            training=True,
            norm_obs=norm_args["normalize_input"],
            norm_reward=norm_args.get("normalize_value", False),
            clip_obs=norm_args.get("clip_obs", 100.0),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )
    vec_norm = env if isinstance(env, VecNormalize) else None
    if frame_stack > 1:
        # the policy sees each drone's last `frame_stack` (normalized) observations
        print(f"Stacking the last {frame_stack} observations")
        env = VecFrameStack(env, n_stack=frame_stack)

    agent = PPO(policy_arch, env, verbose=1, tensorboard_log=log_dir, **agent_cfg)
    if args_cli.checkpoint is not None:
        agent = agent.load(args_cli.checkpoint, env, print_system_info=True)

    # CheckpointCallback counts vec-env steps (one per iteration step), not transitions
    callbacks.append(
        CheckpointCallback(
            save_freq=args_cli.save_interval * agent_cfg["n_steps"],
            save_path=log_dir,
            name_prefix="model",
            save_vecnormalize=True,
            verbose=2,
        )
    )
    with contextlib.suppress(KeyboardInterrupt):
        agent.learn(total_timesteps=n_timesteps, callback=callbacks, progress_bar=True, log_interval=1)

    agent.save(os.path.join(log_dir, "model"))
    print(f"Saving to:\n{os.path.join(log_dir, 'model.zip')}")
    if vec_norm is not None:
        print("Saving normalization")
        vec_norm.save(os.path.join(log_dir, "model_vecnormalize.pkl"))
    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
