"""Stable-Baselines3 wrapper that trains one policy shared by every agent of a MARL env.

Isaac Lab's ``Sb3VecEnvWrapper`` only takes single-agent envs, and its
``multi_agent_to_single_agent`` concatenates all agents into one centralized policy (16-D
action for four drones), which no single real drone could run. Here each agent of each
env becomes its own SB3 env row instead: row = env * num_agents + agent. Every drone then
runs the same policy on its own observation, which is what gets flown on the Tellos.

All agents of an env share its reset, so a row is done exactly when its env is.

Episode statistics the env publishes in ``extras['log']`` are collected and written to the
SB3 logger by :class:`LogExtrasCallback`.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn

from isaaclab.envs import DirectMARLEnv


class Sb3SharedPolicyVecEnv(VecEnv):
    """Flatten a :class:`DirectMARLEnv` with homogeneous agents into num_envs * num_agents SB3 rows."""

    def __init__(self, env: gym.Env):
        if not isinstance(env.unwrapped, DirectMARLEnv):
            raise ValueError(f"Expected a DirectMARLEnv, got {type(env.unwrapped)}")
        self.env = env
        self.unwrapped_env: DirectMARLEnv = env.unwrapped
        self.agents = list(self.unwrapped_env.possible_agents)
        self.num_agents = len(self.agents)
        self.num_isaac_envs = self.unwrapped_env.num_envs
        self.sim_device = self.unwrapped_env.device

        obs_spaces = [self.unwrapped_env.observation_spaces[a] for a in self.agents]
        act_spaces = [self.unwrapped_env.action_spaces[a] for a in self.agents]
        if any(s.shape != obs_spaces[0].shape for s in obs_spaces) or any(
            s.shape != act_spaces[0].shape for s in act_spaces
        ):
            raise ValueError("A shared policy needs every agent to have the same observation and action shapes.")
        observation_space = gym.spaces.Box(-np.inf, np.inf, shape=obs_spaces[0].shape, dtype=np.float32)
        action_space = act_spaces[0]
        super().__init__(self.num_isaac_envs * self.num_agents, observation_space, action_space)

        self._ep_rew_buf = np.zeros(self.num_envs)
        self._ep_len_buf = np.zeros(self.num_envs)
        self.log_buffer: list[dict[str, float]] = []

    # -- helpers ------------------------------------------------------------------

    def _flatten(self, per_agent: dict[str, torch.Tensor]) -> torch.Tensor:
        """dict of (E, ...) tensors -> (E * A, ...), row = env * A + agent."""
        stacked = torch.stack([per_agent[a] for a in self.agents], dim=1)
        return stacked.reshape(self.num_envs, *stacked.shape[2:])

    def _to_numpy(self, x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()

    # -- VecEnv API ---------------------------------------------------------------

    def reset(self) -> VecEnvObs:  # noqa: D102
        obs_dict, _ = self.env.reset()
        self._ep_rew_buf[:] = 0.0
        self._ep_len_buf[:] = 0
        return self._to_numpy(self._flatten(obs_dict))

    def step_async(self, actions):  # noqa: D102
        if not isinstance(actions, torch.Tensor):
            actions = torch.from_numpy(np.asarray(actions))
        actions = actions.to(device=self.sim_device, dtype=torch.float32)
        actions = actions.view(self.num_isaac_envs, self.num_agents, -1)
        self._async_actions = {a: actions[:, i] for i, a in enumerate(self.agents)}

    def step_wait(self) -> VecEnvStepReturn:  # noqa: D102
        obs_dict, rew_dict, term_dict, trunc_dict, extras = self.env.step(self._async_actions)
        obs = self._to_numpy(self._flatten(obs_dict))
        rewards = self._to_numpy(self._flatten(rew_dict))
        terminated = self._to_numpy(self._flatten(term_dict))
        truncated = self._to_numpy(self._flatten(trunc_dict))
        dones = terminated | truncated

        if extras.get("log"):
            self.log_buffer.append(dict(extras["log"]))

        self._ep_rew_buf += rewards
        self._ep_len_buf += 1
        infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        for idx in dones.nonzero()[0]:
            infos[idx]["episode"] = {"r": self._ep_rew_buf[idx], "l": self._ep_len_buf[idx]}
            infos[idx]["TimeLimit.truncated"] = bool(truncated[idx] and not terminated[idx])
            # Isaac Lab resets inside step(), so this is the first observation of the next
            # episode, exactly as with Isaac Lab's own Sb3VecEnvWrapper.
            infos[idx]["terminal_observation"] = obs[idx]
        self._ep_rew_buf[dones] = 0.0
        self._ep_len_buf[dones] = 0
        return obs, rewards, dones, infos

    def close(self):  # noqa: D102
        self.env.close()

    def get_attr(self, attr_name, indices=None):  # noqa: D102
        n = self.num_envs if indices is None else len(self._get_indices(indices))
        return [getattr(self.unwrapped_env, attr_name)] * n

    def set_attr(self, attr_name, value, indices=None):  # noqa: D102
        raise NotImplementedError("Setting attributes is not supported.")

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):  # noqa: D102
        if method_name == "render":
            return self.env.render()
        return getattr(self.unwrapped_env, method_name)(*method_args, **method_kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None):  # noqa: D102
        return [False] * self.num_envs

    def get_images(self):  # noqa: D102
        raise NotImplementedError("Getting images is not supported.")

    def seed(self, seed: int | None = None):  # noqa: D102
        return [self.unwrapped_env.seed(seed)] * self.num_envs


class LogExtrasCallback(BaseCallback):
    """Write the env's ``extras['log']`` statistics to the SB3 logger once per rollout.

    Each entry is weighted by its ``Episode/count`` (episodes finished that step) when present.
    """

    def __init__(self, vec_env: Sb3SharedPolicyVecEnv, verbose: int = 0):
        super().__init__(verbose)
        self._vec_env = vec_env

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        buf = self._vec_env.log_buffer
        if not buf:
            return
        weights = np.array([entry.get("Episode/count", 1.0) for entry in buf])
        keys = {k for entry in buf for k in entry if k != "Episode/count"}
        for key in sorted(keys):
            vals = np.array([entry.get(key, np.nan) for entry in buf])
            mask = ~np.isnan(vals)
            if mask.any():
                self.logger.record(key, float(np.average(vals[mask], weights=weights[mask])))
        self.logger.record("Episode/count", float(weights.sum()))
        buf.clear()
