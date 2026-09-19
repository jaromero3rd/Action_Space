# Action Space Kit

Isaac Lab starter kit for the **Action Space** anti-drone swarm defense hackathon
(Oct 23-25, Microsoft NERD, Cambridge MA). Train defender policies in simulation on day
one, fly them on real Tello EDU drones on day two.

## The one rule that makes day two work

A real Tello EDU accepts only high-level commands over its SDK: `takeoff`, `land`,
`go x y z`, and `rc a b c d` -- four velocity channels, each -100..100, roughly +/-1 m/s.
**There is no motor-level control.** Isaac Lab's built-in quadcopter task commands body
thrust and torque, so a policy trained on it can never fly a Tello.

Every task here acts through `action_space_kit.control.TelloVelocityController`, whose
action vector maps 1:1 onto the four `rc` channels:

```
action = [vx, vy, vz, yaw_rate], each in [-1, 1]
   vx -> rc b (forward+)    vy -> rc a (left+)
   vz -> rc c (up+)         yaw_rate -> rc d (clockwise+)
```

If you write your own task, keep that action space and your policy stays flyable.

## Tasks

| Task | Type | What it is |
|---|---|---|
| `AS-Tello-Hover-v0` | single-agent | Hold a sampled position. Start here; trains in minutes. |
| `AS-Tello-Waypoint-v0` | single-agent | Chase a moving goal -- the basis of pursuit. |
| `AS-Defend-v0` | multi-agent | The hackathon scenario: defenders intercept attackers before they reach the protected site. |

List them yourself with `python scripts/list_envs.py`.

## Quickstart

```bash
# on the shared server
cd /mnt/data/isaac/action_space_kit
source /mnt/data/isaac/env_isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES

# 1. hover: ~10 min on an A10G, reward should climb from ~15 to ~70
python scripts/skrl/train.py --task AS-Tello-Hover-v0 --headless --num_envs 1024 --max_iterations 200

# 2. watch the trained policy
python scripts/skrl/play.py --task AS-Tello-Hover-v0 --headless --num_envs 16

# 3. the hackathon task, multi-agent PPO
python scripts/skrl/train.py --task AS-Defend-v0 --algorithm IPPO --headless --num_envs 512 --max_iterations 500
```

`--algorithm` accepts `IPPO` (each defender learns independently) or `MAPPO` (shared
critic, usually better coordination). Metrics land in `logs/skrl/<task>/<run>/`; view them
with `tensorboard --logdir logs`.

## Measuring whether it actually defends

Training curves say little about the thing that matters on day two. `scripts/evaluate.py`
runs whole episodes and reports interception rate, breach rate and time-to-intercept:

```bash
# baselines first, so you know what a policy has to beat
python scripts/evaluate.py --task AS-Defend-v0 --baseline zero  --episodes 20
python scripts/evaluate.py --task AS-Defend-v0 --baseline chase --episodes 20

# then your trained policy
python scripts/evaluate.py --task AS-Defend-v0 --checkpoint logs/skrl/tello_defend/<run>/checkpoints/best_agent.pt
```

A policy that cannot beat `--baseline chase` (greedy pursuit of the nearest attacker) is
not yet worth flying.

## Tuning the defend task

Everything worth changing is in
`source/action_space_kit/action_space_kit/tasks/direct/tello_defend/tello_defend_env_cfg.py`:

- `num_defenders`, `num_attackers` -- team sizes (2 v 2 by default)
- `site_radius`, `spawn_radius`, `capture_radius` -- the geometry of the scenario
- `attacker_speed`, `attacker_evasion` -- attacker difficulty; `evasion = 0` is a straight
  dive, raise it towards 1 for weaving. This is your curriculum.
- `trainable_attackers` -- set True to train both sides instead of scripted attackers
- `rew_scale_*` -- reward weights: closing distance, interception, perimeter breach,
  control effort

## Flying on real drones

```bash
# no hardware needed: prints the exact rc commands it would send
python -m action_space_kit.bridge.tello_bridge --dry-run --duration 3
```

`TelloBridge` sends policy actions as `rc` commands at a fixed rate with a geofence, a
speed cap, a dead-man timeout, and `land` on any error. Start every session with
`--dry-run`, then one drone, then a swarm. Tello EDU supports station mode, so several
drones can fly from one computer on the same WiFi network.

## Sim-to-real checklist

1. Same action space: `[vx, vy, vz, yaw_rate]`, nothing lower level.
2. Keep `speed_cap` at 0.5 or below indoors -- the sim has no walls, your venue does.
3. Expect 0.1-0.3 s of latency on real hardware; train with
   `TelloCommandCfg.command_latency_steps > 0` if your policy is twitchy.
4. The drone auto-lands after 15 s without a command; the bridge's timeout is shorter.
5. Battery sag makes a real Tello slower than the sim near the end of a flight.

## Docs

- [`docs/shared-server.md`](docs/shared-server.md) -- using the shared GPU box without
  stepping on other teams
- [`docs/livestream.md`](docs/livestream.md) -- watching the sim from your laptop
- [`THIRD_PARTY.md`](THIRD_PARTY.md) -- the Tello model's BSD-3 licence and attribution
- [`docs/template_readme.md`](docs/template_readme.md) -- the original Isaac Lab template README
