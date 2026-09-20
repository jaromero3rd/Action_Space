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

### Baselines to beat (AS-Defend-v0 defaults, 2 defenders vs 2 attackers)

| Policy | Interceptions | Breaches | Time to intercept | Episode return* |
|---|---|---|---|---|
| `zero` (do nothing) | 0% | 100% | n/a | -60.4 |
| `chase` (greedy pursuit) | 100% | 0% | 3.4 s | +83.0 |

*Episode return sums both defenders, matching what the training logger reports. If your
training curve sits near -60, the policy has not learned to engage yet; near +83 it is
doing about as well as greedy pursuit, and beating that means coordinating -- splitting
targets rather than both chasing the nearest attacker.

## Known issue: PPO converges to passivity on the default difficulty

Out of the box, both IPPO and single-agent PPO drift *below* the do-nothing baseline on
`AS-Defend-v0` and end up flying nowhere. This is the interesting part of the hackathon,
not a broken environment -- the environment itself is verified:

- scripted greedy pursuit (`--baseline chase`) intercepts **100%** of attackers with
  **0%** breaches, so a good policy exists inside the action space
- the reward separates those cases cleanly: **-26.2** for doing nothing, **+135.7** for
  greedy pursuit (both-agent episode totals, as the training logger reports them)
- actions, observations and termination were each checked in isolation

What was already ruled out, so you do not repeat it:

| Suspected cause | Verdict |
|---|---|
| Reward scale (breach dwarfing shaping) | fixed, still diverged |
| Rates vs per-step deltas scaled wrongly | fixed, still diverged |
| Sentinel observations (100.0 for unseen targets) | fixed, still diverged |
| Network too small, optimiser settings | set to Isaac Lab reference values, still diverged |
| Multi-agent machinery | single-agent PPO diverges identically |
| Entropy collapse | ruled out: policy std decays normally (1.0 -> 0.4) |
| Unclipped actions inflating the policy std | real bug, fixed (std now stable at ~0.88) |

The most likely remaining explanation is **discovery**: with a 0.35 m capture radius,
random exploration almost never produces an interception, so the policy sees no positive
signal and settles for minimising penalties. The intended fix is a curriculum -- start
easy, then tighten:

```bash
# easier: bigger capture radius, closer and slower attackers
python scripts/skrl/train.py --task AS-Defend-v0 --algorithm IPPO --headless \
  --num_envs 512 --max_iterations 300 \
  env.capture_radius=1.0 env.spawn_radius=4.0 env.attacker_speed=[0.2,0.4]
```

Then retrain from that checkpoint with the values stepped back towards the defaults
(`capture_radius` 0.35, `spawn_radius` 6.0, `attacker_speed` [0.4, 0.9]). Other levers
worth trying: reward the defender for staying between the attacker and the site, give
each defender its own assigned target instead of the nearest one, or warm-start from
behaviour cloning on the `chase` baseline.

## Known issue: Replicator render variables fail on this server

`AS-Defend-Camera-v0` fails at startup with:

```
TypeError: Unable to write from unknown dtype, kind=f, size=0
```

**What is actually broken:** Omniverse Replicator never receives the render variables the
renderer is supposed to produce, so every consumer gets an empty buffer. Rendering itself
is fine -- capturing the Kit viewport to a PNG works, and the WebRTC livestream shows a
live picture.

Evidence, so nobody repeats the search:

| Test | Result |
|---|---|
| Isaac Lab's own `Isaac-Cartpole-RGB-Camera-Direct-v0` | same failure -- not our code |
| Minimal script: cube + camera + `rgb` annotator, no Isaac Lab | same failure |
| `depth` and `semantic_segmentation` annotators | same failure -- not colour-specific |
| Replicator `BasicWriter` to disk | same failure (`kind=i`) -- not the annotator API |
| `get_data(use_legacy_structure=False)` | same failure -- not the data structure |
| 256x256, 512x512, FXAA instead of DLSS | same failure -- not resolution or anti-aliasing |
| Camera on env root / on a drone body / `replicate_physics=False` | same failure |
| `xvfb-run` virtual display | delays it by minutes, same failure |
| No other simulator process running | same failure |
| **Kit viewport capture to PNG** | **works** |

The Kit log records no errors: the RTX renderer starts, gets a device, and the only hint
is a node-registration warning about `omni.replicator.core.FabricReader` -- the component
that feeds annotators -- shortly before the empty buffer appears.

The machine runs driver **590.52.01** from NVIDIA's cloud-gaming (GRID) branch, while
Isaac Sim 5.1 and 6.0 both list a **580-series** production-branch driver as tested. That
is the most likely culprit, though unproven. Fixing it means swapping to the production
branch, or running camera work elsewhere.

### Workaround for visuals: capture the viewport

`scripts/record_clip.py` records the defend scenario by capturing viewport frames and
stitching them with ffmpeg, bypassing Replicator entirely:

```bash
python scripts/record_clip.py --frames 150 --out /mnt/data/isaac/videos/defend_chase.mp4
```

Two things that look like bugs but are not:

- The recorder launches with **livestream mode on**. In plain headless mode Isaac Lab sets
  `/isaaclab/render/active_viewport` to false and leaves the viewport context unset, so
  captures silently produce no files. Livestream keeps the viewport pipeline alive.
- Frames are stitched **before** the simulator is closed, because `simulation_app.close()`
  exits the process and anything after it never runs.

The livestream (`isaaclab-stream.sh`, or `--livestream 1`) works for the same reason. What
stays blocked is anything needing camera *sensors* as policy input -- `AS-Defend-Camera-v0`
and training a detector -- since those read through Replicator. State-based training is
completely unaffected.

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
