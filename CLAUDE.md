# CLAUDE.md: install and run guide for AI agents

> **Repo layout (moved 2026-10-07):** the Isaac Lab **simulation** now lives in
> `simulation_infra/` — prefix the paths in the steps below (`setup.sh`, `source/`,
> `scripts/`, `policies/`, `assets/`, `policy_maker.md`) with `simulation_infra/`
> (e.g. `simulation_infra/setup.sh ~/isaac`). The **real-world drone** code is in
> `drone_deployment_infra/` (calibration, mapping, connect, policies, recordings) —
> see `drone_deployment_infra/README.md`. This guide covers the simulation side.

Isaac Lab kit for the Action Space hackathon. The main demo is **`AS-Tello-Approach-v0`**:
four DJI Tello drones fly to a base, each passes through its own gate on a 3 m keep-out
sphere, then lands on its own pad. The human version of this guide is the
"Installation" section of `README.md`; keep the two in sync when you change either.

Last verified end-to-end: 2026-09-27, RTX 5080 16 GB, driver 580.95.05, Linux Mint 22.3
(Ubuntu 24.04 base), Isaac Sim 5.1.0, Isaac Lab v2.3.2, torch 2.7.0+cu128, Python 3.11.

## Ground rules for agents

- **Run every step and check its pass criterion before moving on.** Don't report success
  from exit codes alone. Isaac Sim often exits 0 after a failure, and it hard-exits on
  `simulation_app.close()`, so buffered stdout gets lost. **Always run Python with `-u`**
  and redirect to a log inside the repo (`outputs/` is gitignored), for example
  `python -u ... > outputs/<name>.log 2>&1`, then grep the log.
- Don't write logs, frames or videos to `/tmp` or `/mnt/data`. Use `outputs/` in the repo.
- `setup.sh` may call `sudo apt-get`, but only for missing packages. If sudo needs a
  password you don't have, check `dpkg -s cmake build-essential git git-lfs curl ffmpeg
  libglu1-mesa libvulkan1 vulkan-tools`. If anything is missing, ask the human to run
  `! sudo apt-get install -y <pkgs>`.
- Filter logs when reading them. Isaac Sim prints hundreds of harmless `[Warning]` lines.
  Useful filter: `grep -vE "Warning\]|WARNING|^\s*$|^\|"`.

## Step 0: preflight

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
df -h ~          # need ~30 GB free
echo "CONDA_PREFIX=$CONDA_PREFIX"   # non-empty is fine: setup.sh unsets it, but you must too (Step 2)
```

Pass: a GPU is listed. RTX 50-series needs driver >= 570. If `nvidia-smi` fails, stop and
tell the human: it's a driver problem (often a kernel upgrade left the module unbuilt).

## Step 1: install

```bash
git clone https://github.com/jaromero3rd/Action_Space.git ~/Action_Space   # skip if already cloned
cd ~/Action_Space
mkdir -p ~/isaac
./setup.sh ~/isaac > ~/isaac/setup.log 2>&1; echo "EXIT=$?" >> ~/isaac/setup.log
grep -E "^== |EXIT=|Traceback|wrote|isaaclab ok|AS-Tello-Approach" ~/isaac/setup.log
```

Takes 7-40 minutes depending on bandwidth (7 on the reference box). Run it in the
background and watch the `== ` stage markers. Stages, in order: `checking GPU`,
`system packages`, `workspace`, `python 3.11 environment`, `Isaac Sim 5.1.0`,
`PyTorch (CUDA 12.8)`, `Isaac Lab v2.3.2`, `this kit`, `Tello asset`, `verifying`.

Pass, all of:
- `EXIT=0`
- `wrote <ISAAC_ROOT>/assets/tello.usd` (or the file already existed)
- `isaaclab ok, cuda: True`
- a table row containing `AS-Tello-Approach-v0`

The script is idempotent; re-run it after fixing a failure.

What it produces:
| Path | What |
|---|---|
| `~/isaac/env_isaaclab/` | Python 3.11 uv venv with isaacsim, isaaclab, torch, skrl, stable_baselines3, this kit (editable) |
| `~/isaac/IsaacLab/` | Isaac Lab v2.3.2 checkout |
| `~/isaac/assets/tello.usd` | drone model, converted from `assets/tello/tello.urdf` |
| `~/isaac/.cache`, `~/isaac/tmp` | pip/uv caches and TMPDIR used during install |

Data the demo needs that's **already in git** (verify with `file`: they must be
`Zip archive` / `data`, not ASCII LFS pointers):
`policies/approach_it5_room_fixed/model.zip`, `model_vecnormalize.pkl`, `params/*.yaml`,
`command.txt`.

### Known install traps (all handled by the current setup.sh)

| Symptom in the log | Cause | Handling |
|---|---|---|
| `ERROR: pip's dependency resolver ... packaging 23.0` | Isaac Sim pins packaging 23.0 | harmless, ignore |
| `FileNotFoundError: Could not find the isaac-sim directory: .../IsaacLab/_isaac_sim` right after `Setting up vscode settings` | an active conda env: `isaaclab.sh` prefers `$CONDA_PREFIX` python over `$VIRTUAL_ENV` | setup.sh unsets the `CONDA_*` vars |
| `error: Distribution not found at: file://<ISAAC_ROOT>/source/action_space_kit` | old setup.sh resolved `$0` after `cd` | fixed: `KIT_DIR` is resolved at the top |
| `No module named isaaclab` after install | flatdict build failure made Isaac Lab skip the core package | setup.sh pins `setuptools<81`, prebuilds flatdict, then reinstalls `source/isaaclab` |
| `[WARN] Could not find Isaac Sim VSCode settings` | cosmetic | ignore |

## Step 2: shell environment (every new shell / every Bash call)

Shell state doesn't persist between agent tool calls, so prefix each command:

```bash
cd ~/Action_Space && unset CONDA_PREFIX CONDA_DEFAULT_ENV && source ~/isaac/env_isaaclab/bin/activate && export OMNI_KIT_ACCEPT_EULA=YES TELLO_USD_PATH=$HOME/isaac/assets/tello.usd && mkdir -p outputs
```

`TELLO_USD_PATH` is mandatory. Its default (`/mnt/data/isaac/assets/tello.usd`) exists only
on the old shared server.

## Step 3: verify the environment (~2 min)

```bash
python -u scripts/check_approach_env.py --headless --test all > outputs/check_approach.log 2>&1
grep -E "^\s*\[(ok|FAIL)|^== |zero |scripted " outputs/check_approach.log
```

Pass: every check line is `[ok]` (12 geometry, 6 controller, 4 baseline) and the baseline
lines read roughly:
```
zero     : success 0%,   ... team return ~ -800
scripted : success 100%, arrived 4.00/4, landed 4.00/4, gates 4.00/4, illegal entry 0%, collision 0%, ... team return ~ +1540
```
The first ever launch compiles shaders for several minutes; that's not a hang.

## Step 4: score the shipped trained policy (~1 min)

```bash
python -u scripts/sb3/play.py --task AS-Tello-Approach-v0 --headless --num_envs 64 --episodes 256 \
  --checkpoint policies/approach_it5_room_fixed/model.zip \
  env.spawn_azimuth_range=0.0 env.rew_scale_separation=-20.0 env.rew_violation=-100.0 \
  > outputs/play_policy.log 2>&1
grep "episodes\]" outputs/play_policy.log | tail -1
```

Pass (reference run): `near_base_frac=0.897, success_rate=0.864, collision_rate=0.136,
gates_passed=4.000, illegal_entry_rate=0.000, crash_rate=0.000`. Treat `near_base_frac >= 0.80`
as the bar. The three `env.*` overrides are **required**; they match the training
command in `policies/approach_it5_room_fixed/command.txt`. `play.py` loads
`model_vecnormalize.pkl` from next to the checkpoint automatically.

## Step 5: record a video (~1-2 min)

```bash
python -u scripts/record_approach.py --pilot policy \
  --checkpoint policies/approach_it5_room_fixed/model.zip \
  --spawn_azimuth_range 0.0 --set rew_scale_separation=-20.0 rew_violation=-100.0 \
  --episodes 2 --out outputs/videos/approach_policy.mp4 --frame_dir outputs/frames \
  > outputs/record.log 2>&1
grep "\[rec\]" outputs/record.log
```

Pass: `[rec] wrote outputs/videos/approach_policy.mp4 (<~1 MB+> bytes)` and at least one
`episode N: SUCCESS -- gates 4/4, landed 4/4`. An episode may show `cut at 20 s`, which
is the recorder's cap, not an error. **Always pass `--out` and `--frame_dir`**; the
defaults are under `/mnt/data`. To confirm frames aren't blank, extract one and look at it:
`ffmpeg -loglevel error -y -ss 7 -i outputs/videos/approach_policy.mp4 -frames:v 1 outputs/frame.png`.
Expect a grid floor, a red dotted sphere, four colored gate rings and a green touchdown disc.
Delete `outputs/frames` afterwards.

Viewport capture works headless because the recorder enables livestream mode internally.
Replicator camera *sensors* are a separate known issue (see README), and this demo doesn't use them.

## Where things live

| Path | What |
|---|---|
| `source/action_space_kit/action_space_kit/tasks/direct/tello_approach/` | approach task: `tello_approach_env_cfg.py` (geometry, rewards: every `env.*` override), `tello_approach_env.py`, `scripted_pilot.py` |
| `source/action_space_kit/action_space_kit/control/` | Tello firmware/velocity cascade model (`tello_dynamics.py`), rc-channel controller |
| `source/action_space_kit/action_space_kit/rl/sb3_shared.py` | makes all 4 drones share one SB3 policy |
| `source/action_space_kit/action_space_kit/assets/tello.py` | reads `TELLO_USD_PATH` |
| `scripts/sb3/{train,play}.py` | SB3 train / score |
| `scripts/check_approach_env.py`, `scripts/record_approach.py` | verification, video |
| `policies/` | shipped checkpoints (small, tracked in git despite the `.gitignore` defaults) |
| `setup.sh` | the installer; `assets/tello/convert.sh` is URDF→USD |

## Training (optional)

For reward locations, tuning knobs, batch sizing and measured throughput, see `policy_maker.md`.

```bash
python -u scripts/sb3/train.py --task AS-Tello-Approach-v0 --headless --num_envs 512 --max_iterations 800 \
  agent.learning_rate=1e-3 env.spawn_azimuth_range=0.0 env.rew_scale_separation=-20.0 env.rew_violation=-100.0
```

Output goes to `logs/sb3/AS-Tello-Approach-v0/<timestamp>/model.zip`. For runs longer than
your tool timeout, launch detached (`setsid nohup ... > outputs/train.log 2>&1 &`) and
poll the log. Score the result with the Step 4 command, pointing `--checkpoint` at it.
