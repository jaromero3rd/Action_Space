#!/usr/bin/env bash
# Run a trained policy in simulation: score it, or record a video.
#
#   ./run_policy.sh score  [CHECKPOINT_DIR] [EPISODES]   # default: 64 episodes
#   ./run_policy.sh record [CHECKPOINT_DIR] [EPISODES]   # default: 2 episodes -> outputs/videos/
#
# CHECKPOINT_DIR defaults to policies/peer_qualified_84M (4-drone peer policy).
# Needs the install from setup.sh in ~/isaac. Logs go to outputs/.
set -euo pipefail

MODE="${1:-score}"
POLICY_DIR="${2:-policies/peer_qualified_84M}"
EPISODES="${3:-}"
TASK="${TASK:-AS-Tello-Room-Peer-v0}"

cd "$(dirname "$0")"
unset CONDA_PREFIX CONDA_DEFAULT_ENV
source "$HOME/isaac/env_isaaclab/bin/activate"
export OMNI_KIT_ACCEPT_EULA=YES TELLO_USD_PATH="$HOME/isaac/assets/tello.usd"
export PYTHONPATH="$PWD/source/action_space_kit${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p outputs/videos

CKPT="$POLICY_DIR/model.zip"
[ -f "$CKPT" ] || { echo "missing $CKPT"; exit 1; }
NAME="$(basename "$POLICY_DIR")"

case "$MODE" in
  score)
    N="${EPISODES:-64}"
    LOG="outputs/play_${NAME}.log"
    python -u scripts/sb3/play.py --task "$TASK" --headless --num_envs 16 --episodes "$N" \
      --checkpoint "$CKPT" --report "outputs/play_${NAME}.json" > "$LOG" 2>&1 || true
    grep -E "episodes\]|complete|Traceback|Error" "$LOG" | tail -5
    echo "full log: $LOG   report: outputs/play_${NAME}.json (need complete=true)"
    ;;
  record)
    N="${EPISODES:-2}"
    LOG="outputs/record_${NAME}.log"
    python -u scripts/record_room.py --task "$TASK" --checkpoint "$CKPT" --episodes "$N" \
      --out "outputs/videos/${NAME}.mp4" --frame_dir outputs/frames_room > "$LOG" 2>&1 || true
    grep -E "\[rec\]|Traceback|Error" "$LOG" | tail -10
    echo "video: outputs/videos/${NAME}.mp4   log: $LOG"
    ;;
  *) echo "usage: $0 score|record [CHECKPOINT_DIR] [EPISODES]"; exit 1 ;;
esac
