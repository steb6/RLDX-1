#!/bin/bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

MODEL_PATH="${1:?Usage: $0 <model-path>}"
TASK_NAME="${TASK_NAME:-PnPStoveToCounter}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
N_EPISODES="${N_EPISODES:-50}"
PORT="${PORT:-20100}"

export CUDA_VISIBLE_DEVICES
export NO_ALBUMENTATIONS_UPDATE=1

LOG_DIR="${LOG_DIR:-$BASE_DIR/output_final/robocasa/$(basename "$MODEL_PATH")}"
mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/server.log"
ROLLOUT_LOG="$LOG_DIR/${TASK_NAME}.log"

cd "$BASE_DIR"

"$HOME/miniconda3/envs/rldx/bin/uv" run python "$BASE_DIR/rldx/eval/run_rldx_server.py" \
  --model-path "$MODEL_PATH" \
  --embodiment-tag GENERAL_EMBODIMENT \
  --use-sim-policy-wrapper \
  --host 127.0.0.1 \
  --port "$PORT" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
  if ss -lnt | awk '{print $4}' | grep -q ":$PORT$"; then
    break
  fi
  sleep 2
done

"$BASE_DIR/rldx/eval/sim/robocasa/robocasa_uv/.venv/bin/python" \
  "$BASE_DIR/rldx/eval/rollout_policy.py" \
  --n_episodes "$N_EPISODES" \
  --policy_client_host 127.0.0.1 \
  --policy_client_port "$PORT" \
  --max_episode_steps 720 \
  --env_name "robocasa_panda_omron/${TASK_NAME}_PandaOmron_Env" \
  --n_action_steps 16 \
  --n_envs 1 \
  --video_dir "$LOG_DIR/$TASK_NAME" >"$ROLLOUT_LOG" 2>&1

echo "Evaluation finished. Logs:"
echo "  server:  $SERVER_LOG"
echo "  rollout: $ROLLOUT_LOG"
