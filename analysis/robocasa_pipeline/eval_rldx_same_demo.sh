#!/bin/bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

MODEL_PATH="${1:?Usage: $0 <model-path> <demo-hdf5>}"
DEMO_HDF5="${2:?Usage: $0 <model-path> <demo-hdf5>}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PORT="${PORT:-20140}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-300}"
N_ACTION_STEPS="${N_ACTION_STEPS:-16}"

export CUDA_VISIBLE_DEVICES
export NO_ALBUMENTATIONS_UPDATE=1

LOG_DIR="${LOG_DIR:-$BASE_DIR/output_final/same_demo/$(basename "$MODEL_PATH")}"
mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/server.log"
EVAL_LOG="$LOG_DIR/eval.log"

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
  "$BASE_DIR/analysis/robocasa_pipeline/eval_same_demo.py" \
  --demo-hdf5 "$DEMO_HDF5" \
  --policy-host 127.0.0.1 \
  --policy-port "$PORT" \
  --output-dir "$LOG_DIR" \
  --max-episode-steps "$MAX_EPISODE_STEPS" \
  --n-action-steps "$N_ACTION_STEPS" >"$EVAL_LOG" 2>&1

echo "Same-demo evaluation finished. Logs:"
echo "  server:  $SERVER_LOG"
echo "  eval:    $EVAL_LOG"
