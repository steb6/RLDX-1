#!/bin/bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

DATA_DIR="${DATA_DIR:?Set DATA_DIR to the RoboCasa LeRobot dataset root}"
CKPT_NAME="${CKPT_NAME:-rldx_robocasa}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-RLWRLD/RLDX-1-PT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
NUM_GPUS="${NUM_GPUS:-1}"
MAX_STEPS="${MAX_STEPS:-50000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
MODALITY_CONFIG_PATH="${MODALITY_CONFIG_PATH:-$BASE_DIR/rldx/configs/data/robocasa_native_config.py}"
ACTION_HORIZON="${ACTION_HORIZON:-16}"
MAX_STATE_DIM="${MAX_STATE_DIM:-53}"
MAX_ACTION_DIM="${MAX_ACTION_DIM:-12}"

export CUDA_VISIBLE_DEVICES
export NO_ALBUMENTATIONS_UPDATE=1
export WANDB_PROJECT="${WANDB_PROJECT:-rldx-robocasa}"

CKPT_DIR="$BASE_DIR/ckpt/rldx1/finetuned/$CKPT_NAME"

cd "$BASE_DIR"
echo "Training $CKPT_NAME on $DATA_DIR using GPU(s) $CUDA_VISIBLE_DEVICES"

"$HOME/miniconda3/envs/rldx/bin/uv" run python "$BASE_DIR/rldx/experiment/launch_train.py" \
  --base-model-path "$BASE_MODEL_PATH" \
  --output-dir "$CKPT_DIR" \
  --experiment-name "$CKPT_NAME" \
  --dataset-path "$DATA_DIR" \
  --embodiment-tag GENERAL_EMBODIMENT \
  --modality-config-path "$MODALITY_CONFIG_PATH" \
  --num-gpus "$NUM_GPUS" \
  --dataloader-num-workers "$DATALOADER_NUM_WORKERS" \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --gradient-accumulation-steps 1 \
  --learning-rate 1e-4 \
  --lr-scheduler-type cosine \
  --max-steps "$MAX_STEPS" \
  --action-horizon "$ACTION_HORIZON" \
  --save-steps "$SAVE_STEPS" \
  --save-total-limit 5 \
  --max-state-dim "$MAX_STATE_DIM" \
  --max-action-dim "$MAX_ACTION_DIM" \
  --dataset-mode sharded \
  --shard-size 1024 \
  --episode-sampling-rate 0.1 \
  --num-shards-per-epoch 100000 \
  --use-wandb \
  --wandb-project "$WANDB_PROJECT"
