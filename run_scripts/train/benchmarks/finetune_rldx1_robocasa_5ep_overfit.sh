#!/bin/bash
# Overfit RLDX-1 on 5 MimicGen episodes (single A100, GPU 1)

set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export WANDB_PROJECT="${WANDB_PROJECT:-rldx-finetune}"
export NO_ALBUMENTATIONS_UPDATE=1

# ── Recipe ─────────────────────────────────────────────
BASE_MODEL_PATH="${BASE_MODEL_PATH:-RLWRLD/RLDX-1-PT}"
CKPT_NAME="rldx1_ft_robocasa_5ep_overfit"
RUN_NAME="$CKPT_NAME"
# ───────────────────────────────────────────────────────

NUM_GPUS=1
BASE_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
DATA_DIR="${DATA_DIR:-$HOME/datasets/lerobot_pnp_stovetocounter_mg_5ep}"

CKPT_DIR="$BASE_DIR/ckpt/rldx1/finetuned/$CKPT_NAME"
MODALITY_CONFIG_PATH="$BASE_DIR/rldx/configs/data/robocasa_flat_config.py"
COLOR_JITTER_PARAMS="brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08"

cd "$BASE_DIR"
export MASTER_PORT=$(shuf -i 20000-30000 -n 1)
export PATH="/home/iit.local/sberti/miniconda3/envs/rldx/bin:$PATH"
export LD_LIBRARY_PATH="/home/iit.local/sberti/miniconda3/envs/rldx/lib:${LD_LIBRARY_PATH:-}"

echo "============================================"
echo "Overfitting RLDX-1 on 5 MimicGen episodes"
echo "Data:  $DATA_DIR"
echo "GPU:   GPU 1 (single A100)"
echo "Batch: 8 (global), grad-accum: 1"
echo "Steps: 5000, save every 500"
echo "============================================"

uv run torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
    rldx/experiment/launch_train.py \
        --n-cog-tokens 64 \
        --max-state-dim 158 \
        --max-action-dim 12 \
        --video-length 4 \
        --dataset-path "$DATA_DIR" \
        --dataloader-num-workers 4 \
        --embodiment-tag GENERAL_EMBODIMENT \
        --modality-config-path "$MODALITY_CONFIG_PATH" \
        --color-jitter-params $COLOR_JITTER_PARAMS \
        --base-model-path "$BASE_MODEL_PATH" \
        --output-dir "$CKPT_DIR" \
        --num-gpus $NUM_GPUS \
        --save-total-limit 10 \
        --save-steps 500 \
        --max-steps 5000 \
        --global-batch-size 8 \
        --gradient-accumulation-steps 1 \
        --use-wandb \
        --wandb-project "$WANDB_PROJECT" \
        --experiment-name "$RUN_NAME"
