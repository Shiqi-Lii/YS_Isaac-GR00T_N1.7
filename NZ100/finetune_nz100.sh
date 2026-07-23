#!/usr/bin/env bash

set -x -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

# Server-local Hugging Face cache. Override these before launching if needed.
export HF_HOME="${HF_HOME:-/mnt/16T/lisq5005_dir/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

# Offline mode: use only models/files already present in the cache paths above.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
MAX_STEPS="${MAX_STEPS:-30000}"
USE_WANDB="${USE_WANDB:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
SHARD_SIZE="${SHARD_SIZE:-1024}"
NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-100000}"
EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-0.1}"

CONDA_ROOT="/mnt/16T/App_dir/conda_dir/miniconda3"
GR00T_CONDA_ENV="${GR00T_CONDA_ENV:-${CONDA_ROOT}/envs/gr00t_n17_lsq}"

# Point this to the local N1.7 checkpoint path on your server.
BASE_MODEL_PATH="${BASE_MODEL_PATH:-nvidia/GR00T-N1.7-3B}"
DATASET_PATH="${DATASET_PATH:-data/data_open_close_package}"

# 描述ckt
RUN_DESC="${RUN_DESC:-open_close_package}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/16T/lisq5005_dir/YS_Isaac-GR00T_N1.7/checkpoints/nz100_${RUN_DESC}}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/finetune_nz100_${RUN_DESC}.log}"

mkdir -p "$LOG_DIR"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${GR00T_CONDA_ENV}"

echo "Using python: $(which python)"
python -V

WANDB_FLAG=()
if [ "$USE_WANDB" = "1" ]; then
    WANDB_FLAG+=(--use_wandb)
fi

LAUNCH_CMD=(
    gr00t/experiment/launch_finetune.py
    --base_model_path "$BASE_MODEL_PATH"
    --dataset_path "$DATASET_PATH"
    --embodiment_tag NZ100
    --num_gpus 1
    --output_dir "$OUTPUT_DIR"
    --save_steps "$SAVE_STEPS"
    --save_total_limit 10
    --max_steps "$MAX_STEPS"
    --warmup_ratio 0.05
    --weight_decay 1e-5
    --learning_rate 1e-4
    "${WANDB_FLAG[@]}"
    --global_batch_size "$GLOBAL_BATCH_SIZE"
    --shortest_image_edge 256
    --crop_fraction 0.95
    --color_jitter_params brightness 0.3 contrast 0.3 saturation 0.2 hue 0.03
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
    --shard_size "$SHARD_SIZE"
    --num_shards_per_epoch "$NUM_SHARDS_PER_EPOCH"
    --episode_sampling_rate "$EPISODE_SAMPLING_RATE"
)

nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTHONPATH="$REPO_ROOT" python "${LAUNCH_CMD[@]}" \
    > "$LOG_FILE" 2>&1 &

echo "Started NZ100 finetuning in background."
echo "  PID: $!"
echo "  Log: $LOG_FILE"
