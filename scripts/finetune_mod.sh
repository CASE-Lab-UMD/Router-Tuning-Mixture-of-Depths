#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_PATH="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_PATH}"

VERSION=3
SIZE=8
# FOLDER_NAME="llama${VERSION}-${SIZE}b-instruct-mod"
# FOLDER_NAME="qwen-2.5-7b-mod"
FOLDER_NAME="mistral-7b-mod"
# FOLDER_NAME="llama${VERSION}-${SIZE}b-mod"
MODEL_NAME_OR_PATH="${ROOT_PATH}/ckpt/${FOLDER_NAME}"

MOD_N=16
GRANULARITY="attn_sequence"
# GRANULARITY="mlp_sequence"

GRADIENT_SCALE=0.0
LEARNING_RATE=1e-5
WEIGHT_DECAY=0.
NUM_EPOCHS=1
TRUST_REMOTE_CODE=True
ROUTER_ONLY=True

CONFIG_FILE="${ROOT_PATH}/configs/accelerate/deepspeed_llama_mod.yaml"
# Set one dataset name under data/reformatted/, or use "mixed" for data/mixed/data.jsonl.
DATA_TYPE="alpaca"
MAX_TRAIN_SAMPLES=1000
MAX_SEQ_LENGTH=2048

if [[ "${DATA_TYPE}" == "mixed" ]]; then
  DATA_FILE="${ROOT_PATH}/data/${DATA_TYPE}/data.jsonl"
else
  DATA_FILE="${ROOT_PATH}/data/reformatted/${DATA_TYPE}/data.jsonl"
fi

OUTPUT_DIR="${ROOT_PATH}/trained_models/${FOLDER_NAME}/${DATA_TYPE}/${MAX_TRAIN_SAMPLES}/${GRANULARITY}_epoch${NUM_EPOCHS}_lr${LEARNING_RATE}_mod_n${MOD_N}_gradient_scale${GRADIENT_SCALE}_wd${WEIGHT_DECAY}"
mkdir -p "${OUTPUT_DIR}"

NUM_NODES=1
if command -v nvidia-smi >/dev/null 2>&1; then
  DETECTED_GPUS="$(nvidia-smi -L | wc -l | tr -d ' ')"
else
  DETECTED_GPUS=1
fi
NUM_PROCESSES="${NUM_PROCESSES:-${DETECTED_GPUS}}"
PORT="${PORT:-29501}"

BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=64
GRADIENT_ACC_STEPS=$((TOTAL_BATCH_SIZE / NUM_PROCESSES / BATCH_SIZE_PER_GPU))
if [[ "${GRADIENT_ACC_STEPS}" -lt 1 ]]; then
  echo "WARN: computed GRADIENT_ACC_STEPS=${GRADIENT_ACC_STEPS}, forcing to 1 (TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE}, NUM_PROCESSES=${NUM_PROCESSES}, BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU})"
  GRADIENT_ACC_STEPS=1
fi

echo "MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

accelerate launch \
  --config_file "${CONFIG_FILE}" \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines "${NUM_NODES}" \
  --main_process_port "${PORT}" \
  "${ROOT_PATH}/entrypoints/finetune/finetune_mod.py" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --tokenizer_name "${MODEL_NAME_OR_PATH}" \
  --use_fast_tokenizer False \
  --train_file "${DATA_FILE}" \
  --max_seq_length "${MAX_SEQ_LENGTH}" \
  --per_device_train_batch_size "${BATCH_SIZE_PER_GPU}" \
  --gradient_accumulation_steps "${GRADIENT_ACC_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --lr_scheduler_type "cosine" \
  --warmup_ratio 0.03 \
  --weight_decay "${WEIGHT_DECAY}" \
  --evaluation_strategy "no" \
  --logging_steps 10 \
  --save_strategy "steps" \
  --save_steps 100000 \
  --save_total_limit 1 \
  --num_train_epochs "${NUM_EPOCHS}" \
  --preprocessing_num_workers 64 \
  --output_dir "${OUTPUT_DIR}" \
  --router_only "${ROUTER_ONLY}" \
  --bf16 \
  --tf32 True \
  --report_to "tensorboard" \
  --granularity "${GRANULARITY}" \
  --mod_n "${MOD_N}" \
  --gradient_scale "${GRADIENT_SCALE}" \
  --trust_remote_code "${TRUST_REMOTE_CODE}" \
  --overwrite_output_dir \
  --max_train_samples "${MAX_TRAIN_SAMPLES}" \
  --use_flash_attn \
  --do_train
