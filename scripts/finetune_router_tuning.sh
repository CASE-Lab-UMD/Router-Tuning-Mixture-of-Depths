#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_PATH="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_PATH}"

DEFAULT_MODEL_ID="Qwen/Qwen2.5-7B"
# Examples:
# DEFAULT_MODEL_ID="mistralai/Mistral-7B-v0.1"
# DEFAULT_MODEL_ID="meta-llama/Meta-Llama-3-8B"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${DEFAULT_MODEL_ID}}"
TOKENIZER_NAME="${TOKENIZER_NAME:-${MODEL_NAME_OR_PATH}}"
RUN_NAME="${RUN_NAME:-$(basename "${MODEL_NAME_OR_PATH}")}"

ROUTER_LAYERS=16
TARGET_CAPACITY=""
GRANULARITY="attn_sequence"
# Example alternatives: attn_token / mlp_sequence / mlp_token / block_sequence / block_token

GRADIENT_SCALE=0.0
LEARNING_RATE=1e-5
WEIGHT_DECAY=0.
NUM_EPOCHS=1
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-False}"
ROUTER_ONLY=True

CONFIG_FILE="${ROOT_PATH}/configs/accelerate/deepspeed_llama_router_tuning.yaml"
# Set one dataset name under data/reformatted/, or use "mixed" for data/mixed/data.jsonl.
DATA_TYPE="alpaca"
MAX_TRAIN_SAMPLES=1000
MAX_SEQ_LENGTH=2048

if [[ "${DATA_TYPE}" == "mixed" ]]; then
  DATA_FILE="${ROOT_PATH}/data/${DATA_TYPE}/data.jsonl"
else
  DATA_FILE="${ROOT_PATH}/data/reformatted/${DATA_TYPE}/data.jsonl"
fi

OUTPUT_DIR="${ROOT_PATH}/trained_models/${RUN_NAME}/${DATA_TYPE}/${MAX_TRAIN_SAMPLES}/${GRANULARITY}_epoch${NUM_EPOCHS}_router_layers${ROUTER_LAYERS}_lambda${GRADIENT_SCALE}_lr${LEARNING_RATE}_wd${WEIGHT_DECAY}"
if [[ -n "${TARGET_CAPACITY}" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR}_target${TARGET_CAPACITY}"
fi
mkdir -p "${OUTPUT_DIR}"

EXTRA_ROUTER_ARGS=()
if [[ -n "${TARGET_CAPACITY}" ]]; then
  EXTRA_ROUTER_ARGS+=(--target_capacity "${TARGET_CAPACITY}")
fi

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
  "${ROOT_PATH}/entrypoints/finetune/finetune_router_tuning.py" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --tokenizer_name "${TOKENIZER_NAME}" \
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
  --router_layers "${ROUTER_LAYERS}" \
  --gradient_scale "${GRADIENT_SCALE}" \
  --trust_remote_code "${TRUST_REMOTE_CODE}" \
  --overwrite_output_dir \
  --max_train_samples "${MAX_TRAIN_SAMPLES}" \
  --use_flash_attn \
  "${EXTRA_ROUTER_ARGS[@]}" \
  --do_train
