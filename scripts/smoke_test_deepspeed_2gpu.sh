#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_PATH="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_PATH="${ENV_PATH:-${ROOT_PATH}/.venv-router-tuning}"
PORT="${PORT:-29621}"

if [[ -z "${CUDA_HOME:-}" ]] && type module >/dev/null 2>&1; then
  module load cuda/13.1.1
fi
if [[ -z "${CUDA_HOME:-}" ]]; then
  echo "ERROR: DeepSpeed requires CUDA_HOME. Load a CUDA toolkit first."
  exit 1
fi

cd "${ROOT_PATH}"
export HF_HOME="${HF_HOME:-${ROOT_PATH}/.hf-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR:-/tmp}/router-tuning-triton-${USER}-${SLURM_JOB_ID:-local}}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${TRITON_CACHE_DIR}"

"${ENV_PATH}/bin/accelerate" launch \
  --config_file configs/accelerate/deepspeed_llama_router_tuning.yaml \
  --num_processes 2 \
  --num_machines 1 \
  --main_process_port "${PORT}" \
  entrypoints/finetune/finetune_router_tuning.py \
  --model_name_or_path Qwen/Qwen2.5-0.5B \
  --tokenizer_name Qwen/Qwen2.5-0.5B \
  --use_fast_tokenizer False \
  --train_file data/reformatted/alpaca/data.jsonl \
  --max_seq_length 128 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --weight_decay 0 \
  --max_steps 1 \
  --warmup_ratio 0 \
  --evaluation_strategy no \
  --logging_steps 1 \
  --save_strategy no \
  --preprocessing_num_workers 2 \
  --output_dir smoke_results/qwen2.5-0.5b-flash-deepspeed-2gpu \
  --router_only True \
  --bf16 \
  --tf32 True \
  --report_to none \
  --granularity attn_sequence \
  --router_layers 2 \
  --gradient_scale 0 \
  --max_train_samples 4 \
  --use_flash_attn \
  --overwrite_output_dir \
  --do_train
