#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_PATH="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_PATH="${ENV_PATH:-${ROOT_PATH}/.venv-router-tuning}"

cd "${ROOT_PATH}"
export HF_HOME="${HF_HOME:-${ROOT_PATH}/.hf-cache}"
export TOKENIZERS_PARALLELISM=false

"${ENV_PATH}/bin/accelerate" launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision bf16 \
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
  --output_dir smoke_results/qwen2.5-0.5b-flash \
  --router_only True \
  --bf16 \
  --tf32 True \
  --report_to none \
  --granularity attn_sequence \
  --router_layers 2 \
  --gradient_scale 0 \
  --max_train_samples 2 \
  --use_flash_attn \
  --overwrite_output_dir \
  --do_train
