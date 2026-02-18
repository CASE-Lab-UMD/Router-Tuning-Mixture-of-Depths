#!/usr/bin/env bash

##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_path="$(cd "${SCRIPT_DIR}/.." && pwd)"
echo $root_path
cd ${root_path}


version=3
size=8
# folder_name="llama${version}-${size}b-instruct-mod"
# folder_name="qwen-2.5-7b-mod"
folder_name="mistral-7b-mod"
# folder_name="llama${version}-${size}b-mod"
model_name_or_path="${root_path}/ckpt/${folder_name}"


##############################################################################
mod_n=16
granularity=attn_sequence
# granularity=mlp_sequence

gradient_scale=0.0
learning_rate=1e-5
weight_decay=0.
num_epochs=1
trust_remote_code=True
router_only=True
##############################################################################
config_file="${root_path}/configs/accelerate/deepspeed_llama_mod.yaml"
# Set one dataset name under data/reformatted/, or use "mixed" for data/mixed/data.jsonl.
data_type=alpaca
max_train_samples=1000
max_seq_length=2048

if [ "$data_type" = "mixed" ]; then
  data_file="$root_path/data/${data_type}/data.jsonl"
else
  data_file="$root_path/data/reformatted/${data_type}/data.jsonl"
fi


output_dir=$root_path/trained_models/$folder_name/${data_type}/${max_train_samples}/${granularity}_epoch${num_epochs}_lr${learning_rate}_mod_n${mod_n}_gradient_scale${gradient_scale}_wd${weight_decay}
mkdir -p $output_dir

num_nodes=1
if command -v nvidia-smi >/dev/null 2>&1; then
  detected_gpus="$(nvidia-smi -L | wc -l | tr -d ' ')"
else
  detected_gpus=1
fi
num_processes="${NUM_PROCESSES:-$detected_gpus}"
port="${PORT:-29501}"
NUM_GPUS=${num_processes}

BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=64
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE / $NUM_GPUS / $BATCH_SIZE_PER_GPU))

echo ${model_name_or_path}
echo $output_dir

accelerate launch \
  --config_file ${config_file} \
  --num_processes ${num_processes} \
  --num_machines ${num_nodes} \
  --main_process_port ${port} \
  ${root_path}/entrypoints/finetune/finetune_mod.py \
  --model_name_or_path ${model_name_or_path} \
  --tokenizer_name ${model_name_or_path} \
  --use_fast_tokenizer False \
  --train_file ${data_file} \
  --max_seq_length $max_seq_length \
  --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
  --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
  --learning_rate $learning_rate \
  --lr_scheduler_type "cosine" \
  --warmup_ratio 0.03 \
  --weight_decay $weight_decay \
  --evaluation_strategy "no" \
  --logging_steps 10 \
  --save_strategy "steps" \
  --save_steps 100000 \
  --save_total_limit 1 \
  --num_train_epochs $num_epochs \
  --preprocessing_num_workers 64 \
  --output_dir ${output_dir} \
  --router_only ${router_only} \
  --bf16 \
  --tf32 True \
  --report_to "tensorboard" \
  --granularity ${granularity} \
  --mod_n ${mod_n} \
  --gradient_scale ${gradient_scale} \
  --trust_remote_code ${trust_remote_code} \
  --overwrite_output_dir \
  --max_train_samples $max_train_samples \
  --use_flash_attn \
  --do_train


##############################################################################
