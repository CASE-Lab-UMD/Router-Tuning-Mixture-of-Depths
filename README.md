# Router-Tuning: A Simple and Effective Approach for Enabling Dynamic-Depth in Transformers

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2410.13184)
![Conference](https://img.shields.io/badge/EMNLP-2025-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)

Official implementation of [**Router-Tuning**](https://arxiv.org/abs/2410.13184).

Authors: [Shwai He](https://shwai-he.github.io/), [Tao Ge](https://getao.github.io/), [Guoheng Sun](https://s1gh.alphaxiv.io/), [Bowei Tian](https://bowei.netlify.app/#about), [Xiaoyang Wang](https://xyang0.github.io/), [Dong Yu](https://sites.google.com/view/dongyu888/)

Router-Tuning enables dynamic-depth inference by fine-tuning only router-related parameters. Compared with standard MoD-style full tuning, it significantly reduces training cost while keeping model quality competitive.

<p align="center">
  <img src="figures/mod.svg" alt="Router-Tuning and MoD overview" width="68%">
</p>

## News
- Aug 2025: Router-Tuning accepted to **EMNLP 2025** main conference.
- Oct 2024: arXiv preprint and code release.

## Why This Repo
Traditional transformers execute a fixed number of layers for every token, which wastes computation on easy tokens.

[Mixture of Depths (MoD)](https://arxiv.org/abs/2404.02258) addresses this by dynamically skipping less important computations, but two practical issues remain:

1. Existing methods usually tune the whole model, causing high training cost.
2. Aggressive skipping can hurt quality if routing is not well calibrated.

Router-Tuning tackles both by focusing optimization on routing components and introducing routing strategies that better preserve performance-efficiency tradeoffs.

## Results at a Glance
| Setting | Speedup | Quality Drop |
| --- | --- | --- |
| Router-Tuning (reported best tradeoff) | Up to **21%** | Around **0.2%** |

> Numbers are from the Router-Tuning paper: https://arxiv.org/abs/2410.13184

### Main Results
<p align="center">
  <img src="figures/main_results.png" alt="Main benchmark results of Router-Tuning" width="92%">
</p>
Router-Tuning consistently improves the efficiency-quality tradeoff over full-parameter MoD tuning baselines.
The reported best setting reaches notable speedup while keeping quality degradation small.

### Expert Routing Analysis
<p align="center">
  <img src="figures/expert_rt.png" alt="Expert-level routing behavior analysis" width="82%">
</p>
Router specialization becomes clearer after tuning: the model learns more stable token-to-layer routing patterns.
This supports dynamic-depth execution with lower unnecessary computation.

### LoRA Compatibility
<p align="center">
  <img src="figures/lora_rt.png" alt="LoRA and Router-Tuning combined results" width="92%">
</p>
Router-Tuning is compatible with LoRA-based adaptation and can be composed for a better efficiency-performance balance.
In practice, this enables lightweight deployment recipes without full-model retraining.

## Core Methods
1. **Router-Only Fine-Tuning**
- Tune router-related parameters instead of full-model updates.
- Strongly reduces optimization cost for dynamic-depth adaptation.

2. **MoD Attention Routing**
- Uses attention-based routing granularity to improve compute and memory efficiency.
- Preserves output quality under dynamic-depth execution.

## Repository Layout
- `entrypoints/finetune/finetune_mod.py`: main training entrypoint.
- `scripts/finetune_mod.sh`: reproducible launcher with `accelerate` + DeepSpeed.
- `entrypoints/data/reformat_datasets.py`: convert raw datasets to unified `messages` format.
- `entrypoints/data/mix_datasets.py`: build mixed instruction-tuning data.
- `utils/pipeline/customized_trainer.py`: router-focused trainer logic.
- `configs/accelerate/`: distributed training launcher configs.
- `configs/deepspeed/`: DeepSpeed runtime configs.
- `ckpt/`: model config/tokenizer files for supported MoD variants.

## Installation
```bash
conda create -n router-tuning python=3.10 -y
conda activate router-tuning

git clone https://github.com/CASE-Lab-UMD/Router-Tuning-Mixture-of-Depths.git
cd Router-Tuning-Mixture-of-Depths

pip install -r requirements.txt
```

## Quick Start
### 1) Prepare Data
Put raw datasets under `data/raw/` using the expected subdirectory names:
- `vicuna_sharegpt`
- `evol_instruct`
- `slim_orca`
- `meta_math_qa`
- `evol_code_alpaca`
- `alpaca`

Then run:

```bash
python entrypoints/data/reformat_datasets.py \
  --raw_data_root ./data/raw \
  --save_path ./data/reformatted

python entrypoints/data/mix_datasets.py \
  --reformatted_dir ./data/reformatted \
  --save_path ./data/mixed
```

### 2) Run Router-Tuning
```bash
bash scripts/finetune_mod.sh
```

### 3) Minimal Single-Node Override Example
```bash
NUM_PROCESSES=4 PORT=29501 bash scripts/finetune_mod.sh
```

## Training Pipeline
```mermaid
flowchart LR
    A[Raw Data: data/raw] --> B[Reformat<br/>reformat_datasets.py]
    B --> C[Mix Data<br/>mix_datasets.py]
    C --> D[Launch Training<br/>scripts/finetune_mod.sh]
    D --> E[Model Output<br/>trained_models/...]
```

## Training Knobs
`finetune_mod.sh` is the recommended launcher. Commonly adjusted fields:

- `folder_name`: base checkpoint directory under `ckpt/`.
- `data_type`: one dataset under `data/reformatted/` or `mixed`.
- `max_train_samples`: training subset size for quick experiments.
- `mod_n`: MoD keep ratio control (legacy alias `mindskip_n` is still supported in Python entrypoint).
- `granularity`: routing granularity (`attn_sequence` or `mlp_sequence`).
- `router_only`: enable router-only training (default `True`).
- `learning_rate`, `weight_decay`, `num_epochs`.

Distributed launch overrides:
- `NUM_PROCESSES`: number of GPU processes.
- `PORT`: distributed master port.

### Knob Matrix
| Knob | Where | Typical Values | Effect |
| --- | --- | --- | --- |
| `folder_name` | `scripts/finetune_mod.sh` | `mistral-7b-mod`, `qwen-2.5-7b-mod`, `llama3-8b-instruct-mod` | Selects base checkpoint under `ckpt/` |
| `data_type` | `scripts/finetune_mod.sh` | `alpaca`, `mixed`, ... | Chooses training data source |
| `mod_n` | `scripts/finetune_mod.sh` / CLI | `8`, `16`, `32` | Controls dynamic-depth sparsity/keep behavior |
| `granularity` | `scripts/finetune_mod.sh` | `attn_sequence`, `mlp_sequence` | Chooses routing granularity |
| `max_train_samples` | `scripts/finetune_mod.sh` | `1000`, `5000`, `all` (by removing cap) | Controls quick debug vs full tuning |
| `NUM_PROCESSES` | shell env | `1`, `4`, `8` | Number of distributed workers |
| `PORT` | shell env | e.g., `29501` | Master communication port |

## Evaluation
Evaluation is compatible with [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
For strict reproduction used in earlier experiments, see [s1ghhh/lm-evaluation-harness](https://github.com/s1ghhh/lm-evaluation-harness).

## Repro Checklist
- Python 3.10 environment with `pip install -r requirements.txt`.
- Valid local model path under `ckpt/` (or customize `folder_name`).
- Reformatted/mixed data exists at `data/reformatted/*/data.jsonl` or `data/mixed/data.jsonl`.
- `accelerate` config selected in `scripts/finetune_mod.sh` matches your hardware.
- `NUM_PROCESSES` and GPU memory are consistent with `max_seq_length` and batch setup.

## Citation
```bibtex
@misc{he2024routertuningsimpleeffectiveapproach,
  title={Router-Tuning: A Simple and Effective Approach for Enabling Dynamic-Depth in Transformers},
  author={Shwai He and Tao Ge and Guoheng Sun and Bowei Tian and Xiaoyang Wang and Dong Yu},
  year={2024},
  eprint={2410.13184},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2410.13184}
}
```

## Contact
- Shwai He: `shwaihe@umd.edu`
