<h1 align="center">[EMNLP 2025] Router-Tuning: A Simple and Effective Approach for Enabling Dynamic-Depth in Transformers</h1>


<p align="center">
  <a href="https://arxiv.org/abs/2410.13184"><img src="https://img.shields.io/badge/arXiv-2410.13184-b31b1b.svg" alt="arXiv"></a>
  <a href="https://aclanthology.org/2025.emnlp-main.99"><img src="https://img.shields.io/badge/EMNLP-2025-blue" alt="EMNLP 2025"></a>
  <img src="https://img.shields.io/badge/Python-3.10+-green" alt="Python 3.10+">
</p>

<p align="center">
  <a href="https://shwai-he.github.io/">Shwai He</a>, <a href="https://getao.github.io/">Tao Ge</a>, <a href="https://s1gh.alphaxiv.io/">Guoheng Sun</a>, <a href="https://bowei.netlify.app/#about">Bowei Tian</a>, <a href="https://xyang0.github.io/">Xiaoyang Wang</a>, <a href="https://sites.google.com/view/dongyu888/">Dong Yu</a>
</p>

<p align="center">
  <a href="#-introduction">📖 Introduction</a> •
  <a href="#-news">📰 News</a> •
  <a href="#-why-this-repo">✨ Why</a> •
  <a href="#-results">📈 Results</a> •
  <a href="#-quick-start">🚀 Quick Start</a> •
  <a href="#-citation">📄 Citation</a>
</p>


## 📖 Introduction

This is the official implementation of the paper [**Router-Tuning: A Simple and Effective Approach for Enabling Dynamic-Depth in Transformers**](https://arxiv.org/abs/2410.13184), accepted at **EMNLP 2025**. We provide a practical framework for efficient dynamic-depth training and inference in Transformers.

Router-Tuning enables dynamic-depth inference by fine-tuning only router-related parameters. Compared with standard full-model dynamic-depth tuning, it significantly reduces training cost while keeping model quality competitive.

<p align="center">
  <img src="figures/router_tuning.svg" alt="Router-Tuning overview" width="68%">
</p>

## 📰 News
- Aug 2025: Router-Tuning accepted to **EMNLP 2025** main conference.
- Oct 2024: arXiv preprint and code release.

## ✨ Why This Repo
Traditional transformers execute a fixed number of layers for every token, which wastes computation on easy tokens.

[Mixture of Depths (MoD)](https://arxiv.org/abs/2404.02258) addresses this by dynamically skipping less important computations, but two practical issues remain:

1. Existing methods usually tune the whole model, causing high training cost.
2. Aggressive skipping can hurt quality if routing is not well calibrated.

Router-Tuning tackles both by focusing optimization on routing components and introducing routing strategies that better preserve performance-efficiency tradeoffs.

## 📈 Results

### 🏁 Main Results
<p align="center">
  <img src="figures/main_results.png" alt="Main benchmark results of Router-Tuning" width="92%">
</p>
Router-Tuning consistently improves the efficiency-quality tradeoff over full-parameter dynamic-depth tuning baselines.
The reported best setting reaches notable speedup while keeping quality degradation small.

### 🔬 Expert Routing Analysis
<p align="center">
  <img src="figures/expert_rt.png" alt="Expert-level routing behavior analysis" width="82%">
</p>
Router specialization becomes clearer after tuning: the model learns more stable token-to-layer routing patterns.
This supports dynamic-depth execution with lower unnecessary computation.

### 🔗 LoRA Compatibility
<p align="center">
  <img src="figures/lora_rt.png" alt="LoRA and Router-Tuning combined results" width="92%">
</p>
Router-Tuning is compatible with LoRA-based adaptation and can be composed for a better efficiency-performance balance.
In practice, this enables lightweight deployment recipes without full-model retraining.

## 🔍 Core Methods
1. **Router-Only Fine-Tuning**
- Tune router-related parameters instead of full-model updates.
- Strongly reduces optimization cost for dynamic-depth adaptation.

2. **Attention-Based Dynamic Depth**
- Uses attention-based routing granularity to improve compute and memory efficiency.
- Preserves output quality under dynamic-depth execution.

## 📦 Repository Layout
- `entrypoints/finetune/finetune_router_tuning.py`: main training entrypoint.
- `scripts/finetune_router_tuning.sh`: reproducible launcher with `accelerate` + DeepSpeed.
- `entrypoints/data/reformat_datasets.py`: convert raw datasets to unified `messages` format.
- `entrypoints/data/mix_datasets.py`: build mixed instruction-tuning data.
- `utils/pipeline/customized_trainer.py`: router-focused trainer logic.
- `configs/accelerate/`: distributed training launcher configs.
- `configs/deepspeed/`: DeepSpeed runtime configs.
- `ckpt/`: model config/tokenizer files for supported Router-Tuning checkpoints.

## ⚙️ Installation
```bash
conda create -n router-tuning python=3.10 -y
conda activate router-tuning

git clone https://github.com/CASE-Lab-UMD/Router-Tuning-Mixture-of-Depths.git
cd Router-Tuning-Mixture-of-Depths

pip install -r requirements.txt
```

## 🚀 Quick Start
### 1) 🧹 Prepare Data
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

### 2) 🏃 Run Router-Tuning
```bash
bash scripts/finetune_router_tuning.sh
```

### 3) 🖥️ Minimal Single-Node Override Example
```bash
NUM_PROCESSES=4 PORT=29501 bash scripts/finetune_router_tuning.sh
```

## 🎛️ Training Knobs
`finetune_router_tuning.sh` is the recommended launcher. Commonly adjusted fields:

- `folder_name`: base checkpoint directory under `ckpt/`.
- `data_type`: one dataset under `data/reformatted/` or `mixed`.
- `max_train_samples`: training subset size for quick experiments.
- `router_layers`: number of deep layers enabled for router tuning in the current implementation.
- `target_capacity`: optional target activation ratio for router regularization.
- `granularity`: routing granularity (`attn_sequence` or `mlp_sequence`).
- `router_only`: enable router-only training (default `True`).
- `learning_rate`, `weight_decay`, `num_epochs`.
- Legacy compatibility: `--mod_n` and `--mod_capacity` remain accepted, but `--router_layers` and `--target_capacity` are the preferred names.

Distributed launch overrides:
- `NUM_PROCESSES`: number of GPU processes.
- `PORT`: distributed master port.

### 🧭 Knob Matrix
| Knob | Where | Typical Values | Effect |
| --- | --- | --- | --- |
| `folder_name` | `scripts/finetune_router_tuning.sh` | `mistral-7b-mod`, `qwen-2.5-7b-mod`, `llama3-8b-instruct-mod` | Selects base checkpoint under `ckpt/` |
| `data_type` | `scripts/finetune_router_tuning.sh` | `alpaca`, `mixed`, ... | Chooses training data source |
| `router_layers` | `scripts/finetune_router_tuning.sh` / CLI | `8`, `16`, `32` | Controls how many deeper layers use router tuning |
| `target_capacity` | `scripts/finetune_router_tuning.sh` / CLI | `0.5`, `0.75`, unset | Sets the router activation target used by regularization |
| `granularity` | `scripts/finetune_router_tuning.sh` | `attn_sequence`, `mlp_sequence` | Chooses routing granularity |
| `max_train_samples` | `scripts/finetune_router_tuning.sh` | `1000`, `5000`, `all` (by removing cap) | Controls quick debug vs full tuning |
| `NUM_PROCESSES` | shell env | `1`, `4`, `8` | Number of distributed workers |
| `PORT` | shell env | e.g., `29501` | Master communication port |

## 🧪 Evaluation
Evaluation is compatible with [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
For strict reproduction used in earlier experiments, see [s1ghhh/lm-evaluation-harness](https://github.com/s1ghhh/lm-evaluation-harness).

## ✅ Repro Checklist
- Python 3.10 environment with `pip install -r requirements.txt`.
- Valid local model path under `ckpt/` (or customize `folder_name`).
- Reformatted/mixed data exists at `data/reformatted/*/data.jsonl` or `data/mixed/data.jsonl`.
- `accelerate` config selected in `scripts/finetune_router_tuning.sh` matches your hardware.
- `NUM_PROCESSES` and GPU memory are consistent with `max_seq_length` and batch setup.

## 📄 Citation
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

## 📬 Contact
- Shwai He: `shwaihe@umd.edu`
