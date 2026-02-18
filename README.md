# Router-Tuning: A Simple and Effective Approach for Enabling Dynamic-Depth in Transformers

Official implementation of [Router-Tuning](https://arxiv.org/abs/2410.13184).

Authors: [Shwai He](https://shwai-he.github.io/), [Tao Ge](https://getao.github.io/), [Guoheng Sun](https://s1gh.alphaxiv.io/), [Bowei Tian](https://bowei.netlify.app/#about), [Xiaoyang Wang](https://xyang0.github.io/), [Ang Li](https://www.ang-li.com/), [Dong Yu](https://sites.google.com/view/dongyu888/)

Router-Tuning fine-tunes only router modules to enable dynamic depth with low training cost. This repo also includes MoD variants for multiple base models.

## TL;DR
Official implementation of Router-Tuning for dynamic-depth Transformers.
Compared with standard MoD-style training, Router-Tuning focuses on tuning router-related parameters for lower training cost while preserving model quality.

## Introduction

Traditional transformer models allocate a fixed amount of computational resources to every input token, leading to inefficient and unnecessary computation.
To address this inefficiency, [**Mixture of Depths (MoD)**](https://arxiv.org/abs/2404.02258) was introduced, dynamically adjusting computational depth by skipping less important layers.
While promising, current MoD approaches face two significant challenges:

1. **High Training Costs**: Existing methods require training the entire model alongside routers, which determine which layers to skip, resulting in substantial computational overhead.
2. **Risk of Performance Degradation**: Bypassing important layers can lead to a drop in model performance.

To overcome these challenges, we introduce [**Router-Tuning**](https://arxiv.org/abs/2410.13184), a method that fine-tunes only the router on a small dataset, drastically reducing training costs.
Additionally, we propose **MoD attention routing**, which preserves model performance while significantly enhancing computational and memory efficiency.

Our approach delivers competitive results, achieving up to **21% speedup** with only a **0.2% performance drop**, demonstrating effectiveness in balancing efficiency and performance.

![Diagram of MoD](mod.svg)

## News
- Aug 2025: Accepted to EMNLP 2025 main conference.
- Oct 2024: arXiv preprint and code release.

## Installation

```bash
conda create -n router-tuning python=3.10 -y
conda activate router-tuning

git clone https://github.com/CASE-Lab-UMD/Router-Tuning-Mixture-of-Depths.git
cd Router-Tuning-Mixture-of-Depths
pip install -r requirements.txt
```

## Repository Layout

- `entrypoints/finetune/finetune_mod.py`: main training entry (MoD router tuning).
- `scripts/finetune_mod.sh`: training launcher example.
- `entrypoints/data/reformat_datasets.py`: convert raw datasets to message format.
- `entrypoints/data/mix_datasets.py`: build mixed instruction-tuning set.
- `utils/pipeline/customized_trainer.py`: custom trainer for router-focused optimization.
- `ckpt/`: model config/tokenizer files for MoD variants.

## Data Preparation

1. Put raw datasets under a root folder (default: `data/raw`).
2. Reformat datasets:

```bash
python entrypoints/data/reformat_datasets.py \
  --raw_data_root ./data/raw \
  --save_path ./data/reformatted
```

3. Mix datasets:

```bash
python entrypoints/data/mix_datasets.py \
  --reformatted_dir ./data/reformatted \
  --save_path ./data/mixed
```

## Training

Run the default launcher:

```bash
bash scripts/finetune_mod.sh
```

Useful environment overrides:
- `NUM_PROCESSES`: number of GPUs used by `accelerate`.
- `PORT`: master port for distributed training.

The script defaults to router-only training and uses paths relative to the repository root.
Use `--mod_n` in training commands (`--mindskip_n` is still accepted as a backward-compatible alias).

## Evaluation

Evaluation follows [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
For strict reproduction, use [s1ghhh/lm-evaluation-harness](https://github.com/s1ghhh/lm-evaluation-harness).

## Citation

```bibtex
@misc{he2024routertuningsimpleeffectiveapproach,
  title={Router-Tuning: A Simple and Effective Approach for Enabling Dynamic-Depth in Transformers},
  author={Shwai He and Tao Ge and Guoheng Sun and Bowei Tian and Xiaoyang Wang and Ang Li and Dong Yu},
  year={2024},
  eprint={2410.13184},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2410.13184}
}
```

## Contact

- Shwai He: `shwaihe@umd.edu`
