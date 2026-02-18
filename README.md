# Router-Tuning: A Simple and Effective Approach for Enabling Dynamic-Depth in Transformers

Official implementation of [Router-Tuning](https://arxiv.org/abs/2410.13184).

Authors: [Shwai He](https://shwai-he.github.io/), [Tao Ge](https://getao.github.io/), [Guoheng Sun](https://s1gh.alphaxiv.io/), [Bowei Tian](https://bowei.netlify.app/#about), [Xiaoyang Wang](https://xyang0.github.io/), [Ang Li](https://www.ang-li.com/), [Dong Yu](https://sites.google.com/view/dongyu888/)

Router-Tuning fine-tunes only router modules to enable dynamic depth with low training cost. This repo also includes MoD variants for multiple base models.

![Diagram of MoD](mindskip.svg)

## News
- Aug 2025: Accepted to EMNLP 2025 main conference.
- Oct 2024: arXiv preprint and code release.

## Installation

```bash
conda create -n router-tuning python=3.10 -y
conda activate router-tuning

git clone https://github.com/CASE-Lab-UMD/Router-Tuning.git
cd Router-Tuning
pip install -r requirements.txt
```

## Repository Layout

- `entrypoints/finetune/finetune_mindskip.py`: main training entry (MoD router tuning).
- `scripts/finetune_mindskip.sh`: training launcher example.
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
bash scripts/finetune_mindskip.sh
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
