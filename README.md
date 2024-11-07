## Router-Tuning: A Simple and Effective Approach for Enabling Dynamic-Depth in Transformers

**[Shwai He](https://shwai-he.github.io/), Tao Ge, Guoheng Sun, [Bowei Tian](https://bowei.netlify.app/#about), Xiaoyang Wang, [Ang Li](https://www.ang-li.com/), Dong Yu**

## TL;DR
The open-source Mixture of Depths code and the official implementation of the paper "Router-Tuning: A Simple and Effective Approach for Enabling Dynamic Depth in Transformers."


## Introduction

Traditional transformer models allocate a fixed amount of computational resources to every input token, leading to inefficient and unnecessary computation. To address this inefficiency, [Mixture of Depths (MoD)](https://arxiv.org/abs/2404.02258) was introduced, dynamically adjusting computational depth by skipping less important layers. While promising, current MoD approaches face two significant challenges:

1. **High Training Costs**: Existing methods require training the entire model alongside routers, which determine which layers to skip, resulting in substantial computational overhead.
2. **Risk of Performance Degradation**: Bypassing important layers can lead to a drop in model performance.

To overcome these challenges, we introduce [**Router-Tuning**](https://arxiv.org/abs/2410.13184), a method that fine-tunes only the router on a small dataset, drastically reducing the training costs. Additionally, we propose **Mindskip** (Attention with Dynamic Depths), which preserves model performance while significantly enhancing computational and memory efficiency. 

Our approach delivers competitive results, achieving up to **21% speedup** with only a **0.2% performance drop**, demonstrating its effectiveness in balancing efficiency and performance.

![Diagram of Mindskip](mindskip.svg)


## News

- **Oct 2024**: Published preprint on [arXiv](https://arxiv.org/abs/2410.13184) along with the related codebase.

## Quick Start

#### Installation

```bash
conda create -n router-tuning python=3.10
conda activate router-tuning

git clone https://github.com/CASE-Lab-UMD/Router-Tuning

cd ./Router-Tuning
pip install -e .
```


## Train

```bash
sh /scripts/finetune_mndskip.sh
```

## Evaluation 

The evaluation code is based on [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). To fully reproduce our results, please use [this version](https://github.com/s1ghhh/lm-evaluation-harness). It samples few-shot based on the index of the samples, avoiding the issue of result variation with the number of processes during data parallel inference.

## Citation

```latex
@misc{he2024routertuningsimpleeffectiveapproach,
      title={Router-Tuning: A Simple and Effective Approach for Enabling Dynamic-Depth in Transformers}, 
      author={Shwai He and Tao Ge and Guoheng Sun and Bowei Tian and Xiaoyang Wang and Ang Li and Dong Yu},
      year={2024},
      eprint={2410.13184},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.13184}, 
}
```


## Contact Us

If you have any questions, please contact:

- Shwai He: shwaihe@umd.edu