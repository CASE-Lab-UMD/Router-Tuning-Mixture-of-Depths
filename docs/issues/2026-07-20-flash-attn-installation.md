# FlashAttention installation failure reported by Ashley Yuan

## Source

Email from Ashley Yuan, a PhD student at Tsinghua University, reporting a
reproduction failure:

```text
ERROR: Failed to build 'flash-attn' when getting requirements to build wheel
```

The email asks for:

1. The PyTorch, CUDA, and FlashAttention versions used for the paper.
2. A recommended FlashAttention version.
3. A workaround or alternative attention implementation.

## Diagnosis

The repository previously placed `torch==2.1.1` and `flash-attn==2.6.3` in the
same `requirements.txt` and instructed users to install that file in one pip
command. FlashAttention's build setup imports PyTorch. With pip build isolation,
PyTorch may not be visible in the temporary build environment, producing the
reported failure before wheel compilation starts.

The launcher also enabled `--use_flash_attn` unconditionally, so a successful
base installation still could not run without the optional package.

## Environment information recoverable from the repository

- Python: 3.10
- PyTorch: 2.1.1
- Transformers: 4.40.1
- FlashAttention: 2.6.3
- CUDA toolkit/driver minor version: not recorded

The exact historical CUDA version cannot be reconstructed from the checked-in
files. It should not be guessed in a reproduction response.

## Resolution

- Removed FlashAttention from the base requirements.
- Added `requirements-flash-attn.txt` with the historical `2.6.3` pin.
- Documented installation after PyTorch with `--no-build-isolation`.
- Made FlashAttention opt-in through `USE_FLASH_ATTN=True`.
- Added a runtime availability check and safe fallback to the model's default
  Transformers attention implementation.

## Recommended reproduction commands

```bash
conda create -n router-tuning python=3.10 -y
conda activate router-tuning
pip install -r requirements.txt

# Optional acceleration, after torch imports successfully:
pip install --no-build-isolation -r requirements-flash-attn.txt
USE_FLASH_ATTN=True bash scripts/finetune_router_tuning.sh
```

Without FlashAttention:

```bash
bash scripts/finetune_router_tuning.sh
```

## Local verification limits

The login environment had Python 3.12.13, no installed PyTorch, no visible
NVIDIA GPU, and no CUDA compiler. Validation was therefore moved to a Slurm
worker with one NVIDIA L40S GPU.

## Validated environment

- Date: 2026-07-20
- GPU: NVIDIA L40S, 46 GB
- NVIDIA driver: 610.43.02
- Python: 3.10.20
- PyTorch: 2.1.1+cu121
- PyTorch CUDA runtime: 12.1
- Transformers: 4.40.1
- FlashAttention: 2.6.3
- NumPy: 1.26.4
- pip: 24.2
- setuptools: 69.5.1
- CUDA toolkit visible during FlashAttention installation: 13.1.1

FlashAttention installed successfully with:

```bash
pip install --no-build-isolation flash-attn==2.6.3
```

The installation required three conditions that were missing or implicit in
the original instructions:

1. PyTorch must already be installed.
2. `setuptools<70` is required by the PyTorch 2.1.1 extension helper.
3. `CUDA_HOME` and `nvcc` must be available while FlashAttention prepares its
   package metadata.

## Experiment result

The following checks passed on the L40S:

- FlashAttention kernel forward and backward in BF16.
- Transformers `LlamaForCausalLM` with `flash_attention_2`, including backward.
- Repository training entrypoint with `Qwen/Qwen2.5-0.5B`.
- Router-Tuning runtime patch on two attention layers.
- Two Alpaca samples, sequence length 128, one optimizer step.

Observed one-step training result:

```text
loss: 0.4950186312198639
train_runtime: 0.7567 seconds
train_steps_per_second: 1.322
```

Two additional repository bugs were found and corrected during the experiment:

- The training entrypoint could not import `utils` when launched through
  Accelerate because the repository root was missing from `sys.path`.
- Trainer removed `input_ids` and `attention_mask` after the runtime model
  wrapper changed the visible forward signature. Preprocessing now removes
  metadata columns explicitly and preserves encoded model inputs.

Debug sample selection now also happens before tokenization, avoiding
preprocessing all 52,002 Alpaca examples for a two-sample smoke test.

## Two-GPU DeepSpeed validation

A second experiment validated the repository's checked-in DeepSpeed Accelerate
configuration:

- Slurm job: 100075
- Node: ihccs402
- GPUs: 2 x NVIDIA L40S
- Distributed backend: NCCL
- World size: 2
- Precision: BF16
- Attention: FlashAttention 2
- DeepSpeed: 0.15.1
- ZeRO optimizer: Stage 1
- Per-device batch size: 1
- Global train batch size: 2
- Samples: 4
- Optimizer steps: 1

Observed result:

```text
loss: 0.8497879505157471
train_runtime: 0.6480 seconds
train_samples_per_second: 3.087
train_steps_per_second: 1.543
Slurm state: COMPLETED
exit code: 0:0
```

The saved checkpoint contains only the two expected Router-Tuning parameters:

```text
model.layers.21.router.weight  shape=(1, 896)  nonzero=896
model.layers.22.router.weight  shape=(1, 896)  nonzero=895
```

This confirms that both ranks participated in training, the DeepSpeed ZeRO
optimizer completed an update, and consolidated router-only checkpoint saving
worked.

DeepSpeed initially failed when `CUDA_HOME` was absent. Loading the cluster CUDA
module fixed initialization:

```bash
module load cuda/13.1.1
```

The tested medium QoS limits also required at most 8 requested CPUs and 64 GB
host memory for this job.
