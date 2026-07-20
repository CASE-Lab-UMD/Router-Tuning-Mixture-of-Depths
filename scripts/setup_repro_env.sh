#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_PATH="$(cd "${SCRIPT_DIR}/.." && pwd)"
UV_BIN="${UV_BIN:-${ROOT_PATH}/.tools/uv}"
ENV_PATH="${ENV_PATH:-${ROOT_PATH}/.venv-router-tuning}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"

mkdir -p "${ROOT_PATH}/.tools" "${ROOT_PATH}/.uv-cache" "${ROOT_PATH}/.pip-cache"

if [[ ! -x "${UV_BIN}" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh |
    env UV_INSTALL_DIR="${ROOT_PATH}/.tools" sh
fi

export UV_CACHE_DIR="${ROOT_PATH}/.uv-cache"
"${UV_BIN}" python install 3.10.20
"${UV_BIN}" venv --python 3.10.20 "${ENV_PATH}"
"${UV_BIN}" pip install \
  --python "${ENV_PATH}/bin/python" \
  -r "${ROOT_PATH}/requirements-repro.txt"

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc is required while installing flash-attn."
    echo "Load a CUDA toolkit or rerun with INSTALL_FLASH_ATTN=0."
    exit 1
  fi

  export PIP_CACHE_DIR="${ROOT_PATH}/.pip-cache"
  MAX_JOBS="${MAX_JOBS:-4}" \
    "${ENV_PATH}/bin/python" -m pip install \
      --no-build-isolation \
      -r "${ROOT_PATH}/requirements-flash-attn.txt"
fi

"${ENV_PATH}/bin/python" -c \
  'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda)'
