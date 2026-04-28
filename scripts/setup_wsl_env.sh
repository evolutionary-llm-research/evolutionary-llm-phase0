#!/usr/bin/env bash
set -euo pipefail

# Bootstrap WSL2 training environment for Evolutionary LLM Research.
# Run inside Ubuntu (WSL2) from repository root.

ENV_NAME="evollm-wsl"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda not found. Install Miniconda/Anaconda in WSL2 first." >&2
  exit 1
fi

conda env create -f environment.wsl.yml || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_runtime', torch.version.cuda)"
python -c "import unsloth; print('unsloth_ok')"

echo "WSL2 environment '${ENV_NAME}' is ready."
