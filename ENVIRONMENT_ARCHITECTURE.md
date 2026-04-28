# Environment Architecture Decision

## Status
Accepted in pre-0.

## Decision
Use a dual-environment architecture:

- Windows + conda (`evolllm`) for inference through Ollama, metric computation, data pipeline, and result analysis.
- WSL2 Ubuntu + conda (`evollm-wsl`) for LoRA fine-tuning with Unsloth and training workloads that directly stress the GPU stack.

## Rationale
- Unsloth is developed and tested primarily for Linux environments.
- Windows import/runtime behavior for Unsloth can be unstable, while CUDA training stack is consistently supported on WSL2.
- RTX 4090 is fully available from WSL2 through NVIDIA CUDA integration.
- VS Code Remote - WSL preserves the same development workflow and Copilot experience.

## Scope by project phase
- Pre-0 and Phase 0: run primarily on Windows environment unless GPU-training tests are explicitly required.
- Phase 1 and Phase 2 (LoRA fine-tuning starts): training jobs should run on WSL2 environment.
- Phase 3 and Phase 4: keep training/evolution jobs on WSL2; analysis can remain on Windows.

## Operating rules
- Do not mix `pip` and `conda` installations for `torch`, `torchvision`, and `torchaudio` in one environment lifecycle.
- Keep reproducibility artifacts for both environments (`conda --explicit`, `pip freeze`).
- Treat environment changes as controlled updates and re-run smoke tests after each change.

## Setup commands (WSL2)
```bash
# Ubuntu WSL2
conda create -n evollm-wsl python=3.11
conda activate evollm-wsl
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install unsloth accelerate peft trl datasets sentence-transformers
```

## Verification checklist
- `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"` returns `True` and CUDA 12.x.
- `python -c "import unsloth; print('ok')"` completes without crash in WSL2.
- Training smoke run (single mini-batch) executes on GPU.
