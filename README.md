# Evolutionary LLM Research

Research project studying evolutionary dynamics of LLM populations under information pressure.

## Quick start

1. Create and activate a Python 3.11 environment.
2. Install dependencies:
   pip install -r requirements.txt
3. Review current phase configuration:
   config/phase0_metrics_validation.yaml
4. Run Phase 0 metric validation:
   python -m src.analysis.phase0_metric_validation --config config/phase0_metrics_validation.yaml
5. Run tests:
   pytest -q

## Pre-0 environment baseline

Use this before Phase 0 to validate environment parameters and reproducibility.

### Dual environment policy

- Windows + conda (`evolllm`): Ollama inference, metrics, data pipeline, and analysis.
- WSL2 Ubuntu + conda (`evollm-wsl`): LoRA fine-tuning with Unsloth and GPU training workloads.

### Windows baseline setup

1. Create the conda environment from the pinned spec:
   conda env create -f environment.pre0.yml
2. Activate environment:
   conda activate evolllm
3. Run pre-0 smoke test:
   python scripts/pre0_smoke_test.py

Smoke-test statuses:
- PASS: all required modules and CUDA checks are healthy.
- PASS_WITH_WARNINGS: required stack is healthy, but optional module checks report issues.
- FAIL: required module imports or CUDA checks failed.

Pre-0 artifacts are written to:
- experiments/pre0_environment/pre0_smoke_test_report.json
- experiments/pre0_environment/conda-explicit-spec.txt
- experiments/pre0_environment/pip-freeze.txt

### WSL2 training setup

1. Open Ubuntu in WSL2.
2. Create the WSL conda environment from pinned spec:
   conda env create -f environment.wsl.yml
3. Activate environment:
   conda activate evollm-wsl
4. Validate CUDA in WSL2:
   python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
5. Validate Unsloth import in WSL2:
   python -c "import unsloth; print('ok')"

Optional bootstrap script (run inside WSL2 Ubuntu from repository root):
- bash scripts/setup_wsl_env.sh

See environment architecture decision details in `ENVIRONMENT_ARCHITECTURE.md`.

## Repository layout

- config/ : Experiment parameters (YAML)
- data/ : Raw and processed datasets and biome containers
- src/ : Source code (data, models, evolution, metrics, analysis)
- experiments/ : Raw results and logs per generation
- tests/ : Unit tests
- papers/ : Publication drafts

## Notes

- Keep experiments reproducible: fixed random seeds and full parameter logging.
- Do not pass content-type tags (food/predator/noise) to model inputs.
