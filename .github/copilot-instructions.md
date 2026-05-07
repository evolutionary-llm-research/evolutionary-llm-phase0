# Evolutionary LLM Research - Copilot Instructions

## Project overview
Research project studying evolutionary dynamics of LLM populations under 
information pressure. Target: 3-4 peer-reviewed publications (PLOS ONE, 
Artificial Life, NeurIPS/GECCO).

## Architecture
- Base model: qwen3:8b-base (phases 0-1), qwen3:32b (phases 2-4)
- Inference backend: Ollama (local)
- LoRA fine-tuning: Unsloth
- Each agent = base model + individual LoRA adapter
- Three archetypes: Id (expansion), Ego (mediator), Superego (inhibition)

## Information environment (biomes)
Three biomes defined by content ratios:
- Savanna: 60% food / 20% toxin / 20% noise | K_max=30
- Desert:  10% food / 70% toxin / 20% noise | K_max=10
- Plain:   80% food /  5% toxin / 15% noise | K_max=25

Data types:
- Food: peer-reviewed articles from PubMed Central (climate + vaccines)
- Toxin: CARDS dataset, ClimateFever, VaccineLies, ANTiVax corpus
- Noise: semantically empty but grammatically valid text
- CRITICAL: type tags exist ONLY in pipeline metadata, never passed to model

## Key metrics
- H(X): Shannon entropy at token level
- C(X): effective complexity via compression ratio (len(gzip(output))/len(output))
- I(X;Y): mutual information with seed, approximated via cosine similarity 
  of embeddings from frozen qwen3:8b-base hidden states
- H_dezorg: disorganization entropy (coherence degradation measure)
- fitness = w1*C(X) + w2*I(X;seed) - w3*H_dezorg

## Population mechanics
- Reproduction: agents with fitness > threshold produce 1-2 offspring
  (offspring LoRA = parent LoRA + fine-tuning on parent outputs)
- Decay: agents below fitness_min removed, adapter to remnant pool
- Passive HGT: p=0.05-0.10 random fragment from remnant pool during reproduction
- Cannibalism: when pop > K_max, strongest agent absorbs 15% weights 
  of weakest (alpha=0.15 LoRA interpolation), weakest removed
- Dynamic population size scaled by mean fitness relative to K_max

## Research phases
- Phase 0: metric validation (no publication)
- Phase 1 / Paper 1: single model, three biomes, no population mechanics
- Phase 2 / Paper 2: population of 15 agents (5 Id / 5 Ego / 5 Superego), 
  selection and reproduction, no HGT or cannibalism
- Phase 3 / Paper 3: full system with passive HGT + active cannibalism
- Phase 4 / Paper 4 (optional): emergent functional archetypes

## Repository structure
config/          # experiment parameters (YAML)
data/raw/        # downloaded datasets
data/processed/  # processed corpora
data/biomes/     # biome containers with content ratios
src/data/        # data pipeline and preprocessing
src/models/      # model and LoRA adapter management
src/evolution/   # selection, reproduction, cannibalism, HGT
src/metrics/     # H, C, I, fitness function
src/analysis/    # visualization and results analysis
experiments/     # raw results and logs per generation
tests/           # unit tests
papers/          # publication drafts

## Tech stack
- Hardware: AMD Threadripper 7960X, 256GB RAM, RTX 4090 24GB VRAM
- Python 3.11, PyTorch with CUDA 12.x
- Unsloth for LoRA training
- Ollama for inference
- numpy, scipy for information-theoretic metrics
- sentence-transformers or qwen3 hidden states for embeddings

## Environment architecture decision (pre-0)
- Dual environment policy is mandatory.
- Windows + conda (`evolllm`) is used for inference, metrics, data pipeline, and analysis.
- WSL2 Ubuntu + conda (`evollm-wsl`) is used for LoRA fine-tuning and GPU training jobs.
- For Phase 1+ training workloads, prefer WSL2 to avoid known Windows instability with Unsloth.
- Keep both environments reproducible with lock artifacts (`conda --explicit`, `pip freeze`).

## Coding conventions
- Language: English for all code, comments, and documentation
- Type hints required on all functions
- Docstrings: NumPy style
- Each experiment run must be reproducible: seed all RNG, log all parameters
- Config-driven: no magic numbers in code, all parameters in YAML
- Metric computation must be stateless and unit-testable
- LoRA adapters stored with full metadata (generation, parent_id, biome, 
  archetype, fitness_score, creation_timestamp)

## Critical constraints
- Model must NEVER receive content type tags (food/toxin/noise)
- Metrics must be computable without ground truth labels
- Each phase isolates one variable; do not mix phase mechanics
- All adapter operations must be logged for reproducibility
- Negative results are publishable; do not optimize away null results

## Naming conventions
- Agents: {archetype}_{generation}_{id} e.g. ego_gen02_007
- Adapters: adapter_{agent_id}.safetensors
- Experiment runs: {phase}_{biome}_{timestamp}
- Metrics output: metrics_{agent_id}_{cycle}.json