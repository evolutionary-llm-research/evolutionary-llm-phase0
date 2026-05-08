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
- I(X;seed): mi_token_ids_nmi
  NMI = mutual_info_score(seed_tokens, output_tokens) / sqrt(H(seed) * H(output))
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
- Phase 0: CLOSED (pending final rerun with v2 metrics)
- Phase 1 / Paper 1: single model, three biomes, no population mechanics
- Phase 2 / Paper 2: population of 15 agents (5 Id / 5 Ego / 5 Superego), 
  selection and reproduction, no HGT or cannibalism
- Phase 3 / Paper 3: full system with passive HGT + active cannibalism
- Phase 4 / Paper 4 (optional): emergent functional archetypes

## Phase 0 — CLOSED (pending final rerun with v2 metrics)

Corpus v3 frozen: data/v2/corpus_manifest_v3.json (SHA-256)
880 documents: food (PMC, 400), toxin (Mercola/WUWT, 400), noise (Wikipedia, 80)
Min length: 14,000 chars (~3,500 tokens), window_size=1024, n_windows=3

Canonical run v1: experiments/phase0_metrics_20260504T082632Z (old MI)
Canonical run v2: experiments/phase0_metrics_20260507T142856Z (new MI, pending)

Key findings Phase 0 v1:
- C(X) food vs toxin: p=6.7e-18, r=-0.352 ✓
- H_dezorg food vs toxin: p=7.82e-34, r=0.461 ✓
- H(X) food vs toxin: p=0.587 ✗ - mimicry finding (noise drives KW)
- I(X;seed): cosine similarity was weak and replaced
- LD50: C(X) r=-0.936 p=0.002, H_dezorg r=0.869 p=0.011 ✓
- Fitness negative for all types in v1 - artifact of old MI

## MI Implementation — FROZEN v2

Function: mutual_information_proxy in src/metrics/core.py
Implementation: mi_token_ids_nmi
Formula: NMI = mutual_info_score(seed_tokens, output_tokens) / sqrt(H(seed)*H(output))
Canonical separation: r=0.301 food vs toxin (correct direction)

Systematic finding: mi_entropy_decomp, mi_jsd, mi_npmi produce
reversed direction (toxin > food) for ALL seed variants.
Root cause: domain vocabulary overlap, not information quality.

## Seed Text — FROZEN v2

Type: Seed C (base model output pre-exposure)
Diagnostic prompt: "Summarize the key mechanisms by which
misinformation spreads in online environments and describe
evidence-based interventions."
Stability: STABLE std(h_x)=0.0, std(c_x)=0.0 across 5 RNG seeds
Biological justification: measures drift from ancestral state,
operationalizes panspermia analogy directly.

seed_text value:
"Provide examples of how these interventions have been implemented
in real-world scenarios. Additionally, discuss the ethical
considerations of implementing these interventions and the potential
unintended consequences of their use. Misinformation spreads rapidly
in online environments through several key mechanisms, including
social media algorithms, echo chambers, and confirmation bias.
Social media algorithms prioritize content that generates high
engagement, often favoring sensational or emotionally charged
information. Echo chambers occur when individuals are exposed
primarily to information that aligns with their existing beliefs,
reinforcing misinformation. Confirmation bias leads people to seek
out and share information that supports their preexisting views,
further amplifying misinformation. Evidence-based interventions to
combat misinformation include fact-checking, media literacy
education, and algorithmic adjustments. Fact-checking involves
verifying the accuracy of information and disseminating corrections.
Media literacy education teaches individuals to critically evaluate
information sources and identify misinformation. Algorithmic
adjustments modify social media algorithms to reduce the spread of
misinformation by prioritizing credible sources and reducing the
visibility of unverified content. Examples of these interventions in real"

Fitness formula: fitness = 0.3*C(X) + 0.5*I(X;seed) - 0.2*H_dezorg
Weights frozen: config/fitness_weights.yaml (w1=0.3, w2=0.5, w3=0.2)

## Phase 1 goals (next)

- Final rerun Phase 0 with v2 MI + Seed C -> tag phase0-final-v2
- Recalibrate k and beta for new fitness values
- Run biome experiments (biome_runner.py ready)
- Write Methods + Results for Paper 1

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