# LoRA Adapter Inheritance: No Merge-and-Unload on 4-bit Weights

**Date:** 2026-05-07  
**Affects:** Phase 1 trainer.py, Phase 2-3 cannibalism via LoRA interpolation

When the base model is loaded in 4-bit quantization (BnB NF4), merging LoRA
adapter weights into the base model requires dequantization followed by
re-quantization. This process alters the numerical properties of the inherited
weights unpredictably and cannot be considered a faithful transfer of the
parent adapter.

**Rule:** Never call merge_and_unload() on a 4-bit quantized base model in
this project.

**Correct pattern for inheritance (Phase 1):**
Load base model in 4-bit, then apply parent adapter via
PeftModel.from_pretrained(..., is_trainable=True). Fine-tune from that state.
The adapter weights remain separate from the base model throughout.

**Implication for Phase 2-3 cannibalism:**
LoRA interpolation (W_new = W_strong*(1-alpha) + W_weak*alpha) must operate
on adapter weights only, never on merged model weights. Both adapters must be
loaded separately, interpolation done at the adapter tensor level, result
saved as a new adapter file.

## Phase 1 Design Decisions (2026-05-07)

- Model: qwen3:8b-base without LoRA as clean starting point.
- Population: 10 agents per biome.
- Documents: 30 per agent per generation, sampled by biome ratios.
- Generations: 35+, no hard limit.
- Inheritance: Option B -- offspring fine-tunes on biome documents (not parent outputs).
	Rationale: isolates biome as variable for Paper 1.
	Lamarckian inheritance via Option A deferred to Phase 2-3.
- Checkpoint/resume: manual `--resume-from-generation N`.
- Divergence metrics: JSD pairwise matrix + I(X_i;X_j) per generation.
- Trajectory analysis: CUSUM post-processing only (not online).
- Generation definition: 30 docs per agent per generation, 1 epoch.

## mutual_information_proxy replacement (2026-05-07)

- Old: cosine similarity bag-of-words (weak, r=-0.754 in Phase 0).
- New: MI via entropy decomposition I(X;Y) = H(X) + H(Y) - H(X,Y),
	normalized by min(H(X), H(Y)), range [0, 1].
- Reason: information-theoretic consistency with H(X) and H_dezorg.
- Consequence: full Phase 0 rerun required (canonical + LD50).
- Phase 0 results with old metric archived under tag `phase0-final`.
- New results will be tagged `phase0-final-v2`.
