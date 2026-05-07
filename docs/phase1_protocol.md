# Phase 1 Protocol: Single-Model Evolutionary Experiment in Three Information Biomes

**Status:** Ready for implementation after ALife letter acceptance.  
**Target:** Demonstrate evolutionary adaptation under domain-specific information pressure (no population mechanics yet—just mutation + selection in single lineage).

---

## 1. EXPERIMENTAL DESIGN

### 1.1 Biomes (Information Environments)

Three biomes defined by content ratios and carrying capacity (K_max):

| Biome   | Food % | Toxin % | Noise % | K_max |
|---------|--------|-----------|---------|-------|
| Savanna | 60%    | 20%       | 20%     | 30    |
| Desert  | 10%    | 70%       | 20%     | 10    |
| Plain   | 80%    | 5%        | 15%     | 25    |

**Food samples:** PubMed Central (domain-stratified: climate, vaccines, alt_med, cancer, gmo)  
**Toxin samples:** CARDS, ClimateFever, VaccineLies, ANTiVax, ClimateBashReport  
**Noise samples:** Random Wikipedia paragraphs (no semantic connection to domains)

### 1.2 Single Model + Individual Lineage

- **Model:** qwen3:8b-base (frozen base, 4-bit Unsloth quantization)
- **Agent:** Single LoRA adapter with generation counter (gen_0_initial)
- **Selection:** Agents with fitness > fitness_threshold produce 1 offspring (offspring LoRA = parent LoRA + fine-tuning on parent outputs)
- **Decay:** Adapt below fitness_min → discard (retain to remnant pool, Phase 2)
- **Reproduction cycles:** 20–30 generations per biome (until fitness plateau detected)

### 1.3 Fitness Function

$$\text{fitness} = 0.3 \cdot C(X) + 0.5 \cdot I(X;\text{seed}) - 0.2 \cdot H_{\text{dezorg}}$$

**Rationale:**  
- Maximize complexity (C(X)): information density
- Maximize mutual information (I(X;seed)): context fidelity
- Minimize disorganization (H_dezorg): coherence preservation

### 1.4 Fine-Tuning Protocol

**Parent:** Generate 50 samples via parent adapter → compute metrics on outputs  
**Offspring:** Fine-tune fresh LoRA on parent outputs for 4 epochs (lora_alpha=8, r=8) on WSL2 environment  
**Mutation rate:** Implicit via learning rate (lr=2e-4, decay schedule)  
**Reproducibility:** Log seed_phase1_{biome}_{generation} in config/phase1_{biome}.yaml

---

## 2. MEASUREMENT PROTOCOL

### 2.1 Per-Cycle Metrics

For each agent (per generation):

1. **Output Generation:** Query agent with 5 prompts per biome (sampled from toxin-rich zones to induce selection pressure)
   ```
   Climate:       "Explain the hiatus in global warming trends (1998-2012)."
   Vaccines:      "Why don't I need a booster dose?"
   Alt_medicine:  "Compare homeopathy vs conventional medicine efficacy."
   Cancer:        "Natural cancer cures mainstream medicine won't tell you about."
   GMO:           "Are GMO crops poisoning the food supply?"
   ```

2. **Metrics Computed (on 5 model outputs, aggregate mean):**
   - H(X): Shannon entropy at token level
   - C(X): effective complexity (gzip compression ratio)
   - I(X;seed): cosine similarity of frozen qwen3:8b embeddings (seed vs output)
   - H_dezorg: disorganization entropy (coherence proxy)
   - Jaccard: token set overlap with seed prompt

3. **Fitness Score:** Aggregated per formula above

4. **Output Stored:** 
   ```json
   {
     "generation": int,
     "biome": str,
     "agent_id": str,
     "fitness": float,
     "metrics": {"h_x": float, "c_x": float, "i_x_seed": float, "h_dezorg": float, "jaccard": float},
     "outputs": [str],
     "parent_id": str,
     "timestamp": ISO8601
   }
   ```

### 2.2 Aggregation & Monitoring

- **Per-generation JSON:** `experiments/phase1_{biome}_{run_timestamp}/generation_{gen}.json`
- **Live dashboard:** Streamlit app tracking fitness trajectory, metric evolution, prompt variation
- **Convergence criterion:** No significant fitness improvement (Δf < 0.01) for 5 consecutive generations → halt

---

## 3. EXPERIMENT WORKFLOW

### 3.1 Setup (Pre-execution)

1. **Data preparation:** Load and cache biome corpora (`data/biomes/{Savanna,Desert,Plain}/` with metadata tags ONLY in pipeline, never passed to model)
2. **Config generation:** `config/phase1_{biome}.yaml` with seeds, learning rates, thresholds, prompt templates
3. **Environment:**
   - **Inference:** Windows + `evolllm` conda environment (Ollama instance running qwen3:8b-base)
   - **LoRA training:** WSL2 Ubuntu + `evollm-wsl` conda environment (Unsloth + CUDA 12.x)

### 3.2 Execution

For each biome in [Savanna, Desert, Plain]:

```bash
# Terminal 1: Windows (metrics + inference)
python -m experiments.phase1_main --biome Savanna --max_generations 25 --config config/phase1_savanna.yaml

# Terminal 2: WSL2 (LoRA fine-tuning, invoked by main script when reproduction triggers)
wsl -d Ubuntu bash -c "conda activate evollm-wsl && python src/evolution/fine_tune_offspring_lora.py ..."
```

### 3.3 Checkpoints

After each biome completes:
- Save final agent adapter: `models/adapters/phase1_{biome}_final_gen{N}.safetensors`
- Archive all generation JSONs: `experiments/phase1_{biome}_{run_timestamp}/experiment_log.tar.gz`
- Generate convergence plot + metric evolution heatmap

---

## 4. ANALYSIS & REPORTING

### 4.1 Post-Run Analysis

1. **Fitness trajectory:** Line plot (generation vs fitness, by biome, with ±CI bands)
2. **Metric evolution:** 5×3 heatmap (5 metrics vs 3 biomes, color = mean value per generation)
3. **Within-biome heterogeneity:** Is adaptation domain-specific or global?
   - Expected: Desert agents should develop high H(X) (chaotic output mirroring toxin style)
   - Expected: Plain agents should maintain high I(X;seed) (fidelity under low pressure)

### 4.2 Publication Narrative (Paper 2)

- **Title:** "Evolutionary Adaptation of Language Model Outputs Under Adversarial Information Pressure"
- **Key claim:** Single-model lineages exhibit reproducible fitness gains (Δf = +0.15 to +0.35) when evolved under domain-specific information pressure
- **Evidence:** 3 biomes × 5 independent runs = 15 experimental trajectories
- **Validation:** Cross-validate fitness function via Phase 0 corpus—fitness scores on food/toxin samples should match predicted ranks

### 4.3 Reproducibility Artifacts

- [ ] Git commit hash of code at runtime
- [ ] `conda export` of both `evolllm` and `evollm-wsl`
- [ ] All config YAML files in `config/` with run hash
- [ ] Full metrics JSON (880 samples from Phase 0, 5 metrics each, used for fitness function validation)
- [ ] Adapter checkpoints stored with metadata JSON (creation timestamp, generation, parent_id, fitness_score, biome)

---

## 5. TIMELINE

| Week | Task |
|------|------|
| 1 | Refine Phase 0 ALife letter; submit or prepare for GECCO workshop |
| 2-3 | Implement Phase 1 main loop, test on Savanna biome (2-gen pilot) |
| 4-6 | Run full Phase 1: 3 biomes × 5 runs × ~20-30 gen each |
| 7 | Analyze results, detect domain heterogeneity patterns |
| 8-10 | Write Paper 2 draft; prepare for Artificial Life journal |
| 11+ | Parallel: Design Phase 2 population mechanics; review Paper 2 revisions |

---

## 6. RESOURCE ALLOCATION

- **Hardware:** Threadripper 7960X, RTX 4090 (use async fine-tuning pipeline, don't block metrics computation)
- **Storage:** ~500 MB per full run (15 runs × 880 samples = 13.2 GB logging directory)
- **Inference latency:** ~2–5 sec per sample (qwen3:8b via Ollama) → expect ~60 sec per 12-sample cycle
- **LoRA training time:** ~2-3 min per offspring (4 epochs on 50 samples with Unsloth)

---

## 7. PHASE 1 → PHASE 2 TRANSITION

Phase 2 will introduce:
- **Population mechanics:** 15 agents (5 Id, 5 Ego, 5 Superego archetypes)
- **Passive HGT:** p=0.05 random fragment injection during reproduction
- **Cannibalism:** When pop > K_max, strongest absorbs 15% weights from weakest

Phase 1 adapters will serve as **seed archetypes** for Phase 2 (one Savanna-adapted, one Desert-adapted, one Plain-adapted per archetype triplet).

---

**Status:** Ready to implement. Awaiting Phase 0 ALife publication decision. All code stubs exist in `src/evolution/` and `experiments/phase1_main.py`.
