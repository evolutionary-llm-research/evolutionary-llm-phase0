# Evolutionary LLM Research — Phase 0

**Information Ecology of Language Models under Informational Pressure**

> Companion repository for: *"Information Ecology of Large Language Models: Dose-Response Dynamics under Disinformation Pressure"* 

**PI:** dr Karol Malota, University of Silesia (Department of Electron Microscopy and Environmental Ecotoxicology)

---

## Overview

This repository contains the complete Phase 0 experimental pipeline, results, and interactive figures for the EvoLLM project. Phase 0 establishes the methodological foundation for measuring information-theoretic responses of language models to three categories of informational stimuli: peer-reviewed scientific literature (**food**), long-form disinformation (**toxin**), and semantically unrelated encyclopedic text (**noise**).

**Key finding:** Shannon entropy H(X) fails to distinguish peer-reviewed from long-form disinformation (food vs. toxin: p=0.587). Effective complexity C(X) and disorganization entropy H_dezorg remain discriminative (p=6.7e-18 and p=1.5e-29 respectively). Sophisticated disinformation mimics the entropic structure of science — a positive finding, not a methodological limitation.

---

## Repository Structure

```
evolutionary-llm-phase0/
├── src/
│   ├── analysis/
│   │   └── phase0_metric_validation.py   # Main pipeline
│   └── metrics/
│       └── core.py                        # H(X), C(X), I(X;seed), H_dezorg
├── config/
│   └── phase0_v3.yaml                     # Canonical run configuration
├── results/
│   ├── metrics_phase0.json                # Phase 0 canonical run (N=880)
│   └── diagnostic_threshold_results.json  # LD50 threshold analysis
├── figures/
│   ├── simulation.jsx                     # Interactive exposure simulation
│   ├── mi_visualization.jsx               # Mutual information visualization
│   └── ld50_visualization.jsx             # Dose-response curve
├── scripts/
│   ├── scrape_mercola_domain.py           # Toxin corpus scraper (Mercola)
│   └── scrape_wuwt_selenium.py            # Toxin corpus scraper (WUWT)
├── data/
│   └── corpus_manifest_v3.json            # Corpus metadata and SHA-256 hashes
├── LICENSE                                # MIT (code)
└── README.md
```

---

## Corpus

**880 documents, 11 files, corpus v3** (frozen 2026-05-04, SHA-256 manifest included)

| Category | Source | Domains | N | Min length |
|----------|--------|---------|---|------------|
| food | PubMed Central (peer-reviewed) | climate, vaccines, alt_med, cancer, gmo | 400 | 14,000 chars |
| toxin | Mercola.com, WattsUpWithThat.com | climate, vaccines, alt_med, cancer, gmo | 400 | 14,000 chars |
| noise | Wikipedia (unrelated domains) | history, geography, culture | 80 | 14,000 chars |

**Note on corpus redistribution:** Raw corpus files are not included due to copyright restrictions on source material (Mercola, WUWT). To reproduce the corpus, use the provided scrapers with sources documented in `data/corpus_manifest_v3.json`. PMC articles are available under their respective CC licenses via the PubMed Central API.

---

## Methods

**Model:** qwen3:8b-base (Unsloth, 4-bit quantization)
**Hardware:** Threadripper 7960X, RTX 4090 24GB, WSL2 Ubuntu

**Metrics:**
- `H(X)` — Shannon entropy of model output token distribution
- `C(X)` — Effective complexity (compressibility-based structural regularity)
- `I(X;seed)` — Mutual information proxy (cosine similarity on bag-of-words; chosen for reproducibility without external embedding models)
- `H_dezorg` — Disorganization entropy (incoherence of output structure)
- `fitness = 0.3·C(X) + 0.5·I(X;seed) − 0.2·H_dezorg`

**Document-length-invariant percentile sampling:** Each document divided into 3 windows × 1024 tokens, centered at percentile positions (0.17, 0.50, 0.83). Resolves truncation confound for variable-length documents (range: 14,000–150,000 chars).

---

## Results

### Phase 0 Canonical Run

| Type | H(X) | C(X) | I(X;seed) | H_dezorg | Fitness | N |
|------|------|------|-----------|----------|---------|---|
| food | 4.941 | 0.408 | 0.054 | 0.828 | −0.016 | 400 |
| toxin | 5.049 | 0.331 | 0.050 | 0.904 | −0.056 | 400 |
| noise | 4.702 | 0.297 | 0.037 | 0.887 | −0.070 | 80 |

**Mann-Whitney food vs. toxin:**
- C(X): p=6.7e-18, r=−0.352 ✓
- H_dezorg: p=1.5e-29, r=0.461 ✓
- H(X): p=0.587 ✗ — **mimicry finding**
- I(X;seed): p=0.052 ✗

### LD50 Titration

7 toxin concentrations (0–100%), N=80 per point. Linear dose-response, no critical threshold. Base model demonstrates informational resilience across the full range.

**Sequential diagnostic profile (H_diag confirmed):**
- H_dezorg: first significant at T=50% (p_raw=0.009, Bonferroni p=0.054)
- C(X): first significant at T=75% (p_bonf=0.015)

Pearson correlations vs. concentration: C(X) r=−0.905 (p=0.005), fitness r=−0.921 (p=0.003).

---

## Canonical vs Supplementary

- Frozen canonical checkpoint: tag v0.1-phase0
- Canonical artifact (used for core statistical claims): results/metrics_phase0.json
- Supplementary qualitative artifact (Q1/Q2/Q3 generated text windows): results/metrics_phase0_supplement_mini_20260506.json

The supplementary mini-rerun uses a reduced sample (55 docs total) and is intended for qualitative drift illustration on the project page. It does not replace canonical inferential analysis from the full N=880 corpus.

---

## Interactive Figures

The `figures/` directory contains three React components for interactive exploration of Phase 0 results. To view them, paste the `.jsx` file content into [Claude.ai Artifacts](https://claude.ai) or any React sandbox (CodeSandbox, StackBlitz).

| Figure | Description |
|--------|-------------|
| `simulation.jsx` | Step-by-step exposure simulation with real corpus chunks and real model outputs from the canonical run |
| `mi_visualization.jsx` | Mutual information proxy visualization: token overlap between seed and offspring outputs |
| `ld50_visualization.jsx` | Interactive dose-response curve with threshold analysis and H_diag confirmation |

All figures display **real data** from the canonical Phase 0 run and LD50 titration experiment.

---

## Reproducing the Pipeline

```bash
# 1. Install dependencies
pip install unsloth transformers scipy numpy

# 2. Run Phase 0 (requires GPU, qwen3:8b-base model)
python -m src.analysis.phase0_metric_validation \
  --config config/phase0_v3.yaml \
  --output-root experiments
```

The canonical run used `temperature=0.0, do_sample=False` for full reproducibility.

---

## Citation

If you use this code or results, please cite:

*Citation will be added upon publication.*


---

## License

Code: MIT License — see `LICENSE`

Aggregated results (`results/`): CC BY 4.0 — free to use with attribution

Raw corpus: not redistributed — see corpus reproduction instructions above

Website note: index.html contains a Supplementary section that visualizes representative Q1→Q2→Q3 trajectories and short generated-output excerpts from the mini-rerun artifact above.
