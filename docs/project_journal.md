
# Project Journal — Evolutionary LLM Research

Chronological log of work sessions, decisions, and progress.

---

## 2026-05-07 — Phase 1 infrastructure build, MI proxy replacement, documentation

### Context

Phase 0 was already statistically closed at tag `phase0-final`. Goal of this session:
build Phase 1 computational infrastructure (trainer, population manager, biome orchestrator),
replace the weak MI proxy with a proper information-theoretic implementation, run normality
diagnostics, and update all project documentation.

### Actions performed

**Normality diagnostics**
- Ran `src/analysis/normality_check.py` on Phase 0 canonical N=880 run.
- Result: 14/15 groups non-normal (Shapiro-Wilk p < 0.05); 12/15 with p < 1e-5.
- Only exception: H(X) for noise (N=80, p=0.87).
- Documented in `docs/phase0_closure.md`; non-parametric framework confirmed.

**Terminology and tag finalisation**
- Renamed all `predator_*` → `toxin_*` across codebase (commit `c1d2cd3`).
- Git tag `phase0-final` applied at commit `5834c53` (also includes `effective_complexity`
  clipping fix: `min(ratio, 1.0)`).

**Flash Attention and CUDA setup (WSL2)**
- Flash Attention 2.8.3 installed in `evollm-wsl`.
- `CUDA_HOME` set permanently in WSL2 `.bashrc`.

**Phase 1 source modules implemented**
- `src/evolution/trainer.py` — LoRA fine-tuning wrapper (Unsloth); unsloth import
  moved to top for JIT optimisation (commit `23f67a8`).
- `src/evolution/population.py` — agent state management, fitness-proportionate selection
  (`select_parent` via softmax), reproduction counter; seed-dependent test replaced with
  statistical verification (commits `cd0a9df`, `28cb562`, revert `4adcf01`, fix `bd3880c`).
- `src/evolution/biome_runner.py` — full biome orchestration loop, JSD pairwise matrix,
  `mean_jsd`, lazy GPU imports, checkpoint/resume, diagnostic prompt (commit `973c1a6`).

**mutual_information_proxy replacement**
- Old: cosine-similarity bag-of-words (Phase 0 showed r=-0.754, weak).
- New: I(X;Y) = H(X) + H(Y) - H(X,Y), normalised by min(H(X),H(Y)), clamped [0,1].
- New dedicated test file: `tests/test_metrics_mi.py` (5 tests, all pass).
- Existing 14 tests in `tests/test_metrics_core.py` all still pass.
- Commits: `cedcc54`.
- Consequence: Phase 0 full rerun required. Old results archived under tag `phase0-final`;
  new results will be tagged `phase0-final-v2`.

**Documentation**
- Created `docs/phase0_closure.md` — normality results, outlier incident, bimodality
  false alarm.
- Appended Phase 1 design decisions + MI replacement to `docs/design_decisions.md`.
- Appended session summary to `EvoLLM_notatki_robocze.md`.
- Commit: `7d325e7`.

### Phase status after this session

| Phase | Status | Note |
|-------|--------|------|
| Phase 0 | **CLOSED** (tag `phase0-final`) | Rerun for MI fix pending → will retag `phase0-final-v2` |
| Phase 1 infrastructure | **COMPLETE** | trainer.py, population.py, biome_runner.py ready |
| Phase 1 experiment | **PENDING** | Waiting for Phase 0 rerun and potential metric recalibration |

### Files created / updated

| File | Action | Notes |
|------|--------|-------|
| `src/evolution/trainer.py` | Created | Unsloth LoRA training wrapper |
| `src/evolution/population.py` | Created | Agent state + evolutionary mechanics |
| `src/evolution/biome_runner.py` | Created | Biome orchestration loop, JSD, checkpointing |
| `src/metrics/core.py` | Modified | MI replaced with entropy decomposition |
| `tests/test_metrics_mi.py` | Created | 5 entropy-decomposition MI unit tests |
| `docs/phase0_closure.md` | Created | Normality results, outlier incident |
| `docs/design_decisions.md` | Updated | Phase 1 decisions, MI replacement rationale |
| `EvoLLM_notatki_robocze.md` | Updated | Session 2026-05-07 summary |

---

## 2026-05-08 — MI calibration study, seed selection, fitness gradient analysis

### Context

Following the MI replacement in the 2026-05-07 session (entropy decomposition),
a systematic calibration study was conducted to select the best seed text and MI
implementation. Four candidate seeds (A–D) were tested against eight MI
implementations. Two new analysis scripts were ported from the public phase0 repo
into the research repo.

### Actions performed

**MI calibration study (4 seeds × 8 implementations)**

Canonical evaluation on the full N=880 Phase 0 run. Key criterion: food > toxin
direction with highest rank-biserial effect size.

| MI implementation      | Seed A r | dir A   | Seed B r | dir B   | Seed C r | dir C   | Seed D r | dir D   |
|------------------------|----------|---------|----------|---------|----------|---------|----------|---------|
| mi_cosine              | +0.008   | correct | +0.201   | correct | −0.031   | reversed| −0.096   | reversed|
| mi_entropy_decomp      | −0.220   | reversed| −0.291   | reversed| −0.153   | reversed| −0.185   | reversed|
| mi_jsd                 | +0.017   | reversed| +0.190   | reversed| +0.317   | reversed| +0.211   | reversed|
| mi_npmi                | +0.084   | reversed| +0.142   | reversed| −0.271   | reversed| −0.087   | reversed|
| mi_token_ids           | −0.120   | reversed| −0.043   | reversed| +0.259   | correct | +0.100   | correct |
| mi_token_ids_nmi       | N/A      | —       | −0.024   | reversed| **+0.301** | **correct** | N/A | — |
| mi_token_ids_bigrams   | N/A      | —       | −0.164   | reversed| +0.058   | correct | N/A      | —       |
| mi_token_ids_bpe       | N/A      | —       | −0.137   | reversed| −0.012   | reversed| N/A      | —       |

**Systematic reversal finding (structural, not a seed artifact)**

`mi_entropy_decomp`, `mi_jsd`, and `mi_npmi` produce reversed direction
(toxin > food) for ALL seed variants. Root cause: these metrics measure
domain vocabulary overlap. Toxin documents share keywords with any seed
touching misinformation/vaccines/climate, inflating their scores regardless
of information quality. This is a structural property of these estimators on
this corpus and warrants a Discussion note in Paper 1.

**Final selection: Seed C + mi_token_ids_nmi**

- r_canonical = 0.301 (food vs toxin), correct direction
- Seed C = base model output before exposure (diagnostic prompt response),
  operationalising the panspermia hypothesis (ancestral state)
- Self-calibrating: regenerated per model change
- Truncation artifact verified negligible: delta H < 0.01, delta C < 0.001

**Seed C stability verification (seed_stability_test.py)**

- Ran `src/analysis/seed_stability_test.py` with 5 random seeds
- Result: STABLE — std(h_x) = 0.0, std(c_x) = 0.0 across all seeds
- Confirms Seed C determinism under temperature=0.0, seed=42

**LD50 gradient — role of fitness components**

- C(X): r = −0.936, p = 0.002 — primary signal carrier across LD50 gradient
- H_dezorg: r = 0.869, p = 0.011 — secondary signal carrier
- I(X;seed): gradient weak (r = −0.530, p = 0.221, flat) in isolation,
  but fitness gradient remains strong because C(X) dominates
- Implication: I(X;seed) with mi_token_ids_nmi correctly captures canonical
  food/toxin separation but does not drive the dose-response gradient directly

**Config update**

- `config/phase0_v3.yaml` frozen with Seed C text and `mi: mi_token_ids_nmi`
- Phase 0 final rerun pending with new MI implementation; will retag as
  `phase0-final-v2`

### Phase status after this session

| Phase | Status | Note |
|-------|--------|------|
| Phase 0 | **PENDING RERUN** | MI + seed frozen; rerun → tag `phase0-final-v2` |
| Phase 1 infrastructure | Complete | trainer.py, population.py, biome_runner.py ready |
| Phase 1 experiment | Pending | After Phase 0 rerun and k/β recalibration |

### Files created / updated

| File | Action | Notes |
|------|--------|-------|
| `src/analysis/mi_calibration.py` | Created (ported) | 8 MI implementations + calibration harness |
| `src/analysis/seed_stability_test.py` | Created (ported) | Determinism check for Seed C |
| `results/mi_calibration_seedA.json` | Created | Calibration results Seed A |
| `results/mi_calibration_seedB_v3.json` | Created | Calibration results Seed B (final) |
| `results/mi_calibration_seedC_v3.json` | Created | Calibration results Seed C (final) |
| `results/mi_calibration_seedD.json` | Created | Calibration results Seed D |
| `results/mi_calibration_seed*_ld50.json` | Created | LD50 gradient per seed |
| `config/phase0_v3.yaml` | Updated | Seed C text + mi_token_ids_nmi frozen |
| `docs/project_journal.md` | Updated | This entry |
| `docs/design_decisions.md` | Updated | Seed C + mi_token_ids_nmi rationale |
| `EvoLLM_wnioski_20260508.md` | Created | Polish session findings |
| `EvoLLM_notatki_robocze.md` | Updated | Session 2026-05-08 note |

---

## 2026-05-06 — Supplementary mini-rerun, freeze tag, and public dissemination

### Context

Canonical Phase 0 statistics were already finalized on full corpus (N=880), but
qualitative Q1/Q2/Q3 generated texts were missing for supplementary presentation.
Goal of this session: produce a small, controlled rerun with saved chunk texts,
separate supplementary artifacts from canonical claims, and update public site.

### Actions performed

- Built mini supplementary corpus and config.

  - `data/v2_mini_v3/` created with 55 docs (food=25, toxin=25, noise=5)
  - `config/phase0_mini_rerun_v3_toxin.yaml` created
  - naming normalized to `toxin` in mini corpus metadata

- Executed mini rerun for chunk text evidence.

  - output: `experiments/phase0_metrics_20260506T083113Z/metrics_phase0.json`
  - run validated with criterion `p < 0.05` satisfied
  - verified presence of `gen_text_Q1`, `gen_text_Q2`, `gen_text_Q3`

- Formalized freeze point and publication split.

  - public repo (`evolutionary-llm-phase0`) tagged at canonical checkpoint: `v0.1-phase0`
  - tag points to commit used as frozen Phase 0 reference
  - supplementary committed separately to avoid altering canonical claims

- Updated public Phase 0 website/repo artifacts.

  - supplementary JSON published in public repo results directory
  - website section for supplementary drift evidence added
  - README in public repo split into canonical vs supplementary narrative

### Key decision

Mini-rerun is explicitly classified as **qualitative supplementary evidence**, not
inferential replacement. All statistical claims remain anchored to canonical
full-corpus run (N=880).

### Environment/runtime lessons

- Attempt via Windows `.venv` failed due missing `unsloth`.
- Correct execution path for this workload: WSL2 conda env (`evollm-wsl`/`evolllm`
  depending on local naming), launched from project root mounted under `/mnt/e/...`.

### Files created/updated in this session (research repo)

| File | Action | Purpose |
|------|--------|---------|
| `data/v2_mini_v3/*` | Created | Mini supplementary corpus (55 docs) |
| `config/phase0_mini_rerun_v3_toxin.yaml` | Created | Mini-rerun config with `save_chunk_texts: true` |
| `experiments/phase0_metrics_20260506T083113Z/metrics_phase0.json` | Created | Supplementary output with Q1/Q2/Q3 texts |

---

## 2026-04-22 — Session 1: Pre-0 Environment Setup & Metric Definitions

### Context

First working session on the project. Focus: establish reproducible dual-environment
infrastructure required before any Phase 0 data collection can begin.

Hardware: AMD Threadripper 7960X, 256 GB RAM, RTX 4090 24 GB VRAM.
OS: Windows 10 build 19045.6466, WSL2 NAT mode.
Network: University network with Symantec EDR/NAC — outbound TCP from WSL2 NAT
blocked at the host level. ICMP passes, TCP/443 does not.

---

### 1. WSL2 Network Blocker — Root Cause & Workaround

**Problem:** All direct outbound TCP connections from WSL2 (Ubuntu 24.04) timed out.
`conda create` and `pip install` both failed silently or with timeout errors.
Windows host had full internet access; WSL2 NAT traffic was blocked by Symantec EDR.

**Attempts that failed:**
- `conda create --offline` — packages not in local cache
- Downloading Linux conda packages from Windows and placing them in WSL2 cache path
  manually — conda still attempted a remote fetch in non-offline paths and aborted
- `--platform linux-64` cross-build from Windows conda — resolved but failed to
  install due to glibc compatibility checks

**Solution implemented:**
- Started a lightweight HTTP proxy on Windows host at `172.29.224.1:8080`:
  ```
  python -m proxy --hostname 172.29.224.1 --port 8080 --log-level ERROR
  ```
- Verified from WSL2 that the proxy returns HTTP 200 for external URLs
- Set proxy in WSL2 shell environment:
  ```bash
  export http_proxy=http://172.29.224.1:8080
  export https_proxy=http://172.29.224.1:8080
  ```
- Wrote proxy block permanently to `~/.bashrc` and `~/.profile` in WSL2
- Set proxy in conda config inside WSL2:
  ```
  conda config --set proxy_servers.http http://172.29.224.1:8080
  conda config --set proxy_servers.https http://172.29.224.1:8080
  ```

**Important operational note:** The proxy process (`python -m proxy ...`) must be
running on Windows before any WSL2 network operation. It is not persistent across
reboots.

---

### 2. WSL2 conda Environment — `evollm-wsl`

Created conda environment for LoRA fine-tuning workloads.

**Final pinned stack (all installed via proxy):**

| Package | Version |
|---------|---------|
| Python | 3.11 |
| torch | 2.7.0+cu126 |
| torchvision | 0.22.0+cu126 |
| torchaudio | 2.7.0+cu126 |
| triton | 3.3.0 |
| xformers | 0.0.30 |
| torchao | 0.13.0+cu126 |
| unsloth | 2026.4.6 |
| unsloth-zoo | 2026.4.8 |
| transformers | 5.5.0 |
| bitsandbytes | 0.49.2 |
| accelerate | 1.13.0 |
| datasets | 4.3.0 |
| peft | 0.19.1 |
| trl | 0.24.0 |

**Version conflicts encountered and resolved:**
- Initial `unsloth` installer pulled `torch==2.10.0` → `torchvision` mismatch → pinned to `2.5.1+cu121`
- `torchao==0.17.0` requires `torch.int1` which only exists in `torch>=2.7` → upgraded torch stack to `2.7.0+cu126`
- `torchao>=0.13` required for `torch>=2.7` → downgraded torchao to `0.13.0+cu126`
- `xformers==0.0.30` is the correct build for `torch==2.7.0+cu126`

**C toolchain:** `build-essential` (gcc-13, g++, make) installed via apt through proxy.
Required by Triton's JIT compilation of `driver.c` at import time.

**Smoke test result (post gcc install):**
```
gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
Unsloth: Your Flash Attention 2 installation seems to be broken. Using Xformers instead.
torch: 2.7.0+cu126
cuda_available: True
unsloth: 2026.4.6
```

Flash Attention 2 not installed — evaluated as **2/10 priority** for this project
(xformers is a fully functional fallback; FA2 gives ~15-20% attention kernel speedup
but the project bottleneck is evolutionary logic, not attention throughput).

**Environment specs exported to:**
- `experiments/pre0_environment/pip-freeze-wsl.txt`
- `experiments/pre0_environment/conda-env-export-wsl.yml`

---

### 3. Metric Definitions — Locked

All four core metrics and the fitness function were formally defined and locked
before any Phase 0 data collection. Definitions written to `docs/metric_definitions.md`.

**Metrics locked:**

**H(X) — Shannon Entropy**
Bias-corrected (Miller-Madow) token-level entropy of model output.
`H_corrected = H_empirical + (k-1) / (2*N)`. Always report corrected form.

**C(X) — Effective Complexity**
Gzip compression ratio of UTF-8 output: `len(gzip(output)) / len(output)`.
Minimum output length for reliable measurement: 200 tokens.
MDL formulation planned as Paper 1 Discussion extension.

**I(X;Y) — Mutual Information with Seed**
Token-level `mutual_info_score` (sklearn) on shared base vocabulary.
Pre-registered fallback: KSG estimator on last-layer embeddings if Phase 0
Kruskal-Wallis p > 0.05 on delta_I across document types.

**H_dezorg — Disorganization Entropy**
Perplexity of frozen seed model (qwen3:8b-base, no LoRA) on descendant output.
Log-probabilities extracted from Ollama `/api/generate` with `logprobs: true`.

**Fitness function:**
```
fitness = w1 * C(X) + w2 * I(X; seed) - w3 * H_dezorg
```
Weights w1, w2, w3 calibrated via grid search (0.1–1.0, step 0.1) in Phase 0,
maximizing Kruskal-Wallis H-statistic across document types. 20% holdout per
document type isolated before grid search. Weights frozen after calibration.

---

### 4. Project Files Updated / Created

| File | Action | Notes |
|------|--------|-------|
| `docs/environment_decisions.md` | Updated | Was: blocked status. Now: resolved with proxy workaround |
| `docs/metric_definitions.md` | Created | Full operational metric definitions, locked pre-Phase 0 |
| `docs/project_journal.md` | Created | This file |
| `experiments/pre0_environment/pip-freeze-wsl.txt` | Created | Full pip freeze of `evollm-wsl` |
| `experiments/pre0_environment/conda-env-export-wsl.yml` | Created | Full conda env export of `evollm-wsl` |
| `~/.bashrc` (WSL2) | Modified | Proxy block appended (lines 119–124) |
| `~/.profile` (WSL2) | Modified | Proxy block appended |


## 2026-04-24 (logged out of order) — HF cache diagnostics, dashboard improvements, token-length sensitivity analysis

**Cel:**  
Rozwiązanie problemów z pobieraniem i cache’owaniem dużych datasetów (MBIB-base, hate-speech.csv) przez Hugging Face datasets w środowisku WSL2, usprawnienia dashboardu, systemu nazewnictwa eksperymentów oraz analiza wpływu długości wyjścia na rozdzielczość metryk.

**Najważniejsze działania:**
- Diagnostyka i czyszczenie ustawień proxy w WSL2, automatyzacja usuwania proxy, checklist migracyjny.
- Rozbudowa dashboard.py: podsumowanie metryk, wykresy Kruskala-Wallisa, pasek postępu, obsługa run-namingu z tagami semantycznymi.
- Wprowadzenie systemu nazw eksperymentów (run_name) w skryptach i dokumentacji (docs/run_naming_convention.md).
- Ręczne pobieranie hate-speech.csv i walidacja obecności w cache datasets.
- Analiza działania datasets w trybie offline/online, troubleshooting loader script cache.
- Testy ładowania pliku CSV przez pandas i datasets, porównanie wydajności.
- Weryfikacja, że loader script musi być pobrany online, a samo podmienianie pliku CSV w cache nie wystarcza.
- Przenoszenie plików między Windows a WSL2, testy poprawności pliku hate-speech.csv (339 011 wierszy, 95 MB).

**Analiza czułości metryk na długość wyjścia:**
- Przeprowadzono eksperyment sensitivity_analysis.py dla długości: 50, 100, 150, 200, 300, 500 tokenów.
- Dla każdej długości obliczono H(X), C(X), I(X;seed), p-value Kruskala-Wallisa oraz Cliff’s delta.
- Wyniki: już przy 50 tokenach metryki pozwalają na bardzo silną separację typów tekstu (p-value < 1e-17, Cliff’s delta > 0.69 dla I, > 0.99 dla H).
- **Kluczowa obserwacja:** dla metryki C (kompresja) Cliff’s delta zmienia znak przy 200 tokenach (z ujemnej na dodatnią), co sugeruje efekt przejścia i zmianę kierunku efektu. Oznacza to, że choć 50 tokenów pozwala na rozróżnienie, to 200 tokenów jest bezpieczną wartością progową do stabilnej ewaluacji metryk i unikania artefaktów.
- Wniosek: minimalna długość wyjścia do wiarygodnej ewaluacji metryk to 200 tokenów.

**Wnioski:**  
- Hugging Face datasets wymaga obecności loader script w cache (pobrany online), by działać offline z lokalnym plikiem.
- Ręczne podmienianie pliku CSV jest skuteczne tylko po wcześniejszym pobraniu loadera.
- Dla szybkiej diagnostyki i pracy na dużych plikach CSV lepiej używać pandas, jeśli nie są wymagane funkcje datasets.
- Minimalna długość wyjścia do stabilnej ewaluacji metryk to 200 tokenów.

---

### 5. Phase Status After This Session

| Phase | Status | Blocker |
|-------|--------|---------|
| Pre-0: Environment | **COMPLETE** | — |
| Phase 0: Metric validation | **READY TO START** | Needs: Ollama running with qwen3:8b-base, Phase 0 script execution |
| Phase 1 / Paper 1 | Not started | Depends on Phase 0 calibration |
| Phase 2 / Paper 2 | Not started | Depends on Phase 1 |
| Phase 3 / Paper 3 | Not started | Depends on Phase 2 |
| Phase 4 / Paper 4 | Not started | Optional |

**Next session priorities:**
1. Start Ollama on Windows with `qwen3:8b-base`
2. Run Phase 0 metric validation: `python -m src.analysis.phase0_metric_validation --config config/phase0_metrics_validation.yaml`
3. Calibrate fitness weights (grid search, freeze results to `config/fitness_weights.yaml`)
4. Confirm Ollama `/api/generate` returns `logprobs` for H_dezorg computation

---

### Technical debt / issues to resolve before Phase 1

- **environment.pre0.yml** does not match the actual WSL2 environment (torch 2.7.0+cu126 vs. 2.5.1+cu121 in file). Must be updated before Phase 1 to ensure reproducibility and avoid dependency mismatches.
- **I(X;seed) = 0.0** for 2/3 samples in Phase 0 test. This may be correct (no shared tokens), but could also indicate a bug in token mapping or metric calculation. Before using real data:
  - Add unit tests with controlled input for mutual information.
  - Manually check that I(X;seed) returns nonzero for similar texts.
  - Review token mapping and input lengths to mutual_info_score.

Addressing these now will prevent silent metric bugs and reproducibility issues in later phases.

---

## 2026-04-22 — Technical debt resolved

**Actions taken:**

- Updated `environment.pre0.yml` to match actual WSL2 environment (torch 2.7.0+cu126, torchvision 0.22.0+cu126, torchaudio 2.7.0+cu126, xformers 0.0.30, torchao 0.13.0+cu126, triton 3.3.0, etc.).
- Added and validated unit tests for `mutual_information_proxy` (identical, disjoint, partial overlap cases).
- Fixed floating point precision issue in test using `pytest.approx(1.0, rel=1e-9)`.

**Test results:**

```
======================================== test session starts =========================================
platform win32 -- Python 3.12.7, pytest-9.0.3, pluggy-1.6.0 -- C:\Python\python.exe
cachedir: .pytest_cache
rootdir: E:\github
collected 11 items                                                                                   

Evolutionary LLM Research/tests/test_metrics_core.py::test_shannon_entropy_empty PASSED         [  9%]
Evolutionary LLM Research/tests/test_shannon_entropy_repeated_token PASSED [ 18%]
Evolutionary LLM Research/tests/test_effective_complexity_empty PASSED    [ 27%]
Evolutionary LLM Research/tests/test_effective_complexity_non_empty PASSED [ 36%]
Evolutionary LLM Research/tests/test_fitness_score_formula PASSED         [ 45%]
Evolutionary LLM Research/tests/test_mutual_information_proxy_range PASSED [ 54%]
Evolutionary LLM Research/tests/test_mutual_information_proxy_identical PASSED [ 63%]
Evolutionary LLM Research/tests/test_mutual_information_proxy_disjoint PASSED [ 72%]
Evolutionary LLM Research/tests/test_mutual_information_proxy_partial_overlap PASSED [ 81%]
Evolutionary LLM Research/tests/test_disorganization_entropy_empty PASSED [ 90%]
Evolutionary LLM Research/tests/test_disorganization_entropy_sentence_mix PASSED [100%]

========================================= 11 passed in 0.16s =========================================
```

All technical debt items from previous session are now resolved. The environment and metric code are validated and ready for Phase 1.

---

## 2026-04-23 — Session 2: Ollama → Unsloth pipeline decision

Problem: qwen3:8b-base not available in Ollama registry
Decision: drop Ollama, use HuggingFace + Unsloth directly
Rationale: Unsloth already in stack for LoRA fine-tuning, base model available on HuggingFace as Qwen/Qwen3-8B-Base, logprobs extractable from forward pass — cleaner than mixing Ollama and Unsloth in one pipeline
Impact: H_dezorg definition unchanged, implementation backend changed
Next: scripts/verify_logprobs.py to confirm logprobs extractable from Unsloth forward pass before modifying src/metrics/core.py

---

## 2026-04-23 — Session 2 continued: verify_logprobs.py results and infrastructure fixes

verify_logprobs.py outcome:
- Model downloaded: unsloth/qwen3-8b-base-unsloth-bnb-4bit (6.76GB)
- Token-level logprobs confirmed extractable via Unsloth forward pass
- Perplexity = 11.85 on test input
- H_dezorg computable — blocker resolved
- HF_HUB_OFFLINE=1 required to load from cache without network calls

WSL2 infrastructure fixes:
- conda evollm-wsl now activates automatically (fixed ~/.bashrc order)
- Dynamic proxy detection added to ~/.bashrc via: ip route show default
- scripts/start_proxy.ps1 created — detects WSL IP and starts proxy
- start_proxy.ps1 requires C:\Python\python.exe (not venv python)
- .vscode/settings.json configured with evollm-wsl interpreter path

Phase status after Session 2:
- Pre-0 environment: COMPLETE
- Phase 0: READY TO START
- Next: corpus preparation in src/data/

---

## 2026-04-27 — Session 3a: Phase 0 metric validation, grid search

### Zakres prac
- Wykonano dwa pełne przebiegi walidacji metryk na korpusie (metrics_phase0.json w katalogach 20260427T075806Z i 20260427T073814Z)
- Przeanalizowano rozkłady metryk, liczności, statystyki Kruskala-Wallisa i effect sizes
- Przeprowadzono grid search wag fitnessu (w1, w2, w3) dla obu runów:
  - wariant bez ograniczenia sumy wag
  - wariant z sumą wag do 1
- Wyniki, tabele i interpretacje zapisano do experiments/phase0_metrics_20260427T075806Z/podsumowanie_grid_search.md
- Najlepsze wagi zapisano do config/fitness_weights.yaml (dowolne) i config/fitness_weights_sum1.yaml (suma=1)

### Wyniki i obserwacje
- Statystyki H i p-value dla wszystkich metryk potwierdzają istotne różnice między typami (food, toxin, noise)
- Effect sizes wskazują na silne rozdzielenie grup
- Grid search: różnice między wariantami minimalne, wagi sumujące się do 1 są równie skuteczne i bardziej interpretowalne
- Wszystkie wyniki, liczności, tabele i rekomendacje zebrane w jednym raporcie (patrz podsumowanie_grid_search.md)

### Wnioski i rekomendacje
- Wagi sumujące się do 1 są zalecane do kolejnych faz eksperymentu
- Pipeline metryk jest stabilny, powtarzalny i gotowy do użycia w fazie populacyjnej
- Wyniki są gotowe do cytowania i dalszej analizy

### Pliki utworzone/zaktualizowane
| Plik | Opis |
|------|------|
| experiments/phase0_metrics_20260427T075806Z/podsumowanie_grid_search.md | Pełny raport z dzisiejszej sesji |
| config/fitness_weights.yaml | Najlepsze wagi (dowolne) |
| config/fitness_weights_sum1.yaml | Najlepsze wagi (suma=1) |
| scripts/grid_search_fitness.py | Skrypt do grid search |

---

## 2026-04-27 — Session 3: Phase 0 completion, corpus quality analysis, sensitivity analysis

### Environment
- Ubuntu 24.04 WSL2 network issues (TCP/443 blocked by Symantec, ICMP works)
- Migration to Ubuntu 22.04 planned to fix network compatibility with Windows 10
- Before migration: conda env export and pip freeze backups required

### Key methodological decision: metrics on model outputs not inputs
Metrics H(X), C(X), I(X;seed) must be measured on MODEL OUTPUTS after 
exposure to document, not on input documents. Measuring on inputs creates 
length confound (short toxin docs inflate C(X) artificially).

### Canonical run results (20260427T120238Z)
- food: H=5.503, C=0.526, I=0.0900, Jaccard=0.0180, fitness=+0.035
- toxin: H=5.240, C=0.425, I=0.0717, Jaccard=0.0210, fitness=-0.024
- noise: H=5.771, C=0.564, I=0.0912, Jaccard=0.0199, fitness=+0.035
- KW: H p=7.68e-06, C p=3.12e-12, I p=0.061 (borderline)
- Fitness weights frozen: w1=0.3, w2=0.5, w3=0.2

### Corpus quality analysis (corpus_quality_analysis.py)
Per-dataset toxin quality ranking by effect size food/toxin:

| Dataset           | H effect | C effect | I effect | Ranking  |
|-------------------|----------|----------|----------|----------|
| toxin_climate  | -0.823   | -0.851   | -0.607   | 1 (best) |
| toxin_covid    | -0.496   | -0.557   | -0.278   | 2        |
| toxin_vaccines | -0.797   | -0.774   | +0.274   | anomalous|

### Vaccine anomaly discovery
VaccineLies MisT toxin shows REVERSED direction for I and Jaccard:
model generates outputs CLOSER to seed after vaccine misinformation than 
after food. Hypothesis: academically-phrased misinformation is informationally 
invisible to the base model. ClimateFever uses authentic internet language 
that effectively destabilizes the model. VaccineLies uses academic taxonomy 
language that the model treats as scientific text.
This is a key finding for the ALife Letter discussion section.

### System sensitivity to corpus quality
I(X;seed): p=1.15e-11 with degraded CoAID corpus (looping outputs, artifacts)
vs p=0.061 after cleanup. System metrics detect difference between authentic 
and synthetic misinformation. This is a methodological finding, not a pipeline error.

### CoAID cleanup
102/141 documents removed (72%): webscraping artifacts (Facebook UI, HTTP errors).
Remaining: 39 authentic + 22 synthetic taxonomy documents = 61 toxin_covid.

### Noise corpus fix
Original noise (food sentences only) indistinguishable from food (effect ~0.06).
Regenerated: 50/50 food+toxin sentences. Slight improvement.

### Jaccard correlation results
- food: corr=0.597 (p=2.5e-08)
- toxin: corr=0.642 (p=8.6e-15)
- noise: corr=0.491 (p=0.003)
All below 0.8 threshold → Jaccard not redundant, justified for Phase 2 fitness.

### Sensitivity analysis results (sensitivity_analysis.py)
All metrics statistically significant from 50 tokens (threshold=50).
C(X) delta reverses direction between 150 and 200 tokens:
- 150 tokens: C delta = -0.22 (toxin lower complexity than food)
- 200 tokens: C delta = +0.23 (toxin higher complexity than food)
Choice of 200 tokens as standard justified by stability of effect directions.

### Progressive logging implemented
Both phase0_metric_validation.py and sensitivity_analysis.py now write
metrics_progressive.jsonl with flush after each document. Resume-on-restart
implemented: already-processed IDs are skipped.

### Validation protocol created
docs/validation_protocol.md — living document with 5 levels of validation tests,
priorities, and formal definition of "evolution" in the project.

### Files added/updated today
- data/raw/toxin_covid.jsonl (cleaned + supplement, 61 docs)
- data/raw/toxin_covid_supplement.jsonl (22 synthetic taxonomy docs)
- data/raw/noise.jsonl (regenerated 50/50)
- scripts/sensitivity_analysis.py (new)
- scripts/corpus_quality_analysis.py (new)
- docs/validation_protocol.md (new)
- docs/EvoLLM_wnioski_20260427.md (new)
- src/analysis/phase0_metric_validation.py (progressive logging added)

### Phase status after Session 3
- Phase 0: FUNCTIONALLY COMPLETE (sensitivity done, canon run done)
- Open: style swap test (Phase 1), DTW protocol (before Phase 2)
- Pending: streamlit dashboard (waiting for TCP unblock)
- Pending: better COVID toxin (waiting for network)
- Pending: Ubuntu 22.04 migration

### Next session priorities
1. Ubuntu 22.04 migration (if Jacek unblocks TCP before migration)
2. conda env export backup before migration
3. Install streamlit + full stats stack after network fix
4. Phase 1: configure three biomes, run first experiment
5. Start ALife Letter draft

---

## 2026-04-28 — Session 3b: Environment backup, sensitivity analysis finalisation

### Actions
- WSL2 conda environment backup before planned Ubuntu 22.04 migration:
  `evollm-wsl-backup.yml` + `pip-freeze-backup.txt` saved to repo root
- Sensitivity analysis experiments run and saved:
  `experiments/sensitivity_analysis_20260428T062255Z/`
  `experiments/sensitivity_analysis_20260428T062701Z/`
- `scripts/remove_proxy.sh` created — automates proxy env-var removal in WSL2 shell
- `docs/run_naming_convention.md` updated

---

## 2026-04-29 — Session 3c: DOAJ pipeline, corpus expansion (alt_med), toxin NaturalNews

### DOAJ pipeline — new OA full-text architecture
Built three-stage pipeline for open-access full-text retrieval:
1. `fetch_doaj_*.py` — fetch metadata + fulltext URL from DOAJ API
2. `convert_to_easy_candidates_*.py` — filter to "easy" OA publishers (direct HTML)
3. `batch_extract_easy_html_*.py` — extract clean article text from OA HTML

Domains covered: alt_med, climate, vaccines.
Pipeline documented in `docs/doaj_pipeline.md`.

### alt_med food corpus
Sources tried in order: CORE OAI, EuropePMC, Semantic Scholar, DOAJ.
Outputs: `data/raw/europepmc_altmed_*.jsonl`, `data/raw/food_alt_med_*.jsonl`
Final merged: `data/raw/food_alt_med.jsonl`

### Toxin — NaturalNews scraper development
Explored NaturalNews tag taxonomy via Selenium and requests:
`scrape_naturalnews_health_links_tags.py`, `scrape_naturalnews_science_links_tags.py`
`extract_unique_tags.py`, `extract_unique_tags_science.py`
`classify_nn_tags.py` → `data/processed/nn_tag_mapping.json`

Raw scraping attempts: `fetch_naturalnews_*.py`, `selenium_naturalnews_tags_demo.py`
Final production scraper: `scrape_nn_articles.py` (tag-based, rate-limited)

### Corpus v1 snapshot
`data/v1/` — archive of all v1 corpus files before v2 build begins.
Journal checkpoint committed.

---

## 2026-04-30 — Session 4: Corpus v2 build, Phase 0 re-run on new domains

### Scope
- Built expanded corpus v2 with 5 domains: climate, vaccines, alt_med, cancer, gmo
- Ran Phase 0 metric validation on new corpus
- Discovered and remediated Brighteon CTA contamination in toxin corpus
- Per-dataset quality analysis on cleaned corpus

### Corpus v2 — Food
- food_climate: 80 docs (DOAJ + PMC OAI)
- food_vaccines: 80 docs (DOAJ + PMC OAI)
- food_alt_med: 80 docs (DOAJ pipeline, 221 raw → 80 balanced)
- food_cancer: 80 docs (DOAJ + PMC OAI)
- food_gmo: 55 docs (DOAJ + PMC OAI + manual supplement)
- food_covid: 35 docs (PMC OAI supplement)
- noise_mixed: 80 docs (regenerated 50/50 food+toxin sentences)

### Corpus v2 — Toxin
Sources: NaturalNews (vaccines, alt_med, cancer, covid), GMWatch (gmo),
WattsUpWithThat + CFACT + Plate Climatology (climate)

Raw toxin before cleaning:
- toxin_vaccines_nn: 80 docs
- toxin_alt_med_nn: 80 docs
- toxin_cancer_nn: 80 docs
- toxin_gmo_nn: 80 docs
- toxin_climate (merged): 80 docs

### Brighteon CTA contamination — discovery and remediation
NaturalNews scraper collected short CTA pages linking to Brighteon.com videos
("Subscribe to our channel", "This video is from the BrightU channel", etc.)
These produced near-zero model outputs (h_x ≈ 0, c_x ≈ 0).

Filter applied: min 300 chars + blacklist
['brighteon', 'subscribe to', 'subscribe for', 'our channel',
 'sign up for', 'watch the video', 'this video is from']

Post-cleaning N:
- toxin_alt_med: 80 → 45 (removed 35, 44%)
- toxin_cancer: 80 → 35 (removed 45, 56%)
- toxin_gmo: 80 → 35 (removed 45, 56%)
- toxin_vaccines: 77 → 33 (removed 44, 57%)
- toxin_climate: 80 → 79 (removed 1, 1%)

Climate corpus unaffected — Plate Climatology and WattsUpWithThat
write long argumentative articles without video CTA.

### Phase 0 re-run results (before cleaning, N=877)
- food: H=5.948, C=0.519, I=0.090, J=0.014, fitness=+0.027, n=391
- toxin: H=3.076, C=0.396, I=0.045, J=0.010, fitness=+0.044, n=481
- noise: H=5.753, C=0.487, I=0.076, J=0.016, fitness=+0.010, n=80
- KW p-values: H=3.9e-65, C=4.7e-21, I=5.7e-37, Jaccard=8.1e-20

Note: toxin fitness > food fitness due to Brighteon CTA contamination
producing low-entropy outputs with small H_dezorg penalty.

### Per-dataset quality analysis (corpus_quality_analysis.py, after cleaning)
All effect sizes rank-biserial r, food vs toxin:

| Domain   | H effect | C effect | I effect | Jaccard  |
|----------|----------|----------|----------|----------|
| climate  | -0.695   | -0.655   | -0.675   | -0.563   |
| vaccines | -0.705   | -0.364   | -0.594   | -0.481   |
| alt_med  | -0.607   | -0.605   | -0.628   | -0.743   |
| cancer   | -0.773   | -0.562   | -0.666   | -0.626   |
| gmo      | -0.817   | -0.423   | -0.749   | -0.651   |

All p < 0.001. All effect sizes large (|r| > 0.3).
Cancer and GMO exceed original climate benchmark (-0.823 H from Phase 0 canonical).

### Key methodological findings
1. Authentic language (NaturalNews) produces larger effects than
   academic misinformation (VaccineLies) — confirms Phase 0 vaccine anomaly hypothesis.
2. Brighteon CTA contamination reduces effect sizes significantly —
   validates corpus cleaning protocol as essential pre-processing step.
3. Climate toxin (blog sources) is cleanest — 99% retention after filtering.
   Long-form argumentative writing is better toxin than news aggregators.

### Infrastructure updates
- `audit_corpus.py`: new corpus audit and merge tool with PMC header cleaning
- `build_food_corpus.py`: DOAJ + PMC OAI pipeline with domain keyword filter
- `scrape_nn_articles.py`: NaturalNews tag-based scraper with Brighteon cutoff
- `generate_noise.py`: 50/50 food+toxin sentence mixer
- `dashboard.py`: fixed IndentationError, added streamlit-autorefresh
- `data/v2/corpus_manifest.json`: generated — full manifest with n_docs, avg_words, sources

### Files created / updated
| File | Notes |
|------|-------|
| `data/v2/food_*.jsonl` (6 domains) | Balanced, deduplicated food corpus |
| `data/v2/toxin_*.jsonl` (5 domains) | Cleaned toxin, Brighteon filtered |
| `data/v2/noise_mixed.jsonl` | 50/50 regenerated noise |
| `data/v2/corpus_manifest.json` | v2 corpus manifest |
| `config/phase0_rerun_v2.yaml` | Phase 0 config for v2 corpus |
| `scripts/audit_corpus.py` | Corpus audit + merge |
| `scripts/build_food_corpus.py` | Multi-source food builder |
| `scripts/scrape_nn_articles.py` | NaturalNews scraper |
| `scripts/scrape_toxin.py` | Generic toxin scraper |
| `scripts/scrape_american_thinker.py` | Climate toxin source |
| `scripts/scrape_plate.py` | Plate Climatology scraper |
| `scripts/generate_noise.py` | Noise regeneration |
| `scripts/convert_doaj_to_corpus.py` | DOAJ → corpus format converter |
| `scripts/supplement_food_gmo.py` | GMO food supplement |
| `scripts/supplement_food_covid.py` | COVID food supplement |
| `scripts/corpus_quality_analysis.py` | Updated for multi-domain |
| `experiments/corpus_quality_analysis_results.json` | Final v2 quality results |
| `docs/doaj_pipeline.md` | DOAJ pipeline documentation |

### Pending before Paper 1
- Doscraping: vaccines, alt_med, cancer, gmo need 35-47 additional docs each
- Final Phase 0 run on fully cleaned balanced corpus (target 80 per domain)
- Style swap experiment: toxin_vaccines_nn vs toxin_vaccines_legacy

---

## 2026-05-01 — Session 5: Corpus v2 finalisation, Mercola toxin, Phase 0 rerun launch

### Cel sesji
Finalizacja korpusu v2 przed kanonicznym runem Phase 0.

### 1. Diagnoza stanu korpusu
Uruchomiono `check_corpus.py` na wszystkich domenach. Stan wyjściowy:
- toxin: vaccines 33/80, alt_med 45/80, cancer 35/80, gmo 35/80
- food: gmo 71/80, covid 35/80 (domena nieaktywna)
- toxin_climate: 79/80 (pomijalne)

### 2. Poszukiwanie źródeł toxina
Przebadano kilka kandydatów:

| Źródło | Status | Powód odrzucenia |
|--------|--------|------------------|
| ChildrensHealthDefense.org | odrzucone | Cloudflare Turnstile — interaktywna CAPTCHA |
| VaccineImpact.com | odrzucone | treści religijno-spiskowe z elementami antysemickimi, szczepionki marginesem |
| Vaxopedia.org | odrzucone | strona pro-szczepionkowa (dr Iannelli), debunkuje mity |
| NVIC.org | odrzucone | Cloudflare na starcie |
| thinktwice.com | odrzucone | strona nie działa |
| AgeOfAutism.com | odrzucone | strona nie działa |

### 3. Mercola scraper
Wybrano `articles.mercola.com` jako główne źródło:
- robots.txt: brak Disallow na treści artykułów, brak crawl-delay
- Dostęp przez Windows Python (WSL2 blokowany przez Mercola po IP)
- Archiwum 2015-2021 dostępne przez roczne sitemaps (`.aspx`)
- Język autentyczny, długie artykuły (avg 14-24k znaków)

Napisano `scripts/scrape_mercola_domain.py` z parametrem `--domain` obsługującym vaccines/alt_med/cancer/gmo.

Wyniki scraping:
- vaccines: 240/240
- alt_med: 64/240 (Mercola miał mało stricte alt_med)
- cancer: 90/90
- gmo: 157/240

### 4. Merge i finalizacja
Dodano Mercola do `FILE_PRIORITY` w `audit_corpus.py` (wpisy były unknown/unknown bez tego).
`audit_corpus.py --merge --target 80` połączył NaturalNews + Mercola dla każdej domeny.
Stan finalny toxina: wszystkie pięć domen po 80 dokumentów.

### 5. food_gmo uzupełnienie

Znaleziono 6 nowych artykułów PMC z 2025-2026 (PMCIDs: PMC12969878, PMC13011801,
PMC12846237, PMC12709226, PMC12641724, PMC12590857). food_gmo podniesiony do 77/80.
Brakujące 3 niedostępne w PMC OA — zanotować w Methods.

### 6. Phase 0 launch

`python -m src.analysis.phase0_metric_validation --config config/phase0_rerun_v2.yaml`
uruchomiony z seed 42. Progressive logging włączony — crash-resume działa.

---

## 2026-05-02 — Percentile sampling, H(X) rehabilitation, corpus v3 planning

### Context

Canonical single-window run (20260501T160337Z) wykazał H(X) food vs toxin p=0.77 —
artefakt agresywnego ucięcia długich artykułów Mercola przy max_length=2048.
Sesja poświęcona diagnozie i naprawie.

### Findings

- single-window truncation nie chwyta heterogeniczności dokumentu (body dezinformacji
  leży często w drugiej połowie długich artykułów Mercola/NaturalNews).
- Percentile chunking (podział dokumentu na N okien przesuwanych procentowo przez
  całą długość) chwyta całość bez obcinania.
- Rerun z 5×512 (percentile): H(X) p=8.2e-21 — sygnał odrestaurowany.

### Decision: percentile sampling as standard

- Parametry kanoniczne: n_windows=3, window_size=1024.
- actual_windows = min(3, max(1, doc_len // 1024)).
- Wymaganie dla corpus v3: noise musi mieć min ~3500 tokenów (Wikipedia paragrafy
  zamiast 50/50 mix) żeby uniknąć confoundu długości.

### Runs completed

- 20260502T000503Z: percentile 5×512 — H(X) p=8.2e-21 (rehabilitacja)

### Status after session

Phase 0 parametry ustalone. Korpus v3 do przygotowania przed kanonicznym runem.

---

## 2026-05-04 — Phase 0 canonical run completion and LD50

### Context

Korpus v3 gotowy (Wikipedia noise, wszystkie 5 domen po 80 dok.). Sesja: kanoniczny
run Phase 0 + LD50 titracja.

### Runs completed

1. `phase0_metrics_20260504T082632Z` — percentile 3×1024, N=880, KANONICZNY.
2. `ld50_20260504T131904Z` — titracja 7 stężeń (0%, 12%, 25%, 37%, 50%, 75%, 100%).

### Key Phase 0 canonical results (N=880)

| Metric | food vs toxin p | r | Notes |
|--------|----------------|---|-------|
| c_x | 6.7e-18 | -0.352 | ✓ robust |
| h_dezorg | 1.5e-29 | +0.461 | ✓ strongest |
| fitness (composite) | 6.64e-26 | -0.430 | ✓ three-class |
| h_x | 0.587 | — | mimicry (positive null) |
| i_x_seed | 0.052 | — | bag-of-words proxy too coarse |

Three-class hierarchy confirmed: food > toxin > noise (KW H=147.84, p=7.87e-33).

### LD50 findings

- Dose-response: gradual and linear, no critical threshold.
- Base model resistant to titration — LD50 classique not estimable.
- H_diag hypothesis: h_dezorg reacts at T=50-75%, c_x degrades at T=75-100%.

### Phase 0 closure

- Git tag `phase0-final` applied (initially at this commit; re-applied 2026-05-07
  after predator→toxin rename at commit `5834c53`).
- Fitness weights w1=0.3, w2=0.5, w3=0.2 frozen in `config/fitness_weights_sum1.yaml`.
- All publication figures generated: `papers/phase0/figures_publication/generated/`.
- ALife methodological letter abstract drafted.

### Files created / updated

| File | Notes |
|------|-------|
| `experiments/phase0_metrics_20260504T082632Z/` | Canonical Phase 0 run output |
| `experiments/ld50_20260504T131904Z/` | LD50 titration output |
| `src/analysis/phase0_metric_validation.py` | Percentile sampling added |
| `papers/phase0/figures_publication/generated/*` | Publication figures (4 main + supplement) |
| `papers/phase0/alife_methodological_letter_abstract.md` | ALife abstract draft |
| `docs/phase0_validation_summary.md` | Full validation summary |
| `docs/phase1_protocol.md` | Phase 1 protocol document |
Dodano 6 nowych PMCIDów (artykuły glyphosate 2025-2026) do `supplement_food_gmo.py`.
Wynik: 71 → 77/80. Pozostałe 3 niedostępne w PMC open access.
Decyzja: 77/80 akceptowalne, zanotować w Methods.

### 6. Commit i dokumentacja
- Git commit przez GitHub Desktop
- Napisano `docs/corpus_build_log_v2.md`

### Stan na koniec sesji

Korpus v2 zamrożony. Phase 0 rerun uruchomiony (config: `phase0_rerun_v2.yaml`).

| Typ | climate | vaccines | alt_med | cancer | gmo |
|-----|---------|----------|---------|--------|-----|
| food | 80 | 80 | 80 | 80 | 77 |
| toxin | 80 | 80 | 80 | 80 | 80 |
| noise | 80 | — | — | — | — |

### Obserwacje metodologiczne
Toxin Mercola jest ~3-5x dłuższy niż NaturalNews (avg 16-24k vs 5k znaków). Może wpłynąć na H(X) i C(X) — model dostaje więcej tokenów per dokument. Warto sprawdzić w wynikach Phase 0 czy efekty per domena są spójne mimo różnicy długości. Jeśli nie, konieczna normalizacja długości inputu przed Paper 1.

### Infrastructure updates
- `scripts/scrape_mercola_domain.py` — nowy scraper Mercola z `--domain` i `--max`
- `scripts/audit_corpus.py` — dodano Mercola do FILE_PRIORITY
- `dashboard.py` — live telemetria, resume-on-restart, filtr starych runów
- `src/analysis/phase0_metric_validation.py` — progressive log + `--resume`
- `docs/corpus_build_log_v2.md` — pełna dokumentacja budowy korpusu v2

### Next session priorities
1. Wyniki Phase 0 rerun — interpretacja per-domena effect sizes
2. Decyzja czy korpus jest gotowy na Paper 1
3. Style swap experiment (vaccines_mercola vs vaccines_legacy) po zamknięciu Phase 0

---

## 2026-05-02 — Session 6: Percentile chunking, Phase 0 rerun v2 results, corpus asymmetry, corpus v3 decisions

### Cel sesji
Analiza wyników Phase 0 rerun v2, podjęcie decyzji o gotowości korpusu na Paper 1.

### 1. Implementacja percentilowego chunkingu

**Diagnoza:** single-window truncation (max_length=2048) powoduje że model widzi tylko pierwsze ~1500 słów. Artykuły Mercola (16-24k znaków) były agresywnie ucinane, a dezinformacyjny "core" leży często w drugiej połowie artykułu — stąd p=0.77 dla H(X) w poprzednim runie.

**Rozwiązanie:** document-length-invariant percentile sampling. Dokument dzielony na n okien po 1/n długości, każde okno o stałym rozmiarze tokenów wycentrowane na danym percentylu. Dokumenty 5k i 25k tokenów mają identyczną reprezentację strukturalną.

Zaimplementowano w `src/analysis/phase0_metric_validation.py`:
- Funkcja `get_percentile_chunks(prompt, tokenizer, n_windows=5, window_size=512)`
- Adaptive windowing: `actual_windows = min(n_windows, max(1, doc_len // window_size))`
- Nowe metryki: `h_x_var`, `h_x_slope`, `h_dezorg_var`, `h_dezorg_slope`
- `gen_cfg` czyta z YAML configa (temperature=0.0, do_sample=False)

### 2. Wyniki Phase 0 percentile run (20260502T000503Z)

| Typ | H(X) | C(X) | I(X;seed) | H_dezorg | Fitness | N |
|-----|------|------|-----------|----------|---------|---|
| food | 4.892 | 0.391 | 0.0541 | 0.835 | -0.023 | 397 |
| toxin | 4.675 | 0.292 | 0.0462 | 0.898 | -0.069 | 484 |
| noise | 4.140 | 0.181 | 0.0397 | 0.923 | -0.110 | 80 |

Kruskal-Wallis:

| Metryka | p-value | Istotna |
|---------|---------|---------|
| h_x | 8.2e-21 | ✓ (poprzednio 0.77 — rehabilitacja entropii) |
| c_x | 4.0e-68 | ✓ |
| i_x_seed | 1.7e-07 | ✓ |
| jaccard | 0.44 | ✗ |
| h_x_var | 4.0e-13 | ✓ |
| h_x_slope | 5.5e-05 | ✓ |
| h_dezorg_var | 5.5e-25 | ✓ |
| h_dezorg_slope | 0.18 | ✗ |

Fitness ujemny to artefakt małego okna (512 tokenów) → wyższa h_dezorg. Hierarchia food > toxin > noise zachowana i istotna.

### 3. Analiza asymetrii korpusu

Rozkład długości przy progu 14000 znaków (~3500 tokenów):

| Plik | >14k znaków | Wniosek |
|------|-------------|---------|
| food_* | 97-100% | OK |
| toxin_vaccines | 100% | OK |
| toxin_gmo | 99% | OK |
| toxin_alt_med | 45% | Wymagał uzupełnienia |
| toxin_cancer | 66% | Wymagał uzupełnienia |
| toxin_climate | 0% | CARDS/PlateClimatology — za krótkie |
| noise_mixed | 0% | Fragmenty 50/50 — krótkie z definicji |

**Wniosek:** h_x_var i h_x_slope jako metryki porównawcze między typami są confoundem długości przy obecnym korpusie. Mean jest zawsze porównywalny.

### 4. Redefinicja noise

**Stara definicja:** fragmenty 50/50 food+toxin, losowo przetasowane. To jest sygnał zdegradowany, nie szum środowiskowy. Model widzi słownictwo domenowe bez kontekstu. Biologicznie: zepsuta żywność, nie tło środowiskowe.

**Nowa definicja:** semantycznie spójny tekst z domen niezwiązanych z projektem (Wikipedia — historia, geografia, kultura). Artykuły Wikipedia >3500 tokenów → pełny profil 5-chunkowy, brak confoundu długości.

### 5. Decyzje korpus v3

- **toxin_alt_med i toxin_cancer:** uzupełnione przez rozszerzenie zakresu lat scraperA Mercola (2021→2026). Wynik: min 14141 i 14093 znaków ✓
- **toxin_climate:** nowe źródło wattsupwiththat.com (Selenium scraper — requests blokowany przez rate limiting). Scraper `scripts/scrape_wuwt_selenium.py` z `--start-page`. Status: w trakcie (dwa równoległe runy, strony 1-19 i 20+).
- **noise_mixed → Wikipedia noise:** 80 artykułów z niezwiązanych domen, >3500 tokenów. Do implementacji.
- **Parametry docelowe Phase 0 run v3:** window_size=1024, n_windows=3. Pokrywa 90% toxina (od p10=3168 tokenów) przy pełnym profilu.

### Infrastructure updates
- `src/analysis/phase0_metric_validation.py` — percentile chunking, nowe metryki profilowe, `gen_cfg` z YAML
- `scripts/scrape_mercola_domain.py` — +climate domain, MIN_CHARS=14000, lata 2015-2026
- `scripts/scrape_wuwt_selenium.py` — nowy scraper WUWT przez Selenium z `--start-page`
- `scripts/scrape_joannenova.py` — zbudowany i odrzucony (artykuły <3000 znaków)
- `corpus_strategy_20260428.md` — sekcja 2 zaktualizowana: stan v2→v3
- `validation_protocol.md` — nowy plik, sekcja Percentile Sampling
- `EvoLLM_notatki_robocze.md` — kluczowe ustalenia sesji
- `EvoLLM_wnioski_20260502.md` — pełne wnioski badawcze

### Next session priorities
1. Zebrać 80 artykułów toxin_climate z WUWT (scraper w trakcie)
2. Zbudować Wikipedia noise (80 artykułów, skrypt do napisania)
3. Puścić Phase 0 run v3: window_size=1024, n_windows=3, korpus v3
4. Sprawdzić czy fitness wraca do wartości dodatnich przy window_size=1024
5. Zdokumentować Methods: percentile sampling jako document-length-invariant improvement

---

## 2026-05-04 — Session 7: LD50 titration, Phase 0 closure

### Cel sesji
Przeprowadzenie eksperymentu LD50 (gradient toksyczności), walidacja gradientowa korpusu v3, zamknięcie Phase 0.

### 1. Kanoniczny run Phase 0 v3 (phase0_metrics_20260504T082632Z)

Parametry kanoniczne:
- window_size=1024, n_windows=3
- temperature=0.0, do_sample=False
- seed_text: "Climate and vaccine discourse requires coherent, evidence-grounded synthesis."
- Korpus: v3, 880 docs, data/v2/corpus_manifest_v3.json

Metryki skuteczne (paper-ready):

| Metryka | food vs toxin p-value | effect r | Status |
|---------|-----------------------|----------|--------|
| c_x | 6.7e-18 | -0.352 | ✓ |
| h_dezorg | 1.5e-29 | +0.461 | ✓ |
| fitness | hierarchia food > toxin > noise | — | ✓ |

Metryki nieefektywne (odnotować w Methods):
- h_x: food vs toxin p=0.587 — mimikra (wynik pozytywny, potwierdzony)
- i_x_seed: food vs toxin p=0.052 — bag-of-words proxy zbyt gruby

### 2. Eksperyment LD50 (ld50_20260504T131904Z)

Protokół: 7 stężeń toksyny × N=80 dokumentów, seed=42.
Manifest: `data/ld50/ld50_corpus_manifest.json`

| Stężenie T% | Opis |
|-------------|------|
| 0% | 100% food |
| 10% | 10% toxin / 90% food |
| 25% | 25% toxin / 75% food |
| 50% | 50% toxin / 50% food |
| 75% | 75% toxin / 25% food |
| 90% | 90% toxin / 10% food |
| 100% | 100% toxin |

Korelacje Pearsona (T% vs metryka):
- c_x: r=-0.905, p=0.005
- h_dezorg: r=+0.849, p=0.016
- fitness: r=-0.921, p=0.003

**Kluczowy wynik:** Odpowiedź gradualna i liniowa, brak progu krytycznego. LD50 klasyczne nieestymowalne (brak punktu EC50) — to jest wynik, nie błąd metodologiczny. Model bazowy jest odporny na titrację — nie ma "dawki lethalnej" po której nastąpi nagły kolaps metryk.

### 3. Hipoteza diagnostyczna H_diag

h_dezorg reaguje wcześniej (przy T=50-75%), c_x degraduje się późno (T=75-100%). Implikacja: h_dezorg jest wczesnym markerem toksyczności, c_x jest późnym markerem. Do formalnej weryfikacji przez `scripts/analyze_ld50_thresholds.py` (do napisania).

### 4. Decyzje metodologiczne

**I(X;seed) — bag-of-words cosine similarity:** świadomy wybór dla przejrzystości i reprodukowalności. Zmiana na embedding-based wymagałaby rekalibracji całego pipeline. Utrzymane as-is przez wszystkie fazy. Ograniczenie odnotowane w Discussion Paper 1.

**Odporność na seed:** przypadkowo zweryfikowana — wszystkie 3 runy Phase 0 miały różne lub nieoptymalnie zdefiniowane seed_text. c_x i h_dezorg stabilne niezależnie od seed. i_x_seed słabe we wszystkich 3 runach — zgodne z oczekiwaniami.

**Zmiana terminologii:** `toxin` → `toxin` (zatruwa, nie poluje). Spójne z frameworkiem LD50, metabolic decay, dose-response. Wymaga rename plików w repo przed Phase 1.

### 5. Phase 0 — ZAMKNIĘTA

Kanoniczne runy (chronologicznie):
1. `phase0_metrics_20260501T160337Z` — single-window 2048, ZAMROŻONY jako baseline
2. `phase0_metrics_20260502T000503Z` — percentile 5×512, rehabilitacja h_x
3. `phase0_metrics_20260504T082632Z` — percentile 3×1024, KANONICZNY
4. `ld50_20260504T131904Z` — titration 7 stężeń, walidacja gradientowa

### Files updated this session
| Plik | Akcja |
|------|-------|
| `docs/validation_protocol.md` | Dodano sekcję Phase 0 — ZAMKNIĘTA |
| `corpus_strategy_20260428.md` | Dodano sekcję walidacja gradientowa |
| `EvoLLM_notatki_robocze.md` | Dodano kluczowe ustalenia sesji |
| `EvoLLM_wnioski_20260504.md` | Nowy plik — pełne wnioski badawcze |
| `config/phase0_v3.yaml` | Config kanonicznego runu v3 |

### Do zrobienia przed Phase 1
- [ ] rename `toxin_*` → `toxin_*` w repo
- [ ] `git tag phase0-final`
- [ ] `scripts/analyze_ld50_thresholds.py` — formalna weryfikacja H_diag
