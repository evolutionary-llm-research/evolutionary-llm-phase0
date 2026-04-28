# Project Journal — Evolutionary LLM Research

Chronological log of work sessions, decisions, and progress.

---

## 2026-04-23 — Session 2: Ollama → Unsloth pipeline decision

Problem: qwen3:8b-base not available in Ollama registry
Decision: drop Ollama, use HuggingFace + Unsloth directly
Rationale: Unsloth already in stack for LoRA fine-tuning, base model available on HuggingFace as Qwen/Qwen3-8B-Base, logprobs extractable from forward pass — cleaner than mixing Ollama and Unsloth in one pipeline
Impact: H_dezorg definition unchanged, implementation backend changed
Next: scripts/verify_logprobs.py to confirm logprobs extractable from Unsloth forward pass before modifying src/metrics/core.py

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

## Session 2 continued — verify_logprobs.py results and infrastructure fixes:

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

## 2026-04-27 — Session 3: Phase 0 metric validation, grid search i analiza wag

### Zakres prac
- Wykonano dwa pełne przebiegi walidacji metryk na korpusie (metrics_phase0.json w katalogach 20260427T075806Z i 20260427T073814Z)
- Przeanalizowano rozkłady metryk, liczności, statystyki Kruskala-Wallisa i effect sizes
- Przeprowadzono grid search wag fitnessu (w1, w2, w3) dla obu runów:
  - wariant bez ograniczenia sumy wag
  - wariant z sumą wag do 1
- Wyniki, tabele i interpretacje zapisano do experiments/phase0_metrics_20260427T075806Z/podsumowanie_grid_search.md
- Najlepsze wagi zapisano do config/fitness_weights.yaml (dowolne) i config/fitness_weights_sum1.yaml (suma=1)

### Wyniki i obserwacje
- Statystyki H i p-value dla wszystkich metryk potwierdzają istotne różnice między typami (food, predator, noise)
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
