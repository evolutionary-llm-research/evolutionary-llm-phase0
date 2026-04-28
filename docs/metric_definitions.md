# Metric Definitions

**Project:** Evolutionary LLM Research  
**Status:** Locked — definitions registered before Phase 0 data collection (April 2026)  
**Language note:** This file is technical documentation; English only.

> These definitions are operationally fixed. Any post-hoc change requires full justification
> and new data. See `docs/DECISIONS.md` for rationale log.

---

## 1. H(X) — Shannon Entropy

**What it measures:** Predictability of token output. High H = diverse, unpredictable output.
Low H = repetitive, narrow output.

**Input:** Tokenized model output using the base model vocabulary (qwen3:8b-base).

**Formula:**

```
H(X) = -Σ p(t) * log2(p(t))    [sum over unique tokens t in output]
```

**Bias correction (Miller-Madow):**

```python
H_corrected = H_empirical + (k - 1) / (2 * N)
# k = number of unique tokens in output
# N = total sequence length (tokens)
```

Always report `H_corrected`. Retain `H_empirical` for diagnostics.

**Implementation:**

```python
from collections import Counter
import numpy as np

def shannon_entropy(tokens: list[str]) -> tuple[float, float]:
    """Returns (H_empirical, H_corrected)."""
    N = len(tokens)
    counts = Counter(tokens)
    k = len(counts)
    probs = np.array(list(counts.values())) / N
    H_emp = -np.sum(probs * np.log2(probs + 1e-12))
    H_corr = H_emp + (k - 1) / (2 * N)
    return H_emp, H_corr
```

---

## 2. C(X) — Effective Complexity

**What it measures:** Amount of compressible structure (regularity) in output.
High C = structured, patterned output. Low C = either random or maximally repetitive.

**Operationalization:** gzip compression ratio of UTF-8 encoded output text.

**Formula:**

```
C(X) = len(gzip(output_bytes)) / len(output_bytes)
```

Range: (0, 1]. Values near 1 indicate low compressibility (random or highly varied).
Values near 0 indicate high compressibility (repetitive). Effective complexity is
maximized at intermediate values.

**Implementation:**

```python
import gzip

def effective_complexity(text: str) -> float:
    """Gzip compression ratio as proxy for effective complexity."""
    encoded = text.encode("utf-8")
    compressed = gzip.compress(encoded, compresslevel=9)
    return len(compressed) / len(encoded)
```

**Known limitation:** This is a heuristic approximation to the Gell-Mann & Lloyd (1996)
definition. It does not formally separate regularity from randomness; very short outputs
will have artificially high ratios due to gzip header overhead (minimum ~20 bytes).
Minimum output length for reliable C(X): 200 tokens. Outputs below this threshold are
excluded from analysis. A Minimum Description Length formulation is planned as a
methodological extension in Paper 1 Discussion.

---

## 3. I(X;Y) — Mutual Information with Seed

**What it measures:** Informational continuity between a descendant output (X) and the
seed model output (Y) on the same prompt. High I = descendant retained token-level
patterns of the seed. Low I = descendant diverged.

**Operationalization:** Token-level mutual information via scikit-learn
`mutual_info_score`, operating on discrete token frequency distributions over the shared
base model vocabulary.

**Formula:**

```
I(X;Y) = Σ Σ p(x,y) * log[ p(x,y) / (p(x) * p(y)) ]
          x  y
```

**Implementation:**

```python
from sklearn.metrics import mutual_info_score
import numpy as np

def mutual_info_with_seed(
    descendant_tokens: list[str],
    seed_tokens: list[str],
    vocab: list[str]
) -> float:
    """Token-level MI between descendant and seed output."""
    token_to_idx = {t: i for i, t in enumerate(vocab)}
    desc_ids = [token_to_idx.get(t, 0) for t in descendant_tokens]
    seed_ids = [token_to_idx.get(t, 0) for t in seed_tokens]
    # Align lengths by truncating to shorter sequence
    min_len = min(len(desc_ids), len(seed_ids))
    return mutual_info_score(desc_ids[:min_len], seed_ids[:min_len])
```

**Fallback (KSG estimator):** If Phase 0 validation shows that token-level MI fails
to discriminate document types (i.e., Kruskal-Wallis p > 0.05 on delta_I across
pokarm/drapieznik/szum groups), replace with KSG estimator on last-layer embeddings:

```python
# Fallback — activate only if Phase 0 MI discrimination fails
# Requires: pip install knncmi
from knncmi import cmi
# embedding_descendant, embedding_seed: np.ndarray shape (N, d)
I_ksg = cmi(embedding_descendant, embedding_seed, k=5)
```

Document which estimator was used in Methods. Do not switch mid-experiment.

---

## 4. H_dezorg — Disorganization Entropy

**What it measures:** Coherence degradation relative to seed. Operationalized as perplexity of the unmodified seed model on descendant output. High H_dezorg = descendant has drifted far from seed generative patterns. Low H_dezorg = descendant remains close to seed.

**Formula:**

```
H_dezorg = exp( -(1/N) * Σ log p_seed(t_i | t_1, ..., t_{i-1}) )
                           i=1..N
```

where `p_seed` is the conditional token probability assigned by the base model (Qwen/Qwen3-8B-Base, frozen, no LoRA), and the sum runs over all N tokens of the descendant output.

This is standard perplexity with the seed as the reference language model.

**Implementation (via Unsloth FastLanguageModel):**

```python
import numpy as np

def disorganization_entropy(
    descendant_text: str,
    seed_model_logprobs: list[float]  # log p(t_i | context) from seed model
) -> float:
    """
    Perplexity of seed model on descendant output.
    seed_model_logprobs: list of per-token log-probabilities (natural log)
                         returned by Unsloth FastLanguageModel forward pass on descendant_text.
    """
    N = len(seed_model_logprobs)
    avg_nll = -np.mean(seed_model_logprobs)
    return np.exp(avg_nll)
```

**Note on implementation:** Log-probabilities are extracted via Unsloth FastLanguageModel forward pass on the frozen seed model (Qwen/Qwen3-8B-Base, load_in_4bit=True). Perplexity computed as exp(-sum(logprobs) / n_tokens).

---

## 5. Fitness Function

**Formula:**

```
fitness = w1 * C(X) + w2 * I(X; seed) - w3 * H_dezorg
```

**Weight calibration procedure:**

Weights w1, w2, w3 are not fixed a priori. They are calibrated once in Phase 0 and
then frozen for all subsequent phases.

| Parameter | Value |
|-----------|-------|
| Search grid | 0.1 to 1.0, step 0.1, per weight |
| Constraint | w1 + w2 + w3 = 1.0 (normalized) |
| Objective | Maximize Kruskal-Wallis H-statistic across document types |
| Holdout | 20% per document type, randomly sampled before grid search |
| Validation | Grid search on 80% training set; evaluate objective on holdout |

**Important:** Grid search is performed on the training split only. The holdout is
used exclusively to validate that selected weights generalize. Do not re-run grid
search after inspecting holdout results.

Calibrated weights must be reported in the Results section of each paper and committed
to `config/fitness_weights.yaml` before Phase 1 begins.

```yaml
# config/fitness_weights.yaml — populated after Phase 0 calibration
fitness_weights:
  w1_complexity: null      # fill after calibration
  w2_mutual_info: null     # fill after calibration
  w3_disorganization: null # fill after calibration
  calibration_date: null
  calibration_n_per_type: null
  holdout_kw_h_stat: null
```

---

## 6. Summary Table

| Metric | Measures | Range | High value means | Library |
|--------|----------|-------|-----------------|---------|
| H(X) | Output predictability | [0, log2(V)] | Diverse, unpredictable output | numpy |
| C(X) | Structural regularity | (0, 1] | Low compressibility (complex or random) | gzip |
| I(X;Y) | Continuity with seed | [0, ∞) | Output retains seed patterns | sklearn |
| H_dezorg | Coherence degradation | [1, ∞) | Drift from seed, disorganized output | Unsloth logprobs |
| fitness | Adaptive value | unbounded | High complexity + continuity, low drift | computed |

---

## 7. Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04 | C(X) = gzip compression ratio | Reproducible, no external dependencies, defensible to reviewers; MDL planned as extension |
| 2026-04-23 | H_dezorg = seed perplexity on descendant (Unsloth) | Ollama dropped: qwen3:8b-base not available; Unsloth logprobs from Qwen/Qwen3-8B-Base used |
| 2026-04 | I(X;Y) = token-level mutual_info_score | Fast, no distributional assumptions, shared vocabulary available; KSG fallback pre-registered |
| 2026-04 | Fitness weights calibrated via grid search with 20% holdout | Prevents overfitting of fitness function to Phase 0 data |

*Update this log for any deviation from the above definitions, with full justification.*
