# Phase 0 v2 — LD50 Dose-Response Profile: Key Observations

Date: 2026-05-08
Run: phase0_metrics_20260508T1xxxxx (LD50 v2, mi_token_ids_nmi + Seed C)

## Raw dose-response values

| T% | C(X) | H_dezorg | I(X;seed) | fitness |
|---|---|---|---|---|
| 0 | 0.3928 | 0.8483 | 0.8099 | 0.3532 |
| 10 | 0.3729 | 0.8383 | 0.7927 | 0.3405 |
| 25 | 0.3761 | 0.8196 | 0.7878 | 0.3428 |
| 50 | 0.3607 | 0.8709 | 0.7874 | 0.3277 |
| 75 | 0.3379 | 0.8825 | 0.7753 | 0.3125 |
| 90 | 0.3308 | 0.8853 | 0.7690 | 0.3067 |
| 100 | 0.3447 | 0.8938 | 0.7783 | 0.3138 |

## Statistical significance vs baseline (T=0%)

| T% | C(X) | H_dezorg | I(X;seed) | fitness |
|---|---|---|---|---|
| 10 | ns | ns | ns | ns |
| 25 | ns | ns | ns | ns |
| 50 | * | * | * | * |
| 75 | *** | *** | *** | *** |
| 90 | *** | *** | *** | *** |
| 100 | *** | *** | *** | *** |

Critical threshold: T=50% for all metrics simultaneously.

## Pearson r (concentration vs metric, n=7)

| Metryka | r | p |
|---|---|---|
| C(X) | -0.936 | 0.002 |
| H_dezorg | +0.869 | 0.011 |
| I(X;seed) | -0.887 | 0.008 |
| fitness | -0.961 | 0.001 |

## Observations

### 1. Synchronous threshold at T=50%
All three primary metrics (C(X), H_dezorg, I(X;seed)) reach statistical 
significance simultaneously at T=50%. Below this threshold, the base model 
shows no detectable response to toxin presence. This suggests a resilience 
zone (T < 50%) for single-exposure conditions without evolutionary pressure.
Implication: single-exposure informational toxicity requires majority 
contamination to produce measurable effects in a naive base model.

### 2. C(X) non-monotonic compensation at T=10-25%
C(X): 0.3928 (T=0) → 0.3729 (T=10) → 0.3761 (T=25) → monotonic decline.
Small rebound at T=25% before sustained degradation.
Hypothesis: mixed food/toxin environment (75/25) may stimulate more 
diverse generation than near-pure food (90/10), possibly because the 
model encounters conflicting informational patterns. Requires replication 
in Phase 1 under evolutionary pressure.

### 3. H_dezorg paradox at T=0-25%
H_dezorg decreases from T=0 to T=25% (0.8483 → 0.8196) before rising 
sharply at T=50% (0.8709).
Interpretation: low toxin concentration (< 25%) does not disorganize 
model output — the model may "suppress" minority signals. Only at T=50% 
does disorganization emerge as the toxin signal becomes equally weighted.
This is the inverse of the C(X) pattern: C(X) shows early sensitivity, 
H_dezorg shows late sensitivity with an initial suppression effect.

### 4. I(X;seed) and fitness: cleanest monotonic gradients
Both decrease smoothly from T=0 to T=90%, with only minor non-monotonicity 
at T=100% vs T=90%.
I(X;seed) is the most stable single-exposure marker for informational drift.

### 5. T=100% partial recovery paradox
C(X) and I(X;seed) are slightly higher at T=100% than T=90%.
Possible interpretation: pure toxin corpus produces more internally 
consistent (if factually wrong) outputs than a near-pure toxin corpus 
with small food contamination. The 10% food at T=90% may introduce 
conflicting signals that reduce output coherence.
Alternative: statistical noise given N=80 per concentration.
Requires N > 80 to resolve.

## Implications for Phase 1

These observations were made on a naive base model (no LoRA, no evolutionary 
pressure). Phase 1 will test whether:
1. The T=50% threshold shifts under multi-generational evolutionary pressure
2. The C(X) compensation at T=25% amplifies or disappears
3. The H_dezorg suppression at low concentrations persists across generations

These constitute testable predictions for Phase 1 biome experiments.
H0_supplementary: single-exposure threshold profile does not predict 
multi-generational evolutionary dynamics.

Commit: "docs: add Phase 0 v2 LD50 dose-response observations and hypotheses"
