# Phase 0 Closure (2026-05-07)

## Statistical closure summary

Phase 0 is formally closed with non-parametric inference justified by distribution diagnostics.

- Shapiro-Wilk results: 14/15 groups are non-normal (p < 0.05).
- Strong non-normality: 12/15 groups have p < 1e-5.
- Only exception: H(X) for noise (N=80, p=0.87).

## Extreme distribution cases

- c_x food: W=0.162, p=1.2e-38 (clear bimodality signal in raw check).
- h_dezorg: all groups in approximately W=0.60-0.78 range (systematic non-normality).
- fitness food: W=0.255, p=4.5e-37.

These results justify non-parametric testing throughout Phase 0 analyses
(Mann-Whitney U / Kruskal-Wallis + effect sizes), with no reliance on Gaussian assumptions.

## Outlier incident and resolution

A single extreme sample was identified:

- Sample ID: FOOD_ALT_MED_0008
- Observation: c_x_Q2=21.0
- Cause: gzip header overhead on near-empty model output where h_x_Q2=0.0

Fix applied in core metric implementation:

- effective_complexity now clips ratio with min(ratio, 1.0) in core.py.

Impact assessment:

- Median impact after correction: 0.0003
- Substantive statistical conclusions: unchanged

## Bimodality interpretation

The apparent c_x food bimodality was a false alarm caused by one numerical outlier,
not by corpus structure. After correction, there is no evidence that corpus construction
introduced a genuine bimodal regime.

## Final methodological position

- Phase 0 inferential framework remains valid.
- Non-parametric workflow is retained as canonical.
- Corrected complexity clipping is part of baseline code for all reruns and later phases.
