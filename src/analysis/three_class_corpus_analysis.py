"""Three-class corpus discrimination analysis (Phase 0).

Compares food (N=400) vs toxin (N=400) vs noise (N=80) across all metrics.
Performs Kruskal-Wallis test + pairwise Mann-Whitney with effect sizes.

Outputs:
  - corpus_quality_v3_threeclass_stats.json
  - figures for three-class discrimination
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, kruskal

METRICS_FILE = Path(
    r"E:\github\Evolutionary LLM Research\experiments"
    r"\phase0_metrics_20260504T082632Z\metrics_phase0.json"
)
OUT_DIR = Path(r"E:\github\Evolutionary LLM Research\experiments")
METRICS = ["h_x", "c_x", "i_x_seed", "jaccard", "h_dezorg"]


def rank_biserial_r(u: float, n1: int, n2: int) -> float:
    return 1.0 - (2.0 * u) / (n1 * n2)


def main() -> None:
    with METRICS_FILE.open("r", encoding="utf-8") as f:
        d = json.load(f)

    food = [r for r in d["results"] if r["type"] == "food"]
    toxin = [r for r in d["results"] if r["type"] == "toxin"]
    noise = [r for r in d["results"] if r["type"] == "noise"]

    print(f"N: food={len(food)}, toxin={len(toxin)}, noise={len(noise)}\n")

    # Bonferroni: 5 metrics x 3 pairwise = 15 tests
    bonf_alpha = 0.05 / (len(METRICS) * 3)

    results: dict[str, dict] = {}

    print(f"{'Metric':<12} {'KW p':>12} {'Pair':>16} {'r':>7} {'p_pair':>12} {'Bonf':>6}")
    print("-" * 76)

    for metric in METRICS:
        food_vals = np.array([r[metric] for r in food], dtype=float)
        pred_vals = np.array([r[metric] for r in toxin], dtype=float)
        noise_vals = np.array([r[metric] for r in noise], dtype=float)

        # Kruskal-Wallis
        h_stat, p_kw = kruskal(food_vals, pred_vals, noise_vals)

        results[metric] = {
            "kruskal_wallis_p": p_kw,
            "pairwise": {},
            "means": {
                "food": float(np.mean(food_vals)),
                "toxin": float(np.mean(pred_vals)),
                "noise": float(np.mean(noise_vals)),
            },
        }

        for label, g1, g2, v1, v2 in [
            ("food vs toxin", "food", "toxin", food_vals, pred_vals),
            ("food vs noise", "food", "noise", food_vals, noise_vals),
            ("toxin vs noise", "toxin", "noise", pred_vals, noise_vals),
        ]:
            u, p = mannwhitneyu(v1, v2, alternative="two-sided")
            r = rank_biserial_r(u, len(v1), len(v2))
            sig = "†" if p < bonf_alpha else " "

            results[metric]["pairwise"][label] = {
                "effect_r": round(r, 6),
                "p_value": p,
                "bonf_sig": bool(p < bonf_alpha),
                "n1": len(v1),
                "n2": len(v2),
            }

            print(f"{metric:<12} {p_kw:>12.2e} {label:>16} {r:>7.3f} {p:>12.2e} {sig:>6}")

    out_json = OUT_DIR / "corpus_quality_v3_threeclass_stats.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nThree-class stats → {out_json}")


if __name__ == "__main__":
    main()
