"""Rebuild corpus quality analysis from raw metrics_phase0.json.

Computes within-domain food vs toxin comparison (N=80 each) for all 5 domains
PLUS a global comparison (N=400 each), both from the authoritative dataset
phase0_metrics_20260504T082632Z/metrics_phase0.json.

Outputs:
  experiments/corpus_quality_v2_per_domain.json  — per-domain stats
  experiments/corpus_quality_v2_global.json      — global stats
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

METRICS_FILE = Path(
    r"E:\github\Evolutionary LLM Research\experiments"
    r"\phase0_metrics_20260504T082632Z\metrics_phase0.json"
)
OUT_DIR = Path(r"E:\github\Evolutionary LLM Research\experiments")
METRICS = ["h_x", "c_x", "i_x_seed", "jaccard", "h_dezorg"]
DOMAINS = ["CLIMATE", "VACCINES", "ALT", "CANCER", "GMO"]
DOMAIN_LABELS = {"ALT": "alt_med", "CLIMATE": "climate",
                  "VACCINES": "vaccines", "CANCER": "cancer", "GMO": "gmo"}


def rank_biserial_r(u: float, n1: int, n2: int) -> float:
    return 1.0 - (2.0 * u) / (n1 * n2)


def analyse_pair(a: np.ndarray, b: np.ndarray) -> dict:
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    r = rank_biserial_r(u, len(a), len(b))
    return {
        "effect_r": round(r, 6),
        "p_value": p,
        "mean_food": float(np.mean(a)),
        "mean_toxin": float(np.mean(b)),
        "n_food": len(a),
        "n_toxin": len(b),
    }


def main() -> None:
    with METRICS_FILE.open("r", encoding="utf-8") as f:
        d = json.load(f)

    # Bin samples by (domain, type)
    bins: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in d["results"]:
        sid = r["sample_id"]
        parts = sid.split("_")
        typ = parts[0]       # FOOD / TOXIN / NOISE
        domain = parts[1]    # CLIMATE / VACCINES / ALT / CANCER / GMO / WIKIPEDIA
        bins[(domain, typ)].append(r)

    # ----------------------------------------------------------------
    # Per-domain analysis  (Bonferroni: 5 metrics × 5 domains = 25 tests)
    # ----------------------------------------------------------------
    n_tests_per_domain = len(METRICS) * len(DOMAINS)
    bonf_alpha_pd = 0.05 / n_tests_per_domain

    per_domain: dict[str, dict] = {}
    print(f"\n=== Per-domain food vs toxin (Bonferroni α={bonf_alpha_pd:.4f}, n={n_tests_per_domain}) ===")
    print(f"{'Domain':<12} {'Metric':<12} {'r':>7} {'p':>12} {'Bonf':>6} {'MeanF':>8} {'MeanP':>8}")
    print("-" * 72)

    for domain in DOMAINS:
        food_rows = bins[(domain, "FOOD")]
        pred_rows = bins[(domain, "TOXIN")]
        label = DOMAIN_LABELS[domain]
        per_domain[label] = {}

        for metric in METRICS:
            a = np.array([r[metric] for r in food_rows], dtype=float)
            b = np.array([r[metric] for r in pred_rows], dtype=float)
            stats = analyse_pair(a, b)
            stats["bonferroni_sig"] = bool(stats["p_value"] < bonf_alpha_pd)
            per_domain[label][metric] = stats
            sig = "✓" if stats["bonferroni_sig"] else " "
            print(f"{label:<12} {metric:<12} {stats['effect_r']:>7.3f} {stats['p_value']:>12.2e} "
                  f"{sig:>6} {stats['mean_food']:>8.4f} {stats['mean_toxin']:>8.4f}")

    out_pd = OUT_DIR / "corpus_quality_v2_per_domain.json"
    with out_pd.open("w", encoding="utf-8") as f:
        json.dump(per_domain, f, indent=2)
    print(f"\nPer-domain stats → {out_pd}")

    # ----------------------------------------------------------------
    # Global analysis  (Bonferroni: 5 metrics)
    # ----------------------------------------------------------------
    bonf_alpha_global = 0.05 / len(METRICS)
    food_all = [r for r in d["results"] if r["type"] == "food"]
    pred_all = [r for r in d["results"] if r["type"] == "toxin"]

    global_stats: dict[str, dict] = {}
    print(f"\n=== Global food (N={len(food_all)}) vs toxin (N={len(pred_all)}) "
          f"(Bonferroni α={bonf_alpha_global:.4f}) ===")
    print(f"{'Metric':<12} {'r':>7} {'p':>12} {'Bonf':>6} {'MeanF':>8} {'MeanP':>8}")
    print("-" * 56)

    for metric in METRICS:
        a = np.array([r[metric] for r in food_all], dtype=float)
        b = np.array([r[metric] for r in pred_all], dtype=float)
        stats = analyse_pair(a, b)
        stats["bonferroni_sig"] = bool(stats["p_value"] < bonf_alpha_global)
        global_stats[metric] = stats
        sig = "✓" if stats["bonferroni_sig"] else " "
        print(f"{metric:<12} {stats['effect_r']:>7.3f} {stats['p_value']:>12.2e} "
              f"{sig:>6} {stats['mean_food']:>8.4f} {stats['mean_toxin']:>8.4f}")

    out_global = OUT_DIR / "corpus_quality_v2_global.json"
    with out_global.open("w", encoding="utf-8") as f:
        json.dump(global_stats, f, indent=2)
    print(f"\nGlobal stats → {out_global}")


if __name__ == "__main__":
    main()
