#!/usr/bin/env python3
"""Analyze fitness function discrimination across food/toxin/noise."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import kruskal, mannwhitneyu
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS_FILE = REPO_ROOT / "experiments" / "phase0_metrics_20260504T082632Z" / "metrics_progressive.jsonl"


def rank_biserial_r(u: float, n1: int, n2: int) -> float:
    """Compute rank-biserial correlation from Mann-Whitney U statistic."""
    return 1 - (2 * u) / (n1 * n2)


def main() -> None:
    """Analyze fitness discrimination."""
    # Load all samples
    food = []
    toxin = []
    noise = []

    with open(METRICS_FILE) as f:
        for line in f:
            sample = json.loads(line)
            fitness = sample.get("fitness", np.nan)
            
            if np.isnan(fitness):
                continue
            
            sample_type = sample.get("type", "").lower()
            if sample_type == "food":
                food.append(fitness)
            elif sample_type in ("toxin", "toxin"):
                toxin.append(fitness)
            elif sample_type == "noise":
                noise.append(fitness)

    print(f"N: food={len(food)}, toxin={len(toxin)}, noise={len(noise)}\n")

    # Convert to numpy
    food_vals = np.array(food, dtype=float)
    toxin_vals = np.array(toxin, dtype=float)
    noise_vals = np.array(noise, dtype=float)

    # Descriptive statistics
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)
    print(f"Food:  mean={np.mean(food_vals):.6f}, std={np.std(food_vals):.6f}, median={np.median(food_vals):.6f}")
    print(f"Toxin: mean={np.mean(toxin_vals):.6f}, std={np.std(toxin_vals):.6f}, median={np.median(toxin_vals):.6f}")
    print(f"Noise: mean={np.mean(noise_vals):.6f}, std={np.std(noise_vals):.6f}, median={np.median(noise_vals):.6f}")
    print()

    # Kruskal-Wallis test (omnibus)
    h_stat, p_kw = kruskal(food_vals, toxin_vals, noise_vals)
    print(f"Kruskal-Wallis: H={h_stat:.4f}, p={p_kw:.4e}")
    print()

    # Pairwise Mann-Whitney comparisons
    bonf_alpha = 0.05 / 3  # 3 pairwise comparisons
    print("PAIRWISE MANN-WHITNEY U TESTS")
    print("=" * 60)
    print(f"Bonferroni α = {bonf_alpha:.4f}")
    print()

    pairs = [
        ("food vs toxin", food_vals, toxin_vals),
        ("food vs noise", food_vals, noise_vals),
        ("toxin vs noise", toxin_vals, noise_vals),
    ]

    results = {}
    for label, v1, v2 in pairs:
        u, p = mannwhitneyu(v1, v2, alternative="two-sided")
        r = rank_biserial_r(u, len(v1), len(v2))
        sig = "†" if p < bonf_alpha else ""
        print(f"{label:20s}  r={r:7.4f}, p={p:.4e} {sig}")
        results[label] = {"r": r, "p": p, "bonf_sig": p < bonf_alpha}

    print()

    # Output JSON
    output_path = REPO_ROOT / "experiments" / "fitness_discrimination_stats.json"
    output_data = {
        "test": "fitness_discrimination",
        "bonf_alpha": bonf_alpha,
        "kruskal_wallis": {"h_stat": float(h_stat), "p_value": float(p_kw)},
        "pairwise": {k: {kk: (float(vv) if isinstance(vv, (int, float, np.number)) else bool(vv)) 
                         for kk, vv in v.items()} 
                     for k, v in results.items()},
        "means": {
            "food": float(np.mean(food_vals)),
            "toxin": float(np.mean(toxin_vals)),
            "noise": float(np.mean(noise_vals)),
        },
        "medians": {
            "food": float(np.median(food_vals)),
            "toxin": float(np.median(toxin_vals)),
            "noise": float(np.median(noise_vals)),
        },
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Fitness stats → {output_path}")


if __name__ == "__main__":
    main()
