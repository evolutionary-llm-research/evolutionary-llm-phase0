"""Global food vs. toxin statistical analysis (Phase 0).

Loads raw per-sample metrics from metrics_phase0.json, compares food (N=400)
against toxin (N=400) across all domains combined, and outputs:
  - Summary table (console + JSON)
  - Bar plot of effect sizes saved to the figures_publication directory
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import mannwhitneyu

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
METRICS = ["h_x", "c_x", "i_x_seed", "jaccard", "h_dezorg"]
METRIC_LABELS = {
    "h_x": "H(X)\nentropy",
    "c_x": "C(X)\ncomplexity",
    "i_x_seed": "I(X;seed)\nmut. info",
    "jaccard": "Jaccard\noverlap",
    "h_dezorg": "H_dezorg\ndisorg.",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rank_biserial_r(u_stat: float, n1: int, n2: int) -> float:
    """Compute rank-biserial r from Mann-Whitney U statistic.

    Parameters
    ----------
    u_stat : float
        U statistic (from scipy, corresponds to group1).
    n1, n2 : int
        Sample sizes of group1 and group2.

    Returns
    -------
    float
        Rank-biserial r in [-1, 1]; negative means group1 < group2.
    """
    return 1.0 - (2.0 * u_stat) / (n1 * n2)


def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze food vs. toxin discrimination in Phase 0 metrics."
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        required=True,
        help="Path to metrics_phase0.json file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/"),
        help="Output directory for results and figures (default: results/)",
    )
    args = parser.parse_args()
    
    metrics_file = args.metrics
    out_dir = args.output_dir
    
    # Load data
    with metrics_file.open("r", encoding="utf-8") as f:
        d = json.load(f)

    results = d["results"]
    food = [r for r in results if r["type"] == "food"]
    toxin = [r for r in results if r["type"] == "toxin"]

    print(f"food N={len(food)}, toxin N={len(toxin)}\n")

    n_food = len(food)
    n_pred = len(toxin)
    n_tests = len(METRICS)
    bonferroni_alpha = 0.05 / n_tests

    rows: list[dict] = []

    header = f"{'Metric':<14} {'p-value':>12} {'p (Bonf)':>10} {'Sig':>4} {'r':>7} {'Mean Food':>11} {'Mean Pred':>11}"
    print(header)
    print("-" * len(header))

    for metric in METRICS:
        vals_food = np.array([r[metric] for r in food], dtype=float)
        vals_pred = np.array([r[metric] for r in toxin], dtype=float)

        u_stat, p_val = mannwhitneyu(vals_food, vals_pred, alternative="two-sided")
        r = rank_biserial_r(u_stat, n_food, n_pred)
        p_bonf_sig = p_val < bonferroni_alpha

        row = {
            "metric": metric,
            "p_value": p_val,
            "p_bonferroni_sig": bool(p_bonf_sig),
            "effect_r": r,
            "mean_food": float(np.mean(vals_food)),
            "mean_toxin": float(np.mean(vals_pred)),
            "n_food": n_food,
            "n_toxin": n_pred,
        }
        rows.append(row)

        stars = sig_stars(p_val) + ("†" if p_bonf_sig else "")
        print(
            f"{metric:<14} {p_val:>12.2e} {str(p_bonf_sig):>10} {stars:>4} "
            f"{r:>7.3f} {row['mean_food']:>11.4f} {row['mean_toxin']:>11.4f}"
        )

    print(f"\nBonferroni α = {bonferroni_alpha:.4f} (n_tests={n_tests})")
    print("† = survives Bonferroni correction\n")

    # Save JSON summary
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "global_food_vs_toxin_stats.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Stats saved → {out_json}")

    # ------------------------------------------------------------------
    # Bar plot: effect sizes
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.15)

    x = np.arange(len(METRICS))
    r_vals = [row["effect_r"] for row in rows]
    colors = ["#e74c3c" if abs(r) >= 0.5 else "#3498db" for r in r_vals]

    bars = ax.bar(x, r_vals, color=colors, edgecolor="black", linewidth=0.7, width=0.55)

    # Dynamic y-limits with padding for stars
    r_max = max(r_vals)
    r_min = min(r_vals)
    y_top = max(0.65, r_max + 0.15)
    y_bot = min(-0.85, r_min - 0.12)
    ax.set_ylim(y_bot, y_top)

    # Annotate with significance stars INSIDE axes using transform-safe positions
    for bar, row in zip(bars, rows):
        stars = sig_stars(row["p_value"])
        bh = bar.get_height()
        if bh >= 0:
            ypos = bh + 0.03
            va = "bottom"
        else:
            ypos = bh - 0.04
            va = "top"
        # Clamp within ylim
        ypos = min(ypos, y_top - 0.04)
        ypos = max(ypos, y_bot + 0.02)
        ax.text(bar.get_x() + bar.get_width() / 2, ypos, stars,
                ha="center", va=va, fontsize=11, fontweight="bold", clip_on=False)

    # Reference lines — both sides (positive and negative)
    ax.axhline(0, color="black", linewidth=0.9)
    for thresh, ls, label in [
        (0.3, ":", ""), (0.5, "--", ""), (0.7, "-.", ""),
        (-0.3, ":", "small"), (-0.5, "--", "medium"), (-0.7, "-.", "large"),
    ]:
        ax.axhline(thresh, color="gray", linewidth=0.7, linestyle=ls, alpha=0.55)
        if label:
            ax.text(len(METRICS) - 0.4, thresh - 0.01, label,
                    color="gray", fontsize=7, va="top", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=9)
    ax.set_ylabel("Rank-biserial r  (food vs. toxin)", fontsize=10)
    ax.set_title(
        "Global metric discrimination: food (N=400) vs. toxin (N=400)\n"
        "Mann-Whitney U, all domains pooled",
        fontsize=10,
    )

    legend_patches = [
        mpatches.Patch(color="#e74c3c", label="|r| ≥ 0.5 (large effect)"),
        mpatches.Patch(color="#3498db", label="|r| < 0.5 (small–medium)"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="upper left")

    for fmt in ("png", "pdf", "svg"):
        out_fig = out_dir / f"figure_global_effect_sizes.{fmt}"
        fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {out_dir / 'figure_global_effect_sizes.*'}")


if __name__ == "__main__":
    main()
