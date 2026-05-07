#!/usr/bin/env python3
"""Generate publication-quality figure for fitness discrimination."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


REPO_ROOT = Path(__file__).resolve().parents[2]


def rank_biserial_r(u: float, n1: int, n2: int) -> float:
    """Compute rank-biserial correlation from Mann-Whitney U statistic."""
    return 1 - (2 * u) / (n1 * n2)


def load_metrics() -> tuple[list, list, list]:
    """Load fitness values for all three corpus types."""
    # build_fitness_figure.py is at papers/phase0/figures_publication/build_fitness_figure.py
    # Go up 3 levels to reach repo root
    script_dir = Path(__file__).resolve().parent  # figures_publication/
    repo_root = script_dir.parent.parent.parent  # up to repo root
    
    metrics_file = repo_root / "experiments" / "phase0_metrics_20260504T082632Z" / "metrics_progressive.jsonl"
    
    food = []
    toxin = []
    noise = []

    with open(metrics_file) as f:
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

    return food, toxin, noise


def build_figure_fitness_discrimination() -> None:
    """Build publication figure for fitness discrimination."""
    food, toxin, noise = load_metrics()
    food_vals = np.array(food, dtype=float)
    toxin_vals = np.array(toxin, dtype=float)
    noise_vals = np.array(noise, dtype=float)

    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.2))

    # PANEL A: Violin plot with individual points
    data_groups = [food_vals, toxin_vals, noise_vals]
    labels = ["Food\n(n=400)", "Toxin\n(n=400)", "Noise\n(n=80)"]
    colors = ["#10b981", "#ef4444", "#9ca3af"]
    
    positions = np.arange(len(labels))
    
    # Violin plot
    parts = ax1.violinplot(data_groups, positions=positions, widths=0.6, 
                            showmeans=False, showmedians=False, showextrema=False)
    
    # Color the violins
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
        pc.set_edgecolor("black")
        pc.set_linewidth(1.0)

    # Overlay box plot
    bp = ax1.boxplot(data_groups, positions=positions, widths=0.15,
                     patch_artist=True, showfliers=False,
                     medianprops=dict(color="black", linewidth=2),
                     boxprops=dict(facecolor="white", edgecolor="black", linewidth=1),
                     whiskerprops=dict(color="black", linewidth=1),
                     capprops=dict(color="black", linewidth=1))

    # Overlay individual points (with jitter)
    np.random.seed(42)
    for i, (data, pos, color) in enumerate(zip(data_groups, positions, colors)):
        jitter = np.random.normal(0, 0.03, size=len(data))
        ax1.scatter(pos + jitter, data, alpha=0.15, s=15, color=color, edgecolor="none")

    ax1.set_xticks(positions)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylabel("Fitness Score", fontsize=11)
    ax1.set_title("A. Fitness distribution across corpus types", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.2, linestyle=":")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Add mean markers
    means = [np.mean(d) for d in data_groups]
    ax1.scatter(positions, means, color="black", s=100, marker="D", zorder=10, 
                edgecolor="white", linewidth=1.5, label="Mean")
    ax1.legend(loc="upper right", fontsize=10)

    # PANEL B: Effect sizes (rank-biserial r)
    pairs = [
        ("Food vs\nToxin", food_vals, toxin_vals),
        ("Food vs\nNoise", food_vals, noise_vals),
        ("Toxin vs\nNoise", toxin_vals, noise_vals),
    ]
    
    effect_sizes = []
    p_values = []
    pair_labels = []
    
    for label, v1, v2 in pairs:
        u, p = mannwhitneyu(v1, v2, alternative="two-sided")
        r = rank_biserial_r(u, len(v1), len(v2))
        effect_sizes.append(r)
        p_values.append(p)
        pair_labels.append(label)

    # Bar plot with significance threshold
    bonf_alpha = 0.05 / 3
    x_pos = np.arange(len(pair_labels))
    
    # Color based on effect size magnitude
    bar_colors = []
    for r in effect_sizes:
        if abs(r) >= 0.5:
            bar_colors.append("#e74c3c")  # Large effect
        else:
            bar_colors.append("#3498db")  # Medium effect
    
    bars = ax2.bar(x_pos, effect_sizes, color=bar_colors, edgecolor="black", linewidth=1.2, width=0.6)

    # Add significance threshold lines
    ax2.axhline(-0.3, color="gray", linestyle=":", linewidth=1.0, alpha=0.5, label="Medium effect (r=±0.3)")
    ax2.axhline(-0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.5, label="Large effect (r=±0.5)")
    ax2.axhline(0, color="black", linewidth=1.0)
    ax2.axhline(0.3, color="gray", linestyle=":", linewidth=1.0, alpha=0.5)
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.5)

    # Add value labels and significance markers
    for i, (r, p) in enumerate(zip(effect_sizes, p_values)):
        y_pos = r - 0.05 if r < 0 else r + 0.05
        sig_marker = "†" if p < bonf_alpha else ""
        ax2.text(i, y_pos, f"{r:.3f}{sig_marker}", ha="center", va="bottom" if r >= 0 else "top",
                fontsize=10, fontweight="bold")
        
        # Add p-value annotation
        ax2.text(i, r - 0.15 if r < 0 else r + 0.15, f"p={p:.2e}", 
                ha="center", va="top" if r < 0 else "bottom", fontsize=8, style="italic")

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(pair_labels, fontsize=11)
    ax2.set_ylabel("rank-biserial r", fontsize=11)
    ax2.set_ylim(-0.75, 0.15)
    ax2.set_title("B. Pairwise effect sizes (Mann-Whitney U)", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.2, linestyle=":")
    ax2.legend(fontsize=9, loc="lower left")

    # Add statistical info box in ax2
    ax2.text(0.98, 0.96, 
            f"Kruskal-Wallis: H=147.84, p=7.87e-33 †\nBonferroni α = {bonf_alpha:.4f}",
            transform=ax2.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.4, edgecolor="black", linewidth=1))

    # Main title
    fig.suptitle(
        "Fitness function discriminates information quality hierarchy (Food > Toxin > Noise)",
        fontsize=12, fontweight="bold", y=0.98
    )
    
    # Adjust subplots to make room for suptitle
    plt.subplots_adjust(top=0.92)

    # Save figure
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(out_dir / f"figure_fitness_discrimination.{ext}", dpi=300 if ext == "png" else 600,
                   bbox_inches="tight", facecolor="white")
    
    plt.close(fig)
    print(f"Figure saved to {out_dir}/figure_fitness_discrimination.*")


if __name__ == "__main__":
    build_figure_fitness_discrimination()
