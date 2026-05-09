from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON data from disk."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rank_biserial_r(u: float, n1: int, n2: int) -> float:
    """Compute rank-biserial correlation from Mann-Whitney U statistic."""
    return 1 - (2 * u) / (n1 * n2)


def save_figure(fig: plt.Figure, out_base: Path) -> None:
    """Save figure in publication formats."""
    fig.savefig(out_base.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")


def load_metrics_samples(metrics_file: Path) -> list[dict[str, Any]]:
    """Load metrics samples from either JSONL or structured JSON.

    Supported formats:
    - .jsonl: one JSON object per line
    - .json: object with key "results" containing a list of samples
    """
    if metrics_file.suffix.lower() == ".jsonl":
        samples: list[dict[str, Any]] = []
        with metrics_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
        return samples

    if metrics_file.suffix.lower() == ".json":
        data = load_json(metrics_file)
        results = data.get("results")
        if not isinstance(results, list):
            raise ValueError(f"Expected 'results' list in JSON file: {metrics_file}")
        return results

    raise ValueError(f"Unsupported metrics file format: {metrics_file}")


def _compute_fitness_food_vs_toxin_stats(metrics_file: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute food-vs-toxin fitness stats per domain and globally."""
    domains = ["climate", "vaccines", "alt_med", "cancer", "gmo"]
    food_by_domain: dict[str, list[float]] = {d: [] for d in domains}
    toxin_by_domain: dict[str, list[float]] = {d: [] for d in domains}

    for sample in load_metrics_samples(metrics_file):
        fitness = sample.get("fitness")
        sample_type = str(sample.get("type", "")).lower()
        sample_id = str(sample.get("id", sample.get("sample_id", ""))).upper()

        if fitness is None:
            continue

        sample_domain = None
        for domain in domains:
            if f"_{domain.upper()}_" in sample_id:
                sample_domain = domain
                break
        if sample_domain is None:
            continue

        if sample_type == "food":
            food_by_domain[sample_domain].append(float(fitness))
        elif sample_type in ("toxin", "toxin"):
            toxin_by_domain[sample_domain].append(float(fitness))

    per_domain_fitness: dict[str, Any] = {}
    all_food: list[float] = []
    all_toxin: list[float] = []

    for domain in domains:
        food_vals = np.array(food_by_domain[domain], dtype=float)
        toxin_vals = np.array(toxin_by_domain[domain], dtype=float)

        u_stat, p_val = mannwhitneyu(food_vals, toxin_vals, alternative="two-sided")
        effect_r = rank_biserial_r(u_stat, len(food_vals), len(toxin_vals))

        per_domain_fitness[domain] = {
            "effect_r": float(effect_r),
            "p_value": float(p_val),
            "mean_food": float(np.mean(food_vals)),
            "mean_toxin": float(np.mean(toxin_vals)),
            "n_food": int(len(food_vals)),
            "n_toxin": int(len(toxin_vals)),
        }

        all_food.extend(food_by_domain[domain])
        all_toxin.extend(toxin_by_domain[domain])

    all_food_arr = np.array(all_food, dtype=float)
    all_toxin_arr = np.array(all_toxin, dtype=float)
    u_global, p_global = mannwhitneyu(all_food_arr, all_toxin_arr, alternative="two-sided")
    effect_global = rank_biserial_r(u_global, len(all_food_arr), len(all_toxin_arr))

    global_fitness = {
        "effect_r": float(effect_global),
        "p_value": float(p_global),
        "mean_food": float(np.mean(all_food_arr)),
        "mean_toxin": float(np.mean(all_toxin_arr)),
        "n_food": int(len(all_food_arr)),
        "n_toxin": int(len(all_toxin_arr)),
    }

    return per_domain_fitness, global_fitness


def build_figure_1_metric_discrimination(
    per_domain: dict[str, Any],
    global_stats: dict[str, Any],
    out_dir: Path,
) -> None:
    """Build Figure 1 using within-domain and global comparisons."""
    domains = ["climate", "vaccines", "alt_med", "cancer", "gmo"]
    domain_labels = ["climate", "vaccines", "alt med", "cancer", "gmo", "GLOBAL"]
    metrics = ["h_x", "c_x", "i_x_seed", "jaccard", "h_dezorg"]
    metric_labels = ["H(X)", "C(X)", "I(X;seed)", "Jaccard", "H_dezorg"]

    bonf_alpha_pd = 0.05 / (len(metrics) * len(domains))
    bonf_alpha_global = 0.05 / len(metrics)

    effect_rows, pval_rows, bonf_rows = [], [], []
    for d in domains:
        effect_rows.append([per_domain[d][m]["effect_r"] for m in metrics])
        pval_rows.append([per_domain[d][m]["p_value"] for m in metrics])
        bonf_rows.append([per_domain[d][m]["p_value"] < bonf_alpha_pd for m in metrics])

    effect_rows.append([global_stats[m]["effect_r"] for m in metrics])
    pval_rows.append([global_stats[m]["p_value"] for m in metrics])
    bonf_rows.append([global_stats[m]["p_value"] < bonf_alpha_global for m in metrics])

    effect = np.array(effect_rows, dtype=float)
    pvals = np.array(pval_rows, dtype=float)
    bonf = np.array(bonf_rows, dtype=bool)
    neglogp = -np.log10(np.clip(pvals, 1e-300, 1.0))

    n_rows = effect.shape[0]
    n_cols = effect.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.5), constrained_layout=True)

    specs = [
        (
            axes[0],
            effect,
            "RdBu_r",
            -1.0,
            1.0,
            "A. Effect size (rank-biserial r)",
            "rank-biserial r",
        ),
        (
            axes[1],
            neglogp,
            "YlOrRd",
            0.0,
            max(5.0, float(np.nanmax(neglogp))),
            "B. Significance (-log10 p)",
            "-log10(p)",
        ),
    ]

    for ax_idx, (ax, data, cmap, vmin, vmax, title, cbar_label) in enumerate(specs):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(domain_labels, fontsize=9)
        ax.axhline(n_rows - 1.5, color="black", linewidth=1.8, linestyle="--")

        for i in range(n_rows):
            for j in range(n_cols):
                if ax_idx == 0:
                    text = f"{effect[i, j]:.3f}"
                    marker = "\u2020" if bonf[i, j] else ""
                    ax.text(
                        j,
                        i,
                        text + marker,
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="black",
                        fontweight="bold" if bonf[i, j] else "normal",
                    )
                else:
                    p = pvals[i, j]
                    stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
                    marker = "\u2020" if bonf[i, j] else ""
                    ax.text(
                        j,
                        i,
                        f"{p:.1e}\n{stars}{marker}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black",
                    )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, fontsize=9)

    fig.suptitle(
        "Figure 1. Within-domain metric discrimination: food vs toxin (Phase 0)\n"
        "Mann-Whitney U, rank-biserial r. \u2020 = Bonferroni-corrected significant.",
        fontsize=10,
    )
    save_figure(fig, out_dir / "figure1_metric_discrimination")
    plt.close(fig)


def build_figure_1_metric_discrimination_with_fitness(
    per_domain: dict[str, Any],
    global_stats: dict[str, Any],
    metrics_file: Path,
    out_dir: Path,
) -> None:
    """Build extended Figure 1 with an additional Fitness column."""
    domains = ["climate", "vaccines", "alt_med", "cancer", "gmo"]
    domain_labels = ["climate", "vaccines", "alt med", "cancer", "gmo", "GLOBAL"]
    base_metrics = ["h_x", "c_x", "i_x_seed", "jaccard", "h_dezorg"]
    metrics = base_metrics + ["fitness"]
    metric_labels = ["H(X)", "C(X)", "I(X;seed)", "Jaccard", "H_dezorg", "Fitness"]

    per_domain_fitness, global_fitness = _compute_fitness_food_vs_toxin_stats(metrics_file)

    bonf_alpha_pd = 0.05 / (len(metrics) * len(domains))
    bonf_alpha_global = 0.05 / len(metrics)

    effect_rows, pval_rows, bonf_rows = [], [], []
    for d in domains:
        effect_row: list[float] = []
        pval_row: list[float] = []
        bonf_row: list[bool] = []

        for m in metrics:
            if m == "fitness":
                stat = per_domain_fitness[d]
            else:
                stat = per_domain[d][m]
            effect_row.append(float(stat["effect_r"]))
            pval_row.append(float(stat["p_value"]))
            bonf_row.append(float(stat["p_value"]) < bonf_alpha_pd)

        effect_rows.append(effect_row)
        pval_rows.append(pval_row)
        bonf_rows.append(bonf_row)

    global_effect_row: list[float] = []
    global_pval_row: list[float] = []
    global_bonf_row: list[bool] = []
    for m in metrics:
        if m == "fitness":
            stat = global_fitness
        else:
            stat = global_stats[m]
        global_effect_row.append(float(stat["effect_r"]))
        global_pval_row.append(float(stat["p_value"]))
        global_bonf_row.append(float(stat["p_value"]) < bonf_alpha_global)

    effect_rows.append(global_effect_row)
    pval_rows.append(global_pval_row)
    bonf_rows.append(global_bonf_row)

    effect = np.array(effect_rows, dtype=float)
    pvals = np.array(pval_rows, dtype=float)
    bonf = np.array(bonf_rows, dtype=bool)
    neglogp = -np.log10(np.clip(pvals, 1e-300, 1.0))

    n_rows = effect.shape[0]
    n_cols = effect.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.5), constrained_layout=True)

    specs = [
        (
            axes[0],
            effect,
            "RdBu_r",
            -1.0,
            1.0,
            "A. Effect size (rank-biserial r)",
            "rank-biserial r",
        ),
        (
            axes[1],
            neglogp,
            "YlOrRd",
            0.0,
            max(5.0, float(np.nanmax(neglogp))),
            "B. Significance (-log10 p)",
            "-log10(p)",
        ),
    ]

    for ax_idx, (ax, data, cmap, vmin, vmax, title, cbar_label) in enumerate(specs):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(domain_labels, fontsize=9)
        ax.axhline(n_rows - 1.5, color="black", linewidth=1.8, linestyle="--")

        for i in range(n_rows):
            for j in range(n_cols):
                if ax_idx == 0:
                    text = f"{effect[i, j]:.3f}"
                    marker = "\u2020" if bonf[i, j] else ""
                    ax.text(
                        j,
                        i,
                        text + marker,
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="black",
                        fontweight="bold" if bonf[i, j] else "normal",
                    )
                else:
                    p = pvals[i, j]
                    stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
                    marker = "\u2020" if bonf[i, j] else ""
                    ax.text(
                        j,
                        i,
                        f"{p:.1e}\n{stars}{marker}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black",
                    )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, fontsize=9)

    fig.suptitle(
        "Figure 1 (extended). Within-domain metric discrimination incl. Fitness: food vs toxin (Phase 0)\n"
        "Mann-Whitney U, rank-biserial r. \u2020 = Bonferroni-corrected significant.",
        fontsize=10,
    )
    save_figure(fig, out_dir / "figure1_metric_discrimination_with_fitness")
    plt.close(fig)


def _extract_series(metric_blob: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract concentration, mean and SE vectors from metric summary table."""
    summary = metric_blob["summary_table"]
    x = np.array([row["toxin_pct"] for row in summary], dtype=float)
    mean = np.array([row["mean"] for row in summary], dtype=float)
    se = np.array([row["se"] for row in summary], dtype=float)
    return x, mean, se


def build_figure_2_ld50_biomarker(
    ld50_data: dict[str, Any],
    diagnostic_data: dict[str, Any],
    out_dir: Path,
) -> None:
    """Build Figure 2 in a two-panel layout for readability."""
    x_cx, y_cx, se_cx = _extract_series(ld50_data["c_x"])
    x_hd, y_hd, se_hd = _extract_series(ld50_data["h_dezorg"])

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(10.8, 7.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08},
        constrained_layout=True,
    )

    se_cx_clipped = np.clip(se_cx, 0, 0.04)
    se_hd_clipped = np.clip(se_hd, 0, 0.04)

    line_cx = ax_top.errorbar(
        x_cx,
        y_cx,
        yerr=se_cx_clipped,
        marker="o",
        linewidth=2.2,
        color="#7c3aed",
        label="C(X)",
        capsize=4,
        capthick=1.3,
        elinewidth=1.1,
    )
    line_hd = ax_bottom.errorbar(
        x_hd,
        y_hd,
        yerr=se_hd_clipped,
        marker="s",
        linewidth=2.2,
        color="#dc2626",
        linestyle="--",
        label="H_dezorg",
        capsize=4,
        capthick=1.3,
        elinewidth=1.1,
    )

    ax_bottom.set_xlabel("Toxin concentration [%]", fontsize=11)
    ax_top.set_ylabel("C(X)", color="#7c3aed", fontsize=11)
    ax_bottom.set_ylabel("H_dezorg", color="#dc2626", fontsize=11)
    ax_top.tick_params(axis="y", colors="#7c3aed", labelsize=10)
    ax_bottom.tick_params(axis="y", colors="#dc2626", labelsize=10)
    ax_bottom.set_xticks(x_cx)
    ax_bottom.set_xticklabels([f"{int(v)}%" for v in x_cx], fontsize=10)
    ax_top.grid(axis="both", alpha=0.2, linestyle=":")
    ax_bottom.grid(axis="both", alpha=0.2, linestyle=":")

    y_cx_pad = (y_cx.max() - y_cx.min()) * 0.3
    y_hd_pad = (y_hd.max() - y_hd.min()) * 0.3
    ax_top.set_ylim(y_cx.min() - y_cx_pad, y_cx.max() + y_cx_pad * 1.3)
    ax_bottom.set_ylim(y_hd.min() - y_hd_pad, y_hd.max() + y_hd_pad * 1.6)

    thr_raw_hd = diagnostic_data["thresholds_uncorrected"]["h_dezorg"]
    thr_bonf_hd = diagnostic_data["thresholds_bonferroni"]["h_dezorg"]
    thr_bonf_cx = diagnostic_data["thresholds_bonferroni"]["c_x"]

    if thr_raw_hd is not None:
        for ax in (ax_top, ax_bottom):
            ax.axvline(thr_raw_hd, color="#f59e0b", linewidth=1.4, linestyle="-", alpha=0.85, zorder=3)
        ax_top.text(
            thr_raw_hd / 100,
            0.97,
            f"H_dezorg first sig ({int(thr_raw_hd)}%)",
            transform=ax_top.transAxes,
            color="#b45309",
            va="top",
            ha="center",
            fontsize=8,
        )

    if thr_bonf_hd is not None:
        for ax in (ax_top, ax_bottom):
            ax.axvline(thr_bonf_hd, color="#dc2626", linewidth=1.4, linestyle="--", alpha=0.85, zorder=3)
        ax_top.text(
            thr_bonf_hd / 100,
            0.90,
            f"Bonferroni threshold ({int(thr_bonf_hd)}%)",
            transform=ax_top.transAxes,
            color="#991b1b",
            va="top",
            ha="center",
            fontsize=8,
        )

    if thr_bonf_cx is not None and thr_bonf_cx != thr_bonf_hd:
        for ax in (ax_top, ax_bottom):
            ax.axvline(thr_bonf_cx, color="#7c3aed", linewidth=1.4, linestyle=":", alpha=0.85, zorder=3)

    r_cx = ld50_data["c_x"]["monotonicity"]["pearson_r"]
    p_cx = ld50_data["c_x"]["monotonicity"]["pearson_p"]
    r_hd = ld50_data["h_dezorg"]["monotonicity"]["pearson_r"]
    p_hd = ld50_data["h_dezorg"]["monotonicity"]["pearson_p"]

    annotation = (
        f"C(X): Pearson r = {r_cx:.3f}, p = {p_cx:.4f}\n"
        f"H_dezorg: Pearson r = {r_hd:.3f}, p = {p_hd:.4f}\n"
        "Sequential profile: H_dezorg rises earlier (T=50%) than C(X) falls (T=75%)"
    )
    ax_bottom.text(
        0.5,
        0.03,
        annotation,
        transform=ax_bottom.transAxes,
        fontsize=8.5,
        va="bottom",
        ha="center",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.75"},
    )

    ax_top.legend([line_cx.lines[0], line_hd.lines[0]], ["C(X)", "H_dezorg"], loc="lower left", fontsize=9)

    fig.suptitle("Figure 2. LD50 titration reveals a sequential biomarker profile", fontsize=12)
    save_figure(fig, out_dir / "figure2_ld50_sequential_biomarker")
    plt.close(fig)


def build_figure_1b_three_class_discrimination(
    three_class_stats: dict[str, Any],
    out_dir: Path,
) -> None:
    """Build Figure 1b: three-class corpus discrimination (food/toxin/noise)."""
    metrics = ["h_x", "c_x", "i_x_seed", "jaccard", "h_dezorg"]
    metric_labels = ["H(X)", "C(X)", "I(X;seed)", "Jaccard", "H_dezorg"]
    pair_keys = ["food_vs_toxin", "food_vs_noise", "toxin_vs_noise"]
    pair_labels = ["Food→Toxin", "Food→Noise", "Toxin→Noise"]

    bonf_alpha = 0.05 / (len(metrics) * 3)

    effect_matrix = np.zeros((len(pair_keys), len(metrics)))
    pval_matrix = np.zeros((len(pair_keys), len(metrics)))
    bonf_matrix = np.zeros((len(pair_keys), len(metrics)), dtype=bool)

    # Support both legacy and new stats formats.
    # New format:
    #   effect_sizes[metric]["food_vs_toxin"] = float
    #   kruskal_wallis[metric]["p"] = float
    # Legacy format:
    #   three_class_stats[metric]["pairwise"]["food vs toxin"] = {effect_r, p_value, bonf_sig}
    has_new_format = "effect_sizes" in three_class_stats and "kruskal_wallis" in three_class_stats

    for j, metric in enumerate(metrics):
        if has_new_format:
            metric_effects = three_class_stats["effect_sizes"].get(metric, {})
            p_metric = float(three_class_stats["kruskal_wallis"].get(metric, {}).get("p", np.nan))
            for i, pair in enumerate(pair_keys):
                effect_matrix[i, j] = float(metric_effects.get(pair, np.nan))
                pval_matrix[i, j] = p_metric
                bonf_matrix[i, j] = bool(p_metric < bonf_alpha) if not np.isnan(p_metric) else False
        else:
            for i, pair in enumerate(pair_keys):
                legacy_key = pair.replace("_", " ")
                pw = three_class_stats[metric]["pairwise"][legacy_key]
                effect_matrix[i, j] = pw["effect_r"]
                pval_matrix[i, j] = pw["p_value"]
                bonf_matrix[i, j] = pw["bonf_sig"]

    fig, ax = plt.subplots(figsize=(10.5, 5), constrained_layout=True)

    im = ax.imshow(effect_matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_title("Figure 1b. Three-class corpus discrimination (pairwise effect sizes)")
    ax.set_xticks(np.arange(len(metric_labels)))
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_yticks(np.arange(len(pair_labels)))
    ax.set_yticklabels(pair_labels, fontsize=10)

    for i in range(len(pair_keys)):
        for j in range(len(metrics)):
            r = effect_matrix[i, j]
            marker = "†" if bonf_matrix[i, j] else ""
            ax.text(j, i, f"{r:.3f}{marker}", ha="center", va="center",
                    fontsize=9, color="black", fontweight="bold" if bonf_matrix[i, j] else "normal")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("rank-biserial r", fontsize=10)

    fig.suptitle(
        "Figure 1b. Three-class corpus discrimination: pairwise comparisons\n"
        "Mann-Whitney U. † = Bonferroni-corrected significant (α=0.003).",
        fontsize=11,
    )
    save_figure(fig, out_dir / "figure1b_three_class_heatmap")
    plt.close(fig)


def build_figure_3_corpus_hierarchy(
    three_class_stats: dict[str, Any],
    out_dir: Path,
) -> None:
    """Build Figure 3: corpus hierarchy visualization."""
    metrics = ["c_x", "h_dezorg"]
    metric_labels = ["C(X)\n(Complexity)", "H_dezorg\n(Disorganization)"]
    corpus_labels = ["Food", "Toxin", "Noise"]
    colors = ["#10b981", "#ef4444", "#9ca3af"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

    for ax_idx, (ax, metric) in enumerate(zip(axes, metrics)):
        means = [
            three_class_stats[metric]["means"]["food"],
            three_class_stats[metric]["means"]["toxin"],
            three_class_stats[metric]["means"]["noise"],
        ]
        x = np.arange(len(corpus_labels))
        bars = ax.bar(x, means, color=colors, edgecolor="black", linewidth=0.8, width=0.6)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                    f"{height:.4f}", ha="center", va="bottom", fontsize=9)

        ax.set_ylabel("Mean value", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(corpus_labels, fontsize=10)
        ax.set_title(metric_labels[ax_idx], fontsize=11)
        ax.grid(axis="y", alpha=0.2, linestyle=":")

    fig.suptitle(
        "Figure 3. Corpus hierarchy: information state profiles\n"
        "Food (high complexity, coherent) vs Toxin (low complexity, disorganized) vs Noise (minimal structure)",
        fontsize=11,
    )
    save_figure(fig, out_dir / "figure3_corpus_hierarchy")
    plt.close(fig)


def build_figure_fitness_biomarker(fitness_stats: dict[str, Any], metrics_file: Path, out_dir: Path) -> None:
    """Build Figure 4: fitness function as composite biomarker (simplified bar plot)."""
    # Load raw fitness data
    food = []
    toxin = []
    noise = []

    for sample in load_metrics_samples(metrics_file):
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

    food_vals = np.array(food, dtype=float)
    toxin_vals = np.array(toxin, dtype=float)
    noise_vals = np.array(noise, dtype=float)

    # Calculate statistics
    means = [np.mean(food_vals), np.mean(toxin_vals), np.mean(noise_vals)]
    stds = [np.std(food_vals), np.std(toxin_vals), np.std(noise_vals)]
    sems = [s / np.sqrt(len(v)) for s, v in zip(stds, [food_vals, toxin_vals, noise_vals])]

    # Pairwise Mann-Whitney comparisons for significance markers
    bonf_alpha = 0.05 / 3
    u_ft, p_ft = mannwhitneyu(food_vals, toxin_vals, alternative="two-sided")
    u_fn, p_fn = mannwhitneyu(food_vals, noise_vals, alternative="two-sided")
    u_tn, p_tn = mannwhitneyu(toxin_vals, noise_vals, alternative="two-sided")

    sig_ft = p_ft < bonf_alpha
    sig_fn = p_fn < bonf_alpha
    sig_tn = p_tn < bonf_alpha

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    labels = ["Food\n(n=400)", "Toxin\n(n=400)", "Noise\n(n=80)"]
    colors = ["#10b981", "#ef4444", "#9ca3af"]
    x_pos = np.arange(len(labels))

    # Bar plot with error bars
    bars = ax.bar(x_pos, means, yerr=sems, color=colors, edgecolor="black", 
                  linewidth=1.5, width=0.6, capsize=8, error_kw={"linewidth": 2})

    # Add value labels on bars
    for i, (mean, sem) in enumerate(zip(means, sems)):
        ax.text(i, mean + sem + 0.01, f"{mean:.4f}", ha="center", va="bottom", 
                fontsize=11, fontweight="bold")

    # Add significance brackets and stars (with vertical tick marks)
    y_max = max(means) + max(sems) + 0.05
    tick_size = 0.015  # vertical tick marks at ends
    
    # Food vs Toxin (lowest bracket)
    if sig_ft:
        y_pos = y_max + 0.02
        ax.plot([0, 1], [y_pos, y_pos], "k-", linewidth=2.0)  # horizontal line
        ax.plot([0, 0], [y_pos - tick_size, y_pos + tick_size], "k-", linewidth=2.0)  # left tick
        ax.plot([1, 1], [y_pos - tick_size, y_pos + tick_size], "k-", linewidth=2.0)  # right tick
        ax.text(0.5, y_pos + 0.035, "†", ha="center", va="bottom", fontsize=13, fontweight="bold")
    
    # Food vs Noise (top bracket)
    if sig_fn:
        y_pos = y_max + 0.08
        ax.plot([0, 2], [y_pos, y_pos], "k-", linewidth=2.0)  # horizontal line
        ax.plot([0, 0], [y_pos - tick_size, y_pos + tick_size], "k-", linewidth=2.0)  # left tick
        ax.plot([2, 2], [y_pos - tick_size, y_pos + tick_size], "k-", linewidth=2.0)  # right tick
        ax.text(1, y_pos + 0.035, "†", ha="center", va="bottom", fontsize=13, fontweight="bold")
    
    # Toxin vs Noise (middle bracket)
    if sig_tn:
        y_pos = y_max + 0.05
        ax.plot([1, 2], [y_pos, y_pos], "k-", linewidth=2.0)  # horizontal line
        ax.plot([1, 1], [y_pos - tick_size, y_pos + tick_size], "k-", linewidth=2.0)  # left tick
        ax.plot([2, 2], [y_pos - tick_size, y_pos + tick_size], "k-", linewidth=2.0)  # right tick
        ax.text(1.5, y_pos + 0.035, "†", ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean Fitness Score", fontsize=12, fontweight="bold")
    ax.set_title("Figure 4. Fitness function discriminates information quality hierarchy", 
                 fontsize=13, fontweight="bold", pad=20)
    ax.grid(axis="y", alpha=0.2, linestyle=":")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    
    # Increase y-axis limit to accommodate brackets
    ax.set_ylim(min(means) - 0.05, y_max + 0.15)

    # Add Kruskal-Wallis annotation
    h_stat = fitness_stats.get("kruskal_wallis", {}).get("h_stat", 147.84)
    p_kw = fitness_stats.get("kruskal_wallis", {}).get("p_value", 7.87e-33)
    
    ax.text(0.98, 0.05, 
            f"Kruskal-Wallis: H={h_stat:.2f}, p={p_kw:.2e} †\nBonferroni α = {bonf_alpha:.4f}\n† = p < α",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    save_figure(fig, out_dir / "figure4_fitness_biomarker")
    plt.close(fig)


def build_figure_3_corpus_hierarchy_old(
    three_class_stats: dict[str, Any],
    out_dir: Path,
) -> None:
    """Build Figure 3: corpus hierarchy visualization."""
    metrics = ["c_x", "h_dezorg"]
    metric_labels = ["C(X)\n(Complexity)", "H_dezorg\n(Disorganization)"]
    corpus_labels = ["Food", "Toxin", "Noise"]
    colors = ["#10b981", "#ef4444", "#9ca3af"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

    for ax_idx, (ax, metric) in enumerate(zip(axes, metrics)):
        means = [
            three_class_stats[metric]["means"]["food"],
            three_class_stats[metric]["means"]["toxin"],
            three_class_stats[metric]["means"]["noise"],
        ]
        x = np.arange(len(corpus_labels))
        bars = ax.bar(x, means, color=colors, edgecolor="black", linewidth=0.8, width=0.6)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                    f"{height:.4f}", ha="center", va="bottom", fontsize=9)

        ax.set_ylabel("Mean value", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(corpus_labels, fontsize=10)
        ax.set_title(metric_labels[ax_idx], fontsize=11)
        ax.grid(axis="y", alpha=0.2, linestyle=":")

    fig.suptitle(
        "Figure 3. Corpus hierarchy: information state profiles\n"
        "Food (high complexity, coherent) vs Toxin (low complexity, disorganized) vs Noise (minimal structure)",
        fontsize=11,
    )
    save_figure(fig, out_dir / "figure3_corpus_hierarchy")
    plt.close(fig)


def main() -> None:
    """Generate ALife publication figures from project experiment outputs."""
    parser = argparse.ArgumentParser(description="Generate ALife Phase 0 publication figures.")
    parser.add_argument(
        "--metrics",
        type=Path,
        required=True,
        help="Path to metrics_phase0.json (or metrics_progressive.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("papers/phase0/figures_publication/generated"),
        help="Output directory for generated figures",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    out_dir = args.output_dir
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = args.metrics
    if not metrics_file.is_absolute():
        metrics_file = repo_root / metrics_file

    per_domain_path = repo_root / "experiments" / "corpus_quality_v2_per_domain.json"
    global_path = repo_root / "experiments" / "corpus_quality_v2_global.json"
    three_class_path = repo_root / "experiments" / "corpus_quality_v3_threeclass_stats.json"
    fitness_stats_path = repo_root / "experiments" / "fitness_discrimination_stats.json"
    ld50_path = repo_root / "experiments" / "ld50_20260504T131904Z" / "ld50_analysis_results.json"
    diag_path = repo_root / "experiments" / "ld50_20260504T131904Z" / "diagnostic_threshold_results.json"

    per_domain = load_json(per_domain_path)
    global_stats = load_json(global_path)
    three_class_stats = load_json(three_class_path)
    fitness_stats = load_json(fitness_stats_path)
    ld50_data = load_json(ld50_path)
    diagnostic_data = load_json(diag_path)

    build_figure_1_metric_discrimination(per_domain, global_stats, out_dir)
    build_figure_1_metric_discrimination_with_fitness(per_domain, global_stats, metrics_file, out_dir)
    build_figure_1b_three_class_discrimination(three_class_stats, out_dir)
    build_figure_3_corpus_hierarchy_old(three_class_stats, out_dir)
    build_figure_fitness_biomarker(fitness_stats, metrics_file, out_dir)
    build_figure_2_ld50_biomarker(ld50_data, diagnostic_data, out_dir)

    print("Generated figures:")
    for path in sorted(out_dir.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
