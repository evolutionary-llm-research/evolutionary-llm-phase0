#!/usr/bin/env python3
"""
analyze_ld50_thresholds.py
==========================
Test H_diag: do h_dezorg and c_x reach statistical significance
at different toxin concentrations (sequential diagnostic profile)?

For each metric and each concentration T > 0%:
  Mann-Whitney U test: metric values at T% vs metric values at T=0%.
  Find lowest T where p < 0.05 (Bonferroni-corrected and uncorrected).

Output:
  - Threshold table printed to stdout
  - JSON results
  - Figure: p-value curves per metric vs concentration

Usage:
    python src/analysis/analyze_ld50_thresholds.py \
        --run-dir experiments/ld50_20260504T131904Z \
        [--alpha 0.05]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CONCENTRATIONS = [0, 10, 25, 50, 75, 90, 100]
METRICS = ["c_x", "h_dezorg", "fitness", "i_x_seed"]

METRIC_LABELS = {
    "c_x":      "C(X) effective complexity",
    "h_dezorg": "H_dezorg disorganization entropy",
    "fitness":  "Fitness",
    "i_x_seed": "I(X;seed)",
}

EXPECTED_DIRECTION = {
    "c_x":      "decrease",   # food > toxin → should drop with T
    "h_dezorg": "increase",   # food < toxin → should rise with T
    "fitness":  "decrease",
    "i_x_seed": "decrease",
}


# ---------------------------------------------------------------------------
# Data loading (same logic as analyze_ld50.py)
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_concentration(run_dir: Path, toxin_pct: int) -> list[dict] | None:
    conc_dir = run_dir / f"concentration_{toxin_pct:03d}"
    if conc_dir.exists():
        for subdir in sorted(conc_dir.iterdir()):
            if subdir.is_dir() and subdir.name.startswith("phase0_metrics_"):
                jsonl = subdir / "metrics_progressive.jsonl"
                if jsonl.exists():
                    return load_jsonl(jsonl)
    log.warning(f"No data for T={toxin_pct}%")
    return None


def load_all_concentrations(run_dir: Path) -> dict[int, list[dict]]:
    data = {}
    for t in CONCENTRATIONS:
        records = load_concentration(run_dir, t)
        if records:
            data[t] = records
            log.info(f"T={t:3d}%: {len(records)} records")
    return data


# ---------------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------------

def extract_values(records: list[dict], metric: str) -> np.ndarray:
    vals = [r[metric] for r in records if metric in r and np.isfinite(r[metric])]
    return np.array(vals, dtype=float)


def run_threshold_analysis(
    data: dict[int, list[dict]],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    For each metric × concentration: Mann-Whitney vs T=0%.
    Returns DataFrame with columns:
      metric, concentration, n_t0, n_t, U, p_raw, p_bonf,
      significant_raw, significant_bonf,
      direction_correct, median_t0, median_t
    """
    if 0 not in data:
        raise ValueError("No T=0% data found.")

    n_comparisons = len(CONCENTRATIONS) - 1  # T=10..100

    rows = []
    for metric in METRICS:
        vals_t0 = extract_values(data[0], metric)
        expected_dir = EXPECTED_DIRECTION[metric]

        for t in CONCENTRATIONS[1:]:
            if t not in data:
                continue
            vals_t = extract_values(data[t], metric)

            # Mann-Whitney — two-sided
            U, p_raw = mannwhitneyu(vals_t0, vals_t, alternative="two-sided")

            p_bonf = min(p_raw * n_comparisons, 1.0)

            median_t0 = float(np.median(vals_t0))
            median_t = float(np.median(vals_t))
            delta = median_t - median_t0

            if expected_dir == "decrease":
                direction_correct = delta < 0
            else:
                direction_correct = delta > 0

            rows.append({
                "metric": metric,
                "concentration": t,
                "n_t0": len(vals_t0),
                "n_t": len(vals_t),
                "U": float(U),
                "p_raw": float(p_raw),
                "p_bonf": float(p_bonf),
                "significant_raw": p_raw < alpha,
                "significant_bonf": p_bonf < alpha,
                "direction_correct": direction_correct,
                "median_t0": median_t0,
                "median_t": median_t,
                "delta_median": float(delta),
            })

    return pd.DataFrame(rows)


def find_threshold(df: pd.DataFrame, metric: str, corrected: bool = False) -> int | None:
    """
    Find lowest concentration where metric first reaches significance
    AND direction is correct.
    """
    p_col = "p_bonf" if corrected else "p_raw"
    sub = df[
        (df["metric"] == metric) &
        (df["significant_bonf" if corrected else "significant_raw"]) &
        (df["direction_correct"])
    ].sort_values("concentration")
    if sub.empty:
        return None
    return int(sub.iloc[0]["concentration"])


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_threshold_table(df: pd.DataFrame, alpha: float) -> None:
    log.info("\n" + "=" * 70)
    log.info("THRESHOLD ANALYSIS — Mann-Whitney vs T=0%")
    log.info(f"alpha = {alpha}, Bonferroni n = {len(CONCENTRATIONS) - 1}")
    log.info("=" * 70)

    for metric in METRICS:
        sub = df[df["metric"] == metric].sort_values("concentration")
        log.info(f"\n{METRIC_LABELS[metric]} ({metric})")
        log.info(f"  Expected direction: {EXPECTED_DIRECTION[metric]}")
        log.info(f"  {'T%':>5}  {'p_raw':>10}  {'p_bonf':>10}  {'sig_raw':>8}  {'sig_bonf':>9}  {'dir_ok':>7}  {'delta_med':>10}")
        log.info("  " + "-" * 65)
        for _, row in sub.iterrows():
            sig_r = "YES" if row["significant_raw"] else "no"
            sig_b = "YES" if row["significant_bonf"] else "no"
            dir_ok = "OK" if row["direction_correct"] else "WRONG"
            log.info(
                f"  {row['concentration']:>5.0f}  "
                f"{row['p_raw']:>10.4f}  "
                f"{row['p_bonf']:>10.4f}  "
                f"{sig_r:>8}  "
                f"{sig_b:>9}  "
                f"{dir_ok:>7}  "
                f"{row['delta_median']:>+10.4f}"
            )

        t_raw = find_threshold(df, metric, corrected=False)
        t_bonf = find_threshold(df, metric, corrected=True)
        log.info(f"  First significant (uncorrected): T={t_raw}%" if t_raw else "  First significant (uncorrected): not reached")
        log.info(f"  First significant (Bonferroni):  T={t_bonf}%" if t_bonf else "  First significant (Bonferroni):  not reached")

    # Diagnostic profile summary
    log.info("\n" + "=" * 70)
    log.info("DIAGNOSTIC PROFILE SUMMARY")
    log.info("=" * 70)
    thresholds = {}
    for metric in METRICS:
        t = find_threshold(df, metric, corrected=False)
        thresholds[metric] = t
        log.info(f"  {metric:12s}: first significant at T={t}%" if t else f"  {metric:12s}: not significant")

    # Test H_diag: is h_dezorg threshold lower than c_x threshold?
    t_hdez = thresholds.get("h_dezorg")
    t_cx = thresholds.get("c_x")
    log.info("\nH_diag test: h_dezorg threshold < c_x threshold?")
    if t_hdez is None and t_cx is None:
        log.info("  INCONCLUSIVE — neither metric reaches significance")
    elif t_hdez is None:
        log.info(f"  REJECTED — h_dezorg not significant, c_x significant at T={t_cx}%")
    elif t_cx is None:
        log.info(f"  PARTIAL — h_dezorg significant at T={t_hdez}%, c_x not reached")
    elif t_hdez < t_cx:
        log.info(f"  SUPPORTED — h_dezorg T={t_hdez}% < c_x T={t_cx}% (sequential profile)")
    elif t_hdez == t_cx:
        log.info(f"  NOT SUPPORTED — both reach threshold at T={t_hdez}% (simultaneous)")
    else:
        log.info(f"  REVERSED — c_x T={t_cx}% < h_dezorg T={t_hdez}% (opposite order)")
    log.info("=" * 70)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pvalue_curves(df: pd.DataFrame, output_dir: Path, alpha: float) -> None:
    concentrations = sorted(df["concentration"].unique())
    x = np.array(concentrations)

    colors = {
        "c_x":      "#2166ac",
        "h_dezorg": "#d6604d",
        "fitness":  "#4dac26",
        "i_x_seed": "#8073ac",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: raw p-values
    ax = axes[0]
    for metric in METRICS:
        sub = df[df["metric"] == metric].sort_values("concentration")
        p_vals = sub["p_raw"].values
        ax.plot(x, p_vals, "o-", color=colors[metric], lw=2, ms=6,
                label=metric)
        # Mark first significant point with direction correct
        t_thresh = find_threshold(df, metric, corrected=False)
        if t_thresh:
            p_at_thresh = sub[sub["concentration"] == t_thresh]["p_raw"].values[0]
            ax.plot(t_thresh, p_at_thresh, "*", color=colors[metric], ms=14, zorder=5)

    ax.axhline(alpha, color="black", lw=1.2, ls="--", label=f"α={alpha}")
    ax.set_yscale("log")
    ax.set_xlabel("Toxin concentration (%)", fontsize=11)
    ax.set_ylabel("p-value (log scale)", fontsize=11)
    ax.set_title("Mann-Whitney p-values vs T=0% (uncorrected)", fontsize=11)
    ax.set_xticks(concentrations)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.25, lw=0.5)

    # Panel B: Bonferroni-corrected
    ax = axes[1]
    for metric in METRICS:
        sub = df[df["metric"] == metric].sort_values("concentration")
        p_vals = sub["p_bonf"].values
        ax.plot(x, p_vals, "o-", color=colors[metric], lw=2, ms=6,
                label=metric)
        t_thresh = find_threshold(df, metric, corrected=True)
        if t_thresh:
            p_at_thresh = sub[sub["concentration"] == t_thresh]["p_bonf"].values[0]
            ax.plot(t_thresh, p_at_thresh, "*", color=colors[metric], ms=14, zorder=5)

    ax.axhline(alpha, color="black", lw=1.2, ls="--", label=f"α={alpha}")
    ax.set_yscale("log")
    ax.set_xlabel("Toxin concentration (%)", fontsize=11)
    ax.set_ylabel("p-value (log scale)", fontsize=11)
    ax.set_title("Mann-Whitney p-values vs T=0% (Bonferroni corrected)", fontsize=11)
    ax.set_xticks(concentrations)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.25, lw=0.5)

    fig.suptitle("Diagnostic threshold analysis — sequential significance profile", fontsize=12)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "diagnostic_thresholds.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figure saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LD50 diagnostic threshold analysis.")
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Path to LD50 experiment run directory",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance threshold (default: 0.05)",
    )
    args = parser.parse_args()

    if not args.run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {args.run_dir}")

    output_dir = args.run_dir / "figures"

    log.info(f"Loading data from: {args.run_dir}")
    data = load_all_concentrations(args.run_dir)

    if len(data) < 4:
        raise ValueError(f"Need at least 4 concentrations, found {len(data)}.")

    df = run_threshold_analysis(data, alpha=args.alpha)
    print_threshold_table(df, alpha=args.alpha)
    plot_pvalue_curves(df, output_dir, alpha=args.alpha)

    # Save results
    results = {
        "run_dir": str(args.run_dir),
        "alpha": args.alpha,
        "n_comparisons_bonferroni": len(CONCENTRATIONS) - 1,
        "thresholds_uncorrected": {
            m: find_threshold(df, m, corrected=False) for m in METRICS
        },
        "thresholds_bonferroni": {
            m: find_threshold(df, m, corrected=True) for m in METRICS
        },
        "h_diag_supported": (
            find_threshold(df, "h_dezorg", corrected=False) is not None and
            find_threshold(df, "c_x", corrected=False) is not None and
            find_threshold(df, "h_dezorg", corrected=False) <
            find_threshold(df, "c_x", corrected=False)
        ),
        "full_table": df.to_dict(orient="records"),
    }

    out_json = args.run_dir / "diagnostic_threshold_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Results saved: {out_json}")


if __name__ == "__main__":
    main()
