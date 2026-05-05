#!/usr/bin/env python3
"""
analyze_ld50.py
===============
Dose-response analysis for the LD50 titration experiment.

Inputs: per-concentration metric JSON files produced by phase0_metric_validation.py.
        Expected in: experiments/ld50_{run_id}/metrics_t{pct:03d}.json

Steps:
  1. Aggregate per-concentration statistics (mean, SE, n).
  2. Fit 4-parameter sigmoid (Hill equation) to C(X) vs toxin concentration.
  3. Estimate LD50 = EC50 from sigmoid: concentration where C(X) = 50% of T=0%.
     Bootstrap 95% CI on LD50.
  4. Test nonlinearity: compare AIC of linear vs sigmoid fit.
     H0: linear model fits as well as sigmoid.
  5. Repeat for H_dezorg and fitness (secondary metrics).
  6. Write results JSON + figures.

Usage:
    python src/analysis/analyze_ld50.py \
        --run-dir experiments/ld50_20260504T120000Z \
        [--metric complexity] [--output-dir experiments/ld50_20260504T120000Z/figures]

Dependencies: numpy, scipy, matplotlib, pandas
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import mannwhitneyu, pearsonr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CONCENTRATIONS = [0, 10, 25, 50, 75, 90, 100]

METRIC_KEYS = {
    "c_x":      "c_x",
    "h_dezorg": "h_dezorg",
    "h_x":      "h_x",
    "fitness":  "fitness",
    "i_x_seed": "i_x_seed",
}

METRIC_LABELS = {
    "c_x":      "C(X) effective complexity",
    "h_dezorg": "H_dezorg disorganization entropy",
    "h_x":      "H(X) Shannon entropy",
    "fitness":  "Fitness",
    "i_x_seed": "I(X;seed) mutual information with seed",
}


# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------

def sigmoid_4param(x, bottom, top, ec50, hill):
    """4-parameter logistic (Hill equation)."""
    return bottom + (top - bottom) / (1.0 + (ec50 / np.maximum(x, 1e-9)) ** hill)


def linear_model(x, a, b):
    return a * x + b


def aic(n, rss, k):
    """AIC = n*ln(RSS/n) + 2k"""
    return n * np.log(rss / n) + 2 * k


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metrics_for_concentration(run_dir: Path, toxin_pct: int) -> list[dict] | None:
    """
    Load per-document metric records for a given toxin concentration.

    Actual pipeline output structure:
      {run_dir}/concentration_{NNN}/phase0_metrics_{timestamp}/metrics_progressive.jsonl

    Also tries flat patterns for flexibility.
    Each line of metrics_progressive.jsonl is a standalone JSON record.
    """
    conc_dir = run_dir / f"concentration_{toxin_pct:03d}"

    # Pattern 1: concentration_NNN/phase0_metrics_*/metrics_progressive.jsonl
    if conc_dir.exists():
        for subdir in sorted(conc_dir.iterdir()):
            if subdir.is_dir() and subdir.name.startswith("phase0_metrics_"):
                jsonl = subdir / "metrics_progressive.jsonl"
                if jsonl.exists():
                    return _read_jsonl(jsonl)

    # Pattern 2: flat JSONL directly in concentration dir
    flat_jsonl = conc_dir / "metrics_progressive.jsonl"
    if flat_jsonl.exists():
        return _read_jsonl(flat_jsonl)

    # Pattern 3: legacy JSON formats
    for p in [
        run_dir / f"metrics_t{toxin_pct:03d}.json",
        conc_dir / f"metrics_t{toxin_pct:03d}.json",
        conc_dir / "metrics.json",
    ]:
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "results" in data:
                return data["results"]

    log.warning(f"No metrics file found for T={toxin_pct}% in {run_dir}")
    return None


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log.info(f"  Loaded {len(records)} records from {path.relative_to(path.parent.parent.parent)}")
    return records


def extract_metric_values(records: list[dict], metric_key: str) -> np.ndarray:
    values = []
    for r in records:
        v = r.get(metric_key)
        if v is not None and np.isfinite(v):
            values.append(float(v))
    return np.array(values)


def build_dose_response_table(run_dir: Path, metric_key: str) -> pd.DataFrame:
    rows = []
    for t in CONCENTRATIONS:
        records = load_metrics_for_concentration(run_dir, t)
        if records is None:
            log.warning(f"Skipping T={t}% (no data).")
            continue
        vals = extract_metric_values(records, metric_key)
        if len(vals) == 0:
            log.warning(f"T={t}%: no valid {metric_key} values.")
            continue
        rows.append(
            {
                "toxin_pct": t,
                "n": len(vals),
                "mean": vals.mean(),
                "std": vals.std(ddof=1),
                "se": vals.std(ddof=1) / np.sqrt(len(vals)),
                "median": np.median(vals),
                "values": vals,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sigmoid fitting
# ---------------------------------------------------------------------------

def fit_sigmoid(x: np.ndarray, y: np.ndarray, y0: float) -> dict:
    """
    Fit 4-parameter sigmoid to (x, y).
    Returns fit parameters, predicted curve, and RSS.
    y0: value at T=0% (reference for LD50 calculation).
    """
    # Initial guesses
    p0 = [y.min(), y.max(), 50.0, 1.0]  # bottom, top, EC50, hill
    bounds = (
        [0.0,       0.0,     0.1,  0.01],
        [np.inf,    np.inf,  100., 20.0],
    )

    try:
        popt, pcov = curve_fit(
            sigmoid_4param, x, y, p0=p0, bounds=bounds, maxfev=10000
        )
        y_pred = sigmoid_4param(x, *popt)
        rss = np.sum((y - y_pred) ** 2)
        perr = np.sqrt(np.diag(pcov))
        return {
            "success": True,
            "params": {"bottom": popt[0], "top": popt[1], "ec50": popt[2], "hill": popt[3]},
            "param_se": {"bottom": perr[0], "top": perr[1], "ec50": perr[2], "hill": perr[3]},
            "y_pred": y_pred,
            "rss": rss,
        }
    except (RuntimeError, ValueError) as e:
        log.warning(f"Sigmoid fit failed: {e}")
        return {"success": False}


def fit_linear(x: np.ndarray, y: np.ndarray) -> dict:
    try:
        popt, pcov = curve_fit(linear_model, x, y)
        y_pred = linear_model(x, *popt)
        rss = np.sum((y - y_pred) ** 2)
        perr = np.sqrt(np.diag(pcov))
        return {
            "success": True,
            "params": {"slope": popt[0], "intercept": popt[1]},
            "param_se": {"slope": perr[0], "intercept": perr[1]},
            "y_pred": y_pred,
            "rss": rss,
        }
    except (RuntimeError, ValueError) as e:
        log.warning(f"Linear fit failed: {e}")
        return {"success": False}


# ---------------------------------------------------------------------------
# LD50 calculation
# ---------------------------------------------------------------------------

def calculate_ld50_from_params(params: dict, y0: float) -> dict:
    """
    LD50 = concentration where C(X) = y0 * 0.5 (50% drop from T=0% baseline).
    From Hill equation: x = EC50 * ((top - target) / (target - bottom))^(1/hill)

    Returns dict with keys:
      ld50        : float or None (None if algebraically unreachable)
      extrapolated: bool — True if LD50 > 100 (outside titration range)
      target      : the C(X) value that defines LD50
      achievable  : bool — False if sigmoid floor > target (drop never reaches 50%)
    """
    bottom = params["bottom"]
    top = params["top"]
    ec50 = params["ec50"]
    hill = params["hill"]
    target = y0 * 0.5

    result = {
        "ld50": None,
        "extrapolated": False,
        "target_cx": target,
        "achievable": target > bottom,  # sigmoid can in principle reach target
    }

    if not result["achievable"]:
        # Sigmoid floor is above target — a 50% drop is never reached.
        # Report the concentration for the maximum achievable drop instead.
        max_drop_pct = (top - bottom) / y0 * 100 if y0 != 0 else 0
        result["max_drop_pct"] = max_drop_pct
        result["note"] = (
            f"50% drop target ({target:.4f}) below sigmoid floor ({bottom:.4f}). "
            f"Max achievable drop: {max_drop_pct:.1f}%. LD50 undefined; "
            f"EC50 ({ec50:.1f}%) reported as inflection point."
        )
        result["ec50_inflection"] = ec50
        return result

    numerator = top - target
    if numerator <= 0:
        result["note"] = "Target above sigmoid ceiling."
        return result

    try:
        ld50 = ec50 * (numerator / (target - bottom)) ** (1.0 / hill)
    except (ZeroDivisionError, ValueError):
        return result

    result["ld50"] = float(ld50)
    result["extrapolated"] = ld50 > 100.0
    if result["extrapolated"]:
        result["note"] = (
            f"LD50 ({ld50:.1f}%) exceeds titration range [0, 100]. "
            "Extrapolated — interpret cautiously. "
            f"EC50 (inflection) = {ec50:.1f}%."
        )
    return result


def bootstrap_ld50(
    x: np.ndarray,
    all_values: dict[int, np.ndarray],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float] | None:
    """
    Bootstrap CI for LD50 by resampling per-concentration values.
    all_values: dict of toxin_pct -> array of per-document metric values.
    Returns (ci_lower, ci_upper) at 95%, or None if LD50 is not achievable.
    Only uses bootstrap samples where LD50 is within [0, 150] (allows mild extrapolation).
    """
    rng = np.random.default_rng(seed)
    ld50_samples = []

    for _ in range(n_bootstrap):
        y_boot = np.array(
            [rng.choice(all_values[t], size=len(all_values[t]), replace=True).mean()
             for t in x]
        )
        y0_boot = y_boot[0]

        fit = fit_sigmoid(x, y_boot, y0_boot)
        if not fit["success"]:
            continue
        ld50_result = calculate_ld50_from_params(fit["params"], y0_boot)
        ld50_val = ld50_result.get("ld50")
        if ld50_val is not None and 0 <= ld50_val <= 150:
            ld50_samples.append(ld50_val)

    if len(ld50_samples) < n_bootstrap * 0.3:
        log.warning(
            f"Bootstrap: only {len(ld50_samples)}/{n_bootstrap} valid LD50 samples. "
            "LD50 may be outside titration range."
        )
        return None

    arr = np.array(ld50_samples)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


# ---------------------------------------------------------------------------
# Nonlinearity test
# ---------------------------------------------------------------------------

def test_nonlinearity(
    x: np.ndarray,
    y: np.ndarray,
    sigmoid_fit: dict,
    linear_fit: dict,
) -> dict:
    """
    Compare AIC of linear vs sigmoid fit.
    H0: linear model is sufficient (delta_AIC <= 2).
    delta_AIC = AIC_linear - AIC_sigmoid; positive = sigmoid preferred.
    """
    n = len(x)
    if not sigmoid_fit["success"] or not linear_fit["success"]:
        return {"success": False, "reason": "fit failure"}

    aic_linear = aic(n, linear_fit["rss"], k=2)    # slope + intercept
    aic_sigmoid = aic(n, sigmoid_fit["rss"], k=4)  # bottom, top, EC50, hill

    delta_aic = aic_linear - aic_sigmoid  # positive = sigmoid better

    return {
        "success": True,
        "aic_linear": aic_linear,
        "aic_sigmoid": aic_sigmoid,
        "delta_aic": delta_aic,
        "sigmoid_preferred": delta_aic > 2.0,
        "interpretation": (
            "Nonlinear (sigmoid) model preferred (delta_AIC > 2); H0 rejected."
            if delta_aic > 2.0
            else "Linear model cannot be rejected (delta_AIC <= 2)."
        ),
    }


# ---------------------------------------------------------------------------
# Pearson correlation (dose-response monotonicity check)
# ---------------------------------------------------------------------------

def test_monotonicity(x: np.ndarray, y: np.ndarray) -> dict:
    r, p = pearsonr(x, y)
    return {"pearson_r": float(r), "pearson_p": float(p)}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_dose_response(
    df: pd.DataFrame,
    sigmoid_fit: dict,
    linear_fit: dict,
    ld50: float | None,
    ci: tuple | None,
    metric_key: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    x_data = df["toxin_pct"].values.astype(float)
    y_data = df["mean"].values
    y_se = df["se"].values

    ax.errorbar(
        x_data, y_data, yerr=y_se,
        fmt="o", color="#2166ac", ms=7, lw=1.5, capsize=4,
        label="Mean ± SE",
        zorder=3,
    )

    x_smooth = np.linspace(0, 100, 500)

    if linear_fit["success"]:
        y_lin = linear_model(x_smooth, *list(linear_fit["params"].values()))
        ax.plot(x_smooth, y_lin, "--", color="#999999", lw=1.2, label="Linear fit", zorder=2)

    if sigmoid_fit["success"]:
        p = sigmoid_fit["params"]
        y_sig = sigmoid_4param(x_smooth, p["bottom"], p["top"], p["ec50"], p["hill"])
        ax.plot(x_smooth, y_sig, "-", color="#d6604d", lw=2.0, label="Sigmoid fit (4PL)", zorder=4)

    # LD50 marker
    if ld50 is not None and 0 <= ld50 <= 100:
        y_at_ld50 = y_data[0] * 0.5
        ax.axvline(ld50, color="#d6604d", lw=1.0, ls=":", alpha=0.7)
        ax.axhline(y_at_ld50, color="#d6604d", lw=1.0, ls=":", alpha=0.7)
        label_ld50 = f"LD50 = {ld50:.1f}%"
        if ci:
            label_ld50 += f"\n95% CI [{ci[0]:.1f}, {ci[1]:.1f}]"
        ax.annotate(
            label_ld50,
            xy=(ld50, y_at_ld50),
            xytext=(ld50 + 5, y_at_ld50 + (y_data.max() - y_data.min()) * 0.05),
            fontsize=9,
            color="#d6604d",
            arrowprops=dict(arrowstyle="->", color="#d6604d", lw=0.8),
        )

    ax.set_xlabel("Toxin concentration (%)", fontsize=11)
    ax.set_ylabel(METRIC_LABELS.get(metric_key, metric_key), fontsize=11)
    ax.set_title(f"Dose-response: {METRIC_LABELS.get(metric_key, metric_key)}", fontsize=12)
    ax.set_xlim(-3, 103)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.25, lw=0.5)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Figure saved: {output_path}")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_metric(
    run_dir: Path,
    metric_key: str,
    output_dir: Path,
    n_bootstrap: int = 1000,
) -> dict:
    log.info(f"\n=== Analyzing metric: {metric_key} ===")

    df = build_dose_response_table(run_dir, metric_key)
    if df.empty or len(df) < 4:
        log.error(f"Insufficient data for {metric_key} (need >=4 concentrations).")
        return {"metric": metric_key, "error": "insufficient_data"}

    x = df["toxin_pct"].values.astype(float)
    y = df["mean"].values
    y0 = y[0]  # T=0%

    log.info(f"Concentrations available: {x.tolist()}")
    log.info(f"C(X) at T=0%: {y0:.4f}")
    log.info(f"C(X) at T=100%: {y[-1]:.4f}  (drop: {(y0 - y[-1]) / y0 * 100:.1f}%)")

    sigmoid_fit = fit_sigmoid(x, y, y0)
    linear_fit = fit_linear(x, y)

    ld50_result = None
    ci = None
    ld50 = None
    if sigmoid_fit["success"]:
        ld50_result = calculate_ld50_from_params(sigmoid_fit["params"], y0)
        ld50 = ld50_result.get("ld50")
        if ld50_result.get("achievable", False):
            log.info(f"LD50: {ld50:.2f}%" + (" [EXTRAPOLATED >100%]" if ld50_result.get("extrapolated") else ""))
            all_values = {int(row["toxin_pct"]): row["values"] for _, row in df.iterrows()}
            ci = bootstrap_ld50(x, all_values, n_bootstrap=n_bootstrap)
            if ci:
                log.info(f"LD50 95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
        else:
            log.warning(ld50_result.get("note", "LD50 not achievable within titration range."))
            log.info(f"EC50 (inflection point): {ld50_result.get('ec50_inflection', sigmoid_fit['params'].get('ec50', 'N/A'))}")

    nonlin = test_nonlinearity(x, y, sigmoid_fit, linear_fit)
    log.info(f"Nonlinearity test: {nonlin.get('interpretation', 'N/A')}")
    log.info(f"  delta_AIC = {nonlin.get('delta_aic', float('nan')):.2f}")

    mono = test_monotonicity(x, y)
    log.info(f"Monotonicity (Pearson r): {mono['pearson_r']:.3f}, p={mono['pearson_p']:.4e}")

    # Summary table to log
    log.info("\nDose-response summary:")
    log.info(f"{'T%':>5}  {'n':>5}  {'mean':>8}  {'SE':>7}")
    for _, row in df.iterrows():
        log.info(f"{row['toxin_pct']:>5.0f}  {row['n']:>5}  {row['mean']:>8.4f}  {row['se']:>7.4f}")

    # Plot
    plot_path = output_dir / f"dose_response_{metric_key}.png"
    plot_dose_response(df, sigmoid_fit, linear_fit, ld50, ci, metric_key, plot_path)

    # Collate result
    result = {
        "metric": metric_key,
        "y0_t0pct": float(y0),
        "y_t100pct": float(y[-1]),
        "relative_drop_pct": float((y0 - y[-1]) / y0 * 100) if y0 != 0 else None,
        "ld50_result": ld50_result,
        "ld50_ci_95": list(ci) if ci else None,
        "sigmoid_fit": {
            k: v for k, v in sigmoid_fit.items() if k not in ("y_pred",)
        } if sigmoid_fit["success"] else {"success": False},
        "linear_fit": {
            k: v for k, v in linear_fit.items() if k not in ("y_pred",)
        } if linear_fit["success"] else {"success": False},
        "nonlinearity_test": nonlin,
        "monotonicity": mono,
        "summary_table": df.drop(columns=["values"]).to_dict(orient="records"),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="LD50 dose-response analysis.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to experiment run directory (experiments/ld50_*/)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="c_x",
        choices=list(METRIC_KEYS.keys()),
        help="Primary metric for LD50 calculation (default: c_x = C(X))",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for figures and results JSON (default: run-dir/figures/)",
    )
    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=1000,
        help="Bootstrap iterations for LD50 CI (default: 1000)",
    )
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help="Analyze all metrics (c_x, h_dezorg, h_x, fitness, i_x_seed)",
    )
    args = parser.parse_args()

    if not args.run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {args.run_dir}")

    output_dir = args.output_dir or args.run_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_to_analyze = list(METRIC_KEYS.keys()) if args.all_metrics else [args.metric]

    all_results = {}
    for m in metrics_to_analyze:
        res = analyze_metric(args.run_dir, m, output_dir, n_bootstrap=args.bootstrap_n)
        all_results[m] = res

    # --- Primary metric summary ---
    primary = all_results.get(args.metric, {})
    ld50_result = primary.get("ld50_result", {}) or {}
    ld50 = ld50_result.get("ld50")
    ci = primary.get("ld50_ci_95")
    nonlin = primary.get("nonlinearity_test", {})

    log.info("\n" + "=" * 55)
    log.info("RESULTS SUMMARY")
    log.info("=" * 55)
    log.info(f"Primary metric: {METRIC_LABELS.get(args.metric, args.metric)}")
    log.info(f"LD50: {f'{ld50:.2f}%' if ld50 else 'not estimable'}")
    if ci:
        log.info(f"LD50 95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
    log.info(f"H0 (linear): {nonlin.get('interpretation', 'N/A')}")
    log.info("=" * 55)

    # Write JSON
    results_path = args.run_dir / "ld50_analysis_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"\nResults written: {results_path}")


if __name__ == "__main__":
    main()