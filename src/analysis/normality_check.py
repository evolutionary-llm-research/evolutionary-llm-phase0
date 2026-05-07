"""
Retroactive normality check (Shapiro-Wilk) for Phase 0 metrics.
Run on canonical metrics_phase0.json (N=880).

Usage:
    python src/analysis/normality_check.py
"""

import json
import numpy as np
from scipy.stats import shapiro

METRICS_PATH = "e:/github/evolutionary-llm-phase0/results/metrics_phase0.json"
METRICS = ["h_x", "c_x", "i_x_seed", "h_dezorg", "fitness"]
TYPES = ["food", "toxin", "noise"]


def main() -> None:
    with open(METRICS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    records = data["results"]

    # Group by type
    groups: dict[str, list[dict]] = {t: [] for t in TYPES}
    for rec in records:
        t = rec.get("type", "")
        if t in groups:
            groups[t].append(rec)

    print(f"{'Metric':<14} {'Type':<10} {'N':>5}  {'W':>8}  {'p':>12}  Normal?")
    print("-" * 60)

    results_out: list[dict] = []

    for metric in METRICS:
        for t in TYPES:
            vals = np.array([r[metric] for r in groups[t] if r.get(metric) is not None], dtype=float)
            n = len(vals)
            if n < 3:
                continue
            # Shapiro-Wilk is reliable up to N=5000; for large N subsample 5000
            sample = vals if n <= 5000 else vals[np.random.default_rng(42).choice(n, 5000, replace=False)]
            w, p = shapiro(sample)
            normal = "YES" if p >= 0.05 else "NO "
            print(f"{metric:<14} {t:<10} {n:>5}  {w:>8.4f}  {p:>12.4e}  {normal}")
            results_out.append({
                "metric": metric,
                "type": t,
                "n": n,
                "shapiro_W": round(float(w), 6),
                "shapiro_p": float(p),
                "normal_at_0.05": bool(p >= 0.05),
            })

    # Save results
    out_path = "e:/github/Evolutionary LLM Research/results/normality_shapiro.json"
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"description": "Shapiro-Wilk normality test, Phase 0 canonical metrics", "results": results_out}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
