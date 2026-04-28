import pandas as pd
import numpy as np
from scipy.stats import kruskal
from sklearn.model_selection import train_test_split


def run_grid_search(json_path):
    import json
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data["results"])
    train, val = train_test_split(df, test_size=0.2, stratify=df["type"], random_state=42)
    grid = np.arange(0.1, 1.1, 0.1)
    best_H = -np.inf
    best_w = None
    best_p = None
    for w1 in grid:
        for w2 in grid:
            for w3 in grid:
                if abs(w1 + w2 + w3 - 1.0) > 1e-6:
                    continue  # tylko kombinacje sumujące się do 1
                fitness = w1 * train["c_x"] + w2 * train["i_x_seed"] - w3 * train["h_dezorg"]
                groups = [fitness[train["type"] == t] for t in ["food", "predator", "noise"]]
                H, p = kruskal(*groups)
                if H > best_H:
                    best_H = H
                    best_w = (w1, w2, w3)
                    best_p = p
    # Walidacja na holdoucie
    val_fitness = best_w[0] * val["c_x"] + best_w[1] * val["i_x_seed"] - best_w[2] * val["h_dezorg"]
    val_groups = [val_fitness[val["type"] == t] for t in ["food", "predator", "noise"]]
    val_H, val_p = kruskal(*val_groups)
    return {
        "best_w": best_w,
        "H_train": best_H,
        "p_train": best_p,
        "H_val": val_H,
        "p_val": val_p
    }

if __name__ == "__main__":
    result1 = run_grid_search("experiments/phase0_metrics_20260427T075806Z/metrics_phase0.json")
    result2 = run_grid_search("experiments/phase0_metrics_20260427T073814Z/metrics_phase0.json")
    print("Run 20260427T075806Z:", result1)
    print("Run 20260427T073814Z:", result2)
