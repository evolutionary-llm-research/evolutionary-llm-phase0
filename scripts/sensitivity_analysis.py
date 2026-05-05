"""
Sensitivity analysis for minimum output token length at which metric separation
(food/toxin/noise) remains statistically detectable.

- Loads all corpus files from config/phase0_metrics_validation.yaml
- Loads Qwen3-8B-Base via Unsloth (HF_HUB_OFFLINE=1, load_in_4bit=True)
- For each sample_size in [50, 100, 150, 200, 300, 500]:
    - Generates outputs with max_new_tokens=sample_size, temperature=0.0
    - Computes H(X), C(X), I(X;seed) on outputs
    - Runs Kruskal-Wallis food vs toxin vs noise for each metric
    - Computes Cliff's delta for food/toxin pair
    - Records: sample_size, metric, p-value, cliff_delta
- Saves results to experiments/sensitivity_analysis_[timestamp].json
- Prints summary table and threshold

Usage: python scripts/sensitivity_analysis.py
"""

import argparse
import os
import json
import yaml
import random
import time
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from cliffs_delta import cliffs_delta

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
os.environ["HF_HUB_OFFLINE"] = "1"

# --- Load config and corpus files ---
CONFIG_PATH = "config/phase0_metrics_validation.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
corpus_files = config["corpus_files"] if "corpus_files" in config else config.get("phase0_validation", {}).get("corpus_files", [])

# --- Load all samples and group by type ---
def load_corpus(files):
    data = defaultdict(list)
    for path in files:
        label = None
        if "food" in path:
            label = "food"
        elif "toxin" in path:
            label = "toxin"
        elif "noise" in path:
            label = "noise"
        else:
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    text = obj.get("content") or obj.get("text")
                    if text:
                        data[label].append(text)
                except Exception:
                    continue
    return data

corpus = load_corpus(corpus_files)

# --- Load Qwen3-8B-Base via Unsloth ---
from unsloth import FastLanguageModel
from transformers import AutoTokenizer


MODEL_NAME = "unsloth/qwen3-8b-base-unsloth-bnb-4bit"
# Unsloth returns (model, tokenizer)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    load_in_4bit=True,
    trust_remote_code=True,
    device_map="auto"
)

# --- Metric functions ---
from src.metrics.core import shannon_entropy, effective_complexity, mutual_information_proxy

SEED_TEXT = config.get("phase0_validation", {}).get("seed_text", "Climate and vaccine discourse requires coherent, evidence-grounded synthesis.")


SAMPLE_SIZES = [50, 100, 150, 200, 300, 500]
results = []


# --- Argument parsing ---
parser = argparse.ArgumentParser(description="Sensitivity analysis for output token length.")
parser.add_argument("--run-name", type=str, default=None, help="Optional run name for experiment folder and dashboard labeling.")
args = parser.parse_args()

from pathlib import Path
from datetime import timezone
timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
if args.run_name:
    run_folder = f"sensitivity_analysis_{args.run_name}"
else:
    run_folder = f"sensitivity_analysis_{timestamp}"
run_dir = Path("experiments") / run_folder
run_dir.mkdir(parents=True, exist_ok=False)
progressive_path = run_dir / "metrics_progressive.jsonl"

# If progressive file exists (resume), load processed (sample_size, doc_id)
processed_pairs = set()
if progressive_path.exists():
    with open(progressive_path, "r", encoding="utf-8") as pf:
        for line in pf:
            try:
                rec = json.loads(line)
                processed_pairs.add((rec.get("sample_size"), rec.get("id")))
            except Exception:
                continue

for sample_size in SAMPLE_SIZES:
    outputs = {k: [] for k in corpus}
    doc_ids = {k: [] for k in corpus}
    for label, texts in corpus.items():
        for prompt in tqdm(texts, desc=f"{label} (tokens={sample_size})"):
            # Use prompt as unique id if no id available
            doc_id = hash(prompt)
            if (sample_size, doc_id) in processed_pairs:
                outputs[label].append(None)
                doc_ids[label].append(doc_id)
                continue
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            gen = model.generate(
                input_ids=input_ids,
                max_new_tokens=sample_size,
                temperature=0.0,
                do_sample=False
            )
            output = tokenizer.decode(gen[0], skip_special_tokens=True)
            outputs[label].append(output)
            doc_ids[label].append(doc_id)
            # Compute metrics and progressive log
            h_x = shannon_entropy(output)
            c_x = effective_complexity(output)
            i_x = mutual_information_proxy(SEED_TEXT, output)
            record = {
                "sample_size": sample_size,
                "id": doc_id,
                "type": label,
                "h_x": h_x,
                "c_x": c_x,
                "i_x_seed": i_x,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with open(progressive_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')
                f.flush()
    # Compute metrics for Kruskal/Cliff only on new outputs
    metrics = {k: {"H": [], "C": [], "I": []} for k in outputs}
    for label, outs in outputs.items():
        for out in outs:
            if out is not None:
                metrics[label]["H"].append(shannon_entropy(out))
                metrics[label]["C"].append(effective_complexity(out))
                metrics[label]["I"].append(mutual_information_proxy(SEED_TEXT, out))
            else:
                metrics[label]["H"].append(np.nan)
                metrics[label]["C"].append(np.nan)
                metrics[label]["I"].append(np.nan)
    # Kruskal-Wallis
    for metric in ["H", "C", "I"]:
        data = [
            [v for v in metrics[k][metric] if not np.isnan(v)]
            for k in ["food", "toxin", "noise"]
        ]
        if all(len(d) > 0 for d in data):
            stat, pval = kruskal(*data)
            # Cliff's delta food vs toxin
            delta, _ = cliffs_delta(
                [v for v in metrics["food"][metric] if not np.isnan(v)],
                [v for v in metrics["toxin"][metric] if not np.isnan(v)]
            )
            results.append({
                "sample_size": sample_size,
                "metric": metric,
                "p_value": pval,
                "cliff_delta": delta
            })
        else:
            results.append({
                "sample_size": sample_size,
                "metric": metric,
                "p_value": None,
                "cliff_delta": None
            })

# --- Save results ---

out_path = run_dir / "sensitivity_analysis.json"
payload = {
    "run_name": args.run_name if args.run_name else timestamp,
    "created_utc": timestamp,
    "results": results
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

# --- Print summary table ---
df = pd.DataFrame(results)
table = []
for sz in SAMPLE_SIZES:
    row = [sz]
    for metric in ["H", "C", "I"]:
        p = df[(df["sample_size"]==sz) & (df["metric"]==metric)]["p_value"].values[0]
        d = df[(df["sample_size"]==sz) & (df["metric"]==metric)]["cliff_delta"].values[0]
        row.extend([f"{p:.3g}", f"{d:.2f}"])
    table.append(row)
print("| tokens | H p-val | C p-val | I p-val | H delta | C delta | I delta |")
print("|--------|---------|---------|---------|---------|---------|---------|")
for row in table:
    print(f"| {row[0]:<6} | {row[1]:<7} | {row[3]:<7} | {row[5]:<7} | {row[2]:<7} | {row[4]:<7} | {row[6]:<7} |")
# Threshold: first sample_size where all three p < 0.05
for sz in SAMPLE_SIZES:
    vals = [df[(df["sample_size"]==sz) & (df["metric"]==m)]["p_value"].values[0] for m in ["H","C","I"]]
    if all(p < 0.05 for p in vals):
        print(f"\nThreshold: {sz} tokens (all metrics p < 0.05)")
        break
