
"""
Per-dataset quality analysis of predator corpus vs food corpus using Phase 0 pipeline (model inference).

- Runs Phase 0 pipeline separately for each predator source (climate, vaccines, covid)
- Compares each predator subset to combined food corpus
- Computes: H(X), C(X), I(X;seed), Jaccard, effect size (rank-biserial r), Mann-Whitney U p-value
- All metrics are computed on model-generated outputs (not raw text)
- Outputs: table (stdout), JSON (results)

Usage:
    python scripts/corpus_quality_analysis.py --config config/phase0_metrics_validation.yaml --output-root experiments
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu
import os

from src.config_validation import load_yaml_config, validate_config
from src.metrics.core import shannon_entropy, effective_complexity, mutual_information_proxy, disorganization_entropy, jaccard_similarity

# Import model loading/generation logic from phase0_metric_validation.py
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
import torch

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model, _ = FastLanguageModel.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    torch.set_grad_enabled(False)
    return model, tokenizer

def generate_output(model, tokenizer, prompt, gen_cfg):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_cfg)
    gen_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen_text

def compute_metrics_on_outputs(docs, seed_text, model, tokenizer, gen_cfg, weights):
    results = []
    for doc in docs:
        prompt = doc["content"]
        gen_text = generate_output(model, tokenizer, prompt, gen_cfg)
        h_x = shannon_entropy(gen_text)
        c_x = effective_complexity(gen_text)
        i_x_seed = mutual_information_proxy(seed_text, gen_text)
        h_dezorg = disorganization_entropy(gen_text)
        fit = weights["w1"] * c_x + weights["w2"] * i_x_seed - weights["w3"] * h_dezorg
        jaccard = jaccard_similarity(seed_text, gen_text)
        results.append({
            "id": doc.get("id", "unknown"),
            "h_x": h_x,
            "c_x": c_x,
            "i_x_seed": i_x_seed,
            "h_dezorg": h_dezorg,
            "fitness": fit,
            "jaccard": jaccard,
            "model_output": gen_text,
        })
    return results

def effect_size_and_p(a, b):
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = len(a), len(b)
    r = 1 - (2 * u) / (n1 * n2)
    return r, p

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/phase0_metrics_validation.yaml")
    parser.add_argument("--output-root", type=str, default="experiments")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    validate_config(config)
    phase0_cfg = config["phase0_validation"]
    seed_text = str(phase0_cfg["seed_text"])
    weights = config["metrics"]["weights"]
    weights = {
        "w1": float(weights["w1_complexity"]),
        "w2": float(weights["w2_mutual_info"]),
        "w3": float(weights["w3_disorganization"]),
    }

    # Model config (reuse phase0 logic)
    os.environ["HF_HUB_OFFLINE"] = "1"
    model_name = "unsloth/qwen3-8b-base-unsloth-bnb-4bit"
    model, tokenizer = load_model_and_tokenizer(model_name)
    gen_cfg = dict(
        max_new_tokens=200,
        do_sample=False,
        temperature=0.0,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # File paths
    food_files = [
    "data/v2/food_climate.jsonl",
    "data/v2/food_vaccines.jsonl",
    "data/v2/food_alt_med.jsonl",
    "data/v2/food_cancer.jsonl",
    "data/v2/food_gmo.jsonl",
]
    predator_files = [
        ("climate",  "data/v2/predator_climate.jsonl"),
        ("vaccines", "data/v2/predator_vaccines.jsonl"),
        ("alt_med",  "data/v2/predator_alt_med.jsonl"),
        ("cancer",   "data/v2/predator_cancer.jsonl"),
        ("gmo",      "data/v2/predator_gmo.jsonl"),
    ]

    # Load corpora
    food_docs = []
    for f in food_files:
        food_docs.extend(load_jsonl(f))
    print(f"Generating model outputs for {len(food_docs)} food documents...")
    food_metrics = compute_metrics_on_outputs(food_docs, seed_text, model, tokenizer, gen_cfg, weights)
    food_stats = {k: np.array([d[k] for d in food_metrics]) for k in ["h_x", "c_x", "i_x_seed", "jaccard"]}

    results = {}
    for name, path in predator_files:
        pred_docs = load_jsonl(path)
        print(f"Generating model outputs for {len(pred_docs)} predator ({name}) documents...")
        pred_metrics = compute_metrics_on_outputs(pred_docs, seed_text, model, tokenizer, gen_cfg, weights)
        pred_stats = {k: np.array([d[k] for d in pred_metrics]) for k in ["h_x", "c_x", "i_x_seed", "jaccard"]}
        results[name] = {"mean": {}, "effect_size": {}, "p_value": {}}
        for metric in ["h_x", "c_x", "i_x_seed", "jaccard"]:
            mean_food = float(np.mean(food_stats[metric]))
            mean_pred = float(np.mean(pred_stats[metric]))
            r, p = effect_size_and_p(food_stats[metric], pred_stats[metric])
            results[name]["mean"][metric] = {"food": mean_food, "predator": mean_pred}
            results[name]["effect_size"][metric] = r
            results[name]["p_value"][metric] = p
    # Print table
    print("\nPer-dataset predator vs food corpus quality analysis (Phase 0 metrics, model outputs):\n")
    print(f"{'Predator':<10} {'Metric':<10} {'Food Mean':>10} {'Pred Mean':>10} {'r':>8} {'p':>10}")
    for name in predator_files:
        pname = name[0]
        for metric in ["h_x", "c_x", "i_x_seed", "jaccard"]:
            m = results[pname]["mean"]
            r = results[pname]["effect_size"]
            p = results[pname]["p_value"]
            print(f"{pname:<10} {metric:<10} {m[metric]['food']:10.4f} {m[metric]['predator']:10.4f} {r[metric]:8.3f} {p[metric]:10.3g}")
    # Save JSON
    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "corpus_quality_analysis_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()

    config = load_yaml_config(args.config)
    validate_config(config)
    phase0_cfg = config["phase0_validation"]
    seed_text = str(phase0_cfg["seed_text"])
    weights = config["metrics"]["weights"]
    # Map to float keys
    weights = {
        "w1": float(weights["w1_complexity"]),
        "w2": float(weights["w2_mutual_info"]),
        "w3": float(weights["w3_disorganization"]),
    }
    # File paths
    food_files = [
    "data/v2/food_climate.jsonl",
    "data/v2/food_vaccines.jsonl",
    "data/v2/food_alt_med.jsonl",
    "data/v2/food_cancer.jsonl",
    "data/v2/food_gmo.jsonl",
]
    predator_files = [
        ("climate",  "data/v2/predator_climate.jsonl"),
        ("vaccines", "data/v2/predator_vaccines.jsonl"),
        ("alt_med",  "data/v2/predator_alt_med.jsonl"),
        ("cancer",   "data/v2/predator_cancer.jsonl"),
        ("gmo",      "data/v2/predator_gmo.jsonl"),
    ]
    # Load corpora
    food_docs = []
    for f in food_files:
        food_docs.extend(load_jsonl(f))
    food_metrics = compute_metrics(food_docs, seed_text, weights)
    food_stats = {k: np.array([d[k] for d in food_metrics]) for k in ["h_x", "c_x", "i_x_seed", "jaccard"]}

    results = {}
    for name, path in predator_files:
        pred_docs = load_jsonl(path)
        pred_metrics = compute_metrics(pred_docs, seed_text, weights)
        pred_stats = {k: np.array([d[k] for d in pred_metrics]) for k in ["h_x", "c_x", "i_x_seed", "jaccard"]}
        results[name] = {"mean": {}, "effect_size": {}, "p_value": {}}
        for metric in ["h_x", "c_x", "i_x_seed", "jaccard"]:
            mean_food = float(np.mean(food_stats[metric]))
            mean_pred = float(np.mean(pred_stats[metric]))
            r, p = effect_size_and_p(food_stats[metric], pred_stats[metric])
            results[name]["mean"][metric] = {"food": mean_food, "predator": mean_pred}
            results[name]["effect_size"][metric] = r
            results[name]["p_value"][metric] = p
    # Print table
    print("\nPer-dataset predator vs food corpus quality analysis (Phase 0 metrics):\n")
    print(f"{'Predator':<10} {'Metric':<10} {'Food Mean':>10} {'Pred Mean':>10} {'r':>8} {'p':>10}")
    for name in predator_files:
        pname = name[0]
        for metric in ["h_x", "c_x", "i_x_seed", "jaccard"]:
            m = results[pname]["mean"]
            r = results[pname]["effect_size"]
            p = results[pname]["p_value"]
            print(f"{pname:<10} {metric:<10} {m[metric]['food']:10.4f} {m[metric]['predator']:10.4f} {r[metric]:8.3f} {p[metric]:10.3g}")
    # Save JSON
    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "corpus_quality_analysis_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
