
from __future__ import annotations
"""Phase 0 runner for metric validation experiments."""

import unsloth  # Must be imported before transformers for Unsloth patching

import argparse
import json

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import os
import json
import numpy as np
from collections import defaultdict
from scipy.stats import kruskal


# Unsloth model loading
import torch
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

from src.config_validation import load_yaml_config, validate_config
from src.metrics.core import disorganization_entropy, effective_complexity, fitness_score, mutual_information_proxy, shannon_entropy, jaccard_similarity


@dataclass(frozen=True)
class Phase0SampleResult:
    """Single sample metric output for Phase 0 validation."""

    sample_id: str
    h_x: float
    c_x: float
    i_x_seed: float
    h_dezorg: float
    fitness: float



def run_phase0_metric_validation(config: dict[str, Any], output_root: str | Path = "experiments") -> Path:
    """Run Phase 0 metric validation and persist results (corpus-aware)."""
    validate_config(config)

    # Ensure offline mode for model loading
    os.environ["HF_HUB_OFFLINE"] = "1"

    project = config.get("project", {})
    if project.get("phase") != 0:
        raise ValueError("Phase 0 runner requires project.phase == 0.")

    seed = int(project.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    weights = config.get("metrics", {}).get("weights", {})
    w1 = float(weights.get("w1_complexity", 1.0))
    w2 = float(weights.get("w2_mutual_info", 1.0))
    w3 = float(weights.get("w3_disorganization", 1.0))

    phase0_cfg = config.get("phase0_validation", {})
    seed_text = str(phase0_cfg.get("seed_text", ""))
    corpus_files = phase0_cfg.get("corpus_files", [])



    # Load Unsloth Qwen3-8B-Base in 4bit
    model_name = "unsloth/qwen3-8b-base-unsloth-bnb-4bit"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model, _ = FastLanguageModel.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    torch.set_grad_enabled(False)

    group_metrics = defaultdict(list)
    all_results = []
    doc_counter = defaultdict(int)

    gen_cfg = dict(
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    for file_path in corpus_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                doc_id = doc.get("id", "unknown")
                doc_type = doc.get("type", "unknown")
                prompt = doc.get("content", "")
                # Generate model output
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
                inputs = {k: v.cuda() for k, v in inputs.items()}
                with torch.no_grad():
                    output_ids = model.generate(**inputs, **gen_cfg)
                # Remove prompt from output
                gen_text = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                # Generate seed output (frozen base model, same prompt)
                # In Phase 0, seed model = base model, so seed output = model output
                seed_text_out = gen_text  # For future phases: generate with frozen model
                # Compute metrics on generated output
                h_x = shannon_entropy(gen_text)
                c_x = effective_complexity(gen_text)
                i_x_seed = mutual_information_proxy(seed_text=seed_text, output_text=gen_text)
                h_dezorg = disorganization_entropy(gen_text)
                fit = fitness_score(
                    complexity=c_x,
                    mutual_info=i_x_seed,
                    disorganization=h_dezorg,
                    w1=w1,
                    w2=w2,
                    w3=w3,
                )
                # Jaccard similarity (diagnostic)
                jaccard = jaccard_similarity(seed_text, gen_text)
                result = {
                    "sample_id": doc_id,
                    "type": doc_type,
                    "h_x": h_x,
                    "c_x": c_x,
                    "i_x_seed": i_x_seed,
                    "h_dezorg": h_dezorg,
                    "fitness": fit,
                    "jaccard": jaccard,
                    "model_output": gen_text,
                }
                group_metrics[doc_type].append(result)
                all_results.append(result)
                doc_counter[doc_type] += 1


    # Kruskal-Wallis test for H(X), C(X), I(X;seed), Jaccard
    def get_metric_list(metric):
        return [ [doc[metric] for doc in group_metrics[typ]] for typ in ("food", "predator", "noise") if group_metrics[typ] ]

    kw_results = {}
    for metric in ["h_x", "c_x", "i_x_seed", "jaccard"]:
        data = get_metric_list(metric)
        if len(data) == 3:
            stat, p = kruskal(*data)
            kw_results[metric] = {"stat": stat, "p": p}
        else:
            kw_results[metric] = {"stat": None, "p": None}

    # Effect size (rank-biserial r) for each pair (food vs predator, food vs noise, predator vs noise)
    def rank_biserial(a, b):
        # Mann-Whitney U effect size
        from scipy.stats import mannwhitneyu
        u, _ = mannwhitneyu(a, b, alternative="two-sided")
        n1, n2 = len(a), len(b)
        r = 1 - (2 * u) / (n1 * n2)
        return r

    effect_sizes = {}
    for metric in ["h_x", "c_x", "i_x_seed", "jaccard"]:
        vals = {typ: [doc[metric] for doc in group_metrics[typ]] for typ in ("food", "predator", "noise")}
        pairs = [("food", "predator"), ("food", "noise"), ("predator", "noise")]
        effect_sizes[metric] = {}
        for a, b in pairs:
            if vals[a] and vals[b]:
                effect_sizes[metric][f"{a}_vs_{b}"] = rank_biserial(vals[a], vals[b])
            else:
                effect_sizes[metric][f"{a}_vs_{b}"] = None

    # Mean metrics per type
    mean_metrics = {}
    for typ, docs in group_metrics.items():
        if docs:
            mean_metrics[typ] = {
                "h_x": float(np.mean([d["h_x"] for d in docs])),
                "c_x": float(np.mean([d["c_x"] for d in docs])),
                "i_x_seed": float(np.mean([d["i_x_seed"] for d in docs])),
                "h_dezorg": float(np.mean([d["h_dezorg"] for d in docs])),
                "fitness": float(np.mean([d["fitness"] for d in docs])),
                "jaccard": float(np.mean([d["jaccard"] for d in docs])),
                "count": len(docs),
            }

    # Korelacja Jaccard vs I(X;seed) per typ
    from scipy.stats import pearsonr
    jaccard_corr = {}
    for typ, docs in group_metrics.items():
        if docs:
            j_vals = [d["jaccard"] for d in docs]
            i_vals = [d["i_x_seed"] for d in docs]
            if len(j_vals) > 1 and len(i_vals) > 1:
                corr, p = pearsonr(j_vals, i_vals)
                jaccard_corr[typ] = {"corr": corr, "p": p}
            else:
                jaccard_corr[typ] = {"corr": None, "p": None}

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"phase0_metrics_{timestamp}"
    run_dir = Path(output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    output_path = run_dir / "metrics_phase0.json"
    payload = {
        "project": project,
        "run": {
            "name": run_name,
            "created_utc": timestamp,
            "seed": seed,
            "doc_count": dict(doc_counter),
        },
        "mean_metrics": mean_metrics,
        "kruskal_wallis": kw_results,
        "effect_sizes": effect_sizes,
        "jaccard_corr": jaccard_corr,
        "results": all_results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Print summary
    print("Documents per type:", dict(doc_counter))
    print("Mean metrics per type:", mean_metrics)
    print("Kruskal-Wallis p-values:", {k: v['p'] for k, v in kw_results.items()})
    print("Phase 0 validation criterion met (p < 0.05):", any(v['p'] is not None and v['p'] < 0.05 for v in kw_results.values()))
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 0 metric validation.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/phase0_metrics_validation.yaml",
        help="Path to Phase 0 YAML config.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="experiments",
        help="Directory where run artifacts are written.",
    )
    return parser


def main() -> None:
    """CLI entrypoint for Phase 0 metric validation."""
    args = _build_parser().parse_args()
    config = load_yaml_config(args.config)
    output = run_phase0_metric_validation(config=config, output_root=args.output_root)
    print(f"Phase 0 metric validation complete: {output}")


if __name__ == "__main__":
    main()
