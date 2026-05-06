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
import numpy as np
from collections import defaultdict
from scipy.stats import kruskal

import torch
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

from src.config_validation import load_yaml_config, validate_config
from src.metrics.core import (
    disorganization_entropy, effective_complexity, fitness_score,
    mutual_information_proxy, shannon_entropy, jaccard_similarity
)


@dataclass(frozen=True)
class Phase0SampleResult:
    """Single sample metric output for Phase 0 validation."""
    sample_id: str
    h_x: float
    c_x: float
    i_x_seed: float
    h_dezorg: float
    fitness: float


def get_percentile_chunks(prompt: str, tokenizer, n_windows: int = 5, window_size: int = 512):
    """
    Dzieli dokument na n_windows rownych procentowo okien.
    Zwraca liste (quartile_id, chunk_text).
    Rozwiazuje problem gestosc informacji: dokument 5k i 25k maja
    identyczna reprezentacje strukturalna 5 x 512 tokenow.
    """
    tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    doc_len = len(tokens)
    chunks = []
    for i in range(n_windows):
        center = int(doc_len * (i + 0.5) / n_windows)
        start = max(0, center - window_size // 2)
        end = min(doc_len, start + window_size)
        chunk_text = tokenizer.decode(tokens[start:end], skip_special_tokens=True)
        chunks.append((i + 1, chunk_text))
    return chunks


def _load_checkpoint(prog_path: Path) -> tuple[set[str], list[dict]]:
    """Load already-processed sample IDs and their results from a progressive log."""
    processed_ids: set[str] = set()
    loaded_results: list[dict] = []
    if not prog_path.exists():
        return processed_ids, loaded_results
    with open(prog_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                sid = rec.get("id")
                if sid:
                    processed_ids.add(sid)
                    loaded_results.append(rec)
            except Exception:
                pass
    return processed_ids, loaded_results


def run_phase0_metric_validation(
    config: dict[str, Any],
    output_root: str | Path = "experiments",
    resume_dir: str | Path | None = None,
) -> Path:
    """Run Phase 0 metric validation and persist results (corpus-aware).

    Parameters
    ----------
    config : dict
        Parsed YAML configuration.
    output_root : str | Path
        Root directory for new experiment runs.
    resume_dir : str | Path | None
        Path to an existing incomplete run directory to resume.
        If provided, already-processed documents are skipped and
        results are appended to the existing metrics_progressive.jsonl.
    """
    validate_config(config)

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
    save_chunk_texts = bool(phase0_cfg.get("save_chunk_texts", False))

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

    group_metrics: dict = defaultdict(list)
    all_results: list[dict] = []
    doc_counter: dict = defaultdict(int)

    # ── Resume or new run ─────────────────────────────────────────────────────
    if resume_dir is not None:
        run_dir = Path(resume_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume directory not found: {run_dir}")
        prog_path = run_dir / "metrics_progressive.jsonl"
        processed_ids, checkpoint_results = _load_checkpoint(prog_path)
        run_name = run_dir.name
        timestamp = run_name.replace("phase0_metrics_", "")
        print(f"[RESUME] Resuming run '{run_name}' — {len(processed_ids)} documents already done.")
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_name = f"phase0_metrics_{timestamp}"
        run_dir = Path(output_root) / run_name
        run_dir.mkdir(parents=True, exist_ok=False)
        prog_path = run_dir / "metrics_progressive.jsonl"
        processed_ids: set[str] = set()
        checkpoint_results: list[dict] = []
        print(f"[NEW RUN] Starting run '{run_name}'.")

    # Pre-populate all_results and group_metrics from checkpoint
    for rec in checkpoint_results:
        dtype = rec.get("type", "unknown")
        # Convert progressive record back to full result shape expected by aggregation
        full_rec = dict(rec)
        full_rec.setdefault("sample_id", rec.get("id", "unknown"))
        all_results.append(full_rec)
        group_metrics[dtype].append(full_rec)
        doc_counter[dtype] += 1

    gen_cfg_yaml = config.get("generation", {})
    gen_cfg = dict(
        max_new_tokens=int(gen_cfg_yaml.get("max_new_tokens", 200)),
        do_sample=bool(gen_cfg_yaml.get("do_sample", False)),
        temperature=float(gen_cfg_yaml.get("temperature", 0.0)),
        eos_token_id=tokenizer.eos_token_id,
    )
    if gen_cfg["do_sample"]:
        gen_cfg["top_p"] = float(gen_cfg_yaml.get("top_p", 0.95))

    for file_path in corpus_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                doc_id = doc.get("id", "unknown")
                doc_type = doc.get("type", "unknown")
                prompt = doc.get("content", "")

                # Skip already-processed documents (resume support)
                if doc_id in processed_ids:
                    continue

                # --- PERCENTILE CHUNKING (5 x 20% okien) ---
                chunks = get_percentile_chunks(prompt, tokenizer, n_windows=phase0_cfg.get('n_windows', 5), window_size=phase0_cfg.get('window_size', 512))

                chunk_metrics = []
                for q_id, chunk_text in chunks:
                    inputs = tokenizer(
                        chunk_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    )
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                    with torch.no_grad():
                        output_ids = model.generate(**inputs, **gen_cfg)
                    gen_text = tokenizer.decode(
                        output_ids[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )
                    chunk_metrics.append({
                        "q_id": q_id,
                        "h_x": shannon_entropy(gen_text),
                        "c_x": effective_complexity(gen_text),
                        "i_x_seed": mutual_information_proxy(seed_text=seed_text, output_text=gen_text),
                        "h_dezorg": disorganization_entropy(gen_text),
                        "jaccard": jaccard_similarity(seed_text, gen_text),
                        "gen_text": gen_text,
                    })

                # --- AGREGACJA ---
                def _mean(key):
                    return float(np.mean([c[key] for c in chunk_metrics]))

                def _var(key):
                    return float(np.var([c[key] for c in chunk_metrics]))

                def _slope(key):
                    vals = [c[key] for c in chunk_metrics]
                    return float(np.polyfit(np.arange(len(vals)), vals, 1)[0])

                # profil kwartylow (do analizy gdzie sygnal jest najsilniejszy)
                profile = {
                    f"{k}_Q{c['q_id']}": c[k]
                    for c in chunk_metrics
                    for k in ["h_x", "c_x", "i_x_seed", "h_dezorg", "jaccard"]
                }
                chunk_text_profile = (
                    {f"gen_text_Q{c['q_id']}": c["gen_text"] for c in chunk_metrics}
                    if save_chunk_texts
                    else {}
                )

                h_x      = _mean("h_x")
                c_x      = _mean("c_x")
                i_x_seed = _mean("i_x_seed")
                h_dezorg = _mean("h_dezorg")
                jaccard  = _mean("jaccard")

                fit = fitness_score(
                    complexity=c_x,
                    mutual_info=i_x_seed,
                    disorganization=h_dezorg,
                    w1=w1, w2=w2, w3=w3,
                )

                result = {
                    "sample_id": doc_id,
                    "type": doc_type,
                    # backward-compatible mean metrics
                    "h_x": h_x,
                    "c_x": c_x,
                    "i_x_seed": i_x_seed,
                    "h_dezorg": h_dezorg,
                    "fitness": fit,
                    "jaccard": jaccard,
                    # nowe metryki profilu - rehabilitacja entropii
                    "h_x_var": _var("h_x"),
                    "h_x_slope": _slope("h_x"),
                    "h_dezorg_var": _var("h_dezorg"),
                    "h_dezorg_slope": _slope("h_dezorg"),
                    # pelny profil kwartylow
                    **profile,
                    **chunk_text_profile,
                    "model_output": chunk_metrics[-1]["gen_text"],
                }

                group_metrics[doc_type].append(result)
                all_results.append(result)
                doc_counter[doc_type] += 1

                # Progressive log — flush after every document for live dashboard
                prog_record = {
                    "id": doc_id,
                    "type": doc_type,
                    "h_x": h_x,
                    "c_x": c_x,
                    "i_x_seed": i_x_seed,
                    "h_dezorg": h_dezorg,
                    "fitness": fit,
                    "jaccard": jaccard,
                }
                with open(prog_path, "a", encoding="utf-8") as _pf:
                    _pf.write(json.dumps(prog_record) + "\n")
                    _pf.flush()

    # Kruskal-Wallis dla wszystkich metryk (oryginalne + nowe)
    def get_metric_list(metric):
        return [
            [doc[metric] for doc in group_metrics[typ]]
            for typ in ("food", "toxin", "noise")
            if group_metrics[typ]
        ]

    kw_results = {}
    metrics_to_test = ["h_x", "c_x", "i_x_seed", "jaccard",
                       "h_x_var", "h_x_slope", "h_dezorg_var", "h_dezorg_slope"]
    for metric in metrics_to_test:
        data = get_metric_list(metric)
        if len(data) == 3:
            stat, p = kruskal(*data)
            kw_results[metric] = {"stat": stat, "p": p}
        else:
            kw_results[metric] = {"stat": None, "p": None}

    # Effect size (rank-biserial r)
    def rank_biserial(a, b):
        from scipy.stats import mannwhitneyu
        u, _ = mannwhitneyu(a, b, alternative="two-sided")
        n1, n2 = len(a), len(b)
        return 1 - (2 * u) / (n1 * n2)

    effect_sizes = {}
    for metric in ["h_x", "c_x", "i_x_seed", "jaccard"]:
        vals = {typ: [doc[metric] for doc in group_metrics[typ]] for typ in ("food", "toxin", "noise")}
        pairs = [("food", "toxin"), ("food", "noise"), ("toxin", "noise")]
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
            if len(j_vals) > 1:
                corr, p = pearsonr(j_vals, i_vals)
                jaccard_corr[typ] = {"corr": corr, "p": p}
            else:
                jaccard_corr[typ] = {"corr": None, "p": None}

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

    print("Documents per type:", dict(doc_counter))
    print("Mean metrics per type:", mean_metrics)
    print("Kruskal-Wallis p-values:", {k: v["p"] for k, v in kw_results.items()})
    print(
        "Phase 0 validation criterion met (p < 0.05):",
        any(v["p"] is not None and v["p"] < 0.05 for v in kw_results.values()),
    )
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
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="RUN_DIR",
        help=(
            "Path to an existing incomplete run directory to resume. "
            "Already-processed documents are skipped; results are appended "
            "to the existing metrics_progressive.jsonl. "
            "Example: --resume experiments/phase0_metrics_20260501T160337Z"
        ),
    )
    return parser


def main() -> None:
    """CLI entrypoint for Phase 0 metric validation."""
    args = _build_parser().parse_args()
    config = load_yaml_config(args.config)
    output = run_phase0_metric_validation(
        config=config,
        output_root=args.output_root,
        resume_dir=args.resume,
    )
    print(f"Phase 0 metric validation complete: {output}")


if __name__ == "__main__":
    main()