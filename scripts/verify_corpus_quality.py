#!/usr/bin/env python3
"""
verify_corpus_quality.py — Corpus quality verification for EvoLLM project.

Checks each document without running any LLM. Outputs per-document score 0-1
with dimension breakdown and flags, plus corpus-level summary.

Usage:
    python scripts/verify_corpus_quality.py \
        --toxin data/toxin_alt_med.jsonl \
        --food data/food_alt_med.jsonl \
        --out reports/quality_alt_med.json

    # Check single corpus:
    python scripts/verify_corpus_quality.py \
        --toxin data/toxin_vaccines_v2.jsonl \
        --out reports/quality_vaccines_v2.json

    # Check all corpora in data/ directory:
    python scripts/verify_corpus_quality.py --all-dir data/ --out reports/
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_CHARS = 500
MIN_UNIQUE_RATIO = 0.25
MAX_ARTIFACT_DENSITY = 0.05    # max fraction of lines with artifacts
MAX_LOOP_REPEAT = 3            # if 5-gram appears >= N times → loop
MIN_NGRAM_DIVERGENCE = 0.3     # toxin should not be too similar to food

# Artifact patterns (same base as scraper, expanded)
ARTIFACT_PATTERNS = [
    r"javascript is (required|disabled)",
    r"please enable javascript",
    r"see more of .{1,40} on facebook",
    r"log in .{0,20} sign up",
    r"share this (article|post|page)",
    r"click here to (subscribe|read more|continue)",
    r"newsletter sign.?up",
    r"loading\.\.\.",
    r"cookies? (policy|settings|consent)",
    r"privacy policy",
    r"terms of (use|service)",
    r"\[[\.\s]{5,}\]",        # placeholder artifacts
    r"follow us on (twitter|facebook|instagram)",
    r"(?:like|follow) us",
    r"subscribe (to|for) (our|the) (newsletter|updates)",
    r"continue reading",
    r"read (more|full article)",
    r"posted in \w+",
    r"tags?:\s*\w",
    r"related (articles?|posts?|stories?)",
    r"you (may|might) also (like|enjoy)",
    r"comments? \(\d+\)",
    r"leave a (comment|reply)",
    r"\d+ (views|shares|likes|comments)",
]
ARTIFACT_RE = re.compile("|".join(ARTIFACT_PATTERNS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# Individual quality dimensions
# ---------------------------------------------------------------------------

def check_length(text: str) -> dict:
    n = len(text)
    words = len(text.split())
    ok = n >= MIN_CHARS
    score = min(1.0, n / 2000)  # saturates at 2000 chars
    return {
        "ok": ok,
        "score": round(score, 3),
        "chars": n,
        "words": words,
        "flag": None if ok else f"too_short ({n} chars, min {MIN_CHARS})",
    }


def check_loop(text: str) -> dict:
    """Detect repeated 5-gram phrases (symptom of scraping loops or model artifacts)."""
    words = text.lower().split()
    if len(words) < 10:
        return {"ok": True, "score": 1.0, "max_repeat": 0, "flag": None}

    ngram_counts = Counter()
    n = 5
    for i in range(len(words) - n):
        gram = " ".join(words[i:i+n])
        ngram_counts[gram] += 1

    max_repeat = ngram_counts.most_common(1)[0][1] if ngram_counts else 0
    ok = max_repeat < MAX_LOOP_REPEAT
    # Score: 1.0 when no repeats, degrades toward 0
    score = max(0.0, 1.0 - (max_repeat - 1) / 5)
    return {
        "ok": ok,
        "score": round(score, 3),
        "max_repeat": max_repeat,
        "most_repeated": ngram_counts.most_common(1)[0][0] if not ok else None,
        "flag": f"loop_detected (max_repeat={max_repeat})" if not ok else None,
    }


def check_lexical_diversity(text: str) -> dict:
    """Unique token ratio — detects low-entropy repetitive content."""
    words = text.lower().split()
    if not words:
        return {"ok": False, "score": 0.0, "unique_ratio": 0.0, "flag": "empty"}
    ratio = len(set(words)) / len(words)
    ok = ratio >= MIN_UNIQUE_RATIO
    score = min(1.0, ratio / 0.5)  # normalize: 0.5 unique ratio → full score
    return {
        "ok": ok,
        "score": round(score, 3),
        "unique_ratio": round(ratio, 3),
        "vocab_size": len(set(words)),
        "flag": f"low_lexical_diversity ({ratio:.2f})" if not ok else None,
    }


def check_artifacts(text: str) -> dict:
    """Density of web artifact patterns."""
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return {"ok": False, "score": 0.0, "artifact_density": 1.0,
                "artifact_count": 0, "flag": "empty"}

    artifact_lines = sum(1 for l in lines if ARTIFACT_RE.search(l))
    density = artifact_lines / len(lines)
    ok = density <= MAX_ARTIFACT_DENSITY
    score = max(0.0, 1.0 - density / MAX_ARTIFACT_DENSITY)
    return {
        "ok": ok,
        "score": round(score, 3),
        "artifact_density": round(density, 3),
        "artifact_lines": artifact_lines,
        "total_lines": len(lines),
        "flag": f"high_artifact_density ({density:.2%})" if not ok else None,
    }


def compute_bigrams(text: str) -> Counter:
    words = re.sub(r"[^a-z\s]", "", text.lower()).split()
    return Counter(zip(words, words[1:]))


def check_ngram_overlap(toxin_text: str,
                         food_bigrams: Optional[Counter]) -> dict:
    """
    Check if toxin is not too similar to food (by bigram Jaccard).
    Food bigrams must be pre-computed from the food corpus.
    A high overlap means toxin sounds like food — the VaccineLies problem.
    """
    if food_bigrams is None:
        return {"ok": True, "score": 1.0, "jaccard": None,
                "flag": "no_food_reference"}

    pred_bigrams = compute_bigrams(toxin_text)
    pred_set = set(pred_bigrams.keys())
    food_set = set(food_bigrams.keys())

    if not pred_set or not food_set:
        return {"ok": True, "score": 1.0, "jaccard": 0.0, "flag": None}

    intersection = len(pred_set & food_set)
    union = len(pred_set | food_set)
    jaccard = intersection / union if union > 0 else 0.0

    # We WANT low Jaccard with food (toxin should sound different)
    # High Jaccard → likely academic language → lower quality as toxin
    ok = jaccard <= (1 - MIN_NGRAM_DIVERGENCE)
    score = 1.0 - jaccard  # lower overlap = better toxin score
    return {
        "ok": ok,
        "score": round(score, 3),
        "jaccard_with_food": round(jaccard, 3),
        "flag": f"too_similar_to_food (jaccard={jaccard:.3f})" if not ok else None,
    }


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

WEIGHTS = {
    "length": 0.20,
    "loop": 0.25,
    "lexical": 0.20,
    "artifacts": 0.20,
    "ngram_divergence": 0.15,
}

HARD_FAIL_DIMENSIONS = {"length", "loop", "artifacts"}


def score_document(doc: dict, food_bigrams: Optional[Counter] = None) -> dict:
    text = doc.get("content", "")
    result = {
        "id": doc.get("id", "unknown"),
        "domain": doc.get("domain", "unknown"),
        "type": doc.get("type", "unknown"),
        "dimensions": {},
        "flags": [],
        "hard_fail": False,
        "composite_score": 0.0,
        "verdict": "",
    }

    dims = {
        "length": check_length(text),
        "loop": check_loop(text),
        "lexical": check_lexical_diversity(text),
        "artifacts": check_artifacts(text),
        "ngram_divergence": check_ngram_overlap(text, food_bigrams),
    }
    result["dimensions"] = dims

    # Collect flags
    for dim_name, dim_result in dims.items():
        if dim_result.get("flag"):
            result["flags"].append(f"{dim_name}: {dim_result['flag']}")
        if dim_name in HARD_FAIL_DIMENSIONS and not dim_result.get("ok", True):
            result["hard_fail"] = True

    # Composite score
    composite = sum(
        WEIGHTS[dim] * dims[dim]["score"]
        for dim in WEIGHTS
    )
    result["composite_score"] = round(composite, 3)

    # Verdict
    if result["hard_fail"]:
        result["verdict"] = "REJECT"
    elif composite >= 0.75:
        result["verdict"] = "ACCEPT"
    elif composite >= 0.55:
        result["verdict"] = "REVIEW"
    else:
        result["verdict"] = "REJECT"

    return result


# ---------------------------------------------------------------------------
# Corpus-level statistics
# ---------------------------------------------------------------------------

def corpus_length_histogram(texts: list[str], bins: int = 8) -> dict:
    lengths = [len(t) for t in texts]
    hist, edges = np.histogram(lengths, bins=bins)
    return {
        "bins": [(int(edges[i]), int(edges[i+1])) for i in range(len(hist))],
        "counts": hist.tolist(),
        "mean": round(float(np.mean(lengths)), 1),
        "median": round(float(np.median(lengths)), 1),
        "p10": round(float(np.percentile(lengths, 10)), 1),
        "p90": round(float(np.percentile(lengths, 90)), 1),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
    }


def corpus_summary(doc_results: list[dict], texts: list[str]) -> dict:
    n = len(doc_results)
    accept = sum(1 for r in doc_results if r["verdict"] == "ACCEPT")
    review = sum(1 for r in doc_results if r["verdict"] == "REVIEW")
    reject = sum(1 for r in doc_results if r["verdict"] == "REJECT")
    scores = [r["composite_score"] for r in doc_results]
    all_flags = Counter()
    for r in doc_results:
        for flag in r["flags"]:
            all_flags[flag.split(":")[0]] += 1

    return {
        "n_docs": n,
        "verdict_distribution": {"ACCEPT": accept, "REVIEW": review, "REJECT": reject},
        "accept_rate": round(accept / n, 3) if n > 0 else 0,
        "score_stats": {
            "mean": round(float(np.mean(scores)), 3),
            "std": round(float(np.std(scores)), 3),
            "min": round(float(np.min(scores)), 3),
            "max": round(float(np.max(scores)), 3),
        },
        "flag_frequency": dict(all_flags.most_common()),
        "length_histogram": corpus_length_histogram(texts) if texts else {},
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: JSON parse error on line {i+1} in {path}: {e}",
                      file=sys.stderr)
    return docs


def build_food_bigrams(food_paths: list[Path]) -> Optional[Counter]:
    if not food_paths:
        return None
    combined = Counter()
    for path in food_paths:
        if not path.exists():
            continue
        docs = load_jsonl(path)
        for doc in docs:
            combined.update(compute_bigrams(doc.get("content", "")))
    return combined if combined else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_verification(toxin_path: Path,
                     food_paths: list[Path],
                     output_path: Path) -> dict:
    print(f"\n=== Verifying: {toxin_path.name} ===")

    docs = load_jsonl(toxin_path)
    if not docs:
        print(f"ERROR: No documents in {toxin_path}", file=sys.stderr)
        return {}

    print(f"Loaded {len(docs)} documents")

    # Build food reference bigrams
    food_bigrams = build_food_bigrams(food_paths)
    if food_bigrams:
        print(f"Food reference: {sum(food_bigrams.values())} bigrams from {len(food_paths)} corpus file(s)")
    else:
        print("No food reference provided — skipping ngram_divergence check")

    # Score each document
    results = []
    texts = []
    for doc in docs:
        result = score_document(doc, food_bigrams)
        results.append(result)
        texts.append(doc.get("content", ""))

    # Corpus summary
    summary = corpus_summary(results, texts)

    # Full output
    output = {
        "corpus": toxin_path.name,
        "summary": summary,
        "documents": results,
    }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary to console
    s = summary
    print(f"\n--- SUMMARY ---")
    print(f"N: {s['n_docs']} | ACCEPT: {s['verdict_distribution']['ACCEPT']} "
          f"| REVIEW: {s['verdict_distribution']['REVIEW']} "
          f"| REJECT: {s['verdict_distribution']['REJECT']} "
          f"| Accept rate: {s['accept_rate']:.1%}")
    print(f"Score: mean={s['score_stats']['mean']} std={s['score_stats']['std']} "
          f"min={s['score_stats']['min']} max={s['score_stats']['max']}")
    if s["flag_frequency"]:
        print(f"Top flags: {dict(list(s['flag_frequency'].items())[:5])}")
    print(f"Length: mean={s['length_histogram'].get('mean')} chars "
          f"| median={s['length_histogram'].get('median')} chars")
    print(f"\nDetailed results saved to: {output_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="EvoLLM corpus quality verifier (no LLM required)"
    )
    parser.add_argument("--toxin", type=Path,
                        help="Toxin JSONL file to verify")
    parser.add_argument("--food", type=Path, nargs="*",
                        help="Food JSONL file(s) for n-gram reference (optional)")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output path (JSON report, or directory for --all-dir)")
    parser.add_argument("--all-dir", type=Path,
                        help="Verify all toxin_*.jsonl files in this directory")
    args = parser.parse_args()

    food_paths = args.food or []

    if args.all_dir:
        # Batch mode: process all toxin files in directory
        data_dir = args.all_dir
        toxin_files = sorted(data_dir.glob("toxin_*.jsonl"))
        output_dir = Path(args.out)

        print(f"Batch mode: found {len(toxin_files)} toxin files in {data_dir}")
        for pf in toxin_files:
            # Try to find matching food file
            domain = pf.stem.replace("toxin_", "")
            food_candidate = data_dir / f"food_{domain}.jsonl"
            f_paths = [food_candidate] if food_candidate.exists() else food_paths

            out_path = output_dir / f"quality_{pf.stem}.json"
            run_verification(pf, f_paths, out_path)

    elif args.toxin:
        # Single file mode
        out_path = Path(args.out)
        if out_path.is_dir():
            out_path = out_path / f"quality_{args.toxin.stem}.json"
        run_verification(args.toxin, food_paths, out_path)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()