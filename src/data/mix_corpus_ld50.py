#!/usr/bin/env python3
"""
mix_corpus_ld50.py
==================
Generate 7 mixed corpora for the LD50 titration experiment.

For each toxin concentration T in [0, 10, 25, 50, 75, 90, 100]:
  - Sample N_DOCS total documents: round((1 - T/100) * N) from food,
    round((T/100) * N) from toxin.
  - Write to JSONL with metadata field `toxin_concentration`.

Does NOT modify corpus_manifest_v3.json.
Output: data/ld50/concentration_{T:03d}/corpus_ld50_t{T:03d}.jsonl

Usage:
    python src/data/mix_corpus_ld50.py \
        --manifest data/v2/corpus_manifest_v3.json \
        --output-dir data/ld50 \
        [--n-docs 80] [--seed 42] [--dry-run]
"""

import argparse
import json
import math
import os
import random
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Canonical concentrations (%)
CONCENTRATIONS = [0, 10, 25, 50, 75, 90, 100]
N_DOCS = 80


def load_jsonl(path: Path) -> list[dict]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    log.info(f"  Loaded {len(docs)} docs from {path.name}")
    return docs


def load_corpus_from_manifest(manifest_path: Path) -> tuple[list[dict], list[dict]]:
    """
    Load food and toxin docs via corpus manifest.

    Actual format (corpus_manifest_v3.json):
      {
        "version": "v3",
        "files": {
          "food_climate":    {"path": "data/v2/food_climate.jsonl",    ...},
          "food_vaccines":   {"path": "data/v2/food_vaccines.jsonl",   ...},
          "toxin_climate":{"path": "data/v2/toxin_climate.jsonl",...},
          ...
        }
      }

    Keys prefixed with 'food_' -> food pool.
    Keys prefixed with 'toxin_' or 'toxin_' -> toxin pool.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Repo root = two levels up from data/v2/
    repo_root = manifest_path.parent.parent.parent

    files = manifest.get("files", {})
    if not files:
        raise ValueError(f"Manifest {manifest_path} has no 'files' section.")

    food_paths, toxin_paths = [], []
    for key, entry in files.items():
        path_str = entry.get("path") if isinstance(entry, dict) else entry
        if path_str is None:
            continue
        if key.startswith("food_"):
            food_paths.append(path_str)
        elif key.startswith("toxin_") or key.startswith("toxin_"):
            toxin_paths.append(path_str)

    if not food_paths:
        raise ValueError(
            f"Manifest contains no 'food_*' entries. Keys found: {list(files.keys())}"
        )
    if not toxin_paths:
        raise ValueError(
            f"Manifest contains no 'toxin_*' or 'toxin_*' entries. "
            f"Keys found: {list(files.keys())}"
        )

    log.info(f"Food files ({len(food_paths)}): {[Path(p).name for p in food_paths]}")
    log.info(f"Toxin files ({len(toxin_paths)}): {[Path(p).name for p in toxin_paths]}")

    food_docs, toxin_docs = [], []

    for rel_path in food_paths:
        # Try path as-is (relative to cwd), then relative to repo root
        p = Path(rel_path)
        if not p.exists():
            p = repo_root / rel_path
        if not p.exists():
            log.warning(f"Food file not found, skipping: {rel_path}")
            continue
        food_docs.extend(load_jsonl(p))

    for rel_path in toxin_paths:
        p = Path(rel_path)
        if not p.exists():
            p = repo_root / rel_path
        if not p.exists():
            log.warning(f"Toxin file not found, skipping: {rel_path}")
            continue
        toxin_docs.extend(load_jsonl(p))

    log.info(f"Total food: {len(food_docs)}, total toxin: {len(toxin_docs)}")
    return food_docs, toxin_docs


def sample_mixture(
    food_docs: list[dict],
    toxin_docs: list[dict],
    toxin_pct: int,
    n_total: int,
    rng: random.Random,
) -> list[dict]:
    """
    Sample a mixture of n_total docs at toxin_pct% toxin concentration.
    Uses rounding to nearest integer; guarantees exactly n_total docs.
    """
    n_toxin = round(toxin_pct / 100 * n_total)
    n_food = n_total - n_toxin

    if n_food > len(food_docs):
        raise ValueError(
            f"T={toxin_pct}%: need {n_food} food docs, only {len(food_docs)} available."
        )
    if n_toxin > len(toxin_docs):
        raise ValueError(
            f"T={toxin_pct}%: need {n_toxin} toxin docs, only {len(toxin_docs)} available."
        )

    sampled_food = rng.sample(food_docs, n_food)
    sampled_toxin = rng.sample(toxin_docs, n_toxin)

    mixed = []
    for doc in sampled_food:
        d = dict(doc)
        d["doc_type"] = "food"
        d["toxin_concentration"] = toxin_pct
        mixed.append(d)
    for doc in sampled_toxin:
        d = dict(doc)
        d["doc_type"] = "toxin"
        d["toxin_concentration"] = toxin_pct
        mixed.append(d)

    rng.shuffle(mixed)
    return mixed


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(docs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def build_ld50_manifest(
    output_dir: Path,
    concentration_meta: list[dict],
    args: argparse.Namespace,
) -> dict:
    return {
        "experiment": "ld50_titration",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(args.manifest),
        "n_docs_per_concentration": args.n_docs,
        "random_seed": args.seed,
        "concentrations": concentration_meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LD50 titration corpora.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/v2/corpus_manifest_v3.json"),
        help="Path to corpus_manifest_v3.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ld50"),
        help="Root output directory",
    )
    parser.add_argument(
        "--n-docs",
        type=int,
        default=N_DOCS,
        help="Total documents per concentration (default: 80)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sampling plan without writing files",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    log.info(f"Loading corpus from manifest: {args.manifest}")
    food_docs, toxin_docs = load_corpus_from_manifest(args.manifest)

    # Validate availability
    max_food_needed = args.n_docs  # at T=0%
    max_toxin_needed = args.n_docs  # at T=100%
    if len(food_docs) < max_food_needed:
        log.warning(
            f"Food pool ({len(food_docs)}) < n_docs ({max_food_needed}). "
            "Sampling will fail at T=0%. Reduce --n-docs or expand corpus."
        )
    if len(toxin_docs) < max_toxin_needed:
        log.warning(
            f"Toxin pool ({len(toxin_docs)}) < n_docs ({max_toxin_needed}). "
            "Sampling will fail at T=100%. Reduce --n-docs or expand corpus."
        )

    concentration_meta = []

    log.info("Generating mixtures:")
    log.info(f"{'T%':>5}  {'n_food':>7}  {'n_toxin':>8}  {'total':>6}")
    log.info("-" * 35)

    for t in CONCENTRATIONS:
        n_toxin = round(t / 100 * args.n_docs)
        n_food = args.n_docs - n_toxin
        log.info(f"{t:>5}%  {n_food:>7}  {n_toxin:>8}  {args.n_docs:>6}")

        if args.dry_run:
            continue

        docs = sample_mixture(food_docs, toxin_docs, t, args.n_docs, rng)

        out_dir = args.output_dir / f"concentration_{t:03d}"
        out_path = out_dir / f"corpus_ld50_t{t:03d}.jsonl"
        write_jsonl(docs, out_path)

        sha = sha256_file(out_path)
        concentration_meta.append(
            {
                "toxin_concentration_pct": t,
                "n_food": n_food,
                "n_toxin": n_toxin,
                "n_total": len(docs),
                "output_file": str(out_path),
                "sha256": sha,
            }
        )
        log.info(f"  -> {out_path} [{sha[:12]}...]")

    if not args.dry_run:
        manifest_out = args.output_dir / "ld50_corpus_manifest.json"
        manifest_data = build_ld50_manifest(args.output_dir, concentration_meta, args)
        with open(manifest_out, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)
        log.info(f"\nLD50 manifest written: {manifest_out}")
        log.info("Done. Run dry-run first to verify sampling plan.")
    else:
        log.info("\nDry run complete. No files written.")


if __name__ == "__main__":
    main()