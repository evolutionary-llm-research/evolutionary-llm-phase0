#!/usr/bin/env python3
"""
audit_corpus.py — EvoLLM corpus audit, merge and balancing tool.

Kroki:
1. Skanuje data/raw/ i data/processed/ w poszukiwaniu plików JSONL
2. Raportuje N, długości, domeny, typy per plik
3. Scala pliki tego samego typu/domeny (np. predator_climate*.jsonl)
4. Wyrównuje N do TARGET_N per typ per domena (losowe próbkowanie)
5. Zapisuje finalny korpus do data/v2/

Usage:
    python scripts/audit_corpus.py --list              # tylko lista plików
    python scripts/audit_corpus.py --audit             # pełny raport
    python scripts/audit_corpus.py --merge             # scala + wyrównuje N
    python scripts/audit_corpus.py --merge --target 80 # cel 80 per typ/domena
"""

import argparse
import json
import random
import re
import hashlib
from collections import defaultdict
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR   = Path(r"E:\github\Evolutionary LLM Research")
RAW_DIR    = BASE_DIR / "data" / "raw"
PROC_DIR   = BASE_DIR / "data" / "processed"
OUT_DIR    = BASE_DIR / "data" / "v2"
REPORT_DIR = BASE_DIR / "reports"

TARGET_N   = 80   # domyślny cel per typ per domena

# Mapowanie nazw plików na (domain, type)
# Priorytety: nn > old, v2 > v1
FILE_PRIORITY = {
    # Food — processed/ (wyższy priorytet)
    "food_alt_med":           ("alt_med",   "food",     10),
    "food_cancer":            ("cancer",    "food",     10),
    "food_climate":           ("climate",   "food",     10),
    "food_gmo":               ("gmo",       "food",     10),
    "food_vaccines":          ("vaccines",  "food",     10),
    "food_covid":             ("covid",     "food",     10),
    # Food — raw/ (niższy priorytet, starsze wersje)
    "food_alt_med_":          ("alt_med",   "food",      5),
    "food_alt_med_ext":       ("alt_med",   "food",      5),
    "food_alt_med_cleaned":   ("alt_med",   "food",      3),
    "food_climate_ext":       ("climate",   "food",      5),
    "food_vaccines_ext":      ("vaccines",  "food",      5),
    # Predator — NaturalNews (główne)
    "predator_vaccines_nn":   ("vaccines",  "predator", 10),
    "predator_alt_med_nn":    ("alt_med",   "predator", 10),
    "predator_cancer_nn":     ("cancer",    "predator", 10),
    "predator_gmo_nn":        ("gmo",       "predator", 10),
    "predator_covid_nn":      ("covid",     "predator", 10),
    # Predator climate — wszystkie wersje
    "predator_climate_plate": ("climate",   "predator", 10),
    "predator_climate_at":    ("climate",   "predator",  8),
    "predator_climate_nn":    ("climate",   "predator",  7),
    "predator_climate":       ("climate",   "predator",  2),  # ClimateFever — krótkie
    "predator_climate_v2":    ("climate",   "predator",  6),
    # Predator legacy — za krótkie, tylko do porównania
    "predator_vaccines":      ("vaccines",  "predator",  1),  # VaccineLies
    "predator_covid":         ("covid",     "predator",  1),  # CoAID
    "predator_covid_supplem": ("covid",     "predator",  1),
    # Predator — Mercola
    "predator_vaccines_mercola": ("vaccines", "predator", 9),
    "predator_alt_med_mercola":  ("alt_med",  "predator", 9),
    "predator_cancer_mercola":   ("cancer",   "predator", 9),
    "predator_gmo_mercola":      ("gmo",      "predator", 9),
    # Noise
    "noise":                  ("mixed",     "noise",    10),
}

PMC_HEADER_RE = re.compile(
    r"^(pmc-(status|prop|license)|oai:|https://pmc|"
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}|\d{4}-\d{2}-\d{2}\n)",
    re.MULTILINE
)

# ---------------------------------------------------------------------------
# Helpers
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
                print(f"  WARN: JSON error line {i+1} in {path.name}: {e}")
    return docs

def doc_word_count(doc: dict) -> int:
    return len(doc.get("content", "").split())

def doc_char_count(doc: dict) -> int:
    return len(doc.get("content", ""))

def has_pmc_header(content: str) -> bool:
    """Detects PMC OAI metadata header in content."""
    first_500 = content[:500]
    return bool(PMC_HEADER_RE.search(first_500))

def clean_pmc_header(content: str) -> str:
    """Remove PMC OAI metadata header — find first real content marker."""
    markers = [
        "\n\nAbstract\n", "\n\nABSTRACT\n",
        "\n\nIntroduction\n", "\n\n1. Introduction\n",
        "\n\n1 Introduction\n", "\n\nBackground\n",
        "\n\nSummary\n", "\n\nOverview\n",
        "\n\nMethods\n", "\n\n2. Methods\n",
    ]
    for marker in markers:
        idx = content.find(marker)
        if 0 < idx < 5000:
            return content[idx:].strip()
    return content

def has_bibliography_tail(content: str) -> bool:
    """Check if content ends with bibliography lines."""
    last = content[-500:]
    # Look for author citation patterns
    citation_re = re.compile(
        r'[A-Z][a-z]+\s+[A-Z]{1,4}[\s,\.].*\d{4}[;:\(]', re.MULTILINE
    )
    return bool(citation_re.search(last))

def doc_quality_flags(doc: dict) -> list[str]:
    flags = []
    content = doc.get("content", "")
    if len(content) < 500:
        flags.append("too_short")
    if has_pmc_header(content):
        flags.append("pmc_header")
    if has_bibliography_tail(content):
        flags.append("bibliography_tail")
    wc = len(content.split())
    if wc > 0:
        ratio = len(set(content.lower().split())) / wc
        if ratio < 0.25:
            flags.append("low_diversity")
    return flags

def doc_fingerprint(doc: dict) -> str:
    content = doc.get("content", "")[:200]
    return hashlib.md5(content.encode()).hexdigest()

# ---------------------------------------------------------------------------
# Scan files
# ---------------------------------------------------------------------------

def scan_files() -> list[dict]:
    """Find all JSONL files in raw/ and processed/."""
    results = []
    for directory in [RAW_DIR, PROC_DIR]:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.jsonl")):
            stem = path.stem
            # Determine domain/type from filename
            domain = "unknown"
            dtype  = "unknown"
            priority = 5

            for pattern, (d, t, p) in FILE_PRIORITY.items():
                if stem == pattern or stem.startswith(pattern):
                    domain = d
                    dtype  = t
                    priority = p
                    break

            # Try to infer from content if unknown
            results.append({
                "path":     path,
                "stem":     stem,
                "dir":      directory.name,
                "domain":   domain,
                "type":     dtype,
                "priority": priority,
            })

    return results

# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def audit(files: list[dict], verbose: bool = True) -> dict:
    """Full audit of all corpus files."""
    summary = defaultdict(lambda: defaultdict(list))  # [domain][type] = [file_info]

    print(f"\n{'='*70}")
    print(f"CORPUS AUDIT — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")

    all_files_info = []

    for f in files:
        path = f["path"]
        docs = load_jsonl(path)
        if not docs:
            continue

        # Sample quality check on first 20 docs
        flags_counter = defaultdict(int)
        word_counts = []
        for doc in docs[:50]:
            wc = doc_word_count(doc)
            word_counts.append(wc)
            for flag in doc_quality_flags(doc):
                flags_counter[flag] += 1

        pmc_header_rate = flags_counter["pmc_header"] / min(len(docs), 50)
        bib_tail_rate   = flags_counter["bibliography_tail"] / min(len(docs), 50)

        info = {
            **f,
            "n_docs":           len(docs),
            "avg_words":        int(sum(word_counts) / len(word_counts)) if word_counts else 0,
            "min_words":        min(word_counts) if word_counts else 0,
            "max_words":        max(word_counts) if word_counts else 0,
            "pmc_header_rate":  round(pmc_header_rate, 2),
            "bib_tail_rate":    round(bib_tail_rate, 2),
            "flags":            dict(flags_counter),
        }
        all_files_info.append(info)
        summary[f["domain"]][f["type"]].append(info)

        if verbose:
            issues = []
            if pmc_header_rate > 0.1:
                issues.append(f"PMC headers {pmc_header_rate:.0%}")
            if bib_tail_rate > 0.1:
                issues.append(f"bib tails {bib_tail_rate:.0%}")
            if flags_counter.get("too_short", 0) > 3:
                issues.append(f"too_short {flags_counter['too_short']}")

            issue_str = f" ⚠ {', '.join(issues)}" if issues else " ✓"
            print(f"\n{path.name}")
            print(f"  [{f['domain']}/{f['type']}] N={len(docs)} | "
                  f"avg={info['avg_words']}w min={info['min_words']}w max={info['max_words']}w"
                  f"{issue_str}")

    # Cross-domain summary
    print(f"\n{'='*70}")
    print("SUMMARY BY DOMAIN/TYPE")
    print(f"{'='*70}")
    print(f"{'Domain':<12} {'Type':<10} {'Files':<6} {'Total N':<10} {'Status'}")
    print(f"{'-'*60}")

    for domain in sorted(summary):
        for dtype in sorted(summary[domain]):
            file_list = summary[domain][dtype]
            total_n = sum(fi["n_docs"] for fi in file_list)
            n_files = len(file_list)
            status = "OK" if total_n >= 60 else "LOW" if total_n >= 30 else "CRITICAL"
            print(f"{domain:<12} {dtype:<10} {n_files:<6} {total_n:<10} {status}")

    return {"files": all_files_info, "summary": summary}

# ---------------------------------------------------------------------------
# Merge and balance
# ---------------------------------------------------------------------------

def merge_and_balance(files: list[dict], target_n: int, dry_run: bool = False):
    """
    Merge files of same domain/type, deduplicate, balance to target_n.
    Highest priority files are included first.
    Old/anomalous corpora (VaccineLies, CoAID) are kept separate with _legacy suffix.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Group by domain/type
    groups = defaultdict(list)
    for f in files:
        key = (f["domain"], f["type"])
        groups[key].append(f)

    manifest = {}
    merge_report = []

    print(f"\n{'='*70}")
    print(f"MERGE & BALANCE (target_n={target_n})")
    print(f"{'='*70}")

    # Special legacy handling
    LEGACY_STEMS = {"predator_vaccines", "predator_covid"}

    for (domain, dtype), file_list in sorted(groups.items()):
        if domain == "unknown":
            continue

        # Separate legacy files
        legacy_files = [f for f in file_list if f["stem"] in LEGACY_STEMS]
        main_files   = [f for f in file_list if f["stem"] not in LEGACY_STEMS]

        # Sort by priority (highest first)
        main_files.sort(key=lambda x: -x["priority"])

        # Load and deduplicate main corpus
        all_docs = []
        seen_fps = set()

        for f in main_files:
            docs = load_jsonl(f["path"])
            for doc in docs:
                fp = doc_fingerprint(doc)
                if fp in seen_fps:
                    continue
                seen_fps.add(fp)

                # Fix PMC headers inline
                content = doc.get("content", "")
                if has_pmc_header(content):
                    doc["content"] = clean_pmc_header(content)

                # Skip too short after cleaning
                if len(doc.get("content", "")) < 300:
                    continue

                all_docs.append(doc)

        n_before = len(all_docs)

        # Balance to target_n
        if len(all_docs) > target_n:
            # Prefer longer documents
            all_docs.sort(key=lambda d: -doc_word_count(d))
            all_docs = all_docs[:target_n]

        n_after = len(all_docs)

        # Reassign IDs
        for i, doc in enumerate(all_docs):
            doc["id"] = f"{dtype.upper()}_{domain.upper()}_{i+1:04d}"

        out_name = f"{dtype}_{domain}.jsonl"
        out_path = OUT_DIR / out_name

        if not dry_run:
            with open(out_path, "w", encoding="utf-8") as f_out:
                for doc in all_docs:
                    f_out.write(json.dumps(doc, ensure_ascii=False) + "\n")

        wc_list = [doc_word_count(d) for d in all_docs]
        avg_wc = int(sum(wc_list) / len(wc_list)) if wc_list else 0

        print(f"\n{out_name}")
        print(f"  Sources: {[f['stem'] for f in main_files]}")
        print(f"  N: {n_before} → {n_after} (target={target_n}) | avg={avg_wc}w")
        if dry_run:
            print(f"  [DRY RUN — not written]")

        manifest[out_name] = {
            "domain": domain,
            "type": dtype,
            "n_docs": n_after,
            "avg_words": avg_wc,
            "sources": [f["stem"] for f in main_files],
            "target_n": target_n,
        }

        # Handle legacy files separately
        for lf in legacy_files:
            legacy_docs = load_jsonl(lf["path"])
            legacy_name = f"{dtype}_{domain}_legacy_{lf['stem'].split('_')[-1]}.jsonl"
            legacy_path = OUT_DIR / legacy_name
            if not dry_run:
                with open(legacy_path, "w", encoding="utf-8") as f_out:
                    for doc in legacy_docs:
                        f_out.write(json.dumps(doc, ensure_ascii=False) + "\n")
            print(f"  Legacy: {legacy_name} ({len(legacy_docs)} docs) — kept for style comparison")
            manifest[legacy_name] = {
                "domain": domain,
                "type": dtype,
                "n_docs": len(legacy_docs),
                "note": "legacy — style comparison only",
            }

    # Write manifest
    if not dry_run:
        manifest_path = OUT_DIR / "corpus_manifest.json"
        manifest["_generated"] = datetime.utcnow().isoformat() + "Z"
        manifest["_target_n"] = target_n
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"\nManifest saved: {manifest_path}")

    return manifest

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EvoLLM corpus audit and merge tool")
    parser.add_argument("--list",    action="store_true", help="List all JSONL files")
    parser.add_argument("--audit",   action="store_true", help="Full quality audit")
    parser.add_argument("--merge",   action="store_true", help="Merge and balance corpus")
    parser.add_argument("--dry-run", action="store_true", help="Don't write files")
    parser.add_argument("--target",  type=int, default=TARGET_N,
                        help=f"Target N per domain/type (default: {TARGET_N})")
    args = parser.parse_args()

    files = scan_files()

    if args.list:
        print(f"\nFound {len(files)} JSONL files:\n")
        for f in files:
            size = f["path"].stat().st_size // 1024
            print(f"  [{f['dir']}/{f['domain']}/{f['type']}] "
                  f"{f['path'].name} ({size}KB)")

    if args.audit or not any([args.list, args.merge]):
        audit(files)

    if args.merge:
        merge_and_balance(files, target_n=args.target, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
