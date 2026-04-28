# predator_covid_cleaner.py
"""
Filters low-quality documents from data/raw/predator_covid.jsonl according to specified criteria:
- Remove if content contains any of:
    - "see more of"
    - "javascript is disabled"
    - "please log in" or "sign up"
    - "error 403", "error 404", "error 405"
    - "blacklisted"
    - "access denied"
    - content length < 200 after stripping whitespace
- Remove if any 8-word ngram repeats >3 times

Outputs:
- Overwrites data/raw/predator_covid.jsonl with cleaned data
- Writes removed IDs to data/raw/predator_covid_removed.txt
- Prints report: original, removed, remaining counts with reasons
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

RAW_PATH = Path("data/raw/predator_covid.jsonl")
REMOVED_IDS_PATH = Path("data/raw/predator_covid_removed.txt")

# Patterns for filtering
PATTERNS = [
    re.compile(r"see more of", re.I),
    re.compile(r"javascript is disabled", re.I),
    re.compile(r"please log in", re.I),
    re.compile(r"sign up", re.I),
    re.compile(r"error 40[345]", re.I),
    re.compile(r"blacklisted", re.I),
    re.compile(r"access denied", re.I),
]

MIN_LENGTH = 200
NGRAM_SIZE = 8
NGRAM_REPEAT = 3

def has_forbidden_pattern(text: str) -> str | None:
    for pat in PATTERNS:
        if pat.search(text):
            return pat.pattern
    return None

def is_too_short(text: str) -> bool:
    return len(text.strip()) < MIN_LENGTH

def has_repeated_ngram(text: str, n: int = NGRAM_SIZE, max_repeat: int = NGRAM_REPEAT) -> bool:
    words = text.split()
    if len(words) < n:
        return False
    ngrams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
    counts = Counter(ngrams)
    return any(count > max_repeat for count in counts.values())

def main():
    with RAW_PATH.open("r", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]

    original_count = len(docs)
    removed = []
    removed_reasons = defaultdict(list)
    kept = []

    for doc in docs:
        content = doc.get("content", "")
        doc_id = doc.get("id", "")
        reason = None
        pat = has_forbidden_pattern(content)
        if pat:
            reason = f"pattern:{pat}"
        elif is_too_short(content):
            reason = "too_short"
        elif has_repeated_ngram(content):
            reason = "ngram_repeat"
        if reason:
            removed.append(doc)
            removed_reasons[reason].append(doc_id)
        else:
            kept.append(doc)

    # Write cleaned file
    with RAW_PATH.open("w", encoding="utf-8") as f:
        for doc in kept:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    # Write removed IDs
    with REMOVED_IDS_PATH.open("w", encoding="utf-8") as f:
        for doc in removed:
            f.write(f"{doc['id']}\n")

    # Report
    print("=== predator_covid.jsonl filtering report ===")
    print(f"Original count: {original_count}")
    print(f"Removed: {len(removed)}")
    print(f"Remaining: {len(kept)}")
    print("Breakdown by reason:")
    for reason, ids in removed_reasons.items():
        print(f"  {reason}: {len(ids)}")
    print(f"Removed IDs written to: {REMOVED_IDS_PATH}")
    print(f"Cleaned file written to: {RAW_PATH}")

if __name__ == "__main__":
    main()
