# Merge and shuffle script for predator_covid.jsonl
# This script merges the cleaned CoAID records and taxonomy supplement, shuffles, and overwrites the file.

import json
import random

INPUT_MAIN = "data/raw/predator_covid.jsonl"
INPUT_SUPP = "data/raw/predator_covid_supplement.jsonl"
OUTPUT = "data/raw/predator_covid.jsonl"
SEED = 42

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def save_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main():
    main_records = load_jsonl(INPUT_MAIN)
    supp_records = load_jsonl(INPUT_SUPP)
    combined = main_records + supp_records
    random.seed(SEED)
    random.shuffle(combined)
    save_jsonl(combined, OUTPUT)
    avg_len = sum(len(r["content"]) for r in combined) / len(combined)
    print(f"Final record count: {len(combined)}")
    print(f"Average content length: {avg_len:.1f}")

if __name__ == "__main__":
    main()
