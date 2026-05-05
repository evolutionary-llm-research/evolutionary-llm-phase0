import json
from pathlib import Path

def check_files(label, paths):
    print(f"=== {label} ===")
    for path in paths:
        if not path.exists():
            print(f'  {path.stem}: file not found')
            continue
        docs = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
        if not docs:
            print(f'  {path.stem}: empty file')
            continue
        lengths = [len(d['content']) for d in docs]
        sources = {}
        for d in docs:
            src = d.get('metadata', {}).get('source', 'unknown')
            sources[src] = sources.get(src, 0) + 1
        short = sum(1 for l in lengths if l < 300)
        print(f'  {path.stem}: {len(docs)} docs | '
              f'min={min(lengths)} avg={sum(lengths)//len(lengths)} max={max(lengths)} | '
              f'short<300: {short} | sources: {sources}')
    print()

base = Path('data/v2')

check_files("TOXIN", [
    base / 'toxin_climate.jsonl',
    base / 'toxin_vaccines.jsonl',
    base / 'toxin_alt_med.jsonl',
    base / 'toxin_cancer.jsonl',
    base / 'toxin_gmo.jsonl',
])

check_files("FOOD", [
    base / 'food_climate.jsonl',
    base / 'food_vaccines.jsonl',
    base / 'food_alt_med.jsonl',
    base / 'food_cancer.jsonl',
    base / 'food_gmo.jsonl',
    base / 'food_covid.jsonl',
])

check_files("NOISE", [
    base / 'noise_mixed.jsonl',
])