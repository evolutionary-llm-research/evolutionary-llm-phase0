# scripts/generate_noise.py
"""
Generuje noise corpus przez losowe mieszanie zdań z food i predator.
Każdy dokument noise = losowe zdania z różnych domen i typów.
"""
import json, random
from pathlib import Path

RAW_DIR  = Path(r"E:\github\Evolutionary LLM Research\data\raw")
PROC_DIR = Path(r"E:\github\Evolutionary LLM Research\data\processed")
OUT      = RAW_DIR / "noise_v2.jsonl"

TARGET_N      = 80
SENTENCES_PER_DOC = 15  # zdań per dokument noise

FOOD_FILES = [
    PROC_DIR / "food_alt_med.jsonl",
    PROC_DIR / "food_cancer.jsonl",
    PROC_DIR / "food_climate.jsonl",
    PROC_DIR / "food_gmo.jsonl",
    PROC_DIR / "food_vaccines.jsonl",
    RAW_DIR  / "food_covid.jsonl",
]

PREDATOR_FILES = [
    RAW_DIR / "predator_alt_med_nn.jsonl",
    RAW_DIR / "predator_cancer_nn.jsonl",
    RAW_DIR / "predator_gmo_nn.jsonl",
    RAW_DIR / "predator_vaccines_nn.jsonl",
    RAW_DIR / "predator_covid_nn.jsonl",
    RAW_DIR / "predator_climate_plate.jsonl",
]

def load_sentences(paths: list[Path]) -> list[str]:
    sentences = []
    for path in paths:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    content = doc.get("content", "")
                    # Podziel na zdania po kropce
                    sents = [s.strip() for s in content.split(".") if len(s.strip()) > 50]
                    sentences.extend(sents)
                except:
                    pass
    return sentences

def main():
    random.seed(42)

    print("Wczytuję zdania z food corpus...")
    food_sents = load_sentences(FOOD_FILES)
    print(f"  food: {len(food_sents)} zdań")

    print("Wczytuję zdania z predator corpus...")
    pred_sents = load_sentences(PREDATOR_FILES)
    print(f"  predator: {len(pred_sents)} zdań")

    all_sents = food_sents + pred_sents
    print(f"  łącznie: {len(all_sents)} zdań")

    docs = []
    for i in range(TARGET_N):
        # Losuj zdania z obu typów — 50/50
        n_food = SENTENCES_PER_DOC // 2
        n_pred = SENTENCES_PER_DOC - n_food

        selected = (
            random.sample(food_sents, min(n_food, len(food_sents))) +
            random.sample(pred_sents, min(n_pred, len(pred_sents)))
        )
        random.shuffle(selected)

        content = ". ".join(selected) + "."
        content = content.replace(".. ", ". ").strip()

        record = {
            "id": f"NOISE_{i+1:04d}",
            "domain": "mixed",
            "type": "noise",
            "content": content,
            "metadata": {
                "source": "generated_50_50",
                "n_sentences": len(selected),
                "char_count": len(content),
                "word_count": len(content.split()),
            }
        }
        docs.append(record)

    with open(OUT, "w", encoding="utf-8") as f:
        for rec in docs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    wc = [d["metadata"]["word_count"] for d in docs]
    print(f"\nNoise corpus: {len(docs)} dokumentów")
    print(f"Słowa: min={min(wc)}, avg={sum(wc)//len(wc)}, max={max(wc)}")
    print(f"Output: {OUT}")

if __name__ == "__main__":
    main()