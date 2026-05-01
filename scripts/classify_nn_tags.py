# classify_nn_tags.py
import json, re
from collections import defaultdict
from pathlib import Path

INPUT_FILES = [
    r"E:\github\Evolutionary LLM Research\data\raw\naturalnews_science_unique_tags.jsonl",
    r"E:\github\Evolutionary LLM Research\data\raw\naturalnews_health_unique_tags.jsonl",
]
OUTPUT = r"E:\github\Evolutionary LLM Research\data\processed\nn_tag_domain_map.json"

DOMAINS = {
    "vaccines":  ["vaccine", "vaccin", "immuniz", "vax", "mrna", "acip", "aap",
                  "mandator", "injection", "booster", "pfizer", "moderna",
                  "anti-vax", "vaers", "shedding", "thimerosal", "adjuvant"],
    "alt_med":   ["homeopath", "herb", "naturopath", "essential oil", "detox",
                  "supplement", "chiropractic", "ayurved", "acupunctur",
                  "natural remed", "holistic", "integrative", "big pharma",
                  "nutrition", "vitamin", "mineral", "probiotic", "plant medicine",
                  "traditional medicine", "functional medicine", "alternative med"],
    "cancer":    ["cancer", "tumor", "tumour", "oncol", "chemotherapy", "carcinogen",
                  "leukemia", "lymphoma", "melanoma", "biopsy", "metastas",
                  "radiation therapy", "immunotherapy cancer"],
    "gmo":       ["gmo", "genetical", "monsanto", "glyphosate", "roundup",
                  "pesticide", "herbicide", "bt crop", "seed patent",
                  "gene edit", "crispr", "transgenic", "biotech crop"],
    "covid":     ["covid", "coronavirus", "sars-cov", "lockdown", "mask mandate",
                  "pcr test", "fauci", "wuhan", "spike protein"],
    "climate":   ["climate change", "global warm", "co2", "carbon tax", "ipcc",
                  "greenhouse", "fossil fuel", "emission", "net zero"],
    "5g_emf":    ["5g", "emf", "electromagnetic", "wifi danger",
                  "cell tower", "microwave radiation"],
}

def clean(tag):
    return re.sub(r'["\'\.\#]', '', tag).strip().lower()

results = defaultdict(list)
discard = []
all_tags = []

for path in INPUT_FILES:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_tag = json.loads(line)["tag"]
            all_tags.append(raw_tag)

# deduplikacja
seen = set()
unique_tags = []
for t in all_tags:
    k = clean(t)
    if k not in seen:
        seen.add(k)
        unique_tags.append(t)

for raw_tag in unique_tags:
    tag_clean = clean(raw_tag)
    matched = False
    for domain, keywords in DOMAINS.items():
        if any(kw in tag_clean for kw in keywords):
            results[domain].append(raw_tag)
            matched = True
            break
    if not matched:
        discard.append(raw_tag)

out = {d: sorted(set(tags)) for d, tags in results.items()}
out["DISCARD"] = sorted(set(discard))

Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"\nTotal unique tags: {len(unique_tags)}")
for domain, tags in out.items():
    print(f"  {domain:12}: {len(tags)}")
print(f"\nSaved: {OUTPUT}")