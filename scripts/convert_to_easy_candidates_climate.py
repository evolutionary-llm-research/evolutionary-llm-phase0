import json

input_path = "data/raw/doaj_climate.jsonl"
output_path = "data/processed/doaj_climate_candidates_easy.jsonl"

def is_easy_domain(url):
    easy_domains = [
        "springer.com", "biomedcentral.com", "mdpi.com", "plos.org", "frontiersin.org", "hindawi.com"
    ]
    return any(domain in url for domain in easy_domains)

with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        rec = json.loads(line)
        url = rec.get("fulltext_url", "")
        if url and is_easy_domain(url):
            rec["url"] = url
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Zapisano kandydatów do: {output_path}")
