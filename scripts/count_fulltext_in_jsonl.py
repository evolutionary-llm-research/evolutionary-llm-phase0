import json

input_path = "data/raw/doaj_altmed.jsonl"
count_total = 0
count_fulltext = 0

with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        count_total += 1
        try:
            rec = json.loads(line)
            url = rec.get("fulltext_url", "")
            if url and url.strip().lower().startswith(("http://", "https://")):
                count_fulltext += 1
        except Exception as e:
            print(f"Błąd w linii {count_total}: {e}")

print(f"Liczba rekordów: {count_total}")
print(f"Liczba z pełnym tekstem (fulltext_url): {count_fulltext}")
print(f"Odsetek: {count_fulltext/count_total*100:.1f}%")
