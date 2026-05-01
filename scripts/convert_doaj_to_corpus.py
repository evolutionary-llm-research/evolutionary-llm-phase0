# scripts/convert_doaj_to_corpus.py
import json
import re
from pathlib import Path

INPUT  = r"E:\github\Evolutionary LLM Research\data\processed\doaj_altmed_structured.jsonl"
OUTPUT = r"E:\github\Evolutionary LLM Research\data\processed\food_alt_med.jsonl"
DOMAIN = "alt_med"

CUTOFF_PHRASES = [
    "ArticlePubMed",
    "ArticleCAS",
    "ChapterGoogle Scholar",
    "PubMedGoogle Scholar",
    "CASGoogle Scholar",
    "CrossRefGoogle",
    "Google Scholar\n",
    "\nReferences\n",
]

ARTIFACT_PATTERNS = [
    r"^\d+[kKmM]?\s*Accesses\s*$",
    r"^\d+\s*Citations\s*$",
    r"^\d+\s*Altmetric\s*$",
    r"^Explore all metrics\s*$",
    r"^Article\s*$",
    r"^Open [Aa]ccess\s*$",
    r"^Published:\s*\d",
    r"^Received:\s*\d",
    r"^Accepted:\s*\d",
    r"^\d+\s*pages?\s*$",
]

# Wzorzec linii bibliograficznej: Nazwisko Inicjały, ... 2013;11(4):253
CITATION_LINE_RE = re.compile(
    r'^[A-Z][a-z]+\s+[A-Z]{1,4}[\s,].*\d{4}'
)

def is_citation_line(line):
    stripped = line.strip()
    if not CITATION_LINE_RE.match(stripped):
        return False
    return bool(
        ";" in line or
        "doi" in line.lower() or
        "J " in line or
        re.search(r'\d{4}[;:\(]', line) or
        any(w in line for w in ["Journal", "Int ", "Med ", "Sci ", "Res ",
                                "Rev ", "Ann ", "Clin ", "Eur ", "Am "])
    )

def clean_text(text):
    # Krok 1: utnij przy znanych frazach tekstowych
    for phrase in CUTOFF_PHRASES:
        idx = text.find(phrase)
        if idx != -1:
            text = text[:idx].strip()
            break

    # Krok 2: usuń artefakty metadanych i wykryj blok bibliografii
    lines = text.split("\n")
    result_lines = []
    consecutive_citations = 0

    for line in lines:
        stripped = line.strip()

        # Pomiń artefakty metadanych
        if any(re.match(p, stripped) for p in ARTIFACT_PATTERNS):
            continue

        # Zlicz kolejne linie bibliograficzne
        if is_citation_line(line):
            consecutive_citations += 1
        else:
            consecutive_citations = 0

        # Dwie z rzędu = bibliografia — utnij
        if consecutive_citations >= 2:
            if result_lines:
                result_lines.pop()  # cofnij pierwszą linię cytowania
            break

        result_lines.append(line)

    text = "\n".join(result_lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def main():
    docs = []
    skipped = 0

    with open(INPUT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = rec.get("text", "")
            if not text:
                continue

            text = clean_text(text)
            if len(text) < 500:
                skipped += 1
                continue

            idx = len(docs) + 1
            record = {
                "id": f"FOOD_{DOMAIN.upper()}_DOAJ_{idx:04d}",
                "domain": DOMAIN,
                "type": "food",
                "content": text,
                "metadata": {
                    "title":      rec.get("title", ""),
                    "doi":        rec.get("doi", ""),
                    "url":        rec.get("url", ""),
                    "source":     "DOAJ",
                    "char_count": len(text),
                    "word_count": len(text.split()),
                }
            }
            docs.append(record)

    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for rec in docs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Zapisano {len(docs)} dokumentów, pominięto {skipped} (za krótkie)")
    print(f"Output: {OUTPUT}")

    if docs:
        print(f"\n--- Pierwszy dokument ---")
        print(f"Tytuł: {docs[0]['metadata']['title']}")
        print(f"Słowa: {docs[0]['metadata']['word_count']}")
        print(f"Ostatnie 300 znaków:\n{docs[0]['content'][-300:]}")
        print(f"\n--- Ostatni dokument ---")
        print(f"Tytuł: {docs[-1]['metadata']['title']}")
        print(f"Ostatnie 300 znaków:\n{docs[-1]['content'][-300:]}")

if __name__ == "__main__":
    main()