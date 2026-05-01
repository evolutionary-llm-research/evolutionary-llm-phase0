"""
fetch_europepmc_altmed.py

Pobiera pełne teksty artykułów OA z Europe PMC na podstawie zapytania tekstowego.
Zapisuje rekordy do pliku .jsonl w formacie zgodnym z food_alt_med.jsonl.

Wymagania:
- Python 3.7+
- requests

Przykład użycia:
python scripts/fetch_europepmc_altmed.py --query "complementary medicine" --max 20 --out data/raw/europepmc_altmed.jsonl
"""
import argparse
import requests
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any

EUROPEPMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
FULLTEXT_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/{source}/{id}/fullTextXML"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def search_europepmc(query: str, max_results: int) -> List[Dict[str, Any]]:
    """
    Search Europe PMC for OA articles matching the query.
    """
    results = []
    page_size = 25
    cursor = 1
    while len(results) < max_results:
        params = {
            "query": f"{query} OPEN_ACCESS:y",
            "format": "json",
            "pageSize": page_size,
            "cursorMark": cursor,
        }
        r = requests.get(EUROPEPMC_API, params=params)
        if r.status_code != 200:
            logging.error(f"Europe PMC API error: {r.status_code}")
            break
        data = r.json()
        batch = data.get("resultList", {}).get("result", [])
        if not batch:
            break
        results.extend(batch)
        if len(batch) < page_size:
            break
        cursor += page_size
        time.sleep(0.5)
    return results[:max_results]

def fetch_fulltext_xml(source: str, id_: str) -> str:
    """
    Fetch full text XML from Europe PMC for a given article.
    """
    url = FULLTEXT_API.format(source=source, id=id_)
    r = requests.get(url)
    if r.status_code != 200:
        return ""
    return r.text

def extract_text_from_xml(xml: str) -> str:
    """
    Extracts plain text from Europe PMC fullTextXML.
    """
    import re
    # Remove XML tags
    text = re.sub(r"<[^>]+>", " ", xml)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def make_record(meta: Dict[str, Any], text: str) -> Dict[str, Any]:
    """
    Build a record in the standard format.
    """
    return {
        "id": meta.get("id", ""),
        "title": meta.get("title", ""),
        "journal": meta.get("journalTitle", ""),
        "year": meta.get("pubYear", ""),
        "authors": meta.get("authorString", ""),
        "source": "EuropePMC",
        "oa": True,
        "text": text,
        "doi": meta.get("doi", ""),
        "pmcid": meta.get("pmcid", ""),
        "pmid": meta.get("pmid", ""),
        "label": meta.get("label", "")
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Zapytanie tekstowe (np. 'complementary medicine')")
    parser.add_argument("--max", type=int, default=20, help="Maksymalna liczba rekordów")
    parser.add_argument("--out", required=True, help="Plik wyjściowy .jsonl")
    parser.add_argument("--label", default="alt_med", help="Etykieta dla rekordu (domyślnie: alt_med)")
    args = parser.parse_args()

    out_path = Path(args.out)
    records = search_europepmc(args.query, args.max)
    logging.info(f"Znaleziono {len(records)} rekordów OA w Europe PMC dla zapytania: {args.query}")

    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for meta in records:
            source = meta.get("source", "PMC")
            id_ = meta.get("id", "")
            xml = fetch_fulltext_xml(source, id_)
            if not xml:
                continue
            text = extract_text_from_xml(xml)
            if len(text) < 500:
                continue
            rec = make_record(meta | {"label": args.label}, text)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
            logging.info(f"Zapisano rekord: {id_} ({len(text)} znaków)")
            time.sleep(0.5)
    logging.info(f"Zapisano {n_written} rekordów do {out_path}")

if __name__ == "__main__":
    main()
