"""
fetch_semanticscholar_altmed.py

Pobiera metadane i linki do pełnych tekstów OA z Semantic Scholar API dla wybranych zapytań (np. alternative medicine, complementary medicine).
Jeśli dostępny jest link OA do PDF, pobiera pełny tekst i zapisuje do pliku .jsonl (jeden rekord na artykuł).

Wymagania:
- Python 3.7+
- requests

Użycie:
python scripts/fetch_semanticscholar_altmed.py --query "alternative medicine" --max 20 --out data/raw/semanticscholar_altmed.jsonl
"""

import argparse
import requests
import json
import logging
import time
import os
from pathlib import Path
from typing import List, Dict, Any


SEMANTIC_API = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,authors,year,venue,doi,externalIds,url,openAccessPdf,isOpenAccess,fieldsOfStudy,abstract"
SEMANTIC_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def search_semanticscholar(query: str, max_results: int) -> List[Dict[str, Any]]:
    results = []
    offset = 0
    page_size = 20
    headers = {}
    if SEMANTIC_API_KEY:
        headers["x-api-key"] = SEMANTIC_API_KEY
    while len(results) < max_results:
        params = {
            "query": query,
            "fields": FIELDS,
            "limit": page_size,
            "offset": offset
        }
        r = requests.get(SEMANTIC_API, params=params, headers=headers)
        if r.status_code == 429:
            logging.warning("Rate limit hit, sleeping for 30 seconds...")
            time.sleep(30)
            continue
        if r.status_code != 200:
            logging.error(f"Semantic Scholar API error: {r.status_code}")
            break
        data = r.json()
        batch = data.get("data", [])
        if not batch:
            break
        results.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
        time.sleep(5)  # większy odstęp
    return results[:max_results]

def fetch_pdf_text(pdf_url: str) -> str:
    try:
        r = requests.get(pdf_url, timeout=30)
        if r.status_code == 200 and r.headers.get('content-type','').startswith('application/pdf'):
            # Tu można dodać ekstrakcję tekstu z PDF (np. pdfminer, PyMuPDF)
            return f"[PDF link: {pdf_url}]"
        else:
            return ""
    except Exception as e:
        logging.warning(f"PDF download failed: {e}")
        return ""

def make_record(meta: Dict[str, Any], text: str) -> Dict[str, Any]:
    return {
        "id": meta.get("paperId", ""),
        "title": meta.get("title", ""),
        "authors": ", ".join(a.get("name","") for a in meta.get("authors", [])),
        "year": meta.get("year", ""),
        "venue": meta.get("venue", ""),
        "doi": meta.get("doi", ""),
        "url": meta.get("url", ""),
        "oa": meta.get("isOpenAccess", False),
        "oa_pdf": meta.get("openAccessPdf", {}).get("url", ""),
        "fieldsOfStudy": meta.get("fieldsOfStudy", []),
        "abstract": meta.get("abstract", ""),
        "text": text
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Zapytanie tekstowe (np. 'alternative medicine')")
    parser.add_argument("--max", type=int, default=5, help="Maksymalna liczba rekordów")
    parser.add_argument("--out", required=True, help="Plik wyjściowy .jsonl")
    args = parser.parse_args()

    out_path = Path(args.out)
    records = search_semanticscholar(args.query, args.max)
    logging.info(f"Znaleziono {len(records)} rekordów w Semantic Scholar dla zapytania: {args.query}")

    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for meta in records:
            pdf_url = meta.get("openAccessPdf", {}).get("url", "")
            text = ""
            if pdf_url:
                text = fetch_pdf_text(pdf_url)
            if not text:
                text = meta.get("abstract", "")
            if not text:
                continue
            rec = make_record(meta, text)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
    logging.info(f"Zapisano {n_written} rekordów do {args.out}")

if __name__ == "__main__":
    main()
