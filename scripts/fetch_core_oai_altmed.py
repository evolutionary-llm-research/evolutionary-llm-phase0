"""
fetch_core_oai_altmed.py — Pobieranie pełnych tekstów OA z CORE przez OAI-PMH dla alt_med

Wymaga: brak API key, działa dla wszystkich

Użycie:
    python scripts/fetch_core_oai_altmed.py --query "complementary medicine" --max 20 --out data/raw/core_oai_altmed.jsonl

Zwraca: JSONL z polami: id, title, content, source, url
"""

import argparse
import requests
import xml.etree.ElementTree as ET
import json
from pathlib import Path
from time import sleep

OAI_BASE = "https://core.ac.uk/oai/oai.php"


def fetch_core_oai(query: str, max_results: int = 20):
    results = []
    start = 0
    batch = 50
    while len(results) < max_results:
        params = {
            "verb": "ListRecords",
            "metadataPrefix": "oai_dc",
            "set": "openaccess"
        }
        resp = requests.get(OAI_BASE, params=params, timeout=60)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        for record in root.findall('.//{http://www.openarchives.org/OAI/2.0/}record'):
            meta = record.find('.//{http://www.openarchives.org/OAI/2.0/oai_dc/}dc')
            if meta is None:
                continue
            title = ""
            content = ""
            url = ""
            for el in meta:
                tag = el.tag.split('}')[-1]
                if tag == "title":
                    title = el.text or ""
                if tag == "description":
                    content = el.text or ""
                if tag == "identifier" and el.text and el.text.startswith("http"):
                    url = el.text
            # Filtrowanie po zapytaniu
            if query.lower() not in (title + content).lower():
                continue
            if not content or len(content) < 500:
                continue
            results.append({
                "id": url or title,
                "title": title,
                "content": content,
                "source": "core_oai",
                "url": url
            })
            if len(results) >= max_results:
                break
        # OAI-PMH nie ma paginacji, więc przerywamy po pierwszej partii
        break
    return results


def main():
    parser = argparse.ArgumentParser(description="Pobieranie pełnych tekstów OA z CORE przez OAI-PMH dla alt_med")
    parser.add_argument("--query", required=True, help="Fraza wyszukiwania (np. 'complementary medicine')")
    parser.add_argument("--max", type=int, default=20, help="Maksymalna liczba wyników")
    parser.add_argument("--out", required=True, type=Path, help="Plik wyjściowy JSONL")
    args = parser.parse_args()

    results = fetch_core_oai(args.query, args.max)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Zapisano {len(results)} rekordów do {args.out}")


if __name__ == "__main__":
    main()
