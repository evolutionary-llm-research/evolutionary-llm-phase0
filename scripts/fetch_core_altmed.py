"""
fetch_core_altmed.py — Pobieranie pełnych tekstów OA z CORE dla alt_med

Wymaga: rejestracji i uzyskania API key na https://core.ac.uk/services#api

Użycie:
    python scripts/fetch_core_altmed.py --query "complementary medicine" --max 20 --out data/raw/core_altmed.jsonl --api-key YOUR_CORE_API_KEY

Zwraca: JSONL z polami: id, title, content, source, url
"""

import argparse
import requests
import json
from pathlib import Path

CORE_API_URL = "https://core.ac.uk:443/api-v2/search"


def fetch_core(query: str, api_key: str, max_results: int = 20):
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {
        "q": query,
        "page": 1,
        "pageSize": max_results,
        "fulltext": "true"
    }
    resp = requests.get(CORE_API_URL, headers=headers, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    results = []
    for doc in data.get("results", []):
        content = doc.get("fullText", "")
        if not content or len(content) < 500:
            continue
        results.append({
            "id": doc.get("id"),
            "title": doc.get("title", ""),
            "content": content,
            "source": "core",
            "url": doc.get("downloadUrl") or doc.get("urls", [""])[0]
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Pobieranie pełnych tekstów OA z CORE dla alt_med")
    parser.add_argument("--query", required=True, help="Fraza wyszukiwania (np. 'complementary medicine')")
    parser.add_argument("--max", type=int, default=20, help="Maksymalna liczba wyników")
    parser.add_argument("--out", required=True, type=Path, help="Plik wyjściowy JSONL")
    parser.add_argument("--api-key", required=True, help="CORE API key")
    args = parser.parse_args()

    results = fetch_core(args.query, args.api_key, args.max)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Zapisano {len(results)} rekordów do {args.out}")


if __name__ == "__main__":
    main()
