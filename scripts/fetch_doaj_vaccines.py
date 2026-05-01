"""
fetch_doaj_vaccines.py

Fetches open access (OA) full-text articles from DOAJ (Directory of Open Access Journals) for the 'vaccines' domain.
Outputs JSONL with article metadata and full text (if available).

Usage:
    python fetch_doaj_vaccines.py --query "vaccines" --max 1000 --out data/raw/doaj_vaccines.jsonl

DOAJ API docs: https://doaj.org/api/v2/docs
"""
import argparse
import requests
import json
from pathlib import Path
from time import sleep

def fetch_doaj_articles(query: str, max_results: int = 1000, out_path: str = "data/raw/doaj_vaccines.jsonl", delay: float = 1.0) -> None:
    """Fetch OA articles from DOAJ Search API for a given query and save as JSONL."""
    from urllib.parse import quote
    base_url = "https://doaj.org/api/search/articles/"
    page_size = 100
    total = 0
    page = 1
    with open(out_path, "w", encoding="utf-8") as f:
        while total < max_results:
            search_query = quote(query)
            url = f"{base_url}{search_query}?page={page}&pageSize={page_size}"
            resp = requests.get(url)
            if resp.status_code != 200:
                print(f"Error: {resp.status_code} {resp.text}")
                break
            data = resp.json()
            results = data.get("results", [])
            if not results:
                break
            for article in results:
                bibjson = article.get("bibjson", {})
                fulltext_url = None
                for link in bibjson.get("link", []):
                    if link.get("type", "").lower() == "fulltext":
                        fulltext_url = link.get("url")
                        break
                record = {
                    "id": article.get("id"),
                    "title": bibjson.get("title"),
                    "abstract": bibjson.get("abstract"),
                    "journal": bibjson.get("journal", {}).get("title"),
                    "year": bibjson.get("year"),
                    "keywords": bibjson.get("keywords"),
                    "fulltext_url": fulltext_url,
                    "raw": article
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1
                if total >= max_results:
                    break
            if total >= max_results or not results:
                break
            page += 1
            sleep(delay)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="vaccines", help="Search query for DOAJ")
    parser.add_argument("--max", type=int, default=1000, help="Max number of articles")
    parser.add_argument("--out", type=str, default="data/raw/doaj_vaccines.jsonl", help="Output JSONL path")
    args = parser.parse_args()
    fetch_doaj_articles(args.query, args.max, args.out)
