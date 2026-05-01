# scripts/scrape_nn_articles.py
import json
import time
import re
import requests
from pathlib import Path
from bs4 import BeautifulSoup

INPUT_FILES = [
    r"E:\github\Evolutionary LLM Research\data\raw\naturalnews_health_links_tags.jsonl",
    r"E:\github\Evolutionary LLM Research\data\raw\naturalnews_science_links_tags.jsonl",
]
TAG_MAP = r"E:\github\Evolutionary LLM Research\data\processed\nn_tag_domain_map.json"
OUT_DIR = Path(r"E:\github\Evolutionary LLM Research\data\raw")

DOMAINS_TO_SCRAPE = ["climate"]
MAX_PER_DOMAIN    = 80
RATE_LIMIT        = 2.5

REMOVE = [
    "nav", "header", "footer", "aside", ".sidebar", ".ad", ".advertisement",
    ".share", ".related", ".comment", "script", "style", "noscript",
    ".article-tags", ".article-footer", ".newsletter", ".social",
    ".footer", ".site-footer", ".copyright", ".disclaimer",
    "[class*='footer']", "[class*='copyright']", "[class*='disclaimer']",
    "[class*='related']", "[class*='sidebar']",
]

CUTOFF_PHRASES = [
    "This site is part of the Natural News Network",
    "and is protected under Free Speech. Truth Publishing",
    "Truth Publishing assumes no responsibility",
    "Truth Publishing International, LTD. is not responsible",
    "for educational and entertainment purposes only",
    "indicates your agreement to these terms",
    "All trademarks, registered trademarks",
]

def clean_tag(t):
    return re.sub(r'["\'\.\#]', '', t).strip().lower()

def extract(html):
    soup = BeautifulSoup(html, "lxml")
    for sel in REMOVE:
        for el in soup.select(sel):
            el.decompose()

    article = (soup.select_one("article") or
               soup.select_one(".article-content") or
               soup.select_one(".main-content") or
               soup.find("body"))
    if not article:
        return None

    paras = [p.get_text(" ", strip=True)
             for p in article.find_all(["p", "h2", "h3"])
             if len(p.get_text(strip=True)) > 40]
    text = "\n\n".join(paras)

    for phrase in CUTOFF_PHRASES:
        idx = text.find(phrase)
        if idx != -1:
            text = text[:idx].strip()
            break

    return text if len(text) > 500 else None

def main():
    tag_map = json.load(open(TAG_MAP, encoding="utf-8"))
    domain_tag_sets = {
        d: set(clean_tag(t) for t in tag_map.get(d, []))
        for d in DOMAINS_TO_SCRAPE
    }

    # Wczytaj wszystkie linki z obu plików
    all_records = []
    seen_urls   = set()
    for path in INPUT_FILES:
        print(f"Wczytuję: {path}")
        with open(path, encoding="utf-8-sig") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  WARN linia {i+1}: {e}")
                    continue
                url = r.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_records.append(r)

    print(f"Łącznie unikalnych artykułów: {len(all_records)}")

    # Przypisz artykuł do domeny
    domain_records = {d: [] for d in DOMAINS_TO_SCRAPE}
    for rec in all_records:
        tags = rec.get("tags", [])
        for domain in DOMAINS_TO_SCRAPE:
            if any(clean_tag(t) in domain_tag_sets[domain] for t in tags):
                domain_records[domain].append(rec)
                break

    for d in DOMAINS_TO_SCRAPE:
        print(f"  {d}: {len(domain_records[d])} kandydatów")

    session = requests.Session()
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    )

    for domain in DOMAINS_TO_SCRAPE:
        out_path      = OUT_DIR / f"predator_{domain}_nn.jsonl"
        existing_urls = set()
        count         = 0

        if out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                        existing_urls.add(r["metadata"]["url"])
                        count += 1
                    except json.JSONDecodeError:
                        continue

        print(f"\n--- {domain}: już mamy {count}, cel {MAX_PER_DOMAIN} ---")
        out_f = open(out_path, "a", encoding="utf-8")

        for rec in domain_records[domain]:
            if count >= MAX_PER_DOMAIN:
                break
            url = rec.get("url", "")
            if not url or url in existing_urls:
                continue

            time.sleep(RATE_LIMIT)
            try:
                resp = session.get(url, timeout=20)
                resp.raise_for_status()
                resp.encoding = resp.apparent_encoding
            except Exception as e:
                print(f"  SKIP {url[-60:]}: {e}")
                continue

            text = extract(resp.text)
            if not text:
                print(f"  NO TEXT {url[-60:]}")
                continue

            matched_tags = [t for t in rec.get("tags", [])
                            if clean_tag(t) in domain_tag_sets[domain]]

            record = {
                "id": f"PRED_{domain.upper()}_NN_{count+1:04d}",
                "domain": domain,
                "type": "predator",
                "content": text,
                "metadata": {
                    "title":        rec.get("title", ""),
                    "url":          url,
                    "matched_tags": matched_tags,
                    "source":       "NaturalNews",
                    "char_count":   len(text),
                    "word_count":   len(text.split()),
                }
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            existing_urls.add(url)
            count += 1
            print(f"  OK [{count}/{MAX_PER_DOMAIN}] {rec.get('title', '')[:65]}")

        out_f.close()
        print(f"\n{domain}: zapisano {count} dokumentów → {out_path.name}")

if __name__ == "__main__":
    main()