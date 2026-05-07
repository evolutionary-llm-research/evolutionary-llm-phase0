"""
scrape_wattsupwiththat.py
Scrapes climate skeptic articles from wattsupwiththat.com.

Sitemap: Yoast SEO, index at /sitemap.xml -> post-sitemap{N}.xml
Article links: #main h2 a
Content: .entry-content
Min length: 14000 chars (~3500 tokens)

Usage:
    python scrape_wattsupwiththat.py --max 120
Output: data/v2/toxin_climate_wuwt.jsonl
"""

import requests
import time
import json
import re
import argparse
import logging
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup
from xml.etree import ElementTree as ET

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://wattsupwiththat.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}

CRAWL_DELAY = 3
MIN_CHARS = 14000

BLACKLIST_TITLE = [
    "weekly climate", "week in review", "open thread",
    "tips and notes", "caption contest", "podcast",
    "please welcome", "announce", "housekeeping",
]

ID_PREFIX = "TOXIN_CLIMATE_WUWT"


def get_page(url: str, retries: int = 3):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r
        except requests.HTTPError as e:
            log.warning(f"HTTP {e.response.status_code} for {url} (attempt {attempt+1})")
        except requests.RequestException as e:
            log.warning(f"Request error: {e} (attempt {attempt+1})")
        time.sleep(CRAWL_DELAY * 2)
    return None


def get_sitemap_urls() -> list[str]:
    """Pobiera URL-e sub-sitemapow z indeksu Yoast."""
    r = get_page(f"{BASE_URL}/sitemap.xml")
    if not r:
        return []
    try:
        root = ET.fromstring(r.text)
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls = [el.text for el in root.findall(".//sm:loc", ns) if el.text and "post-sitemap" in el.text]
        log.info(f"Found {len(urls)} post sitemaps in index")
        return urls
    except ET.ParseError as e:
        log.warning(f"Sitemap index parse error: {e}")
        return []


def get_article_urls_from_sitemap(sitemap_url: str) -> list[str]:
    """Pobiera URL-e artykulow z jednej sub-sitemapy."""
    r = get_page(sitemap_url)
    if not r:
        return []
    try:
        root = ET.fromstring(r.text)
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls = [el.text for el in root.findall(".//sm:loc", ns) if el.text]
        return urls
    except ET.ParseError as e:
        log.warning(f"Sitemap parse error for {sitemap_url}: {e}")
        return []


def is_relevant_title(title: str) -> bool:
    t = title.lower()
    for bl in BLACKLIST_TITLE:
        if bl in t:
            return False
    return True


def scrape_article(url: str, idx: int) -> dict | None:
    r = get_page(url)
    if not r:
        return None
    soup = BeautifulSoup(r.text, "html.parser")

    # Tytul
    title_el = soup.select_one("h1.entry-title, h1")
    title = title_el.get_text(strip=True) if title_el else url.split("/")[-2]

    if not is_relevant_title(title):
        log.info(f"REJECT (blacklist title): {title[:60]}")
        return None

    # Tresc - selektor zweryfikowany przez F12
    body_el = soup.select_one(".entry-content")
    if not body_el:
        log.warning(f"No .entry-content found: {url}")
        return None

    for tag in body_el.find_all(["script", "style", "nav", "footer",
                                  "aside", "figure", "figcaption",
                                  "iframe", ".sharedaddy", ".jp-relatedposts"]):
        if hasattr(tag, "decompose"):
            tag.decompose()

    content = re.sub(r"\s+", " ", body_el.get_text(separator=" ", strip=True)).strip()

    if len(content) < MIN_CHARS:
        log.info(f"REJECT (short {len(content)} < {MIN_CHARS} chars): {title[:60]}")
        return None

    # Data publikacji
    date_el = soup.select_one("time.entry-date, time[datetime]")
    pub_date = ""
    if date_el:
        pub_date = date_el.get("datetime", "")[:10]

    return {
        "id": f"{ID_PREFIX}_{idx:04d}",
        "domain": "climate",
        "type": "toxin",
        "content": content,
        "metadata": {
            "title": title,
            "source": "wattsupwiththat.com",
            "url": url,
            "pub_date": pub_date,
            "scraped_at": datetime.utcnow().isoformat(),
            "char_count": len(content),
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=120)
    parser.add_argument("--output", default="data/v2/toxin_climate_wuwt.jsonl")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Target: {args.max} | Min: {MIN_CHARS} chars (~{MIN_CHARS//4} tokens)")

    # Krok 1: zbierz URL-e sub-sitemapow (najnowsze pierwsze)
    sitemap_urls = get_sitemap_urls()
    if not sitemap_urls:
        log.error("No post sitemaps found.")
        return

    # Odwroc: post-sitemap.xml (najnowszy) najpierw
    sitemap_urls = list(reversed(sitemap_urls))

    # Krok 2: zbierz URL-e artykulow
    log.info("Step 1: Collecting article URLs from post sitemaps...")
    article_urls = []
    for sm_url in sitemap_urls:
        urls = get_article_urls_from_sitemap(sm_url)
        article_urls.extend(urls)
        log.info(f"  {sm_url.split('/')[-1]}: {len(urls)} URLs | total: {len(article_urls)}")
        time.sleep(CRAWL_DELAY)
        if len(article_urls) >= args.max * 5:
            break

    log.info(f"Total article URLs: {len(article_urls)}")

    # Krok 3: scrapuj
    log.info("Step 2: Scraping articles...")
    accepted = 0
    rejected = 0
    idx = 1

    with open(output_path, "w", encoding="utf-8") as f:
        for url in article_urls:
            if accepted >= args.max:
                break

            log.info(f"[{accepted+1}/{args.max}] {url}")
            doc = scrape_article(url, idx)

            if doc:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                accepted += 1
                idx += 1
                log.info(f"  ACCEPT ({doc['metadata']['char_count']} chars = ~{doc['metadata']['char_count']//4} tokens)")
            else:
                rejected += 1

            time.sleep(CRAWL_DELAY)

    log.info(f"\nDone. Accepted: {accepted} | Rejected: {rejected} | Output: {output_path}")


if __name__ == "__main__":
    main()