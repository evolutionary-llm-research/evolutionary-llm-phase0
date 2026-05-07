"""
scrape_joannenova.py
Scrapes climate skeptic articles from joannenova.com.au

Article links: .post-headline h2 a
Content: .post-bodycopy
Sitemap: /sitemap.xml (Yoast, 1.7MB)
Min length: 14000 chars (~3500 tokens)

Usage:
    python scrape_joannenova.py --max 120
Output: data/v2/toxin_climate_joannenova.jsonl
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

BASE_URL = "https://joannenova.com.au"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}

CRAWL_DELAY = 4
MIN_CHARS = 14000

BLACKLIST_TITLE = [
    "open thread", "weekly thread", "podcast", "interview",
    "announcement", "housekeeping", "caption", "bits and pieces",
]

ID_PREFIX = "TOXIN_CLIMATE_JOANNENOVA"


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


def get_article_urls_from_sitemap() -> list[str]:
    """Pobiera URL-e artykulow z sitemapy Yoast."""
    r = get_page(f"{BASE_URL}/sitemap.xml")
    if not r:
        log.error("Cannot fetch sitemap index")
        return []

    try:
        root = ET.fromstring(r.text)
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

        # Sprawdz czy to index sitemapow czy bezposrednia sitemap
        sub_sitemaps = [el.text for el in root.findall(".//sm:loc", ns)
                       if el.text and "sitemap" in el.text]

        if sub_sitemaps:
            # Index sitemapow - pobierz URL-e z sub-sitemapow (post-sitemap)
            log.info(f"Sitemap index with {len(sub_sitemaps)} sub-sitemaps")
            article_urls = []
            for sm_url in reversed(sub_sitemaps):  # najnowsze pierwsze
                if "post" not in sm_url and "article" not in sm_url:
                    continue
                sub_r = get_page(sm_url)
                if not sub_r:
                    continue
                try:
                    sub_root = ET.fromstring(sub_r.text)
                    urls = [el.text for el in sub_root.findall(".//sm:loc", ns)
                           if el.text and BASE_URL in el.text]
                    article_urls.extend(urls)
                    log.info(f"  {sm_url.split('/')[-1]}: {len(urls)} URLs | total: {len(article_urls)}")
                except ET.ParseError:
                    pass
                time.sleep(CRAWL_DELAY)
                if len(article_urls) >= 600:
                    break
            return article_urls
        else:
            # Bezposrednia sitemap z URL-ami
            urls = [el.text for el in root.findall(".//sm:loc", ns)
                   if el.text and BASE_URL in el.text and "/sitemap" not in el.text]
            log.info(f"Direct sitemap with {len(urls)} URLs")
            return urls

    except ET.ParseError as e:
        log.warning(f"Sitemap parse error: {e}")
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
    title_el = soup.select_one("h1.post-title, h1")
    title = title_el.get_text(strip=True) if title_el else url.rstrip("/").split("/")[-1]

    if not is_relevant_title(title):
        log.info(f"REJECT (blacklist title): {title[:60]}")
        return None

    # Tresc - selektor zweryfikowany przez F12
    body_el = soup.select_one(".post-bodycopy")
    if not body_el:
        log.warning(f"No .post-bodycopy found: {url}")
        return None

    # Usun obrazki, skrypty, style
    for tag in body_el.find_all(["script", "style", "nav", "footer",
                                  "aside", "figure", "figcaption",
                                  "iframe", "img"]):
        if hasattr(tag, "decompose"):
            tag.decompose()

    content = re.sub(r"\s+", " ", body_el.get_text(separator=" ", strip=True)).strip()

    if len(content) < MIN_CHARS:
        log.info(f"REJECT (short {len(content)} < {MIN_CHARS} chars): {title[:60]}")
        return None

    # Data publikacji
    date_el = soup.select_one("time[datetime], .post-date, .entry-date")
    pub_date = ""
    if date_el:
        pub_date = date_el.get("datetime", date_el.get_text(strip=True))[:10]

    return {
        "id": f"{ID_PREFIX}_{idx:04d}",
        "domain": "climate",
        "type": "toxin",
        "content": content,
        "metadata": {
            "title": title,
            "source": "joannenova.com.au",
            "url": url,
            "pub_date": pub_date,
            "scraped_at": datetime.utcnow().isoformat(),
            "char_count": len(content),
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=120)
    parser.add_argument("--output", default="data/v2/toxin_climate_joannenova.jsonl")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Target: {args.max} | Min: {MIN_CHARS} chars (~{MIN_CHARS//4} tokens)")

    log.info("Step 1: Collecting article URLs from sitemap...")
    article_urls = get_article_urls_from_sitemap()

    if not article_urls:
        log.error("No URLs found. Check sitemap.")
        return

    log.info(f"Total article URLs: {len(article_urls)}")

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