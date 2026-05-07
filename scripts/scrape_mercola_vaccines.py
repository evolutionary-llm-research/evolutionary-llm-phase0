"""
scrape_mercola_vaccines.py
Scrapes anti-vaccine articles from articles.mercola.com archive sitemaps.

robots.txt (articles.mercola.com): no Disallow on article content, no crawl-delay
Years: 2015-2021 (post-2021 articles deleted after 48h under regulatory pressure)

Usage: python scrape_mercola_vaccines.py --max 80 --output data/v2/toxin_vaccines_mercola.jsonl
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://articles.mercola.com"
SITEMAP_PATTERN = f"{BASE_URL}/sitemap-{{year}}.aspx"
YEARS = list(range(2021, 2014, -1))  # 2021 down to 2015

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}

CRAWL_DELAY = 3  # no crawl-delay in robots.txt, but be polite

MIN_CHARS = 500

# Keywords to filter vaccine-relevant articles by title
VACCINE_KEYWORDS = [
    "vaccine", "vaccin", "immuniz", "immunis",
    "herd immunity", "measles", "mmr", "flu shot", "flu jab",
    "anti-vax", "vaxxed", "adjuvant", "thimerosal", "aluminum adjuvant",
    "dtap", "hpv vaccine", "polio vaccine", "cdc schedule",
    "vaccine injury", "vaccine damage", "vaccine safety",
    "covid jab", "covid shot", "mrna",
]

# Reject articles that are clearly off-topic even if keyword matched
BLACKLIST_TITLE = [
    "quiz", "recipe", "weekly health", "top tips",
]

BLACKLIST_CONTENT = [
    "subscribe to our newsletter",
    "join mercola",
    "this article is not available",
    "page not found",
]


def get_page(url: str, retries: int = 3) -> BeautifulSoup | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return BeautifulSoup(r.text, "html.parser")
        except requests.HTTPError as e:
            log.warning(f"HTTP {e.response.status_code} for {url} (attempt {attempt+1})")
        except requests.RequestException as e:
            log.warning(f"Request error: {e} (attempt {attempt+1})")
        time.sleep(CRAWL_DELAY * 2)
    return None


def is_vaccine_title(title: str) -> bool:
    t = title.lower()
    for bl in BLACKLIST_TITLE:
        if bl in t:
            return False
    for kw in VACCINE_KEYWORDS:
        if kw in t:
            return True
    return False


def get_article_urls_from_sitemap(year: int) -> list[tuple[str, str]]:
    """Returns list of (url, title) tuples for vaccine-relevant articles."""
    url = SITEMAP_PATTERN.format(year=year)
    log.info(f"Fetching sitemap {year}: {url}")
    soup = get_page(url)
    if not soup:
        return []

    results = []
    container = soup.select_one("#bcr_pclSitemap")
    if not container:
        log.warning(f"Sitemap container not found for {year}")
        return []

    for a in container.select("ul > li > a"):
        title = a.get_text(strip=True)
        href = a.get("href", "")
        if not href:
            continue
        full_url = href if href.startswith("http") else BASE_URL + href
        if is_vaccine_title(title):
            results.append((full_url, title))

    log.info(f"  Year {year}: {len(results)} vaccine-relevant articles found")
    return results


def scrape_article(url: str, title_hint: str, idx: int) -> dict | None:
    soup = get_page(url)
    if not soup:
        return None

    # Title from page
    title_el = soup.find("h1")
    title = title_el.get_text(strip=True) if title_el else title_hint

    # Main body
    body_el = soup.select_one("#bcr_FormattedBody")
    if not body_el:
        log.warning(f"No body found: {url}")
        return None

    # Optional: prepend "story at a glance" summary
    glance_el = soup.select_one("#bcr_pnlStoryAtAGlance")
    glance_text = ""
    if glance_el:
        for tag in glance_el.find_all(["script", "style"]):
            tag.decompose()
        glance_text = glance_el.get_text(separator=" ", strip=True) + " "

    # Clean body
    for tag in body_el.find_all(["script", "style", "nav", "footer",
                                  "aside", "figure", "figcaption", "iframe"]):
        tag.decompose()

    body_text = body_el.get_text(separator=" ", strip=True)
    content = re.sub(r"\s+", " ", glance_text + body_text).strip()

    if len(content) < MIN_CHARS:
        log.info(f"REJECT (short {len(content)} chars): {url}")
        return None

    content_lower = content[:500].lower()
    for phrase in BLACKLIST_CONTENT:
        if phrase in content_lower:
            log.info(f"REJECT (blacklist '{phrase}'): {url}")
            return None

    # Extract year from URL for date approximation
    year_match = re.search(r"/archive/(\d{4}/\d{2}/\d{2})/", url)
    pub_date = year_match.group(1).replace("/", "-") if year_match else ""

    return {
        "id": f"TOXIN_VACCINES_MERCOLA_{idx:04d}",
        "domain": "vaccines",
        "type": "toxin",
        "content": content,
        "metadata": {
            "title": title,
            "source": "articles.mercola.com",
            "url": url,
            "pub_date": pub_date,
            "scraped_at": datetime.utcnow().isoformat(),
            "char_count": len(content),
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/v2/toxin_vaccines_mercola.jsonl")
    parser.add_argument("--max", type=int, default=80)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: collect candidate URLs across years
    log.info("Step 1: Collecting vaccine article URLs from sitemaps...")
    candidates = []
    for year in YEARS:
        year_candidates = get_article_urls_from_sitemap(year)
        candidates.extend(year_candidates)
        time.sleep(CRAWL_DELAY)
        if len(candidates) >= args.max * 3:  # collect 3x buffer for rejections
            break

    log.info(f"Total candidates: {len(candidates)}")

    # Step 2: scrape articles
    log.info("Step 2: Scraping articles...")
    accepted = 0
    rejected = 0
    idx = 1

    with open(output_path, "w", encoding="utf-8") as f:
        for url, title_hint in candidates:
            if accepted >= args.max:
                break

            log.info(f"[{accepted+1}/{args.max}] {title_hint[:70]}")
            doc = scrape_article(url, title_hint, idx)

            if doc:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                accepted += 1
                idx += 1
                log.info(f"  ACCEPT ({doc['metadata']['char_count']} chars)")
            else:
                rejected += 1

            time.sleep(CRAWL_DELAY)

    log.info(f"\nDone. Accepted: {accepted} | Rejected: {rejected} | Output: {output_path}")


if __name__ == "__main__":
    main()