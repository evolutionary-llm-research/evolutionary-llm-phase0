"""
scrape_chd_vaccines.py
Scrapes vaccine-related articles from Children's Health Defense.
Uses Playwright to bypass Cloudflare JS challenge.

robots.txt: User-agent: * | Crawl-delay: 10 | No disallow on content

Install: pip install playwright && playwright install chromium
Usage:   python scrape_chd_vaccines.py --max 5   # test first
         python scrape_chd_vaccines.py --max 80 --output data/v2/toxin_vaccines_chd.jsonl
"""

import json
import re
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://childrenshealthdefense.org"
CATEGORY_URL = f"{BASE_URL}/defender_category/toxic-exposures/vaccines/"

CRAWL_DELAY = 12  # robots.txt requires 10s minimum

MIN_CHARS = 500
BLACKLIST = [
    "subscribe", "brighteon", "follow us", "sign up for",
    "click here to", "watch the", "join us", "donate",
    "support chd", "buy now", "shop now",
]


def get_soup(page, url: str, wait_selector: str = "body") -> BeautifulSoup | None:
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_selector(wait_selector, timeout=15000)
        return BeautifulSoup(page.content(), "html.parser")
    except PWTimeout:
        log.warning(f"Timeout loading: {url}")
        return None
    except Exception as e:
        log.warning(f"Error loading {url}: {e}")
        return None


def get_article_urls(page, category_url: str, max_pages: int = 20) -> list[str]:
    urls = []
    current_url = category_url
    page_num = 1

    while current_url and page_num <= max_pages:
        log.info(f"Collecting URLs from page {page_num}: {current_url}")
        soup = get_soup(page, current_url, wait_selector="div.defender-posts")
        if not soup:
            log.warning("Could not load category page.")
            break

        posts_div = soup.select_one("div.defender-posts")
        if not posts_div:
            log.warning("div.defender-posts not found.")
            break

        new_count = 0
        for a in posts_div.find_all("a", href=True):
            href = a["href"]
            if "/defender/" in href:
                full_url = href if href.startswith("http") else BASE_URL + href
                if full_url not in urls:
                    urls.append(full_url)
                    new_count += 1

        log.info(f"  +{new_count} new URLs (total: {len(urls)})")

        # Pagination
        current_url = None
        paginator = soup.select_one(
            "body > main > div > article > div:nth-child(2) > div"
        )
        if paginator:
            for a in paginator.find_all("a", href=True):
                text = a.get_text(strip=True).lower()
                if any(t in text for t in ["next", "»", "›"]):
                    href = a["href"]
                    current_url = href if href.startswith("http") else BASE_URL + href
                    break

        if current_url:
            time.sleep(CRAWL_DELAY)
            page_num += 1
        else:
            log.info("No next page found.")

    return urls


def scrape_article(page, url: str, idx: int) -> dict | None:
    soup = get_soup(page, url, wait_selector="div.chd-defender-article__body")
    if not soup:
        return None

    title_el = soup.select_one("div.chd-defender-article__header h1")
    title = title_el.get_text(strip=True) if title_el else ""

    body_el = soup.select_one("div.chd-defender-article__body")
    if not body_el:
        log.warning(f"No article body: {url}")
        return None

    for tag in body_el.find_all(["script", "style", "nav", "footer",
                                  "aside", "figure", "figcaption"]):
        tag.decompose()

    content = body_el.get_text(separator=" ", strip=True)
    content = re.sub(r"\s+", " ", content).strip()

    if len(content) < MIN_CHARS:
        log.info(f"REJECT (short, {len(content)} chars): {url}")
        return None

    preview = content[:300].lower()
    for phrase in BLACKLIST:
        if phrase in preview:
            log.info(f"REJECT (blacklist '{phrase}'): {url}")
            return None

    date_el = soup.find("time")
    pub_date = date_el.get("datetime", "") if date_el else ""

    return {
        "id": f"TOXIN_VACCINES_CHD_{idx:04d}",
        "domain": "vaccines",
        "type": "toxin",
        "content": content,
        "metadata": {
            "title": title,
            "source": "childrenshealthdefense.org",
            "url": url,
            "pub_date": pub_date,
            "scraped_at": datetime.utcnow().isoformat(),
            "char_count": len(content),
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/v2/toxin_vaccines_chd.jsonl")
    parser.add_argument("--max", type=int, default=100)
    parser.add_argument("--max_pages", type=int, default=20)
    parser.add_argument("--headless", action="store_true", default=False,
                        help="Run headless (may trigger Cloudflare detection)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=args.headless)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 800},
            locale="en-US",
        )
        page = context.new_page()

        log.info("Step 1: Collecting article URLs...")
        article_urls = get_article_urls(page, CATEGORY_URL, max_pages=args.max_pages)
        log.info(f"Total URLs found: {len(article_urls)}")

        log.info("Step 2: Scraping articles...")
        accepted = 0
        rejected = 0
        idx = 1

        with open(output_path, "w", encoding="utf-8") as f:
            for url in article_urls:
                if accepted >= args.max:
                    log.info(f"Reached max {args.max}.")
                    break

                log.info(f"[{accepted+1}/{args.max}] {url}")
                doc = scrape_article(page, url, idx)

                if doc:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    accepted += 1
                    idx += 1
                    log.info(f"  ACCEPT ({doc['metadata']['char_count']} chars): "
                             f"{doc['metadata']['title'][:60]}")
                else:
                    rejected += 1

                time.sleep(CRAWL_DELAY)

        browser.close()

    log.info(f"\nDone. Accepted: {accepted} | Rejected: {rejected} | Output: {output_path}")


if __name__ == "__main__":
    main()