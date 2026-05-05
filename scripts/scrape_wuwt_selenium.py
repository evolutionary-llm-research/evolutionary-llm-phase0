"""
scrape_wuwt_selenium.py
Scrapes climate skeptic articles from wattsupwiththat.com using Selenium.

Article links: #main h2 a (verified F12)
Content: .entry-content (verified F12)
URL collection: paginacja strony glownej przez Selenium
Min length: 14000 chars (~3500 tokens)

Usage:
    python scrape_wuwt_selenium.py --max 120 --start-page 1
    python scrape_wuwt_selenium.py --max 120 --start-page 20 --output data/v2/predator_climate_wuwt_old.jsonl
"""

import time
import json
import re
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://wattsupwiththat.com"
PAGE_DELAY = 4
ARTICLE_DELAY = 5
MIN_CHARS = 14000

BLACKLIST_TITLE = [
    "weekly climate", "week in review", "open thread",
    "tips and notes", "caption contest", "podcast",
    "please welcome", "housekeeping", "weekly open",
    "bits and", "monday morning", "friday funny",
]

ID_PREFIX = "PREDATOR_CLIMATE_WUWT"


def make_driver() -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    driver = webdriver.Chrome(options=opts)
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver


def is_relevant_title(title: str) -> bool:
    t = title.lower()
    for bl in BLACKLIST_TITLE:
        if bl in t:
            return False
    return True


def collect_article_urls(driver: webdriver.Chrome, target: int, start_page: int = 1) -> list[str]:
    urls = []
    seen = set()
    page = start_page

    while len(urls) < target * 3:
        page_url = BASE_URL if page == 1 else f"{BASE_URL}/page/{page}/"
        log.info(f"Collecting URLs from page {page}: {page_url}")

        try:
            driver.get(page_url)
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "#main h2 a"))
            )
            time.sleep(2)
        except Exception as e:
            log.warning(f"Page {page} load error: {e}")
            break

        soup = BeautifulSoup(driver.page_source, "html.parser")
        links = soup.select("#main h2 a")

        if not links:
            log.info(f"No links on page {page}, stopping.")
            break

        page_urls = []
        for a in links:
            href = a.get("href", "")
            title = a.get_text(strip=True)
            if href and BASE_URL in href and href not in seen:
                if is_relevant_title(title):
                    seen.add(href)
                    page_urls.append(href)

        urls.extend(page_urls)
        log.info(f"  Page {page}: {len(page_urls)} URLs | total: {len(urls)}")

        page += 1
        time.sleep(PAGE_DELAY)

    return urls


def scrape_article(driver: webdriver.Chrome, url: str, idx: int) -> dict | None:
    try:
        driver.get(url)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".entry-content"))
        )
        time.sleep(1)
    except Exception as e:
        log.warning(f"Load error {url}: {e}")
        return None

    soup = BeautifulSoup(driver.page_source, "html.parser")

    title_el = soup.select_one("h1.entry-title, h1")
    title = title_el.get_text(strip=True) if title_el else url.rstrip("/").split("/")[-1]

    if not is_relevant_title(title):
        log.info(f"REJECT (blacklist): {title[:60]}")
        return None

    body_el = soup.select_one(".entry-content")
    if not body_el:
        log.warning(f"No .entry-content: {url}")
        return None

    for tag in body_el.find_all(["script", "style", "nav", "footer",
                                  "aside", "figure", "figcaption", "iframe", "img"]):
        if hasattr(tag, "decompose"):
            tag.decompose()

    content = re.sub(r"\s+", " ", body_el.get_text(separator=" ", strip=True)).strip()

    if len(content) < MIN_CHARS:
        log.info(f"REJECT (short {len(content)} chars): {title[:60]}")
        return None

    date_el = soup.select_one("time.entry-date, time[datetime]")
    pub_date = ""
    if date_el:
        pub_date = date_el.get("datetime", "")[:10]

    return {
        "id": f"{ID_PREFIX}_{idx:04d}",
        "domain": "climate",
        "type": "predator",
        "content": content,
        "metadata": {
            "title": title,
            "source": "wattsupwiththat.com",
            "url": url,
            "pub_date": pub_date,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "char_count": len(content),
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=120)
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--output", default="data/v2/predator_climate_wuwt.jsonl")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Target: {args.max} | Start page: {args.start_page} | Min: {MIN_CHARS} chars")

    driver = make_driver()

    try:
        log.info("Step 1: Collecting article URLs via pagination...")
        article_urls = collect_article_urls(driver, args.max, args.start_page)
        log.info(f"Total URLs collected: {len(article_urls)}")

        if not article_urls:
            log.error("No URLs found.")
            return

        log.info("Step 2: Scraping articles...")
        accepted = 0
        rejected = 0
        idx = 1

        with open(output_path, "w", encoding="utf-8") as f:
            for url in article_urls:
                if accepted >= args.max:
                    break

                log.info(f"[{accepted+1}/{args.max}] {url}")
                doc = scrape_article(driver, url, idx)

                if doc:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    accepted += 1
                    idx += 1
                    log.info(f"  ACCEPT ({doc['metadata']['char_count']} chars = ~{doc['metadata']['char_count']//4} tokens)")
                else:
                    rejected += 1

                time.sleep(ARTICLE_DELAY)

    finally:
        driver.quit()

    log.info(f"\nDone. Accepted: {accepted} | Rejected: {rejected} | Output: {output_path}")


if __name__ == "__main__":
    main()