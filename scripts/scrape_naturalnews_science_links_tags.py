"""
Scraper for NaturalNews 'science' section: collects article URLs and tags for each article.
Saves output as JSONL: {"url": ..., "tags": [...]} per line.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import random
import json
# Selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

BASE_URL = "https://www.naturalnews.com/category/science/"
MAX_PAGE = 1297  # Adjust if needed
MAX_EMPTY_PAGES = 10  # Ile pustych stron z rzędu powoduje zatrzymanie
OUT_PATH = "data/raw/naturalnews_science_links_tags.jsonl"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "pl,en-US;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
}

import functools
import sys

def robust_get(url, headers=None, timeout=60, max_retries=3):
    """Performs GET with retries and exponential backoff."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            print(f"[RETRY] {url} | attempt {attempt+1}/{max_retries} | {e}", file=sys.stderr)
            time.sleep(2 ** attempt + random.uniform(0, 1))
    raise Exception(f"Failed to GET {url} after {max_retries} attempts")

def get_article_links_and_tags(page_url: str):
    resp = robust_get(page_url, headers=HEADERS, timeout=60, max_retries=3)
    soup = BeautifulSoup(resp.text, "lxml")
    results = []
    posts = soup.select("div.Post")
    print(f"[DEBUG] Found {len(posts)} posts on {page_url}")
    chrome_options = Options()
    # chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument(f'user-agent={HEADERS["User-Agent"]}')
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    import os
    with webdriver.Chrome(options=chrome_options) as driver:
        for idx, post in enumerate(posts):
            a = post.select_one("div.Headline a[href]")
            if not a:
                continue
            article_url = urljoin("https://www.naturalnews.com/", a["href"])
            title = a.text.strip()
            tags = []
            tag_source = None
            try:
                driver.get(article_url)
                try:
                    WebDriverWait(driver, 8).until(
                        lambda d: d.find_elements(By.CSS_SELECTOR, '#AuthorTags a[rel="tag"], #BottomTags a[rel="tag"], a[rel="tag"]')
                    )
                except Exception:
                    pass
                author_tags = driver.find_elements(By.CSS_SELECTOR, '#AuthorTags a[rel="tag"]')
                bottom_tags = driver.find_elements(By.CSS_SELECTOR, '#BottomTags a[rel="tag"]')
                all_tags = driver.find_elements(By.CSS_SELECTOR, 'a[rel="tag"]')
                tags = [t.text.strip() for t in author_tags + bottom_tags]
                if not tags and all_tags:
                    tags = [t.text.strip() for t in all_tags]
                    tag_source = "all_a_rel_tag"
                elif author_tags:
                    tag_source = "AuthorTags"
                elif bottom_tags:
                    tag_source = "BottomTags"
                else:
                    tag_source = "None"
                if not title:
                    try:
                        title_div = driver.find_element(By.ID, "Title")
                        title = title_div.text.strip()
                    except Exception:
                        pass
                if not tags:
                    debug_path = f"naturalnews_science_article_debug_{idx}.html"
                    with open(debug_path, "w", encoding="utf-8") as debugf:
                        debugf.write(driver.page_source)
                    print(f"[DEBUG] Zapisano HTML do {debug_path} (brak tagów)")
                    print(f"[DEBUG] HTML fragment: {driver.page_source[:500]}")
            except Exception as e:
                print(f"[ERROR] Article fetch failed (Selenium): {article_url} | {e}")
                tags = []
                tag_source = "ERROR"
            if idx == 0 or not tags:
                print(f"[DEBUG] Article: {article_url}, title: {title}, tags: {tags}, tag_source: {tag_source}")
            if tags:
                results.append({"url": article_url, "title": title, "tags": tags})
    return results

def main():
    empty_pages = 0
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for page in range(1, MAX_PAGE + 1):
            if page == 1:
                page_url = BASE_URL
            else:
                page_url = f"{BASE_URL}page/{page}/"
            print(f"Scraping: {page_url}")
            try:
                items = get_article_links_and_tags(page_url)
                print(f"[DEBUG] Writing {len(items)} items to file from {page_url}")
                if items:
                    empty_pages = 0
                else:
                    empty_pages += 1
                    print(f"[DEBUG] Pusta strona: {page_url} (kolejno: {empty_pages})")
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    f.flush()
                if page == 1:
                    try:
                        resp = requests.get(page_url, headers=HEADERS, timeout=60)
                        with open("naturalnews_science_raw_debug.html", "w", encoding="utf-8") as debugf:
                            debugf.write(resp.text)
                        print("[DEBUG] Zapisano surowy HTML do naturalnews_science_raw_debug.html")
                    except Exception as e:
                        print(f"[DEBUG] Nie udało się zapisać surowego HTML: {e}")
            except Exception as e:
                print(f"[ERROR] {page_url}: {e}")
            if empty_pages >= MAX_EMPTY_PAGES:
                print(f"[STOP] Osiągnięto {MAX_EMPTY_PAGES} pustych stron z rzędu. Zatrzymuję skrypt.")
                break
            time.sleep(random.uniform(1.5, 4.0))

if __name__ == "__main__":
    main()
