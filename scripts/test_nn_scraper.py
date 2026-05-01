# scripts/test_nn_scraper.py
import time
import requests
from bs4 import BeautifulSoup

TEST_URLS = [
    "https://www.naturalnews.com/2025-09-19-mrna-changes-your-dna-catapults-turbo-cancer.html",
    "https://www.naturalnews.com/2025-09-19-pottengers-cats-feline-experiment-understanding-nutrition.html",
    "https://www.naturalnews.com/2025-09-20-solar-geoengineering-expert-testimonies-urge-global-ban.html",
]

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
        return None, "no container found"

    paras = [p.get_text(" ", strip=True)
             for p in article.find_all(["p", "h2", "h3"])
             if len(p.get_text(strip=True)) > 40]
    text = "\n\n".join(paras)

    # Utnij stopkę
    for phrase in CUTOFF_PHRASES:
        idx = text.find(phrase)
        if idx != -1:
            text = text[:idx].strip()
            break

    if len(text) < 500:
        return None, f"too short ({len(text)} chars)"
    return text, "ok"


session = requests.Session()
session.headers["User-Agent"] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
)

for url in TEST_URLS:
    print(f"\n{'='*60}")
    print(f"URL: {url[-70:]}")
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding
        print(f"HTTP: {resp.status_code} | size: {len(resp.text)} chars")
    except Exception as e:
        print(f"FETCH FAILED: {e}")
        continue

    text, status = extract(resp.text)
    if text:
        print(f"STATUS: {status}")
        print(f"EXTRACTED: {len(text)} chars, {len(text.split())} słów")
        print(f"\nPIERWSZE 300 ZNAKÓW:\n{text[:300]}")
        print(f"\nOSTATNIE 300 ZNAKÓW:\n{text[-300:]}")
    else:
        print(f"BRAK TEKSTU: {status}")

    time.sleep(2)