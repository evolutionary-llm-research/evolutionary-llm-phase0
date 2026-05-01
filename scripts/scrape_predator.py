# scripts/scrape_predator.py
"""
EvoLLM Predator Corpus Scraper

Usage:
    python scripts/scrape_predator.py --domain climate --max 60 --out data/raw/predator_climate_v2.jsonl --rate-limit 5.0
    python scripts/scrape_predator.py --domain gmo --max 50 --out data/raw/predator_gmo_scraped.jsonl
    python scripts/scrape_predator.py --domain vaccines --max 40 --out data/raw/predator_vaccines_extra.jsonl
"""

import argparse
import json
import logging
import re
import time
import urllib.robotparser
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

try:
    from fake_useragent import UserAgent
    UA = UserAgent()
    def get_ua():
        return UA.random
except ImportError:
    def get_ua():
        return ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain configuration
# ---------------------------------------------------------------------------

DOMAIN_CONFIG = {
    "climate": {
        "sources": [
            {
                "name": "CFACT",
                "base": "https://www.cfact.org",
                "seed_paths": [
                    "/category/climate/",
                    "/category/energy/",
                    "/category/climate/page/2/",
                    "/category/climate/page/3/",
                    "/category/climate/page/4/",
                ],
                "article_pattern": r"/\d{4}/\d{2}/\d{2}/",
            },
            {
                "name": "GlobalResearch",
                "base": "https://www.globalresearch.ca",
                "seed_paths": [
                    "/theme/environment",
                    "/theme/climate-change",
                ],
                "article_pattern": r"/\d{4}/\d{2}/\d{2}/",
            },
            {
                "name": "WattsUpWithThat",
                "base": "https://wattsupwiththat.com",
                "seed_paths": [
                    "/category/climate-change/page/2/",
                    "/category/climate-change/page/4/",
                    "/category/climate-change/page/6/",
                    "/category/ipcc/",
                ],
                "article_pattern": r"/\d{4}/\d{2}/\d{2}/",
            },
        ],
        "keywords": [
            "climate hoax", "global warming lie", "climate scam",
            "CO2 beneficial", "climate fraud", "IPCC corrupt",
            "carbon tax scam", "climate alarmism", "climate hysteria",
            "global cooling", "climate models wrong", "net zero disaster",
            "climate narrative", "climate skeptic", "climate realist",
            "endangerment finding", "climate consensus myth",
            "climate change denial", "global warming pause",
        ],
    },

    "alt_med": {
        "sources": [
            {
                "name": "NaturalNews_AltMed",
                "base": "https://www.naturalnews.com",
                "seed_paths": [
                    "/tag/homeopathy",
                    "/tag/natural-remedies",
                    "/tag/herbal-medicine",
                    "/tag/alternative-medicine",
                ],
                "article_pattern": r"/\d{4}/\d{2}/\d{2}/",
            },
            {
                "name": "GreenMedInfo",
                "base": "https://www.greenmedinfo.com",
                "seed_paths": [
                    "/topic/homeopathy",
                    "/topic/herbal-medicine",
                    "/blog",
                ],
                "article_pattern": r"/blog/",
            },
            {
                "name": "HealthImpactNews_AltMed",
                "base": "https://healthimpactnews.com",
                "seed_paths": [
                    "/category/alternative-medicine/",
                    "/category/health/",
                ],
                "article_pattern": r"/\d{4}/",
            },
        ],
        "keywords": [
            "homeopathy", "homeopathic", "natural remedy", "herbal medicine",
            "essential oil", "detox", "cleanse", "alternative medicine",
            "big pharma", "natural cure", "pharmaceutical conspiracy",
            "holistic", "naturopath", "cancer cure natural",
        ],
    },

    "gmo": {
        "sources": [
            {
                "name": "GMWatch",
                "base": "https://www.gmwatch.org",
                "seed_paths": [
                    "/en/news/latest-news",
                    "/en/gm-crops/gm-food",
                    "/en/news/archive",
                ],
                "article_pattern": r"/en/(news|articles)/\d+",
            },
            {
                "name": "NaturalNews_GMO",
                "base": "https://www.naturalnews.com",
                "seed_paths": [
                    "/tag/gmo",
                    "/tag/monsanto",
                    "/tag/glyphosate",
                    "/tag/pesticides",
                ],
                "article_pattern": r"/\d{4}/\d{2}/\d{2}/",
            },
            {
                "name": "ResponsibleTechnology",
                "base": "https://www.responsibletechnology.org",
                "seed_paths": [
                    "/gmo-education/",
                    "/health-risks/",
                ],
                "article_pattern": r"/(articles|gmo|health)/",
            },
        ],
        "keywords": [
            "GMO", "genetically modified", "Monsanto", "glyphosate",
            "Roundup", "pesticide danger", "GM crop danger",
            "frankenfoods", "GMO ban", "seed patent", "Séralini",
            "corporate agriculture", "chemical farming", "Bt toxin danger",
        ],
    },

    "vaccines": {
        "sources": [
            {
                "name": "NaturalNews_Vax",
                "base": "https://www.naturalnews.com",
                "seed_paths": [
                    "/tag/vaccines",
                    "/tag/vaccine-injury",
                ],
                "article_pattern": r"/\d{4}/\d{2}/\d{2}/",
            },
            {
                "name": "HealthImpactNews_Vax",
                "base": "https://healthimpactnews.com",
                "seed_paths": [
                    "/category/vaccines/",
                ],
                "article_pattern": r"/\d{4}/",
            },
        ],
        "keywords": [
            "vaccine injury", "vaccine damage", "autism vaccines",
            "vaccine danger", "mandatory vaccine", "vaccine truth",
            "CDC corruption", "Big Pharma vaccines", "unvaccinated healthier",
            "mRNA danger", "natural immunity better",
        ],
    },

    "covid": {
        "sources": [
            {
                "name": "GlobalResearch_Covid",
                "base": "https://www.globalresearch.ca",
                "seed_paths": [
                    "/theme/corona-virus",
                    "/theme/covid-19",
                ],
                "article_pattern": r"/\d{4}/\d{2}/\d{2}/",
            },
        ],
        "keywords": [
            "COVID hoax", "COVID scam", "PCR fraud", "lockdown tyranny",
            "COVID vaccine danger", "fauci lies", "COVID conspiracy",
            "pandemic hoax", "COVID truth", "COVID censorship",
        ],
    },
}

# ---------------------------------------------------------------------------
# Artifact patterns
# ---------------------------------------------------------------------------

ARTIFACT_PATTERNS = [
    r"javascript is (required|disabled)",
    r"please enable javascript",
    r"see more of .{1,40} on facebook",
    r"log in .{0,20} sign up",
    r"share this (article|post|page)",
    r"click here to (subscribe|read more)",
    r"newsletter sign.?up",
    r"loading\.\.\.",
    r"cookies? (policy|settings|consent)",
    r"privacy policy",
    r"terms of (use|service)",
    r"\[[\.\s]+\]",
    r"follow us on",
    r"subscribe (to|for) (our|the) (newsletter|updates)",
    r"related (articles?|posts?|stories?)",
    r"you (may|might) also (like|enjoy)",
    r"comments? \(\d+\)",
    r"leave a (comment|reply)",
]
ARTIFACT_RE = re.compile("|".join(ARTIFACT_PATTERNS), re.IGNORECASE)

CUTOFF_PHRASES = [
    "This site is part of the Natural News Network",
    "and is protected under Free Speech. Truth Publishing",
    "Truth Publishing assumes no responsibility",
    "Truth Publishing International, LTD. is not responsible",
    "for educational and entertainment purposes only",
    "indicates your agreement to these terms",
    "All trademarks, registered trademarks",
]

def has_loop(text: str, min_repeats: int = 3) -> bool:
    words = text.split()
    for i in range(len(words) - 5):
        chunk = " ".join(words[i:i+5])
        if text.count(chunk) >= min_repeats:
            return True
    return False

# ---------------------------------------------------------------------------
# Robots.txt cache
# ---------------------------------------------------------------------------

ROBOTS_CACHE: dict = {}

def can_fetch(base_url: str, path: str) -> bool:
    if base_url not in ROBOTS_CACHE:
        rp = urllib.robotparser.RobotFileParser()
        try:
            rp.set_url(urljoin(base_url, "/robots.txt"))
            rp.read()
            ROBOTS_CACHE[base_url] = rp
        except Exception:
            ROBOTS_CACHE[base_url] = None
    rp = ROBOTS_CACHE[base_url]
    if rp is None:
        return True
    return rp.can_fetch("*", urljoin(base_url, path))

# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

SESSION = requests.Session()

def fetch_html(url: str, timeout: int = 20) -> Optional[str]:
    headers = {
        "User-Agent": get_ua(),
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        resp = SESSION.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding
        return resp.text
    except requests.RequestException as e:
        log.warning(f"Fetch failed {url}: {e}")
        return None

# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------

REMOVE_SELECTORS = [
    "nav", "header", "footer", "aside", ".sidebar", ".ad",
    ".advertisement", ".cookie", ".popup", ".modal", ".newsletter",
    ".social", ".share", ".related", ".recommended", "#comments",
    ".comment", ".menu", ".navigation", ".breadcrumb", ".pagination",
    "[class*='widget']", "[class*='banner']", "script", "style", "noscript",
]

def extract_article_text(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, "lxml")
    for sel in REMOVE_SELECTORS:
        for el in soup.select(sel):
            el.decompose()

    content = None
    for sel in ["article", "[class*='article-body']",
                "[class*='entry-content']", "[class*='post-content']",
                "[class*='article-content']", "[class*='story-body']",
                "main", ".content"]:
        el = soup.select_one(sel)
        if el:
            content = el
            break
    if content is None:
        content = soup.find("body")
    if content is None:
        return None

    paras = []
    for p in content.find_all(["p", "h2", "h3", "li"]):
        text = p.get_text(separator=" ", strip=True)
        if len(text) < 30:
            continue
        if ARTIFACT_RE.search(text):
            continue
        paras.append(text)

    full_text = "\n\n".join(paras)

    # Utnij stopki
    for phrase in CUTOFF_PHRASES:
        idx = full_text.find(phrase)
        if idx != -1:
            full_text = full_text[:idx].strip()
            break

    full_text = re.sub(r"\s{3,}", "\n\n", full_text)
    full_text = re.sub(r"[ \t]+", " ", full_text).strip()
    return full_text if full_text else None

def extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for sel in ["h1", "title", "[class*='title']"]:
        el = soup.select_one(sel)
        if el:
            return el.get_text(strip=True)
    return ""

# ---------------------------------------------------------------------------
# URL discovery
# ---------------------------------------------------------------------------

def discover_urls(base_url: str, seed_path: str,
                  article_pattern: str,
                  max_per_seed: int = 25) -> list[str]:
    seed_url = urljoin(base_url, seed_path)
    parsed_base = urlparse(base_url)

    if not can_fetch(base_url, seed_path):
        log.info(f"robots.txt disallows {seed_url}")
        return []

    html = fetch_html(seed_url)
    if not html:
        return []

    soup = BeautifulSoup(html, "lxml")
    article_re = re.compile(article_pattern)
    found = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/"):
            href = urljoin(base_url, href)
        elif not href.startswith("http"):
            continue
        if urlparse(href).netloc != parsed_base.netloc:
            continue
        path = urlparse(href).path
        if article_re.search(path):
            found.add(href)
        if len(found) >= max_per_seed:
            break

    log.info(f"Discovered {len(found)} URLs from {seed_url}")
    return list(found)

# ---------------------------------------------------------------------------
# Keyword filter
# ---------------------------------------------------------------------------

def passes_keywords(text: str, keywords: list[str]) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)

# ---------------------------------------------------------------------------
# Quality check
# ---------------------------------------------------------------------------

def quality_check(text: str) -> tuple[bool, str]:
    if len(text) < 500:
        return False, f"too_short ({len(text)} chars)"
    if has_loop(text):
        return False, "loop_detected"
    word_count = len(text.split())
    unique_ratio = len(set(text.lower().split())) / word_count if word_count else 0
    if unique_ratio < 0.25:
        return False, f"low_lexical_diversity ({unique_ratio:.2f})"
    artifact_count = len(ARTIFACT_RE.findall(text))
    if artifact_count > 5:
        return False, f"too_many_artifacts ({artifact_count})"
    return True, "ok"

# ---------------------------------------------------------------------------
# Main scraping
# ---------------------------------------------------------------------------

def make_doc_id(domain: str, source_name: str, idx: int) -> str:
    abbrev = "".join(w[0].upper() for w in source_name.split("_")[:2])
    return f"PRED_{domain.upper()}_{abbrev}_{idx:04d}"

def scrape_domain(domain: str, max_docs: int, output_path: Path,
                  rate_limit: float = 3.0) -> int:
    if domain not in DOMAIN_CONFIG:
        raise ValueError(f"Unknown domain: {domain}. Options: {list(DOMAIN_CONFIG)}")

    cfg = DOMAIN_CONFIG[domain]
    keywords = cfg["keywords"]

    # Load existing
    existing_urls = set()
    docs = []
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    existing_urls.add(rec["metadata"].get("url", ""))
                    docs.append(rec)
                except json.JSONDecodeError:
                    pass
        log.info(f"Loaded {len(docs)} existing documents")

    doc_idx = len(docs) + 1

    for source in cfg["sources"]:
        if len(docs) >= max_docs:
            break

        name = source["name"]
        base_url = source["base"]
        article_pattern = source["article_pattern"]

        log.info(f"Source: {name}")

        if not can_fetch(base_url, "/"):
            log.warning(f"robots.txt disallows {base_url} — skipping")
            continue

        # Collect URLs from all seeds
        all_urls = []
        for seed_path in source["seed_paths"]:
            time.sleep(rate_limit)
            urls = discover_urls(base_url, seed_path, article_pattern, 25)
            all_urls.extend(urls)

        all_urls = list(dict.fromkeys(all_urls))
        log.info(f"{name}: {len(all_urls)} candidate URLs")

        for url in all_urls:
            if len(docs) >= max_docs:
                break
            if url in existing_urls:
                continue

            time.sleep(rate_limit)

            path = urlparse(url).path
            if not can_fetch(base_url, path):
                continue

            html = fetch_html(url)
            if not html:
                continue

            text = extract_article_text(html)
            if not text:
                log.debug(f"No content: {url}")
                continue

            if not passes_keywords(text, keywords):
                log.debug(f"Keyword filter failed: {url}")
                continue

            ok, reason = quality_check(text)
            if not ok:
                log.info(f"Quality fail [{reason}]: {url}")
                continue

            title = extract_title(html)
            doc_id = make_doc_id(domain, name, doc_idx)

            record = {
                "id": doc_id,
                "domain": domain,
                "type": "predator",
                "content": text,
                "metadata": {
                    "title": title,
                    "url": url,
                    "source": name,
                    "scraped_at": datetime.utcnow().isoformat() + "Z",
                    "char_count": len(text),
                    "word_count": len(text.split()),
                },
            }

            docs.append(record)
            existing_urls.add(url)
            doc_idx += 1
            log.info(f"OK [{doc_idx-1}/{max_docs}] {title[:65]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in docs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log.info(f"Saved {len(docs)} documents to {output_path}")
    return len(docs)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EvoLLM predator corpus scraper")
    parser.add_argument("--domain", required=True,
                        choices=list(DOMAIN_CONFIG))
    parser.add_argument("--max", type=int, default=60)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rate-limit", type=float, default=3.0)
    args = parser.parse_args()

    n = scrape_domain(
        domain=args.domain,
        max_docs=args.max,
        output_path=Path(args.out),
        rate_limit=args.rate_limit,
    )
    print(f"\nDone: {n} documents → {args.out}")

if __name__ == "__main__":
    main()