import argparse
import requests
from bs4 import BeautifulSoup
import json
import time

EASY_DOMAINS = [
    "springer.com", "biomedcentral.com", "mdpi.com", "plos.org", "frontiersin.org", "hindawi.com"
]

INPUT = "data/processed/doaj_climate_candidates_easy.jsonl"
OUTPUT = "data/processed/doaj_climate_structured.jsonl"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}

def is_easy_domain(url):
    return any(domain in url for domain in EASY_DOMAINS)

def extract_doi(soup):
    meta = soup.find("meta", attrs={"name": "citation_doi"})
    if meta and meta.get("content"):
        return meta["content"].strip()
    meta = soup.find("meta", attrs={"property": "og:doi"})
    if meta and meta.get("content"):
        return meta["content"].strip()
    return None

def extract_title(soup):
    meta = soup.find("meta", attrs={"name": "citation_title"})
    if meta and meta.get("content"):
        return meta["content"].strip()
    if soup.title:
        return soup.title.get_text(strip=True)
    return None

def extract_main_text(soup):
    main = soup.find("main")
    if not main:
        main = soup
    paragraphs = main.find_all("p")
    text = "\n\n".join(p.get_text(strip=True) for p in paragraphs)
    stopwords = ["references", "bibliography"]
    for stopword in stopwords:
        idx = text.lower().find(stopword)
        if idx != -1:
            text = text[:idx]
    return text.strip()

def process_url(title, url, logf=None):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(resp.text, "html.parser")
        doi = extract_doi(soup)
        title_extracted = extract_title(soup) or title
        text = extract_main_text(soup)
        if not text or len(text) < 500:
            if logf:
                logf.write(f"SKIP: Too little text | {url}\n")
            return None
        return {"doi": doi, "title": title_extracted, "text": text, "url": url}
    except Exception as e:
        if logf:
            logf.write(f"ERROR: {e} | {url}\n")
        return None

def main():
    with open(INPUT, "r", encoding="utf-8") as fin, \
         open(OUTPUT, "w", encoding="utf-8") as fout, \
         open("data/processed/doaj_climate_structured.log", "w", encoding="utf-8") as logf:
        for line in fin:
            rec = json.loads(line)
            url = rec.get("url", "")
            title = rec.get("title", "")
            if url and is_easy_domain(url):
                result = process_url(title, url, logf)
                if result:
                    result.update({k: rec.get(k) for k in ["id", "domain", "type", "metadata"] if k in rec})
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                time.sleep(1)  # polite delay

if __name__ == "__main__":
    main()
