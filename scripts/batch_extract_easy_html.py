
import argparse
import requests
from bs4 import BeautifulSoup
import json
import time

# Lista domen "łatwych" wydawców OA (HTML):
EASY_DOMAINS = [
    "springer.com", "biomedcentral.com", "mdpi.com", "plos.org", "frontiersin.org", "hindawi.com"
]

INPUT = "data/processed/doaj_altmed_fulltext_candidates.jsonl"
OUTPUT = "data/processed/doaj_altmed_structured.jsonl"
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
            return None  # Za mało tekstu
        if logf:
            logf.write(f"OK: {url}\n")
        return {"title": title_extracted, "doi": doi, "url": url, "text": text}
    except Exception as e:
        if logf:
            logf.write(f"ERROR: {url} | {e}\n")
        print(f"Błąd dla {url}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Batch extract OA HTML articles with target count.")
    parser.add_argument("--target-count", type=int, default=300, help="Number of articles to download (default: 300)")
    parser.add_argument("--input", type=str, default=INPUT, help="Input JSONL file with candidates")
    parser.add_argument("--output", type=str, default=OUTPUT, help="Output JSONL file for structured articles")
    args = parser.parse_args()

    count = 0
    with open(args.input, encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout, open("batch_extract_easy_html.log", "a", encoding="utf-8") as logf:
        for line in fin:
            if count >= args.target_count:
                print(f"Osiągnięto limit {args.target_count} artykułów. Kończę.")
                break
            rec = json.loads(line)
            url = rec.get("url")
            title = rec.get("title")
            if not url or not is_easy_domain(url):
                continue
            print(f"Przetwarzam: {url}")
            logf.write(f"START: {url}\n")
            result = process_url(title, url, logf=logf)
            if result:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                count += 1
            time.sleep(2)  # Delikatny throttle

if __name__ == "__main__":
    main()
