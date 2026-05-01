import argparse
import requests
import time
import json
from bs4 import BeautifulSoup

EASY_DOMAINS = [
    "springer.com", "biomedcentral.com", "mdpi.com", "plos.org", "frontiersin.org", "hindawi.com"
]
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
DOAJ_API = "https://doaj.org/api/v3/articles"


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

def process_url(title, url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(resp.text, "html.parser")
        doi = extract_doi(soup)
        title_extracted = extract_title(soup) or title
        text = extract_main_text(soup)
        if not text or len(text) < 500:
            return None
        return {"title": title_extracted, "doi": doi, "url": url, "text": text}
    except Exception as e:
        print(f"Błąd dla {url}: {e}")
        return None

def fetch_doaj_articles(query, page_size=100, max_results=1000):
    articles = []
    page = 1
    headers = {
        "User-Agent": HEADERS["User-Agent"],
        "Accept": "application/json"
    }
    while len(articles) < max_results:
        params = {
            "page": page,
            "pageSize": page_size,
            "q": query
        }
        resp = requests.get(DOAJ_API, params=params, headers=headers)
        if resp.status_code != 200:
            print(f"Błąd API DOAJ: {resp.status_code}")
            break
        data = resp.json()
        # Obsługa różnych struktur odpowiedzi
        if "results" in data:
            results = data["results"]
        elif "hits" in data:
            results = data["hits"]
        elif "data" in data:
            results = data["data"]
        else:
            print("Nieoczekiwana odpowiedź API:", data)
            break
        if not results:
            break
        articles.extend(results)
        if len(results) < page_size:
            break
        page += 1
        time.sleep(1)
    return articles[:max_results]

def main():
    parser = argparse.ArgumentParser(description="Pobieranie pełnych tekstów z DOAJ online.")
    parser.add_argument("--target-count", type=int, default=300, help="Liczba artykułów do pobrania (default: 300)")
    parser.add_argument("--query", type=str, default="alternative medicine", help="Zapytanie do DOAJ API")
    parser.add_argument("--output", type=str, default="data/processed/doaj_altmed_structured.jsonl", help="Plik wyjściowy JSONL")
    args = parser.parse_args()

    print(f"Pobieram rekordy z DOAJ dla zapytania: {args.query}")
    articles = fetch_doaj_articles(args.query, max_results=args.target_count*3)
    print(f"Znaleziono {len(articles)} rekordów, próbuję pobrać pełne teksty...")
    count = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for rec in articles:
            bibjson = rec.get("bibjson", {})
            title = bibjson.get("title")
            links = bibjson.get("link", [])
            url = None
            for l in links:
                if l.get("type") == "fulltext" and is_easy_domain(l.get("url", "")):
                    url = l["url"]
                    break
            if not url:
                continue
            print(f"Przetwarzam: {url}")
            result = process_url(title, url)
            if result:
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                count += 1
                if count >= args.target_count:
                    print(f"Osiągnięto limit {args.target_count} artykułów. Kończę.")
                    break
            time.sleep(2)
    print(f"Zapisano {count} artykułów do {args.output}")

if __name__ == "__main__":
    main()
