import requests
from bs4 import BeautifulSoup
import json

url = "http://link.springer.com/article/10.1186/s12906-017-2010-y"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

def extract_doi(soup):
    # Springer: meta name="citation_doi"
    meta = soup.find("meta", attrs={"name": "citation_doi"})
    if meta and meta.get("content"):
        return meta["content"].strip()
    # Fallback: og:doi
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
    # Filtracja: usuń tekst po "References" lub "Bibliography"
    stopwords = ["references", "bibliography"]
    for stopword in stopwords:
        idx = text.lower().find(stopword)
        if idx != -1:
            text = text[:idx]
    return text.strip()

doi = extract_doi(soup)
title = extract_title(soup)
text = extract_main_text(soup)

record = {
    "title": title,
    "doi": doi,
    "url": url,
    "text": text
}

with open("springer_article_structured.jsonl", "w", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
print("Zapisano springer_article_structured.jsonl\nDOI:", doi)