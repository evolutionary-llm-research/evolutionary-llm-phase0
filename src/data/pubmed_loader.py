
"""
PubMed Central food article loader for Evolutionary LLM Research (Phase 0)

- Queries two domains: climate change health effects, vaccine safety efficacy
- Uses direct NCBI E-utilities API calls (requests) to fetch PMC full text articles
- Outputs JSONL files: one article per line, with required metadata
- Proxy support: uses HTTPS_PROXY from environment or defaults to http://172.29.224.1:8080
"""
import os
import time
import json
import requests
import pathlib


# --- Config ---
SEARCH_QUERIES = {
    "climate": "climate change health effects AND pmc[sb]",
    "vaccines": "vaccine safety efficacy AND pmc[sb]"
}
OUT_PATHS = {
    "climate": os.path.join("data", "raw", "food_climate.jsonl"),
    "vaccines": os.path.join("data", "raw", "food_vaccines.jsonl")
}
MIN_TOKENS = 200
MAX_ARTICLES = 30
RETRY_LIMIT = 3
RETRY_DELAY = 2  # seconds

# Proxy setup
_https_proxy = os.environ.get("HTTPS_PROXY", "http://172.29.224.1:8080")
os.environ["HTTPS_PROXY"] = _https_proxy
proxies = {"https": _https_proxy}


def fetch_article_ids(query: str, max_count: int):
    """
    Search PMC for article IDs matching the query using E-utilities esearch.
    """
    params = {
        "db": "pmc",
        "term": query,
        "retmax": max_count,
        "retmode": "json"
    }
    for attempt in range(RETRY_LIMIT):
        try:
            resp = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params=params,
                proxies=proxies,
                timeout=15
            )
            resp.raise_for_status()
            data = resp.json()
            ids = data["esearchresult"]["idlist"]
            return ids
        except Exception as e:
            if attempt < RETRY_LIMIT - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise e


def fetch_article_details(pmcid: str):
    """
    Fetch article metadata and full text for a given PMCID using E-utilities efetch.
    """
    import xml.etree.ElementTree as ET
    params = {
        "db": "pmc",
        "id": pmcid,
        "rettype": "full",
        "retmode": "xml"
    }
    for attempt in range(RETRY_LIMIT):
        try:
            resp = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params=params,
                proxies=proxies,
                timeout=20
            )
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            # Find article metadata
            article = root.find(".//article")
            if article is None:
                return None
            # PMCID
            pmcid_val = None
            pmid = None
            doi = None
            for aid in article.findall(".//article-id"):
                if aid.attrib.get("pub-id-type") == "pmc":
                    pmcid_val = aid.text
                elif aid.attrib.get("pub-id-type") == "pmid":
                    pmid = aid.text
                elif aid.attrib.get("pub-id-type") == "doi":
                    doi = aid.text
            # Title
            title_el = article.find(".//title-group/article-title")
            title = title_el.text.strip() if title_el is not None and title_el.text else ""
            # Abstract
            abstract_el = article.find(".//abstract")
            abstract = ""
            if abstract_el is not None:
                abstract = " ".join([t.text.strip() for t in abstract_el.findall(".//p") if t.text])
            # Full text body
            body_el = article.find(".//body")
            full_text = ""
            if body_el is not None:
                paragraphs = [p.text.strip() for p in body_el.findall(".//p") if p.text]
                full_text = "\n".join(paragraphs)
            # Compose content
            content = f"{title}\n\n{abstract}\n\n{full_text}".strip()
            if len(content.split()) < MIN_TOKENS:
                return None
            return {
                "id": f"PMC{pmcid_val or pmcid}",
                "content": content,
                "metadata": {
                    "title": title,
                    "pmid": pmid,
                    "doi": doi,
                    "type": "food"
                }
            }
        except Exception as e:
            if attempt < RETRY_LIMIT - 1:
                time.sleep(RETRY_DELAY)
            else:
                return None


def save_jsonl(records, out_path, domain):
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            rec_out = dict(rec)
            rec_out["domain"] = domain
            f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")


def main():
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)
    for domain, query in SEARCH_QUERIES.items():
        print(f"Fetching articles for domain: {domain}")
        ids = fetch_article_ids(query, MAX_ARTICLES * 2)  # Fetch extra in case of short articles
        articles = []
        for pmcid in ids:
            art = fetch_article_details(pmcid)
            if art:
                articles.append(art)
            if len(articles) >= MAX_ARTICLES:
                break
        print(f"Fetched {len(articles)} valid articles for {domain}")
        save_jsonl(articles, OUT_PATHS[domain], domain)
        print(f"Saved to {OUT_PATHS[domain]}")

if __name__ == "__main__":
    main()
