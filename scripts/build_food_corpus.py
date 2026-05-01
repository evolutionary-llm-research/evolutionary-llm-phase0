# scripts/build_food_corpus.py
"""
Unified food corpus builder for EvoLLM project.
Strategy: DOAJ API → DOI → PMC OAI (primary) → HTML (fallback) → PDF (fallback)

Usage:
    python scripts/build_food_corpus.py --domain alt_med --target 150
    python scripts/build_food_corpus.py --domain gmo --target 100
    python scripts/build_food_corpus.py --domain cancer --target 150
    python scripts/build_food_corpus.py --domain climate --target 100
    python scripts/build_food_corpus.py --domain vaccines --target 100
"""

import argparse
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Domain queries
# ---------------------------------------------------------------------------

DOMAIN_QUERIES = {
    "alt_med": [
        "homeopathy clinical trial",
        "herbal medicine randomized controlled trial",
        "acupuncture systematic review",
        "complementary alternative medicine efficacy",
        "naturopathy evidence based",
        "traditional medicine pharmacological",
        "phytotherapy safety efficacy",
        "dietary supplements clinical evidence",
    ],
    "gmo": [
        "genetically modified crops safety human consumption",
        "transgenic plant food risk assessment",
        "glyphosate herbicide health effects epidemiology",
        "GMO crop environmental biodiversity review",
        "Bt toxin Bacillus thuringiensis human safety",
        "gene editing CRISPR food crop regulatory",
        "herbicide tolerant crop weed resistance review",
        "GM maize soybean rice feeding study",
    ],
    "cancer": [
        "cancer treatment systematic review",
        "oncology clinical trial",
        "tumor immunotherapy review",
        "cancer prevention evidence",
        "chemotherapy efficacy review",
        "radiation therapy outcomes",
        "cancer screening meta-analysis",
        "anticancer drug randomized trial",
    ],
    "climate": [
        "climate change health effects",
        "global warming epidemiology",
        "greenhouse gas emissions evidence",
        "sea level rise coastal flooding",
        "climate attribution study",
        "carbon dioxide temperature review",
        "extreme weather health outcomes",
        "climate change biodiversity",
    ],
    "vaccines": [
        "vaccine safety systematic review",
        "vaccine efficacy clinical trial",
        "immunization adverse events surveillance",
        "vaccine hesitancy intervention",
        "mRNA vaccine immunogenicity",
        "childhood vaccination outcomes",
        "vaccine preventable disease burden",
        "herd immunity vaccination",
    ],
    "covid": [
        "COVID-19 treatment clinical trial",
        "SARS-CoV-2 epidemiology review",
        "coronavirus vaccine efficacy",
        "COVID-19 long term outcomes",
        "pandemic response public health",
        "COVID-19 mortality risk factors",
    ],
}

EASY_DOMAINS = [
    "plos.org", "frontiersin.org", "biomedcentral.com",
    "hindawi.com", "ncbi.nlm.nih.gov",
]

# ---------------------------------------------------------------------------
# Domain keyword filters (minimum 2 matches required)
# ---------------------------------------------------------------------------

DOMAIN_KEYWORDS = {
    "gmo": [
        "genetically modified", "transgenic", "GMO", "glyphosate",
        "Roundup", "Bt crop", "Bt toxin", "CRISPR", "gene edit",
        "herbicide", "biosafety", "biotech crop", "GM crop", "GM food",
        "GM maize", "GM rice", "GM soybean", "bioengineered",
    ],
    "alt_med": [
        "homeopathy", "homeopathic", "herbal medicine", "acupuncture",
        "naturopath", "complementary", "alternative medicine",
        "phytotherapy", "traditional medicine", "supplement",
        "integrative medicine", "ayurved",
    ],
    "cancer": [
        "cancer", "tumor", "tumour", "oncol", "chemotherapy",
        "carcinogen", "leukemia", "lymphoma", "melanoma",
        "radiotherapy", "immunotherapy",
    ],
    "climate": [
        "climate change", "global warming", "greenhouse gas",
        "CO2", "carbon dioxide", "sea level", "temperature rise",
        "IPCC", "emissions", "fossil fuel",
    ],
    "vaccines": [
        "vaccine", "vaccination", "immunization", "mRNA vaccine",
        "adjuvant", "immunogen", "booster", "vaccine efficacy",
        "vaccine safety", "childhood vaccine",
    ],
    "covid": [
        "COVID-19", "COVID", "SARS-CoV", "coronavirus",
        "pandemic", "lockdown", "spike protein",
    ],
}

def passes_domain_filter(text: str, domain: str) -> bool:
    keywords = DOMAIN_KEYWORDS.get(domain, [])
    if not keywords:
        return True
    text_lower = text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    return matches >= 2

def is_mostly_latin(text: str) -> bool:
    if not text:
        return False
    latin_chars = sum(1 for c in text if ord(c) < 591)
    return latin_chars / len(text) > 0.85

# ---------------------------------------------------------------------------
# HTTP session
# ---------------------------------------------------------------------------

SESSION = requests.Session()
SESSION.headers["User-Agent"] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
)

def fetch(url: str, binary: bool = False, timeout: int = 30):
    try:
        resp = SESSION.get(url, timeout=timeout)
        resp.raise_for_status()
        if binary:
            return resp.content
        resp.encoding = resp.apparent_encoding
        return resp.text
    except Exception as e:
        print(f"    Fetch error: {e}")
        return None

# ---------------------------------------------------------------------------
# DOAJ API v2
# ---------------------------------------------------------------------------

def search_doaj(query: str, page: int = 1, page_size: int = 100) -> list[dict]:
    url = "https://doaj.org/api/v2/search/articles/" + requests.utils.quote(query)
    params = {"page": page, "pageSize": page_size}
    try:
        resp = SESSION.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json().get("results", [])
    except Exception as e:
        print(f"  DOAJ error: {e}")
        return []

def extract_doaj_metadata(result: dict) -> dict:
    bib = result.get("bibjson", {})
    authors = [a.get("name", "") for a in bib.get("author", [])[:5]]
    doi = ""
    for ident in bib.get("identifier", []):
        if ident.get("type") == "doi":
            doi = ident.get("id", "")
    return {
        "title":   bib.get("title", ""),
        "authors": authors,
        "journal": bib.get("journal", {}).get("title", ""),
        "year":    bib.get("year", ""),
        "doi":     doi,
        "source":  "DOAJ/PMC",
    }

def extract_doaj_links(result: dict) -> list[dict]:
    return result.get("bibjson", {}).get("link", [])

# ---------------------------------------------------------------------------
# PMC (primary text source)
# ---------------------------------------------------------------------------

def doi_to_pmcid(doi: str) -> str | None:
    try:
        resp = SESSION.get(
            "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
            params={"ids": doi, "format": "json"},
            timeout=15,
        )
        records = resp.json().get("records", [])
        if records:
            pmcid = records[0].get("pmcid", "")
            if pmcid:
                return pmcid
    except Exception:
        pass
    return None

def fetch_pmc_text(pmcid: str) -> str | None:
    clean_id = pmcid.replace("PMC", "")
    try:
        resp = SESSION.get(
            "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi",
            params={
                "verb": "GetRecord",
                "identifier": f"oai:pubmedcentral.nih.gov:{clean_id}",
                "metadataPrefix": "pmc",
            },
            timeout=45,
        )
        text = re.sub(r"<[^>]+>", " ", resp.text)
        for entity, char in [("&amp;", "&"), ("&lt;", "<"),
                              ("&gt;", ">"), ("&quot;", '"'),
                              ("&#39;", "'")]:
            text = text.replace(entity, char)
        text = re.sub(r"&#\d+;", " ", text)
        text = re.sub(r"\s{3,}", "\n\n", text).strip()
        text = clean_text(text)
        return text if len(text) > 800 else None
    except Exception as e:
        print(f"    PMC OAI error: {e}")
        return None

# ---------------------------------------------------------------------------
# HTML extraction (fallback)
# ---------------------------------------------------------------------------

REMOVE_SELECTORS = [
    "nav", "header", "footer", "aside", ".sidebar", ".ad",
    ".advertisement", ".share", ".comment", "script", "style",
    "noscript", ".references", ".ref-list", "#references",
    ".article-references", ".bibliography", ".fn-group",
    "[class*='reference']", "[class*='citation']",
    "[class*='footer']", "[class*='sidebar']",
    ".back-matter", ".article-back",
]

def extract_html(html: str) -> str | None:
    soup = BeautifulSoup(html, "lxml")
    for sel in REMOVE_SELECTORS:
        for el in soup.select(sel):
            el.decompose()
    article = (
        soup.select_one("article") or
        soup.select_one("[class*='article-body']") or
        soup.select_one("[class*='article-content']") or
        soup.select_one("[class*='entry-content']") or
        soup.select_one("[class*='fulltext']") or
        soup.select_one("main") or
        soup.find("body")
    )
    if not article:
        return None
    paras = [
        p.get_text(" ", strip=True)
        for p in article.find_all(["p", "h2", "h3", "h4"])
        if len(p.get_text(strip=True)) > 30
    ]
    text = "\n\n".join(paras)
    text = clean_text(text)
    return text if len(text) > 800 else None

# ---------------------------------------------------------------------------
# PDF extraction (fallback)
# ---------------------------------------------------------------------------

def extract_pdf(pdf_bytes: bytes) -> str | None:
    try:
        import pdfplumber
        import io
        pages = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages[:30]:
                t = page.extract_text()
                if t:
                    pages.append(t)
        text = "\n\n".join(pages)
        text = clean_text(text)
        return text if len(text) > 800 else None
    except Exception as e:
        print(f"    PDF error: {e}")
        return None

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

ARTIFACT_LINE_RES = [re.compile(p, re.IGNORECASE) for p in [
    r"^\d+[kKmM]?\s*(Accesses|Citations|Downloads)\s*$",
    r"^\d+\s*Altmetric\s*$",
    r"^Explore all metrics\s*$",
    r"^(Open Access|Article)\s*$",
    r"^(Published|Received|Accepted|Revised)[\s:]\s*\d",
    r"^ArticleCAS", r"^ArticlePubMed",
    r"^PubMedGoogle", r"^CASGoogle",
    r"^ChapterGoogle", r"^CrossRef",
    r"^Google Scholar\s*$",
    r"^\d+\s*pages?\s*$",
    # PMC OAI metadata patterns
    r"^pmc-(status|prop|license)-",
    r"^oai:pubmedcentral",
    r"^https://pmc\.ncbi",
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$",
]]

CUTOFF_MARKERS = [
    "ArticlePubMed", "ArticleCAS",
    "PubMedGoogle Scholar", "ChapterGoogle Scholar",
    "CrossRefGoogle Scholar", "CASGoogle Scholar",
    "\nReferences\n", "\nBibliography\n",
]

# PMC OAI intro markers (metadata before actual article text)
PMC_INTRO_MARKERS = [
    "\n\nAbstract\n", "\n\nABSTRACT\n",
    "\n\nIntroduction\n", "\n\n1. Introduction\n",
    "\n\n1 Introduction\n", "\n\nBackground\n",
    "\n\nSummary\n", "\n\nOverview\n",
]

CITATION_RE = re.compile(
    r'^[A-Z][a-záéíóúżźćńółęąś\-]+\s+[A-Z]{1,4}[\s,\.].*\d{4}'
)

def is_citation_line(line: str) -> bool:
    s = line.strip()
    if len(s) < 20 or not CITATION_RE.match(s):
        return False
    return bool(
        re.search(r'\d{4}[;:\(]', s) or
        any(w in s for w in ["Journal", " Int ", " Med ", " Sci ",
                              "doi", "DOI", ".org/10."])
    )

def clean_text(text: str) -> str:
    # Krok 1: Usuń nagłówek PMC OAI (metadane przed treścią artykułu)
    for marker in PMC_INTRO_MARKERS:
        idx = text.find(marker)
        if 0 < idx < 5000:
            text = text[idx:].strip()
            break

    # Krok 2: Utnij przy znanych markerach bibliografii
    for marker in CUTOFF_MARKERS:
        idx = text.find(marker)
        if idx > 500:
            text = text[:idx]
            break

    # Krok 3: Przetwarzaj linia po linii
    lines = text.split("\n")
    result = []
    consecutive_citations = 0

    for line in lines:
        stripped = line.strip()

        # Pomiń artefakty metadanych
        if any(r.match(stripped) for r in ARTIFACT_LINE_RES):
            continue

        # Wykryj blok bibliografii
        if is_citation_line(line):
            consecutive_citations += 1
        else:
            consecutive_citations = 0

        if consecutive_citations >= 2:
            if result:
                result.pop()
            break

        result.append(line)

    text = "\n".join(result)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_corpus(domain: str, target: int, out_dir: Path) -> None:
    out_path = out_dir / f"food_{domain}.jsonl"

    existing_dois = set()
    existing_urls = set()
    docs = []

    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    existing_dois.add(r["metadata"].get("doi", ""))
                    existing_urls.add(r["metadata"].get("url", ""))
                    docs.append(r)
                except json.JSONDecodeError:
                    pass
        print(f"Wczytano {len(docs)} istniejących dokumentów")

    queries = DOMAIN_QUERIES.get(domain, [])
    if not queries:
        print(f"ERROR: brak zapytań dla domeny '{domain}'")
        return

    out_f = open(out_path, "a", encoding="utf-8")

    for query in queries:
        if len(docs) >= target:
            break

        print(f"\nQuery: {query}")
        results = search_doaj(query, page_size=100)
        print(f"  DOAJ: {len(results)} wyników")
        time.sleep(1)

        for result in results:
            if len(docs) >= target:
                break

            meta = extract_doaj_metadata(result)
            doi   = meta["doi"]
            links = extract_doaj_links(result)

            if doi and doi in existing_dois:
                continue

            html_url = ""
            pdf_url  = ""
            for link in links:
                url   = link.get("url", "")
                ctype = link.get("content_type", "").lower()
                ltype = link.get("type", "")
                if "pdf" in ctype:
                    pdf_url = url
                elif ltype == "fulltext" and not html_url:
                    html_url = url
                elif "html" in ctype and not html_url:
                    html_url = url

            display_url = html_url or pdf_url
            if display_url in existing_urls:
                continue

            print(f"  [{len(docs)+1}/{target}] {meta['title'][:65]}")

            text = None
            time.sleep(1.5)

            # Próba 1: PMC OAI przez DOI
            if doi:
                print(f"    Próba PMC...")
                pmcid = doi_to_pmcid(doi)
                if pmcid:
                    print(f"    PMCID: {pmcid}")
                    time.sleep(1)
                    text = fetch_pmc_text(pmcid)
                    if text:
                        meta["pmcid"] = pmcid

            # Próba 2: HTML z łatwych domen
            if not text and html_url:
                is_easy = any(d in html_url for d in EASY_DOMAINS)
                if is_easy:
                    print(f"    Próba HTML: {html_url[-60:]}")
                    html = fetch(html_url)
                    if html:
                        text = extract_html(html)

            # Próba 3: PDF
            if not text and pdf_url:
                print(f"    Próba PDF: {pdf_url[-60:]}")
                pdf = fetch(pdf_url, binary=True)
                if pdf:
                    text = extract_pdf(pdf)

            # Próba 4: HTML z trudnych domen (ostatnia szansa)
            if not text and html_url:
                is_easy = any(d in html_url for d in EASY_DOMAINS)
                if not is_easy:
                    print(f"    Próba HTML (trudna domena): {html_url[-60:]}")
                    html = fetch(html_url)
                    if html:
                        text = extract_html(html)

            if not text:
                print(f"    NO TEXT")
                continue

            # Filtr językowy — odrzuć non-Latin (perski, arabski, chiński itp.)
            if not is_mostly_latin(text):
                print(f"    SKIP: non-Latin text ({len(text)} chars)")
                continue

            # Filtr domenowy — minimum 2 dopasowania słów kluczowych
            if not passes_domain_filter(text, domain):
                print(f"    SKIP: domain mismatch")
                continue

            meta["url"]        = display_url
            meta["char_count"] = len(text)
            meta["word_count"] = len(text.split())

            record = {
                "id":       f"FOOD_{domain.upper()}_DOAJ_{len(docs)+1:04d}",
                "domain":   domain,
                "type":     "food",
                "content":  text,
                "metadata": meta,
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            docs.append(record)
            existing_dois.add(doi)
            existing_urls.add(display_url)
            print(f"    OK — {meta['word_count']} słów")

    out_f.close()

    print(f"\n{'='*50}")
    print(f"Domena:    {domain}")
    print(f"Dokumenty: {len(docs)}")
    print(f"Output:    {out_path}")

    if docs:
        wc = [d["metadata"]["word_count"] for d in docs]
        print(f"Słowa: min={min(wc)}, avg={sum(wc)//len(wc)}, max={max(wc)}")
        print(f"\nPierwsze 300 znaków pierwszego dokumentu:")
        print(docs[0]["content"][:300])
        print(f"\nOstatnie 200 znaków ostatniego dokumentu:")
        print(docs[-1]["content"][-200:])

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EvoLLM food corpus builder (DOAJ + PMC)"
    )
    parser.add_argument(
        "--domain", required=True,
        choices=list(DOMAIN_QUERIES),
    )
    parser.add_argument("--target", type=int, default=150)
    parser.add_argument(
        "--out-dir",
        default=r"E:\github\Evolutionary LLM Research\data\processed",
    )
    args = parser.parse_args()

    try:
        import pdfplumber
    except ImportError:
        print("INFO: pdfplumber niedostępny — pip install pdfplumber")

    build_corpus(args.domain, args.target, Path(args.out_dir))

if __name__ == "__main__":
    main()