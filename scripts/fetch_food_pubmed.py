def clean_pmc_text(text: str) -> str:
    """Remove PMC OAI boilerplate, affiliations, license, references, tables, etc."""
    # Usuń linie z linkami, licencjami, afiliacjami
    lines = text.splitlines()
    cleaned = []
    skip = False
    for line in lines:
        l = line.strip()
        # Pomijaj linie z typowymi śmieciami
        if not l:
            continue
        if l.lower().startswith("open access") or l.lower().startswith("this article is licensed"):
            skip = True
        if l.lower().startswith("background"):
            skip = False  # Właściwy tekst
        if any(x in l.lower() for x in ["creativecommons", "orcid.org", "ror.org", "grid.", "public domain dedication", "correspondence:", "affiliation", "author contributions", "funding", "conflict of interest", "ethics approval", "data availability", "supplementary information", "tables", "table ", "references", "reference", "figure "]):
            skip = True
        if skip:
            continue
        cleaned.append(l)
    # Połącz z powrotem
    cleaned_text = "\n".join(cleaned)
    # Dodatkowo przytnij do sekcji 'Background' jeśli istnieje
    idx = cleaned_text.lower().find("background")
    if idx > 0:
        cleaned_text = cleaned_text[idx:]
    # Usuń końcówkę od 'References' jeśli istnieje
    ref_idx = cleaned_text.lower().find("references")
    if ref_idx > 0:
        cleaned_text = cleaned_text[:ref_idx]
    return cleaned_text.strip()
#!/usr/bin/env python3
"""
fetch_food_pubmed.py — PubMed food corpus builder for EvoLLM project.

Downloads full-text articles from PubMed Central for new domains (alt_med, gmo)
and can also augment existing domains (climate, vaccines, covid).

Usage:
    python scripts/fetch_food_pubmed.py --domain alt_med --out data/food_alt_med.jsonl
    python scripts/fetch_food_pubmed.py --domain gmo --out data/food_gmo.jsonl
    python scripts/fetch_food_pubmed.py --domain alt_med --pmcids PMC6788024 PMC7952165 --out data/food_alt_med.jsonl

Requirements:
    pip install requests
    (No external LLM needed — uses NCBI E-utilities API directly)

Note: NCBI rate limit is 3 req/s without API key, 10 req/s with API key.
Set NCBI_API_KEY env variable if you have one.
"""


# --- IMPORTS ---
import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PMC_OA_BASE = "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"
RATE_LIMIT = 0.34 if not NCBI_API_KEY else 0.11  # seconds between requests

SESSION = requests.Session()

# ---------------------------------------------------------------------------
# PubMed search queries per domain
# These have been tested and return relevant peer-reviewed articles.
# ---------------------------------------------------------------------------

DOMAIN_QUERIES = {
    "alt_med": [
        # Core evidence-based medicine assessments of CAM
        ("homeopathy efficacy placebo controlled trial systematic review[Publication Type]", "Homeopathy systematic reviews"),
        ("complementary alternative medicine herbal remedies clinical trial evidence", "CAM herbal clinical trials"),
        ("naturopathy integrative medicine evidence-based evaluation review", "Naturopathy evidence review"),
        ("acupuncture sham controlled randomized trial meta-analysis", "Acupuncture RCT meta-analysis"),
        ("dietary supplements phytotherapy safety efficacy randomized controlled trial", "Supplements RCT"),
        ("traditional herbal medicine pharmacological active compounds clinical evidence", "Traditional herbal pharmacology"),
        ("homeopathy water memory dilution mechanism evidence", "Homeopathy mechanism evidence"),
        ("alternative cancer treatment evidence systematic review oncology", "Alternative cancer treatments evidence"),
        # Nowe szerokie zapytania
        ("alternative therapies systematic review", "Alternative therapies"),
        ("complementary and alternative medicine systematic review", "CAM systematic review"),
    ],
    "alt_med_ext": [
        ("complementary medicine clinical trial review", "Complementary medicine"),
        ("integrative medicine clinical trial review", "Integrative medicine"),
        ("traditional medicine clinical trial review", "Traditional medicine"),
        ("herbal therapy clinical trial review", "Herbal therapy"),
        ("holistic medicine clinical trial review", "Holistic medicine"),
        ("natural remedies systematic review", "Natural remedies"),
        ("mind-body therapies clinical trial review", "Mind-body therapies"),
    ],

    "gmo": [
        # Safety, regulatory, and scientific consensus on GMO crops
        (
            "genetically modified food crops safety human health review",
            "GM food safety review"
        ),
        (
            "transgenic crops biosafety regulatory framework risk assessment",
            "Transgenic crops risk assessment"
        ),
        (
            "glyphosate herbicide safety toxicology human exposure review",
            "Glyphosate toxicology review"
        ),
        (
            "GMO crop environmental impact biodiversity ecosystem review",
            "GM crops environmental review"
        ),
        (
            "genetically modified organism nutritional composition equivalence study",
            "GM food nutritional equivalence"
        ),
        (
            "Bt crops insect resistance management safety review",
            "Bt crops safety"
        ),
        (
            "genome editing CRISPR crop improvement regulatory review",
            "CRISPR crop regulatory"
        ),
        (
            "pesticide residue genetically modified food human health",
            "Pesticide residue GM food"
        ),
    ],

    # Extended queries for existing domains if needed
    "climate_ext": [
        (
            "climate change health effects systematic review epidemiology",
            "Climate health effects"
        ),
        (
            "sea level rise coastal flooding climate attribution study",
            "Sea level rise climate"
        ),
        (
            "greenhouse gas emissions carbon dioxide global temperature evidence",
            "GHG global temperature evidence"
        ),
    ],

    "vaccines_ext": [
        (
            "vaccine safety adverse events surveillance systematic review",
            "Vaccine safety surveillance"
        ),
        (
            "vaccine hesitancy misinformation public health intervention",
            "Vaccine hesitancy intervention"
        ),
    ],
}

# Pre-identified PMCIDs from PubMed search (verified 2026-04-28)
# These are the starting set — add more by running search below
SEED_PMCIDS = {
    "alt_med": [
        "PMC10559431",  # PMID 37805577 - Systematic review homeopathy
        "PMC11345309",  # PMID 39098144 - CAM clinical evidence
        "PMC4233444",   # PMID 25408760 - Alternative medicine review
        "PMC6788024",   # PMID 31601215 - Herbal medicine BMC CAM
        "PMC12045525",  # PMID 40314057 - Cureus CAM evaluation
        "PMC7952165",   # PMID 33763144 - Evid-Based Complement Alternat Med
        "PMC9526286",   # PMID 36180884 - BMC Complement Med Ther 2022
        "PMC6595365",   # PMID 31312224 - CAM systematic review 2019
        "PMC9657145",   # PMID 36358622 - Cancers 2022 CAM in oncology
    ],
    "gmo": [
        "PMC10939142",  # PMID 38471133 - GM Crops Food 2024
        "PMC8441473",   # PMID 34532037 - Food Sci Nutr 2021
        "PMC6918800",   # PMID 31921242 - Front Plant Sci 2019
        "PMC9688552",   # PMID 36354467 - Biosensors 2022
        "PMC10409827",  # PMID 37213044 - Transgenic Res 2023
        "PMC4413729",   # PMID 25972882 - Front Plant Sci 2015
        "PMC6492171",   # PMID 29952140 - J Sci Food Agric 2018
        "PMC7547035",   # PMID 33037482 - Metabolomics 2020
    ],
}

# ---------------------------------------------------------------------------
# NCBI API helpers
# ---------------------------------------------------------------------------

def ncbi_get(url: str, params: dict, retries: int = 3) -> Optional[dict]:
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    for attempt in range(retries):
        try:
            time.sleep(RATE_LIMIT)
            resp = SESSION.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            log.warning(f"NCBI request failed (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return None


def search_pubmed(query: str, max_results: int = 50) -> list[str]:
    """Search PubMed and return PMIDs."""
    resp = ncbi_get(f"{NCBI_BASE}/esearch.fcgi", {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "usehistory": "n",
    })
    if not resp:
        return []
    data = resp.json()
    return data.get("esearchresult", {}).get("idlist", [])


def get_pmcids(pmids: list[str]) -> dict[str, str]:
    """Convert PMIDs to PMCIDs. Returns {pmid: pmcid} for those with PMC full text."""
    if not pmids:
        return {}
    ids_str = ",".join(pmids)
    resp = ncbi_get(f"{NCBI_BASE}/elink.fcgi", {
        "dbfrom": "pubmed",
        "db": "pmc",
        "id": ids_str,
        "retmode": "json",
    })
    if not resp:
        return {}

    result = {}
    try:
        data = resp.json()
        link_sets = data.get("linksets", [])
        for ls in link_sets:
            pmid = ls.get("ids", [None])[0]
            links = ls.get("linksetdbs", [])
            for link_db in links:
                if link_db.get("linkname") == "pubmed_pmc":
                    pmcids_found = link_db.get("links", [])
                    if pmcids_found and pmid:
                        result[str(pmid)] = f"PMC{pmcids_found[0]}"
    except (KeyError, json.JSONDecodeError) as e:
        log.warning(f"Error parsing elink response: {e}")

    return result


def fetch_pmc_full_text(pmcid: str) -> Optional[str]:
    """Fetch full text from PMC OAI. Returns plain text or None."""
    clean_id = pmcid.replace("PMC", "")
    params = {
        "verb": "GetRecord",
        "identifier": f"oai:pubmedcentral.nih.gov:{clean_id}",
        "metadataPrefix": "pmc",
    }
    time.sleep(RATE_LIMIT)
    try:
        resp = SESSION.get(PMC_OA_BASE, params=params, timeout=45)
        resp.raise_for_status()
    except requests.RequestException as e:
        log.warning(f"PMC OAI fetch failed for {pmcid}: {e}")
        return None

    # Extract text from XML
    xml = resp.text
    # Remove XML tags, keeping text content
    text = re.sub(r"<[^>]+>", " ", xml)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"\s{3,}", "\n\n", text)
    text = text.strip()

    if len(text) < 500:
        return None

    # Filter boilerplate XML header artifacts
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    lines = [l for l in lines if not l.startswith("<?xml") and len(l) > 20]
    return "\n\n".join(lines)


def fetch_pmc_via_efetch(pmcid: str) -> Optional[str]:
    """Alternative: fetch via efetch (text format)."""
    clean_id = pmcid.replace("PMC", "")
    resp = ncbi_get(f"{NCBI_BASE}/efetch.fcgi", {
        "db": "pmc",
        "id": clean_id,
        "rettype": "full",
        "retmode": "text",
    })
    if not resp or len(resp.text) < 500:
        return None
    return resp.text[:50000]  # cap at 50k chars for pipeline


def get_article_metadata(pmid: str) -> dict:
    """Get title, authors, journal, doi from PubMed."""
    resp = ncbi_get(f"{NCBI_BASE}/esummary.fcgi", {
        "db": "pubmed",
        "id": pmid,
        "retmode": "json",
    })
    if not resp:
        return {}
    try:
        data = resp.json()
        result = data.get("result", {}).get(pmid, {})
        return {
            "title": result.get("title", ""),
            "authors": [a.get("name", "") for a in result.get("authors", [])[:5]],
            "journal": result.get("fulljournalname", ""),
            "pubdate": result.get("pubdate", ""),
            "doi": result.get("elocationid", ""),
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Document building
# ---------------------------------------------------------------------------

def make_food_record(domain: str, pmcid: str, pmid: str,
                     text: str, meta: dict, idx: int, query_label: str = None) -> dict:
    doc_id = f"FOOD_{domain.upper()}_{idx:04d}"
    metadata = {
        "title": meta.get("title", ""),
        "pmid": pmid,
        "pmcid": pmcid,
        "doi": meta.get("doi", ""),
        "journal": meta.get("journal", ""),
        "authors": meta.get("authors", []),
        "pubdate": meta.get("pubdate", ""),
        "source": "PubMed Central",
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "char_count": len(text),
    }
    if query_label:
        metadata["query_label"] = query_label
    return {
        "id": doc_id,
        "domain": domain,
        "type": "food",
        "content": text,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------


def build_food_corpus(domain: str, output_path: Path,
                      target_n: int = 35,
                      extra_pmcids: list[str] = None) -> int:
    if domain not in DOMAIN_QUERIES and domain not in SEED_PMCIDS:
        log.error(f"Unknown domain: {domain}. Options: {list(DOMAIN_QUERIES)}")
        sys.exit(1)

    # Load existing to avoid duplicates
    existing_pmcids = set()
    docs = []
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    existing_pmcids.add(rec["metadata"].get("pmcid", ""))
                    docs.append(rec)
                except json.JSONDecodeError:
                    pass
        log.info(f"Loaded {len(docs)} existing documents")

    queries = DOMAIN_QUERIES.get(domain, [])
    doc_idx = len(docs) + 1
    for query, label in queries:
        # Collect all candidate PMCIDs for this query
        candidate_pmcids = list(SEED_PMCIDS.get(domain, []))
        if extra_pmcids:
            candidate_pmcids.extend(extra_pmcids)

        log.info(f"Searching: {label}")
        pmids = search_pubmed(query, max_results=30)
        log.info(f"  Found {len(pmids)} PMIDs")
        pmid_to_pmc = get_pmcids(list(set(pmids))) if pmids else {}
        log.info(f"  {len(pmid_to_pmc)} have full text in PMC")
        for pmid, pmcid in pmid_to_pmc.items():
            if pmcid not in candidate_pmcids:
                candidate_pmcids.append(pmcid)

        # Deduplicate
        candidate_pmcids = [p for p in dict.fromkeys(candidate_pmcids)
                            if p not in existing_pmcids]
        log.info(f"Processing {len(candidate_pmcids)} candidate PMCIDs for label '{label}'")

        for pmcid in candidate_pmcids:
            if len(docs) >= target_n:
                break

            log.info(f"Fetching {pmcid}...")


            # Try OAI first, then efetch
            text = fetch_pmc_full_text(pmcid)
            if text:
                text = clean_pmc_text(text)
            if not text or len(text) < 500:
                log.info(f"  OAI failed, trying efetch for {pmcid}")
                text = fetch_pmc_via_efetch(pmcid)
                if text:
                    text = clean_pmc_text(text)

            if not text or len(text) < 500:
                log.warning(f"  No usable text for {pmcid}")
                continue

            # Get PMID for this PMCID (for metadata)
            pmid = ""
            for pid, pmc in pmid_to_pmc.items():
                if pmc == pmcid:
                    pmid = pid
                    break

            meta = get_article_metadata(pmid) if pmid else {}

            record = make_food_record(domain, pmcid, pmid, text, meta, doc_idx, query_label=label)
            docs.append(record)
            existing_pmcids.add(pmcid)
            doc_idx += 1
            log.info(f"  OK [{doc_idx-1}/{target_n}]: {meta.get('title', pmcid)[:70]} (label: {label})")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in docs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log.info(f"Saved {len(docs)} documents to {output_path}")
    return len(docs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# --- CLI ---

def main():
    parser = argparse.ArgumentParser(
        description="EvoLLM PubMed food corpus builder"
    )
    parser.add_argument("--domain", required=True,
                        choices=list(DOMAIN_QUERIES) + ["climate_ext", "vaccines_ext"],
                        help="Target domain")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output JSONL path (use {label} for per-query output)")
    parser.add_argument("--target", type=int, default=35,
                        help="Target document count (default: 35)")
    parser.add_argument("--pmcids", nargs="*",
                        help="Additional PMCIDs to include (e.g. PMC6788024)")
    parser.add_argument("--sequential", action="store_true",
                        help="Run each query separately and save to {label} in output filename")
    args = parser.parse_args()

    if args.sequential:
        queries = DOMAIN_QUERIES[args.domain]
        for query, label in queries:
            # Upewnij się, że label nie jest pusty i nie zawiera niedozwolonych znaków
            safe_label = label.replace(" ", "_").replace("/", "-").replace("\\", "-")
            out_path = Path(str(args.out).replace("{label}", safe_label))
            # Wymuś rozszerzenie .jsonl
            if not str(out_path).endswith(".jsonl"):
                out_path = Path(str(out_path) + ".jsonl")
            print(f"\n--- Running query: {label} ---")
            print(f"Output file: {out_path}")
            n = build_food_corpus(
                domain=args.domain,
                output_path=out_path,
                target_n=args.target,
                extra_pmcids=args.pmcids or [],
            )
            print(f"Done: {n} documents written to {out_path}")
    else:
        n = build_food_corpus(
            domain=args.domain,
            output_path=args.out,
            target_n=args.target,
            extra_pmcids=args.pmcids or [],
        )
        print(f"\nDone: {n} documents written to {args.out}")


if __name__ == "__main__":
    main()