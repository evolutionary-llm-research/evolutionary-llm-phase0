# scripts/supplement_food_gmo.py
"""
Uzupełnia food_gmo.jsonl bezpośrednio z PubMed PMC.
Uruchom gdy build_food_corpus.py da za mało dokumentów dla domeny GMO.
"""
import json, re, time, requests
from pathlib import Path

OUT = Path(r"E:\github\Evolutionary LLM Research\data\processed\food_gmo.jsonl")

# PMCIDs zidentyfikowane manualnie z PubMed — GMO safety, glyphosate, transgenic crops
PMCIDS = [
    # GMO crops safety reviews
    ("PMC7164548",  "10.7759/cureus.7306",          "GMO food safety review"),
    ("PMC10434127", "10.7717/peerj.15808",           "GM crop biosafety assessment"),
    ("PMC6918800",  "10.3389/fpls.2019.01592",       "Transgenic crops front plant sci"),
    ("PMC8164681",  "10.1007/s11248-021-00261-y",    "GM crop regulatory review"),
    ("PMC11225911", "10.1080/21645698.2024.2375664",  "GM crops food 2024"),
    ("PMC8397579",  "10.3389/fpls.2021.718775",      "Front Plant Sci GMO 2021"),
    ("PMC9688552",  "10.3390/bios12110959",           "Biosensors GMO detection"),
    ("PMC10409827", "10.1007/s11248-023-00344-y",    "Transgenic Res 2023"),
    ("PMC4413729",  "10.3389/fpls.2015.00283",       "Front Plant Sci GMO safety 2015"),
    ("PMC6492171",  "10.1002/jsfa.9227",             "J Sci Food Agric GMO 2018"),
    ("PMC7547035",  "10.1007/s11306-020-01733-8",    "Metabolomics GMO 2020"),
    # Glyphosate toxicology
    ("PMC9823069",  "10.1007/s00420-022-01878-0",    "Glyphosate occupational health"),
    ("PMC4756530",  "10.1186/s12940-016-0117-0",     "Glyphosate env health 2016"),
    ("PMC5484035",  "10.1136/jech-2016-208463",      "Glyphosate epidemiology"),
    ("PMC6503538",  "10.1186/s12940-019-0474-6",     "Glyphosate health review 2019"),
    ("PMC4819582",  "10.3109/10408444.2014.1003423",  "Glyphosate toxicology review"),
    ("PMC8082925",  "10.1186/s12940-021-00729-8",    "Glyphosate env health 2021"),
    ("PMC6997716",  "10.2486/indhealth.2018-0111",   "Glyphosate industrial health"),
    # Bt crops
    ("PMC9230539",  "10.3390/toxins14060386",        "Bt toxin safety review"),
    # Nowe — dodaj do listy PMCIDS
    # Nowe — glyphosate reviews 2025-2026 z PMC
    ("PMC12969878", "10.3389/fmicb.2026.1751932",    "Glyphosate microbiome frontiers 2026"),
    ("PMC13011801", "10.1530/RAF-25-0178",             "Glyphosate reproduction review 2025"),
    ("PMC12846237", "10.3390/toxics14010026",           "Glyphosate toxics review 2025"),
    ("PMC12709226", "10.3390/toxics13110971",           "Glyphosate toxicology 2025"),
    ("PMC12641724", "10.3390/jox15060187",              "Glyphosate toxicology 2025"),
    ("PMC12590857", "10.1186/s12940-025-01241-z",       "Glyphosate env health 2025"),
    # Dodaj do listy PMCIDS
    ("PMC7186845",  "10.3389/fpls.2020.00445",       "Front Plant Sci GMO 2020"),
    ("PMC9207611",  "10.1016/j.omtm.2022.05.012",    "Gene editing crops review 2022"),
    ("PMC7658669",  "10.2903/j.efsa.2020.6297",      "EFSA GMO opinion 2020"),
    ("PMC10180588", "10.3390/plants12091764",          "Plants GMO review 2023"),
    ("PMC10797474", "10.2903/j.efsa.2024.8489",      "EFSA GMO opinion 2024"),
    ("PMC7327110",  "10.3389/fpls.2020.00940",       "Front Plant Sci biosafety 2020"),
    ("PMC10911423", "10.3389/falgy.2024.1297547",    "Frontiers allergy GMO 2024"),
    ("PMC11056846", "10.2903/j.efsa.2024.8715",      "EFSA opinion GM crop 2024"),
    ("PMC4395235",  "10.1371/journal.pone.0121636",   "GMO feeding study PLOS ONE 2015"),
    ("PMC7261740",  "10.1007/s00204-019-02400-1",    "GMO toxicology Archives 2019"),
    ("PMC6015625",  "10.1007/s00204-018-2230-z",     "GMO safety Archives 2018"),
    ("PMC4931567",  "10.15252/embr.201642739",        "EMBO Reports GMO review 2016"),
    ("PMC11012018", "10.3390/ijms25073824",           "IJMS GMO safety 2024"),
    ]

CUTOFF_MARKERS = [
    "ArticlePubMed", "ArticleCAS", "PubMedGoogle Scholar",
    "ChapterGoogle Scholar", "CrossRefGoogle Scholar",
    "\nReferences\n", "\nBibliography\n",
]
PMC_INTRO_MARKERS = [
    "\n\nAbstract\n", "\n\nABSTRACT\n",
    "\n\nIntroduction\n", "\n\n1. Introduction\n",
    "\n\n1 Introduction\n", "\n\nBackground\n",
]
ARTIFACT_LINE_RES = [re.compile(p, re.IGNORECASE) for p in [
    r"^pmc-(status|prop|license)-",
    r"^oai:pubmedcentral",
    r"^https://pmc\.ncbi",
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$",
    r"^\d+[kKmM]?\s*(Accesses|Citations|Downloads)\s*$",
    r"^ArticleCAS", r"^ArticlePubMed",
    r"^PubMedGoogle", r"^CASGoogle",
    r"^ChapterGoogle", r"^CrossRef",
    r"^Google Scholar\s*$",
]]
GMO_KEYWORDS = [
    "genetically modified", "transgenic", "GMO", "glyphosate",
    "Roundup", "Bt crop", "Bt toxin", "CRISPR", "gene edit",
    "herbicide", "biosafety", "GM crop", "GM food",
    "GM maize", "GM rice", "GM soybean", "bioengineered",
]

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "Mozilla/5.0 Chrome/120.0.0.0"

def clean_text(text):
    for marker in PMC_INTRO_MARKERS:
        idx = text.find(marker)
        if 0 < idx < 5000:
            text = text[idx:].strip()
            break
    for marker in CUTOFF_MARKERS:
        idx = text.find(marker)
        if idx > 500:
            text = text[:idx]
            break
    lines = text.split("\n")
    result = []
    consec = 0
    for line in lines:
        s = line.strip()
        if any(r.match(s) for r in ARTIFACT_LINE_RES):
            continue
        result.append(line)
    text = "\n".join(result)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def fetch_pmc(pmcid):
    clean_id = pmcid.replace("PMC", "")
    try:
        resp = SESSION.get(
            "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi",
            params={
                "verb": "GetRecord",
                "identifier": f"oai:pubmedcentral.nih.gov:{clean_id}",
                "metadataPrefix": "pmc",
            }, timeout=45
        )
        text = re.sub(r"<[^>]+>", " ", resp.text)
        for e, c in [("&amp;","&"),("&lt;","<"),("&gt;",">"),("&quot;",'"')]:
            text = text.replace(e, c)
        text = re.sub(r"&#\d+;", " ", text)
        text = re.sub(r"\s{3,}", "\n\n", text).strip()
        return clean_text(text)
    except Exception as e:
        print(f"  PMC error: {e}")
        return None

def main():
    # Wczytaj istniejące
    existing_pmcids = set()
    docs = []
    if OUT.exists():
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    existing_pmcids.add(r["metadata"].get("pmcid", ""))
                    docs.append(r)
                except:
                    pass
    print(f"Istniejące dokumenty: {len(docs)}")

    out_f = open(OUT, "a", encoding="utf-8")

    for pmcid, doi, label in PMCIDS:
        if pmcid in existing_pmcids:
            print(f"  SKIP (already have): {pmcid}")
            continue

        print(f"\n[{len(docs)+1}] {pmcid} — {label}")
        time.sleep(1.5)

        text = fetch_pmc(pmcid)
        if not text or len(text) < 500:
            print(f"  NO TEXT")
            continue

        # Filtr domenowy
        text_lower = text.lower()
        matches = sum(1 for kw in GMO_KEYWORDS if kw.lower() in text_lower)
        if matches < 2:
            print(f"  SKIP: domain mismatch ({matches} keywords)")
            continue

        # Filtr językowy
        latin = sum(1 for c in text if ord(c) < 591)
        if latin / len(text) < 0.85:
            print(f"  SKIP: non-Latin")
            continue

        record = {
            "id": f"FOOD_GMO_DOAJ_{len(docs)+1:04d}",
            "domain": "gmo",
            "type": "food",
            "content": text,
            "metadata": {
                "title": label,
                "doi": doi,
                "pmcid": pmcid,
                "source": "PMC",
                "char_count": len(text),
                "word_count": len(text.split()),
            }
        }
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_f.flush()
        docs.append(record)
        existing_pmcids.add(pmcid)
        print(f"  OK — {len(text.split())} słów")

    out_f.close()
    print(f"\nŁącznie GMO: {len(docs)} dokumentów")

if __name__ == "__main__":
    main()