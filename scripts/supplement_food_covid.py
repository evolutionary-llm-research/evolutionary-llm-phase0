# scripts/supplement_food_covid.py
import json, re, time, requests
from pathlib import Path

OUT = Path(r"E:\github\Evolutionary LLM Research\data\raw\food_covid.jsonl")

PMCIDS = [
    ("PMC7390912",  "10.1136/bmj.m2980",                "BMJ COVID treatment review 2020"),
    ("PMC9055450",  "10.1001/jamanetworkopen.2022.8873", "JAMA COVID systematic review 2022"),
    ("PMC10321603", "10.3389/fimmu.2023.1200180",        "Frontiers immunology COVID 2023"),
    ("PMC9215332",  "10.1002/14651858.CD015017.pub3",    "Cochrane COVID treatment 2022"),
    ("PMC7444584",  "10.1371/journal.pone.0237903",      "PLOS ONE COVID 2020"),
    ("PMC8155021",  "10.1038/s41598-021-90551-6",        "Scientific Reports COVID 2021"),
    ("PMC11505156", "10.3390/biomedicines12102206",       "Biomedicines COVID 2024"),
    ("PMC10198008", "10.1080/07853890.2023.2208872",      "Annals Medicine COVID 2023"),
    ("PMC8607540",  "10.1080/14787210.2022.2004118",      "Expert Review COVID 2022"),
    ("PMC9899558",  "10.1002/14651858.CD011511.pub3",    "Cochrane COVID 2023"),
    ("PMC11085542", "10.3390/nu16091345",                "Nutrients COVID 2024"),
    ("PMC9635104",  "10.1186/s12931-022-02186-4",        "Resp Research COVID 2022"),
    ("PMC8611039",  "10.1038/s41598-021-02321-z",        "Scientific Reports COVID 2021"),
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
    r"^ArticleCAS", r"^ArticlePubMed",
    r"^PubMedGoogle", r"^CASGoogle",
    r"^ChapterGoogle", r"^CrossRef",
    r"^Google Scholar\s*$",
]]
COVID_KEYWORDS = [
    "COVID", "SARS-CoV", "coronavirus", "pandemic",
    "COVID-19", "spike protein", "vaccination", "antiviral",
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
    print(f"Istniejące: {len(docs)}")

    out_f = open(OUT, "a", encoding="utf-8")

    for pmcid, doi, label in PMCIDS:
        if pmcid in existing_pmcids:
            print(f"  SKIP: {pmcid}")
            continue

        print(f"\n[{len(docs)+1}] {pmcid} — {label}")
        time.sleep(1.5)

        text = fetch_pmc(pmcid)
        if not text or len(text) < 500:
            print(f"  NO TEXT")
            continue

        text_lower = text.lower()
        matches = sum(1 for kw in COVID_KEYWORDS if kw.lower() in text_lower)
        if matches < 2:
            print(f"  SKIP: domain mismatch")
            continue

        record = {
            "id": f"FOOD_COVID_PMC_{len(docs)+1:04d}",
            "domain": "covid",
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
    print(f"\nŁącznie COVID food: {len(docs)}")

if __name__ == "__main__":
    main()