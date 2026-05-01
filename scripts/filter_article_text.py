# Filtruje tekst artykułu, usuwając metadane, afiliacje, bibliografię itp.
# Użycie: python scripts/filter_article_text.py springer_article.txt filtered_article.txt
import sys

INFILE = sys.argv[1] if len(sys.argv) > 1 else "springer_article.txt"
OUTFILE = sys.argv[2] if len(sys.argv) > 2 else "filtered_article.txt"

SKIP_PHRASES = [
    "Accesses", "Citations", "Altmetric", "Explore all metrics", "Affiliations", "Author information", "Funding", "Acknowledgements"
]
STOP_PHRASES = ["References", "Bibliography"]

def should_skip(line):
    return any(phrase.lower() in line.lower() for phrase in SKIP_PHRASES)

def should_stop(line):
    return any(phrase.lower() in line.lower() for phrase in STOP_PHRASES)

with open(INFILE, encoding="utf-8") as fin, open(OUTFILE, "w", encoding="utf-8") as fout:
    title_written = False
    for line in fin:
        line = line.strip()
        if not line:
            continue
        if should_stop(line):
            break
        if should_skip(line):
            continue
        if not title_written:
            fout.write(line + "\n\n")
            title_written = True
        else:
            fout.write(line + "\n\n")
print(f"Zapisano przefiltrowany tekst do {OUTFILE}")
