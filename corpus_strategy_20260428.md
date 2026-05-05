## Walidacja gradientowa (2026-05-04)

LD50 titration: 7 stężeń × N=80, seed=42.
Manifest: data/ld50/ld50_corpus_manifest.json
Run: experiments/ld50_20260504T131904Z

Korelacje Pearsona (T% vs metryka):
- c_x: r=-0.905, p=0.005
- h_dezorg: r=+0.849, p=0.016
- fitness: r=-0.921, p=0.003

Wniosek: gradient monotonicznie uporządkowany, brak nieciągłości. Kontrast food/toxin jest rzeczywisty i mierzalny ilościowo.
# EvoLLM Corpus: Strategia i Architektura
*Wygenerowane: 2026-04-28 | Uzupełnienie do wnioski_20260427.md*

---

## 1. MBIB-base — Ocena

### Struktura datasetu
MBIB-base (Wessel et al., SIGIR 2023) zawiera 8 splitów:
`fake_news`, `cognitive_bias`, `linguistic_bias`, `political_bias`,
`text_level_bias`, `gender_bias`, `hate_speech`, `racial_bias`.

Schema: `text` (str), `label` (int: 0=unbiased, 1=biased).
Licencja: CC-BY-NC-ND-4.0.

### Werdykt: NIE nadaje się jako główny predator

**Powód 1 — brak filtrowania domenowego.** fake_news split zawiera
twierdzenia z wielu obszarów (polityka, zdrowie, nauka) bez możliwości
filtrowania per domena bez LLM lub manualnej anotacji. Przy 5 domenach
(klimat, szczepionki, COVID, alt_med, GMO) wymagałoby to ~500 manualnych
etykiet.

**Powód 2 — krótkie teksty.** MBIB agreguje LIAR dataset (single claims),
FakeNewsNet (akapity). Większość rekordów poniżej progu 500 znaków.
Naruszałoby to wymóg jakości predatora.

**Powód 3 — styl akademicki.** LIAR-based claims są pisane w stylu neutralnym,
faktograficznym. Na podstawie wyników Phase 0: akademicko sformułowana
dezinformacja jest informatycznie nieodróżnialna od pokarmu (anomalia
szczepionkowa, I effect = +0.274). MBIB fake_news prawdopodobnie
zreplikowałby tę anomalię we wszystkich domenach.

**Powód 4 — licencja ND.** "no derivatives" jest problematyczne jeśli
transformujesz dane do formatu JSONL lub wycinasz fragmenty.

### Gdzie MBIB może być użyteczny
Jedyne zastosowanie: **negatywna kontrola w style swap test** (validation_protocol,
Level 3). Załaduj fake_news split jako "academic-style misinformation",
porównaj efekty z ClimateFever. Potwierdza lub falsyfikuje hipotezę stylu.

---

## 2. Stan korpusu v2 → v3 (aktualizacja 2026-05-02)

| Plik | v2 status | v3 status | Źródło |
|------|-----------|-----------|--------|
| food_* (5 domen) | 77-80 docs | bez zmian | PMC |
| predator_vaccines | 80 docs OK | bez zmian | Mercola |
| predator_gmo | 80 docs OK | bez zmian | Mercola |
| predator_alt_med | 80 docs, 45% za krótkie | 80 docs, min 14141 znaków ✓ | Mercola 2015-2026 |
| predator_cancer | 80 docs, 66% za krótkie | 80 docs, min 14093 znaków ✓ | Mercola 2015-2026 |
| predator_climate | 80 docs CARDS/PlateClim. | W budowie — WUWT Selenium | wattsupwiththat.com |
| noise_mixed | 80 docs, 0% >3500 tokenów | Do wymiany — Wikipedia | Wikipedia API |

**Kryterium długości v3:** MIN_CHARS=14000 znaków (~3500 tokenów), wymóg pełnego profilu 3-chunkowego przy window_size=1024.

---

## 3. Protokół webscraping — konkretne źródła

### Dla alt_med (predator)
Sortowane od najlepszych do najgorszych pod kątem autentyczności języka:

1. **NaturalNews.com** — najbardziej emocjonalny język, długie artykuły
   (>2000 słów typowo), tags: /tag/homeopathy, /tag/natural-remedies.
   Uwaga: bardzo obciążony JavaScript — użyj User-Agent rotation.

2. **GreenMedInfo.com** — blog-style, długie artykuły, cytowalny (znana
   strona w literaturze dezinformacji medycznej). Sekcje /blog/.

3. **HealthImpactNews.com** — kategorie /category/alternative-medicine/,
   dobre ratio tekstu do nawigacji.

4. **Mercola.com** — UWAGA: heavily paywalled od 2021. robots.txt może
   blokować scraping. Sprawdź przed użyciem.

### Dla GMO (predator)
1. **GMWatch.org** — dostępny, długie artykuły, emocjonalny język anty-GMO,
   cytowalny w literaturze naukowej (Hilbeck et al. 2015 cytuje GMWatch).
   Sekcje: /en/news/latest-news

2. **NaturalNews.com** (tagi GMO) — /tag/gmo, /tag/monsanto.

3. **ResponsibleTechnology.org** (Institute for Responsible Technology) —
   Jeffrey Smith's site. Wyraźnie anty-GMO, artykuły długie, prosty HTML.

4. **Séralini blog** (criigen.org lub seralini.fr) — naukowy styl pisania,
   ale spolaryzowany. UWAGA: styl może być zbyt akademicki →
   sprawdź Jaccard z food przed włączeniem.

### Dla vaccines v2 (predator replacement)
1. **NaturalNews.com** /tag/vaccines, /tag/vaccine-injury
2. **HealthImpactNews.com** /category/vaccines/
3. **VAXXED-adjacent blogs** — wyszukaj "vaccine truth blog site:blogspot.com"
   dla mniejszych, autentycznych głosów (trudniejsze do scraping, ale
   bardzo autentyczny język).

---

## 4. PubMed queries dla food corpus

### alt_med (Food) — przetestowane zapytania PubMed (2026-04-28)
Zwracają recenzowane artykuły oceniające CAM z perspektywy EBM:

```
# Query 1 (seed set - 9 artykułów z PMC full text)
homeopathy OR alternative medicine evidence-based systematic review randomized controlled trial

# Query 2
complementary alternative medicine herbal remedies clinical trial evidence

# Query 3 (bardziej specyficzny)
homeopathy efficacy placebo controlled trial systematic review[Publication Type]

# Query 4
naturopathy integrative medicine evidence-based evaluation review

# Query 5
dietary supplements phytotherapy safety efficacy randomized controlled trial

# Query 6
alternative cancer treatment evidence systematic review oncology
```

**Zidentyfikowane PMCIDs (weryfikacja 2026-04-28, 9 artykułów):**
- PMC10559431 (PMID 37805577) — systematic review homeopathy
- PMC11345309 (PMID 39098144) — CAM clinical evidence
- PMC4233444 (PMID 25408760) — alternative medicine review
- PMC6788024 (PMID 31601215) — herbal medicine BMC CAM
- PMC12045525 (PMID 40314057) — Cureus CAM evaluation
- PMC7952165 (PMID 33763144) — Evid-Based Complement Alternat Med
- PMC9526286 (PMID 36180884) — BMC Complement Med Ther 2022
- PMC6595365 (PMID 31312224) — CAM systematic review 2019
- PMC9657145 (PMID 36358622) — Cancers 2022 CAM in oncology

Skrypt fetch_food_pubmed.py automatycznie wyszuka więcej przez NCBI API.

### gmo (Food) — przetestowane zapytania PubMed (2026-04-28)
```
# Query 1 (seed set - 8 artykułów z PMC full text)
genetically modified food crops safety human health review

# Query 2
transgenic crops biosafety regulatory framework risk assessment

# Query 3
glyphosate herbicide safety toxicology human exposure review

# Query 4
GMO crop environmental impact biodiversity ecosystem review

# Query 5
Bt crops insect resistance management safety review

# Query 6
genome editing CRISPR crop improvement regulatory review
```

**Zidentyfikowane PMCIDs (weryfikacja 2026-04-28, 8 artykułów):**
- PMC10939142 (PMID 38471133) — GM Crops Food 2024
- PMC8441473 (PMID 34532037) — Food Sci Nutr 2021
- PMC6918800 (PMID 31921242) — Front Plant Sci 2019
- PMC9688552 (PMID 36354467) — Biosensors 2022
- PMC10409827 (PMID 37213044) — Transgenic Res 2023
- PMC4413729 (PMID 25972882) — Front Plant Sci 2015
- PMC6492171 (PMID 29952140) — J Sci Food Agric 2018
- PMC7547035 (PMID 33037482) — Metabolomics 2020

---

## 5. Rozmiar korpusu dla Paper 1 — uzasadnienie statystyczne

### Minimalne N per domena
Kruskal-Wallis na 3 grupach (food/predator/noise), moc 0.80, α=0.05,
effect size Cohen d=0.5 (umiarkowany): N ≈ 20-25 per grupa (G*Power).

Biorąc pod uwagę oczekiwane niejednorodności w obrębie grupy i potrzebę
testów post-hoc (Benjamini-Hochberg correction): **minimum 30 per typ per domena**.

### Target dla solidnego Paper 1
| Typ      | Per domena | Liczba domen | Total |
|----------|-----------|--------------|-------|
| food     | 30-40     | 5            | 150-200 |
| predator | 30-40     | 5            | 150-200 |
| noise    | 20-30     | global       | 20-30 |

Noise może być współdzielony między domenami (losowy miks).

### Uzasadnienie dla 5 domen vs 3
Każda dodatkowa domena to dodatkowy replikacyjny warunek testu hipotezy zerowej
"jakość informacji nie wpływa na metryki". Przy 5 domenach uzyskujesz:
- Cross-domain replication w obrębie Paper 1
- Możliwość analizy domain specificity (czy efekty są spójne vs. idiosyncratic)
- Silniejszy argument dla recenzentów PLOS ONE

Minimalnie: alt_med i GMO po 30+ daje wystarczające N. Więcej nie zaszkodzi
pod warunkiem że predator jest autentyczny (lekcja z anomalii szczepionkowej).
