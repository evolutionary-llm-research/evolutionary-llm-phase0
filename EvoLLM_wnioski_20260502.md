# EvoLLM — Wnioski badawcze z sesji 2026-05-02

## Status na koniec dnia

Phase 0 percentile rerun zakończony. Entropia zrehabilitowana (p=0.77→8.2e-21). 
Decyzja o przebudowie korpusu predator v3 przed Paper 1. Scraping w trakcie.

---

## 1. Implementacja percentilowego chunkingu

**Problem:** single-window truncation (max_length=2048) powoduje że model widzi tylko pierwsze ~1500 słów. Mercola (16-24k znaków) był agresywnie ucinany. Dezinformacyjny "core" leży często w drugiej połowie artykułu.

**Rozwiązanie:** document-length-invariant percentile sampling. Dokument dzielony na n okien po 1/n długości, każde okno o stałym rozmiarze tokenów wycentrowane na danym percentylu. Dokumenty 5k i 25k tokenów mają identyczną reprezentację strukturalną.

**Implementacja w `src/analysis/phase0_metric_validation.py`:**
- Funkcja `get_percentile_chunks(prompt, tokenizer, n_windows=5, window_size=512)`
- Adaptive windowing: `actual_windows = min(n_windows, max(1, doc_len // window_size))`
- Nowe metryki: `h_x_var`, `h_x_slope`, `h_dezorg_var`, `h_dezorg_slope`
- `gen_cfg` czyta z YAML configa (temperature=0.0, do_sample=False)

---

## 2. Wyniki Phase 0 percentile run (20260502T000503Z)

| Typ | H(X) | C(X) | I(X;seed) | H_dezorg | Fitness | N |
|-----|------|------|-----------|----------|---------|---|
| food | 4.892 | 0.391 | 0.0541 | 0.835 | -0.023 | 397 |
| predator | 4.675 | 0.292 | 0.0462 | 0.898 | -0.069 | 484 |
| noise | 4.140 | 0.181 | 0.0397 | 0.923 | -0.110 | 80 |

**Kruskal-Wallis:**
- h_x: p=8.2e-21 ✓ (poprzednio p=0.77 — rehabilitacja entropii)
- c_x: p=4.0e-68 ✓
- i_x_seed: p=1.7e-07 ✓
- jaccard: p=0.44 ✗
- h_x_var: p=4.0e-13 ✓
- h_x_slope: p=5.5e-05 ✓
- h_dezorg_var: p=5.5e-25 ✓
- h_dezorg_slope: p=0.18 ✗

**Fitness ujemny:** artefakt małego okna kontekstu (512 tokenów) → wyższa h_dezorg. Hierarchia food > predator > noise zachowana i istotna.

---

## 3. Analiza asymetrii korpusu

Rozkład długości przy progu 14000 znaków (~3500 tokenów):

| Plik | N | >14k znaków | Status |
|------|---|-------------|--------|
| food_* | 80 | 97-100% | OK |
| predator_vaccines | 80 | 100% | OK |
| predator_gmo | 80 | 99% | OK |
| predator_alt_med | 80 | 45% | Wymagał uzupełnienia |
| predator_cancer | 80 | 66% | Wymagał uzupełnienia |
| predator_climate | 80 | 0% | Źródła CARDS/PlateClimatology — za krótkie |
| noise_mixed | 80 | 0% | Fragmenty 50/50 — krótkie z definicji |

**Wniosek:** h_x_var i h_x_slope jako metryki porównawcze między typami są confoundem długości przy obecnym korpusie. Mean jest zawsze porównywalny.

---

## 4. Redefinicja noise

**Stara definicja:** fragmenty 50/50 food+predator, losowo przetasowane.

**Problem:** to jest sygnał zdegradowany, nie szum środowiskowy. Model widzi słownictwo domenowe bez kontekstu. Biologicznie: zepsuta żywność, nie tło środowiskowe.

**Nowa definicja:** semantycznie spójny tekst z domen niezwiązanych z projektem (Wikipedia — historia, geografia, kultura). Uzasadnienie biologiczne: tło środowiskowe w biome. Uzasadnienie aplikacyjne: realny internet zawiera szum semantycznie spójny.

**Korzyść techniczna:** artykuły Wikipedia >3500 tokenów → pełny profil 5-chunkowy, brak confoundu długości.

---

## 5. Decyzje korpus v3

### predator_alt_med i predator_cancer
Uzupełnione przez rozszerzenie zakresu lat scraperA Mercola (2021→2026).
- predator_alt_med: 80 docs, min 14141, median 17576 znaków ✓
- predator_cancer: 80 docs, min 14093, median 17147 znaków ✓

### predator_climate
Źródło: wattsupwiththat.com (Selenium scraper — rate limiting przez requests).
Scraper: `scripts/scrape_wuwt_selenium.py` z `--start-page` parametrem.
Status: w trakcie (dwa równoległe runy, strony 1-19 i 20+).

### noise_mixed → Wikipedia noise
80 artykułów z niezwiązanych domenach, >3500 tokenów. Do implementacji.

### Parametry docelowe dla Phase 0 run v3
- window_size=1024, n_windows=3
- Uzasadnienie: 90% predatora (od p10=3168 tokenów) dostaje pełny profil, kontekst generacji 2x lepszy niż 512 tokenów.

---

## 6. Scrapers zbudowane w tej sesji

| Skrypt | Cel | Status |
|--------|-----|--------|
| `scripts/scrape_mercola_domain.py` | Zaktualizowany: +climate domain, MIN_CHARS=14000, lata 2015-2026 | Gotowy |
| `scripts/scrape_wuwt_selenium.py` | WUWT przez Selenium, parametr --start-page | W trakcie |
| `scripts/scrape_joannenova.py` | Jo Nova — odrzucony (artykuły <3000 znaków) | Odrzucony |

---

## 7. Otwarte zadania

- [ ] Zebrać 80 artykułów predator_climate z WUWT (scraper w trakcie)
- [ ] Zbudować Wikipedia noise (80 artykułów, skrypt do napisania)
- [ ] Puścić Phase 0 run v3: window_size=1024, n_windows=3, korpus v3
- [ ] Sprawdzić czy fitness wraca do wartości dodatnich przy window_size=1024
- [ ] Zaktualizować validation_protocol.md o nową metodologię chunkingu
- [ ] Zdokumentować Methods: percentile sampling jako document-length-invariant improvement
