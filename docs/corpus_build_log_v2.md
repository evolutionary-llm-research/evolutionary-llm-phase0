# Corpus Build Log — v2

**Data:** 2026-04-28 / 2026-05-01  
**Status:** finalny, zamrożony przed kanonicznym runem Phase 0

---

## Struktura korpusu

Pięć aktywnych domen: climate, vaccines, alt_med, cancer, gmo.  
Domena covid wykluczona z Paper 1 (legacy, artefakty, nakładanie z vaccines).  
Trzy typy per domena: food (pokarm), predator (drapieżnik), noise (szum).  
Cel: 80 dokumentów per typ per domena.

---

## Stan finalny

| Domena | food | predator | noise |
|--------|------|----------|-------|
| climate | 80 | 80 | — |
| vaccines | 80 | 80 | — |
| alt_med | 80 | 80 | — |
| cancer | 80 | 80 | — |
| gmo | 77* | 80 | — |
| noise_mixed | — | — | 80 |

*food_gmo: 77/80 z powodu ograniczonej dostępności open access w PMC dla tej domeny.

---

## Food corpus

**Źródła:** DOAJ + PubMed Central OAI  
**Pipeline:** `scripts/build_food_corpus.py`, uzupełnienia przez `scripts/supplement_food_*.py`  
**Filtrowanie:** min. długość, usunięcie nagłówków PMC OAI, walidacja słów kluczowych domenowych

| Domena | N | Główne źródła | avg (słowa) |
|--------|---|---------------|-------------|
| climate | 80 | DOAJ/PMC (71), PubMed Central (3) | 8501 |
| vaccines | 80 | DOAJ/PMC (66), PubMed Central (2) | 7861 |
| alt_med | 80 | DOAJ (63), PubMed Central (17) | 8729 |
| cancer | 80 | DOAJ/PMC (80) | 10053 |
| gmo | 77 | PMC (28), DOAJ/PMC (27), PubMed Central (16) | 9419 |

food_gmo uzupełniony o 6 artykułów glyphosate/GMO z PMC 2025-2026 (PMCIDs: PMC12969878, PMC13011801, PMC12846237, PMC12709226, PMC12641724, PMC12590857).

---

## Predator corpus

**Filtr jakości:** min. 300 znaków na input, blacklist CTA patterns (brighteon, subscribe to, etc.)  
**Weryfikacja:** `scripts/verify_corpus_quality.py` (bez LLM, bigram Jaccard)

### Pierwotne źródła (NaturalNews scraper)

| Plik | N | Źródło |
|------|---|--------|
| predator_climate_plate.jsonl | 60 | PlateClimatology.com |
| predator_climate_at.jsonl | 14 | AmericanThinker.com |
| predator_climate_nn.jsonl | 5 | NaturalNews.com |
| predator_vaccines_nn.jsonl | 33 | NaturalNews.com |
| predator_alt_med_nn.jsonl | 45 | NaturalNews.com |
| predator_cancer_nn.jsonl | 35 | NaturalNews.com |
| predator_gmo_nn.jsonl | 35 | NaturalNews.com |

### Mercola scraper (2026-05-01)

**Skrypt:** `scripts/scrape_mercola_domain.py`  
**Źródło:** articles.mercola.com, archiwa 2015-2021  
**robots.txt:** brak Disallow na treści, brak crawl-delay  
**Filtrowanie:** keywords domenowe w tytule artykułu, min. 500 znaków treści  
**Uwaga:** artykuły Mercola są znacznie dłuższe niż NaturalNews (avg 16-24k znaków vs 5k) — odnotować w Methods jako zmienną do kontroli

| Plik | Zebrane | Odrzucone | avg (znaki) |
|------|---------|-----------|-------------|
| predator_vaccines_mercola.jsonl | 240 | 0 | ~24000 |
| predator_alt_med_mercola.jsonl | 64 | 2 | ~14000 |
| predator_cancer_mercola.jsonl | 90 | 0 | ~16000 |
| predator_gmo_mercola.jsonl | 157 | 0 | ~18000 |

### Merge finalny (audit_corpus.py --merge --target 80)

| Domena | Źródła w merge | N finalny |
|--------|---------------|-----------|
| climate | PlateClimatology + AmericanThinker + NaturalNews | 80 |
| vaccines | NaturalNews + Mercola | 80 |
| alt_med | NaturalNews + Mercola | 80 |
| cancer | NaturalNews + Mercola | 80 |
| gmo | NaturalNews + Mercola | 80 |

---

## Noise corpus

**Skrypt:** `scripts/generate_noise.py`  
**Metoda:** 50/50 mieszanie zdań z food i predator  
**N:** 80 dokumentów, avg 350 słów

---

## Legacy (nie używane w Paper 1)

- `predator_vaccines_legacy_vaccines.jsonl` (23 dok, VaccineLies MisT) — zachowany do style swap experiment
- `predator_covid_legacy_covid.jsonl` (61 dok, CoAID) — zachowany do style swap experiment
- `food_covid.jsonl` (35 dok) — domena nieaktywna

---

## Kluczowe odkrycie z Phase 0 (poprzedni run)

Akademicko sformułowana dezinformacja (VaccineLies MisT) jest informatycznie nieodróżnialna od pokarmu. Autentyczny język internetowy (ClimateFever, NaturalNews, Mercola) daje efekty -0.60 do -0.82. Kryterium jakości predatora: język musi być autentyczny (potoczny, emocjonalny), nie akademicki.

---

## Pliki konfiguracyjne

- `config/phase0_rerun_v2.yaml` — aktywny config Phase 0 na korpusie v2
- `config/fitness_weights.yaml` — w1=0.3, w2=0.5, w3=0.2 (zamrożone)
- `data/v2/corpus_manifest.json` — automatycznie generowany przez audit_corpus.py

---

## Odtwarzalność

1. Git commit: "corpus v2 final: Mercola predator (vaccines/alt_med/cancer/gmo), food_gmo=77/80, merge 80 per domain"
2. `data/v2/corpus_manifest.json` rejestruje skład każdego pliku processed
3. Skrypty scraping: `scripts/scrape_mercola_domain.py --domain {vaccines|alt_med|cancer|gmo} --max 240`
4. Merge: `python scripts/audit_corpus.py --merge --target 80`
