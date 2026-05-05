## Phase 0 — ZAMKNIĘTA (2026-05-04)

### Kanoniczne runy (chronologicznie)
1. phase0_metrics_20260501T160337Z — single-window 2048, ZAMROŻONY jako baseline
2. phase0_metrics_20260502T000503Z — percentile 5×512, rehabilitacja h_x
3. phase0_metrics_20260504T082632Z — percentile 3×1024, KANONICZNY
4. ld50_20260504T131904Z — titration 7 stężeń, walidacja gradientowa

### Kanoniczne parametry Phase 0
- window_size=1024, n_windows=3
- temperature=0.0, do_sample=False
- seed_text: "Climate and vaccine discourse requires coherent, evidence-grounded synthesis."
- Korpus: v3, 880 docs, data/v2/corpus_manifest_v3.json

### Metryki skuteczne (paper-ready)
- c_x: food vs toxin p=6.7e-18, r=-0.352 ✓
- h_dezorg: food vs toxin p=1.5e-29, r=0.461 ✓
- fitness: hierarchia food > toxin > noise zachowana ✓

### Metryki nieefektywne (odnotować w Methods)
- h_x: food vs toxin p=0.587 — mimikra (wynik pozytywny)
- i_x_seed: food vs toxin p=0.052 — bag-of-words proxy, zbyt grube

### Do zrobienia przed Phase 1
- [ ] rename toxin_* → toxin_* w repo
- [ ] git tag phase0-final
- [ ] skrypt analyze_ld50_thresholds.py (hipoteza H_diag)
# EvoLLM — Validation Protocol

*Stworzony: 2026-04-27 | Living document — aktualizowany po każdej sesji*

Protokół walidacji definiuje co musi być prawdziwe zanim projekt przejdzie do kolejnej fazy.
Pięć poziomów walidacji, od sanity checks po walidację ewolucyjną.

---

## Definicja "ewolucji" w projekcie

Ewolucja agenta jest stwierdzona gdy:
- Zmiana średniego fitness między kolejnymi generacjami przekracza 0.5 SD populacji
- p < 0.05 (Mann-Whitney U, Benjamini-Hochberg correction) dla dowolnej metryki
- Kierunek zmiany jest spójny przez co najmniej 3 kolejne generacje
- Zmiana nie jest artefaktem inicjalizacji adaptera (wymaga kontroli z losową inicjalizacją)

Ewolucja *nie* jest stwierdzona przez jednorazowy skok fitness lub zmianę bez replikacji.

---

## Level 1 — Unit tests: sanity metryk

**Cel:** Upewnić się że metryki są obliczane poprawnie przed uruchomieniem jakiegokolwiek eksperymentu.

**Testy (tests/test_metrics_core.py):**

| Test | Warunek | Status |
|------|---------|--------|
| shannon_entropy_empty | H("") = 0.0 | ✅ (2026-04-22) |
| shannon_entropy_repeated_token | H("aaa") = 0.0 | ✅ |
| effective_complexity_empty | C("") = 0.0 | ✅ |
| effective_complexity_non_empty | C(tekst) > 0 | ✅ |
| fitness_score_formula | fitness = w1·C + w2·I − w3·H_dezorg | ✅ |
| mutual_information_proxy_range | 0 ≤ I ≤ 1 | ✅ |
| mutual_information_proxy_identical | I(x, x) = 1.0 | ✅ |
| mutual_information_proxy_disjoint | I(x, y) = 0.0 przy disjoint vocab | ✅ |
| mutual_information_proxy_partial_overlap | 0 < I < 1 | ✅ |
| disorganization_entropy_empty | H_dezorg("") = 0.0 | ✅ |
| disorganization_entropy_sentence_mix | H_dezorg > 0 | ✅ |

Wynik: **11/11 passed** (2026-04-22, pytest 9.0.3, Python 3.12.7).

**Próg przejścia:** 11/11. Jakikolwiek failure blokuje Phase 0.

---

## Level 2 — Phase 0 canonical run: separacja typów

**Cel:** Potwierdzić że metryki na wyjściach modelu różnicują typy dokumentów (food / toxin / noise) w sposób statystycznie istotny.

**Protokół:**
- Kruskal-Wallis H-test na trzech grupach (food, toxin, noise)
- Minimalny corpus: 30 dokumentów per typ
- Metryki mierzone na wyjściach modelu (nie inputach — patrz: decyzja metodologiczna sesja 3)
- Temperatura generacji: 0.0 (deterministyczna, reproducibility requirement)

**Próg przejścia:**
- Co najmniej dwie metryki z {H(X), C(X), I(X;seed)} z p < 0.05
- Effect size |r| > 0.3 (rank-biserial) dla przynajmniej jednej metryki

**Wyniki kanoniczne (20260427T120238Z, pre-percentile):**

| Metryka | KW p-value | Próg |
|---------|-----------|------|
| H(X) | 7.68e-06 | ✅ |
| C(X) | 3.12e-12 | ✅ |
| I(X;seed) | 0.061 | ⚠️ borderline |
| Jaccard | istotny | ✅ |

**Known limitation (20260501T160337Z):**
Single-window truncation (max_length=2048) powoduje H(X) p=0.77 dla korpusu v2
z długimi artykułami Mercola. Artefakt truncation, nie brak sygnału — potwierdzony
przez percentile rerun (patrz: Aktualizacja 2026-05-02).

**Wagi fitness (zamrożone po grid search):**
```
w1 = 0.3  (C(X) — complexity)
w2 = 0.5  (I(X;seed) — mutual information)
w3 = 0.2  (H_dezorg — disorganization)
```
Źródło: `config/fitness_weights_sum1.yaml`. Wagi nie mogą być zmieniane po Phase 0.

---

## Level 3 — Style swap test: autentyczność języka toxina

**Cel:** Potwierdzić że autentyczny język internetowy (NaturalNews, Mercola, WUWT)
daje inne efekty niż akademicko sformułowana dezinformacja (VaccineLies MisT, MBIB).

**Protokół:**
- Uruchomić Phase 0 na dwóch wersjach toxina per domena:
  - `toxin_vaccines_legacy` (VaccineLies MisT — akademicki styl)
  - `toxin_vaccines_mercola` (Mercola — autentyczny internet)
- Porównać effect sizes food vs toxin dla każdego wariantu
- Hipoteza: autentyczny język → większy efekt (bardziej negatywna r)

**Odkrycie (sesja 3, 2026-04-27):**
VaccineLies MisT (akademicki): I(X;seed) effect = +0.274 (ODWRÓCONY kierunek)
ClimateFever (autentyczny): I(X;seed) effect = -0.607

Interpretacja: akademicko sformułowana dezinformacja jest informatycznie
nieodróżnialna od pokarmu. Model traktuje styl naukowy jako sygnał jakości
niezależnie od treści. To jest kluczowe odkrycie metodologiczne dla Paper 1.

**Status:** Wstępnie potwierdzony (anomalia szczepionkowa). Formalny test
(style swap na nowym toxinze) do przeprowadzenia po zamknięciu Phase 0 v3.

**Kryterium dla MBIB (negatywna kontrola):**
MBIB fake_news split może być użyty jako kontrola "academic-style misinformation"
w style swap test. Wynik powinien replikować anomalię szczepionkową.

---

## Level 4 — Per-dataset quality analysis: spójność między domenami

**Cel:** Potwierdzić że efekty są spójne we wszystkich pięciu domenach, nie tylko w jednej.

**Protokół:**
- `scripts/corpus_quality_analysis.py` — effect sizes per dataset
- Minimalny próg: |r| > 0.3 dla H(X) lub C(X) w każdej domenie
- Domeny z |r| < 0.1 muszą być zbadane (contamination? styl?)

**Wyniki (korpus v2, po czyszczeniu Brighteon, 2026-04-30):**

| Domena | H effect | C effect | I effect | Jaccard |
|--------|----------|----------|----------|---------|
| climate | -0.695 | -0.655 | -0.675 | -0.563 |
| vaccines | -0.705 | -0.364 | -0.594 | -0.481 |
| alt_med | -0.607 | -0.605 | -0.628 | -0.743 |
| cancer | -0.773 | -0.562 | -0.666 | -0.626 |
| gmo | -0.817 | -0.423 | -0.749 | -0.651 |

Wszystkie p < 0.001. Wszystkie |r| > 0.3. Walidacja zaliczona.

**Known issue — contamination detection:**
Brighteon CTA contamination (2026-04-30) powoduje h_x ≈ 0, c_x ≈ 0 dla
zainfekowanych dokumentów. Filtr: min 300 znaków + blacklist CTA patterns.
Contamination-free corpus wymagany przed każdym kanonicznym runem.

---

## Level 5 — DTW protocol: walidacja trajektorii ewolucyjnych (przed Phase 2)

**Cel:** Zdefiniować metrykę podobieństwa trajektorii fitness przed uruchomieniem
populacji, żeby odróżnić prawdziwą ewolucję od dryftu losowego.

**Status:** Do zaimplementowania przed Phase 2.

**Plan:**
- Dynamic Time Warping (DTW) na time-series fitness per agent
- Baseline: losowo inicjalizowane adaptery (brak ewolucji) → DTW distance > threshold
- Ewolucja: adaptery uczone na własnych outputach → DTW distance < threshold
- Threshold kalibrowany na 100 bootstrap permutacjach w Phase 1

**Narzędzia:** `tslearn` lub `dtw-python` (do zainstalowania przed Phase 2).

---

## Aktualizacja 2026-05-02: Percentile Sampling

**Zmiana metodologiczna:** single-window truncation (max_length=2048) zastąpiony przez
document-length-invariant percentile sampling jako standard od Phase 0 v3.

**Uzasadnienie:** Dokumenty Mercola (16-24k znaków) były agresywnie ucinane.
Single-window truncation nie chwyta heterogeniczności dokumentu — dezinformacyjny
"core" leży często w drugiej połowie artykułu. Wynik: H(X) p=0.77 dla korpusu v2.

**Parametry kanoniczne (korpus v3):**
- n_windows=3, window_size=1024 tokenów
- actual_windows = min(3, max(1, doc_len // 1024))

**Known limitation kanonicznego runu Phase 0 (20260501T160337Z):**
- Single-window 2048 tokenów, temperature=0.0
- H(X) nieistotne (p=0.77) — artefakt truncation, nie brak sygnału
- Potwierdzony przez percentile rerun (20260502T000503Z): H(X) p=8.2e-21

**Metryki profilowe (exploratory, nie do Kruskal-Wallis między typami):**
- h_x_var, h_x_slope, h_dezorg_var, h_dezorg_slope
- Confound długości przy obecnym korpusie: noise_mixed (krótkie fragmenty) vs food (długie PMC)
- Z korpusem v3 (Wikipedia noise >3500 tokenów) confound znika
