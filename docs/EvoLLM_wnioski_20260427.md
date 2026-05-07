# EvoLLM — Wnioski badawcze z sesji 2026-04-27

## Status na koniec dnia

Phase 0 kanonicznie zakończony. Metryki walidują hipotezę (food ≠ toxin ≠ noise).
Odkrycie anomalii szczepionkowej zmienia podejście do budowy korpusu toxin.
System jest gotowy do Phase 1 po rozwiązaniu blockerów infrastrukturalnych.

---

## 1. Kluczowa decyzja metodologiczna: metryki na outputach, nie inputach

**Problem:** Pierwotnie rozważano mierzenie H(X), C(X), I(X;seed) na dokumentach wejściowych.

**Odrzucone z powodu confoundu długości:** Krótkie artykuły toxin (pojedyncze twierdzenia
z MBIB/LIAR) sztucznie zawyżają C(X) — krótszy tekst ma wyższy stosunek kompresji.
Porównanie między typami dokumentów o różnych długościach byłoby artefaktem, nie sygnałem.

**Decyzja:** Metryki mierzone wyłącznie na **wyjściach modelu** po ekspozycji na dokument.
Model generuje odpowiedź; metryka mierzy właściwości tej odpowiedzi. Długość inputu
nie wpływa na wynik — model zawsze generuje ~200 tokenów output.

**Konsekwencja dla paper:** Methods section musi wyraźnie podkreślać że mierzymy
informacyjne właściwości *zachowania modelu*, nie *dokumentów*.

---

## 2. Wyniki kanonicznego runu Phase 0 (20260427T120238Z)

| Typ | H(X) | C(X) | I(X;seed) | Jaccard | Fitness | N |
|-----|------|------|-----------|---------|---------|---|
| food | 5.503 | 0.526 | 0.0900 | 0.018 | +0.035 | 73 |
| toxin | 5.240 | 0.425 | 0.0717 | 0.021 | -0.024 | 116 |
| noise | 5.771 | 0.564 | 0.0912 | 0.020 | +0.035 | 35 |

**Kruskal-Wallis:**
- H(X): p=7.68e-06 ✓
- C(X): p=3.12e-12 ✓
- I(X;seed): p=0.061 ⚠️ (borderline, akceptowalne)

**Wagi fitness zamrożone:** w1=0.3, w2=0.5, w3=0.2 (config/fitness_weights_sum1.yaml).

---

## 3. Anomalia szczepionkowa — kluczowe odkrycie

**Obserwacja:** Toxin szczepionkowy (VaccineLies MisT) daje *odwrócony* kierunek
dla I(X;seed): model po ekspozycji na dezinformację szczepionkową generuje output
*bliższy* seedowi niż po ekspozycji na rzetelne artykuły naukowe.

| Toxin | I(X;seed) effect r | Kierunek |
|---------|-------------------|----------|
| toxin_climate (ClimateFever) | -0.607 | ✓ prawidłowy |
| toxin_vaccines (VaccineLies MisT) | +0.274 | ✗ odwrócony |

**Hipoteza:** VaccineLies używa akademickiego stylu taksonomicznego ("claim X is false
because Y"). Model base traktuje styl naukowy jako sygnał wysokiej jakości,
niezależnie od treści. Efekt dezinformacji jest maskowany przez styl.

ClimateFever i NaturalNews używają autentycznego języka internetowego (emocjonalny,
potoczny, conspiracyjny) który destabilizuje model. To jest właściwy toxin.

**Implikacja dla corpus building:** Źródłem toxina musi być autentyczny język
internetowy, nie akademicka taksonomia twierdzeń. MBIB, LIAR, VaccineLies MisT
są nieodpowiednie jako główne źródła. NaturalNews, Mercola, WUWT — właściwe.

**Wartość dla Paper 1:** To jest strongest finding sekcji Discussion. Pokazuje
że informatyczna nieodróżnialność akademickiej dezinformacji od pokarmu ma
implikacje dla systemów detekcji. Style matters more than content for LLM response.

---

## 4. Analiza jakości korpusu (corpus_quality_analysis.py)

Ranking toxinów według effect size r (food vs toxin), H(X):

| Toxin | H effect | C effect | I effect | Ocena |
|---------|---------|---------|---------|-------|
| toxin_climate (ClimateFever) | -0.823 | -0.851 | -0.607 | ✓ najlepszy |
| toxin_covid (CoAID cleaned) | -0.496 | -0.557 | -0.278 | ✓ dobry |
| toxin_vaccines (VaccineLies) | -0.797 | -0.774 | +0.274 | ✗ anomalny |

**CoAID cleanup:** 102/141 dokumentów usuniętych (72%) — artefakty webscraping
(Facebook UI, HTTP errors, duplicate headers). Pozostałe 39 autentycznych + 22
syntetyczne dokumenty taksonomiczne = 61 total.

**Korelacja Jaccard ↔ I(X;seed):**
- food: r=0.597 (p=2.5e-08)
- toxin: r=0.642 (p=8.6e-15)
- noise: r=0.491 (p=0.003)

Wszystkie poniżej 0.8 → Jaccard nie jest redundantny. Justified for Phase 2 fitness.

---

## 5. Sensitivity analysis — próg 200 tokenów

**Eksperyment:** Różne długości wyjścia: 50, 100, 150, 200, 300, 500 tokenów.
Per każdą długość: H(X), C(X), I(X;seed), KW p-value, Cliff's delta.

**Wyniki:**
- Od 50 tokenów: wszystkie metryki istotne (p < 1e-17 dla H, Cliff's delta > 0.69 dla I)
- C(X) delta odwraca znak między 150 a 200 tokenami:
  - 150 tokenów: C delta = -0.22 (toxin niższa złożoność niż food)
  - 200 tokenów: C delta = +0.23 (toxin wyższa złożoność — stabilna interpretacja)

**Decyzja:** Minimalna długość wyjścia = 200 tokenów. Próg konieczny dla stabilnych
kierunków efektów. Zapisany do `docs/metric_definitions.md`.

---

## 6. Otwarte zadania po sesji 3

- [ ] Style swap test: toxin_vaccines_mercola vs toxin_vaccines_legacy
- [ ] Zbudować alt_med i GMO toxin (5 domen = Paper 1 requirement)
- [ ] DTW protocol implementation przed Phase 2
- [ ] Ubuntu 22.04 migration (TCP blocked w WSL2)
- [ ] Streamlit dashboard deployment
- [ ] ALife Letter first draft
