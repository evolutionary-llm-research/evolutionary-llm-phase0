# EvoLLM — Wnioski badawcze z sesji 2026-05-07

## Status na koniec dnia

Phase 0 formalnie zamknięty (tag `phase0-final`, commit `5834c53`).
Infrastruktura Phase 1 w pełni zaimplementowana.
`mutual_information_proxy` zastąpiony entropijną dekompozycją — rerun Phase 0 w toku.

---

## 1. Normality diagnostics — uzasadnienie testu nieparametrycznego

Uruchomiono `src/analysis/normality_check.py` na kanonicznym runie N=880.

**Wyniki Shapiro-Wilka:**
- 14/15 grup non-normal, p < 0.05
- 12/15 grup z p < 1e-5 (silna non-normalność)
- Jedyny wyjątek: H(X) dla noise (N=80, p=0.87)

**Przypadki ekstremalne:**
- c_x food: W=0.162, p=1.2e-38
- h_dezorg: wszystkie grupy W≈0.60–0.78
- fitness food: W=0.255, p=4.5e-37

**Wniosek:** Cały framework statystyczny Phase 0 (Mann-Whitney U, Kruskal-Wallis)
jest metodologicznie uzasadniony z danych, nie tylko a priori.

---

## 2. Outlier FOOD_ALT_MED_0008 — zidentyfikowany i naprawiony

**Obserwacja:** c_x_Q2=21.0 dla pojedynczego sample (oczekiwany zakres 0–1).

**Przyczyna:** gzip header overhead (~18 bajtów) na prawie pustym wyjściu modelu
gdzie h_x_Q2=0.0 — output był jeden token, gzip produkował plik większy niż input.

**Fix:** `effective_complexity` w `core.py` klipuje wynik `min(ratio, 1.0)`.
Wpływ na medianę: 0.0003 — wnioski niezakłócone.

**Interpretacja bimodalności:** fałszywy alarm. Pojedynczy outlier numeryczny,
nie struktura korpusu. Brak bimodalności c_x w próbie food.

---

## 3. Zmiana implementacji mutual_information_proxy

**Stara implementacja:** cosine similarity bag-of-words tokenów.
- Słabość: w Phase 0 korelacja z H_dezorg i C(X) niska (r=-0.754), informatycznie niespójna.

**Nowa implementacja:** MI przez dekompozycję entropijną
```
I(X;Y) = H(X) + H(Y) - H(X,Y)
NMI = max(0.0, min(1.0, I / min(H(X), H(Y))))
```
- Tokenizacja: whitespace (spójne z `shannon_entropy` i `H_dezorg`).
- Brak nowych importów (math i Counter już w module).
- Sygnatura funkcji bez zmian — pełna kompatybilność wsteczna.

**Testy:** `tests/test_metrics_mi.py` — 5 testów, wszystkie zielone.
`tests/test_metrics_core.py` — 14 testów nadal przechodzi (bez regresji).

**Konsekwencja:** wymagany pełny rerun Phase 0 (canonical + LD50).
Stare wyniki zachowane pod tagiem `phase0-final`.
Nowe wyniki będą tagowane `phase0-final-v2`.

**Uwaga matematyczna:** H(X,Y) obliczane z konkatenacji `seed_text + " " + output_text`
z białymi spacjami — standardowe przybliżenie rozkładu łącznego przy tokenizacji whitespace.
Wynik klipowany do [0,1] ze względu na możliwe ujemne I w zdegenerowanych rozkładach.

---

## 4. Phase 1 — zaimplementowane moduły

### trainer.py (`src/evolution/trainer.py`)
- Wrapper Unsloth LoRA fine-tuning.
- Import unsloth na górze modułu (wymagane dla JIT kernel patches).
- Konfigurowalne przez YAML: lora_r, lora_alpha, lr, epochs, max_seq_length.

### population.py (`src/evolution/population.py`)
- Stan populacji: słownik agentów z fitness, generacją, parent_id.
- `select_parent`: selekcja fitness-proportionate przez softmax z temperaturą beta.
- Test `select_parent`: statystyczna weryfikacja częstości (nie deterministyczny seed) —
  bardziej odporna metoda po doświadczeniach z odwróconym softmax (revert commit).
- Mechanika reprodukcji i rejestracji potomków.

### biome_runner.py (`src/evolution/biome_runner.py`)
- Pełna pętla orchestracji biome per generacja.
- JSD pairwise matrix między agentami (`compute_jsd_matrix`, `mean_jsd`).
- Lazy GPU imports — moduł importowalny bez CUDA.
- Checkpoint/resume: `--resume-from-generation N`.
- `_agent_doc_slice`: deterministyczny offset dokumentów per agent per generacja.
- `DIAGNOSTIC_PROMPT`: stały prompt do śledzenia dryfu między generacjami.

---

## 5. Decyzje projektowe Phase 1 (finalne)

- Model: qwen3:8b-base bez LoRA jako punkt startowy.
- Populacja: 10 agentów per biome.
- Dokumenty: 30 per agent per generacja, losowane w proporcjach biome.
- Generacje: 35+, bez górnego limitu.
- **Dziedziczenie: Opcja B** — potomek uczy się na dokumentach biome (nie na
  wyjściach rodzica). Rationale: izoluje biome jako zmienną dla Paper 1.
  Lamarckian inheritance (Opcja A) odkładana na Phase 2–3.
- Checkpoint/resume: ręczny `--resume-from-generation N`.
- Metryki dywergencji: macierz JSD pairwise + I(X_i;X_j) per generacja.
- Analiza trajektorii: CUSUM post-processing (nie online).

---

## 6. Otwarte zadania

- [ ] Uruchomić Phase 0 rerun z nowym I(X;Y) (canonical + LD50)
- [ ] Sprawdzić wyniki — czy recalibration k i beta konieczna
- [ ] Tag `phase0-final-v2` po walidacji rerunu
- [ ] Zaktualizować `docs/phase0_validation_summary.md` jeśli wyniki istotnie różne
- [ ] Uruchomić Phase 1 pilot (2–3 generacje, biome Savanna)
