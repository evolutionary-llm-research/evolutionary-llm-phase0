# EvoLLM — Notatki robocze (running research log)

*Bieżące ustalenia metodologiczne, decyzje i obserwacje. Pełne wnioski per sesja w osobnych plikach.*

---

## Sesja 2026-05-04 — zamknięcie Phase 0

**LD50 zakończony.** Odpowiedź gradualna i liniowa, brak progu krytycznego. Model bazowy odporny na titrację. LD50 klasyczne nieestymowalne — to jest wynik, nie błąd.

**Hipoteza diagnostyczna H_diag:** h_dezorg to wczesny marker toksyczności (reaguje przy T=50-75%), c_x to późny marker (degraduje przy T=75-100%). Do weryfikacji przez analyze_ld50_thresholds.py.

**Walidacja dwupoziomowa korpusu v3:**
- Binarna: c_x p=6.7e-18, h_dezorg p=1.5e-29
- Gradientowa: r=-0.921 dla fitness, 7 punktów titracji bez nieciągłości

**I(X;seed) jako bag-of-words cosine similarity:** świadomy wybór dla przejrzystości i reprodukowalności. Zmiana na embedding-based wymagałaby rekalibracji całego pipeline. Zostawione as-is przez wszystkie papiery. Ograniczenie odnotowane w Discussion Paper 1.

**Odporność pipeline na nieoptymalny seed:** przypadkowo zweryfikowana — c_x i h_dezorg stabilne niezależnie od seed_text. I(X;seed) słabe we wszystkich 3 runach.

**Terminologia:** toxin → toxin (zatruwa, nie poluje). Spójne z LD50, metabolic decay, dose-response.

---

## Sesja 2026-05-07 — domknięcie decyzji i start rerunu po zmianie MI

- Phase 0 closed: commits `5834c53`, `c1d2cd3`, tag `phase0-final`.
- Flash Attention 2.8.3 installed, `CUDA_HOME` set in `.bashrc`.
- `trainer.py` implemented (commit after import fix).
- `population.py` implemented, statistical test for `select_parent` added.
- `biome_runner.py` implemented with lazy GPU imports.
- `mutual_information_proxy` replaced with entropy decomposition.
- Phase 0 rerun in progress (canonical + LD50).
- Pending: rerun results, recalibration of k and beta if needed,
  tag `phase0-final-v2`, then Phase 1 start.

---

## Sesja 2026-05-08 — kalibracja MI, wybór seeda C, analiza gradientu LD50

- MI calibration complete: 4 seedy × 8 implementacji, wynik zamrożony.
- **Seed C + mi_token_ids_nmi** — r_canonical=0.301, kierunek correct.
- systematic_reversal_finding: mi_entropy_decomp / mi_jsd / mi_npmi zawsze
  odwrócone dla wszystkich seedów — problem strukturalny, nie artefakt seeda.
  Materiał do Discussion Paper 1.
- Seed C STABLE: seed_stability_test.py std(h_x)=0.0, std(c_x)=0.0 × 5 RNG seeds.
- Truncation artifact pomijalny: delta H=0.010, delta C=0.001.
- **C(X) i H_dezorg = primary signal**: r=-0.936 i r=0.869 na gradiencie LD50.
- I(X;seed) = mierzy separację kanoniczną, nie gradient dawka-odpowiedź.
- config/phase0_v3.yaml zaktualizowany z Seed C + mi_token_ids_nmi.
- Pending: final rerun Phase 0, rekalibracja k/β, tag phase0-final-v2, Phase 1.

---

## Sesja 2026-05-06 — mini-rerun pod materiał supplementary

**Cel:** domknąć brakujące evidence jakościowe dryfu Q1/Q2/Q3 bez ruszania canonical run N=880.

**Co zrobiono:**
- utworzono mini-korpus `data/v2_mini_v3/` (55 dokumentów: food=25, toxin=25, noise=5),
- dodano konfigurację `config/phase0_mini_rerun_v3_toxin.yaml` z `save_chunk_texts: true`,
- uruchomiono mini-rerun i zapisano wynik: `experiments/phase0_metrics_20260506T083113Z/metrics_phase0.json`,
- potwierdzono obecność pól `gen_text_Q1`, `gen_text_Q2`, `gen_text_Q3` dla dokumentów.

**Wniosek metodologiczny:**
- dryf ilościowy był już policzony wcześniej na pełnym korpusie,
- mini-rerun służy wyłącznie do ilustracji jakościowej (przykładowe trajektorie i excerpty),
- nie nadpisuje canonical wniosków statystycznych Phase 0.

**Repo/publication workflow:**
- publiczne repo Phase 0 oznaczono tagiem zamrażającym `v0.1-phase0`,
- supplementary opublikowano jako osobny artefakt i sekcję na stronie, aby oddzielić canonical od dodatków.

## Sesja 2026-04-22 — kluczowe ustalenia

**Metryki na wyjściach, nie inputach.** H(X), C(X), I(X;seed) muszą być mierzone na outputach modelu po ekspozycji — nie na samych dokumentach. Measuring na inputach tworzy confound długości.

**Minimum 200 tokenów output.** C(X) delta odwraca znak między 150 a 200 tokenami. Poniżej 200 kierunek efektu jest niestabilny.

**Miller-Madow correction wymagana dla H(X).** H_corrected = H_empirical + (k-1)/(2·N). Zawsze raportować wersję skorygowaną.

---

## Sesja 2026-04-23 — kluczowe ustalenia

**Ollama odrzucona, Unsloth jako backend.** qwen3:8b-base niedostępny w Ollama registry. Unsloth już w stacku dla LoRA, logprobs extractable z forward pass — cleaner architecture.

**H_dezorg = perplexity z forward pass, nie Ollama.** Logprobs z `model(**inputs, labels=inputs["input_ids"])` bezpośrednio. `HF_HUB_OFFLINE=1` wymagany przy ładowaniu z cache.

---

## Sesja 2026-04-27 — kluczowe ustalenia

**Anomalia szczepionkowa.** VaccineLies MisT (akademicki styl): I effect = +0.274 (odwrócony). ClimateFever (autentyczny internet): I effect = -0.607. Styl języka ważniejszy od treści dla odpowiedzi modelu.

**Corpus toxin musi być autentyczny.** NaturalNews, Mercola, WUWT — tak. MBIB, LIAR, VaccineLies MisT — nie. Akademicka taksonomia twierdzeń jest informatycznie nieodróżnialna od food.

**Wagi fitness zamrożone po grid search:** w1=0.3, w2=0.5, w3=0.2. Nie zmieniać po Phase 0.

**Jaccard nie jest redundantny.** Korelacja z I(X;seed) < 0.8 we wszystkich typach. Justified for Phase 2.

---

## Sesja 2026-04-30 — kluczowe ustalenia

**Brighteon CTA contamination.** NaturalNews scraper zbierał krótkie strony CTA (subscribe, video link). Efekt: h_x ≈ 0, c_x ≈ 0 dla zainfekowanych dokumentów, toxin fitness > food fitness (artefakt). Filtr: min 300 znaków + blacklist patterns.

**5 aktywnych domen dla Paper 1:** climate, vaccines, alt_med, cancer, gmo. COVID wykluczone (artefakty, nakładanie z vaccines).

**Climate toxin jest najczystszy** (99% retencja po filtrowaniu). Długie artykuły argumentacyjne (Plate Climatology, WUWT) > news aggregators.

---

## Sesja 2026-05-01 — kluczowe ustalenia

**Mercola przez Windows, nie WSL2.** WSL2 blokowany przez Mercola po IP. Scraper uruchamiać z conda `evolllm` (Windows).

**Artykuły Mercola 3-5× dłuższe niż NaturalNews** (avg 16-24k vs 5k znaków). Potential confound przy single-window truncation. Wymaga normalizacji długości inputu — percentile chunking jako rozwiązanie.

**food_gmo = 77/80 akceptowalne.** Ostatnie 3 niedostępne w PMC OA. Zanotować w Methods.

---

## Sesja 2026-05-02 — kluczowe ustalenia

**Entropia nie była bezużyteczna — była źle mierzona.** Single-window truncation nie chwyta heterogeniczności dokumentu. Percentilowe chunki (5 x 20%) rehabilitują H(X): p=0.77→8.2e-21.

**Noise musi być redefiniowany przed Paper 1.** Fragmenty 50/50 to sygnał zdegradowany, nie tło środowiskowe. Wikipedia noise biologicznie i metodologicznie poprawniejszy.

**Docelowe parametry chunkingu dla korpusu v3:** window_size=1024, n_windows=3. Pokrywa 90% toxina przy pełnym profilu, kontekst generacji 2x lepszy.

**toxin_climate wymaga nowego źródła.** CARDS/PlateClimatology to datasety twierdzeń, nie artykułów. WUWT działa przez Selenium. Mercola nie pisze o klimacie (3 artykuły z 120 prób).
