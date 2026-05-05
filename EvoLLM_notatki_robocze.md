## Sesja 2026-05-04 — zamknięcie Phase 0

**LD50 zakończony.** Odpowiedź gradualna i liniowa, brak progu krytycznego. Model bazowy odporny na titrację. LD50 klasyczne nieestymowalne — to jest wynik, nie błąd.

**Hipoteza diagnostyczna H_diag:** h_dezorg to wczesny marker toksyczności (reaguje przy T=50-75%), c_x to późny marker (degraduje przy T=75-100%). Do weryfikacji przez analyze_ld50_thresholds.py.

**Walidacja dwupoziomowa korpusu v3:**
- Binarna: c_x p=6.7e-18, h_dezorg p=1.5e-29
- Gradientowa: r=-0.921 dla fitness, 7 punktów titracji bez nieciągłości

**I(X;seed) jako bag-of-words cosine similarity:** świadomy wybór dla przejrzystości i reprodukowalności. Zmiana na embedding-based wymagałaby rekalibracji całego pipeline. Zostawione as-is przez wszystkie papiery. Ograniczenie odnotowane w Discussion Paper 1.

**Odporność pipeline na nieoptymalny seed:** przypadkowo zweryfikowana — c_x i h_dezorg stabilne niezależnie od seed_text. I(X;seed) słabe we wszystkich 3 runach.

**Terminologia:** predator → toxin (zatruwa, nie poluje). Spójne z LD50, metabolic decay, dose-response.
# EvoLLM — Notatki robocze (running research log)

*Bieżące ustalenia metodologiczne, decyzje i obserwacje. Pełne wnioski per sesja w osobnych plikach.*

---

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

**Corpus predator musi być autentyczny.** NaturalNews, Mercola, WUWT — tak. MBIB, LIAR, VaccineLies MisT — nie. Akademicka taksonomia twierdzeń jest informatycznie nieodróżnialna od food.

**Wagi fitness zamrożone po grid search:** w1=0.3, w2=0.5, w3=0.2. Nie zmieniać po Phase 0.

**Jaccard nie jest redundantny.** Korelacja z I(X;seed) < 0.8 we wszystkich typach. Justified for Phase 2.

---

## Sesja 2026-04-30 — kluczowe ustalenia

**Brighteon CTA contamination.** NaturalNews scraper zbierał krótkie strony CTA (subscribe, video link). Efekt: h_x ≈ 0, c_x ≈ 0 dla zainfekowanych dokumentów, predator fitness > food fitness (artefakt). Filtr: min 300 znaków + blacklist patterns.

**5 aktywnych domen dla Paper 1:** climate, vaccines, alt_med, cancer, gmo. COVID wykluczone (artefakty, nakładanie z vaccines).

**Climate predator jest najczystszy** (99% retencja po filtrowaniu). Długie artykuły argumentacyjne (Plate Climatology, WUWT) > news aggregators.

---

## Sesja 2026-05-01 — kluczowe ustalenia

**Mercola przez Windows, nie WSL2.** WSL2 blokowany przez Mercola po IP. Scraper uruchamiać z conda `evolllm` (Windows).

**Artykuły Mercola 3-5× dłuższe niż NaturalNews** (avg 16-24k vs 5k znaków). Potential confound przy single-window truncation. Wymaga normalizacji długości inputu — percentile chunking jako rozwiązanie.

**food_gmo = 77/80 akceptowalne.** Ostatnie 3 niedostępne w PMC OA. Zanotować w Methods.

---

## Sesja 2026-05-02 — kluczowe ustalenia

**Entropia nie była bezużyteczna — była źle mierzona.** Single-window truncation nie chwyta heterogeniczności dokumentu. Percentilowe chunki (5 x 20%) rehabilitują H(X): p=0.77→8.2e-21.

**Noise musi być redefiniowany przed Paper 1.** Fragmenty 50/50 to sygnał zdegradowany, nie tło środowiskowe. Wikipedia noise biologicznie i metodologicznie poprawniejszy.

**Docelowe parametry chunkingu dla korpusu v3:** window_size=1024, n_windows=3. Pokrywa 90% predatora przy pełnym profilu, kontekst generacji 2x lepszy.

**predator_climate wymaga nowego źródła.** CARDS/PlateClimatology to datasety twierdzeń, nie artykułów. WUWT działa przez Selenium. Mercola nie pisze o klimacie (3 artykuły z 120 prób).
