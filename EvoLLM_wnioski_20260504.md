# EvoLLM — Wnioski badawcze z sesji 2026-05-04

## Status

Phase 0 zakończony jako canonical baseline. Potwierdzono robust separację metryk C(X) i H_dezorg dla full-corpus run oraz liniowy profil odpowiedzi w LD50 bez klasycznego progu krytycznego.

## Najważniejsze wnioski

1. Shannon entropy H(X) nie rozróżnia food vs toxin na poziomie binarnym (mimicry effect), ale C(X) i H_dezorg są stabilnymi biomarkerami jakości informacji.
2. LD50 dla modelu bazowego ma charakter ciągły (dose-response), co wspiera interpretację odporności bazowego modelu na presję informacyjną.
3. Terminologia toxin -> toxin jest spójniejsza biologicznie i metodologicznie dla dalszych publikacji.
4. I(X;seed) pozostaje świadomie utrzymane jako bag-of-words cosine proxy dla reprodukowalności; ograniczenie odnotowane do Discussion.

## Konsekwencje dla kolejnych działań

- Canonical claims powinny pozostać zakotwiczone w pełnym runie N=880.
- Dodatkowe reruny mogą służyć jako supplementary evidence (trajektorie Q1/Q2/Q3), ale bez zamiany inferencji bazowej.