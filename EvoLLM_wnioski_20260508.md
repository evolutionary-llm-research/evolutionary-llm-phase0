# EvoLLM — Wnioski badawcze z sesji 2026-05-08

## Status na koniec dnia

MI calibration zakończona. Seed C + mi_token_ids_nmi zamrożone.
Phase 0 final rerun pending.

---

## 1. Kalibracja MI — wyniki pełne

Testowano 4 seedy (A/B/C/D) × 8 implementacji MI na kanonicznych N=880 wynikach
Phase 0. Kryterium selekcji: kierunek food > toxin + maksymalny rank-biserial r.

| Implementacja MI       | Seed A r | kier. A  | Seed B r | kier. B  | Seed C r | kier. C  | Seed D r | kier. D  |
|------------------------|----------|----------|----------|----------|----------|----------|----------|----------|
| mi_cosine              | +0.008   | correct  | +0.201   | correct  | −0.031   | reversed | −0.096   | reversed |
| mi_entropy_decomp      | −0.220   | reversed | −0.291   | reversed | −0.153   | reversed | −0.185   | reversed |
| mi_jsd                 | +0.017   | reversed | +0.190   | reversed | +0.317   | reversed | +0.211   | reversed |
| mi_npmi                | +0.084   | reversed | +0.142   | reversed | −0.271   | reversed | −0.087   | reversed |
| mi_token_ids           | −0.120   | reversed | −0.043   | reversed | +0.259   | correct  | +0.100   | correct  |
| mi_token_ids_nmi       | N/A      | —        | −0.024   | reversed | **+0.301** | **correct** | N/A | —     |
| mi_token_ids_bigrams   | N/A      | —        | −0.164   | reversed | +0.058   | correct  | N/A      | —        |
| mi_token_ids_bpe       | N/A      | —        | −0.137   | reversed | −0.012   | reversed | N/A      | —        |

**Zwycięzca: Seed C + mi_token_ids_nmi** (r=0.301, kierunek correct)

### Systematic reversal finding

`mi_entropy_decomp`, `mi_jsd`, `mi_npmi` — odwrócone dla WSZYSTKICH seedów.
Nie jest to artefakt konkretnego seeda, lecz właściwość strukturalna tych
estymatorów na tym korpusie: mierzą nakładanie się słownictwa domenowego,
a toksyny dzielą słowa kluczowe z każdym seedem dotyczącym
dezinformacji/szczepionek/klimatu, niezależnie od jakości informacji.

Wniosek: te trzy implementacje są nienadające się do roli I(X;seed) w tym
projekcie. Wynik do Discussion Paper 1.

---

## 2. Walidacja seeda C

**seed_stability_test.py** — wynik: **STABLE**

- Uruchomiono z 5 różnymi seed RNG
- std(h_x) = 0.0, std(c_x) = 0.0 — pełny determinizm
- Warunek: temperature=0.0, generation seed=42

**Truncation artifact** — pomijalny

- Seed C zawiera ucięty koniec zdania (model output urwany przy max_new_tokens)
- Zmierzono: delta H = 0.010, delta C = 0.001
- Konkluzja: bez wpływu na wyniki kalibracji

**Uzasadnienie biologiczne seeda C**

- Seed C = output modelu bazowego przed ekspozycją na biom
- Odpowiedź na diagnostic prompt:
  *"Summarize the key mechanisms by which misinformation spreads in online
  environments and describe evidence-based interventions."*
- Operacjonalizacja hipotezy panspermii: mierzymy dryft od stanu przodka
- Self-calibrating: regenerowany przy każdej zmianie modelu

---

## 3. LD50 gradient — rola składników fitness

Analiza korelacji Pearsona między stężeniem toksyny a metrykami:

| Składnik  | r        | p       | Interpretacja              |
|-----------|----------|---------|----------------------------|
| C(X)      | −0.936   | 0.002   | **główny nośnik sygnału**  |
| H_dezorg  | +0.869   | 0.011   | wtórny nośnik              |
| I(X;seed) | −0.530   | 0.221   | płaski gradient (flat)     |

**Implikacja:**

I(X;seed) z mi_token_ids_nmi poprawnie separuje klasy kanoniczne (food vs toxin
r=0.301), ale nie jest głównym składnikiem gradientu dawka-odpowiedź.
Gradient fitness jest silny dzięki C(X), nie I(X;seed).
I(X;seed) mierzy separację kanoniczną; C(X) mierzy reakcję na dawkę.

---

## 4. Kolejne kroki

- [ ] Push config changes do GitHub
- [ ] Final rerun Phase 0 (kanoniczny + LD50) z nową MI (mi_token_ids_nmi) i seedem C
- [ ] Rekalibracja k i β dla nowych wartości fitness
- [ ] Tag `phase0-final-v2`
- [ ] Start Phase 1: biome_runner.py + cli.py
