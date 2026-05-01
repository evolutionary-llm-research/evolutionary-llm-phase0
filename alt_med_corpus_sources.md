# alt_med_corpus_sources.md

## Plan pobierania pełnych tekstów naukowych dla alt_med

### Priorytetowe źródła:
1. CORE (https://core.ac.uk/)
2. Europe PMC (https://europepmc.org/)
3. Semantic Scholar (https://www.semanticscholar.org/)

### Kroki:
- [ ] 1. Przygotować skrypt do pobierania artykułów OA z CORE (API lub OAI-PMH)
- [ ] 2. Przygotować skrypt do pobierania pełnych tekstów z Europe PMC (API)
- [ ] 3. Przygotować skrypt do pobierania pełnych tekstów z Semantic Scholar (API)
- [ ] 4. Standaryzować format (JSONL: id, tytuł, treść, źródło, label, ...)
- [ ] 5. Deduplikacja po DOI/PMCID/tytule/treści
- [ ] 6. Czyszczenie tekstu (usuwanie śmieci, sekcji licencyjnych, referencji)
- [ ] 7. Połączenie z istniejącym korpusem PubMed/PMC

### Notatki:
- CORE: API wymaga rejestracji, limity pobrań, OAI-PMH dla masowych dumpów
- Europe PMC: API bezpłatne, można pobierać pełne teksty OA (format XML lub plain)
- Semantic Scholar: API, dostęp do pełnych tekstów OA, limity pobrań

---

Kolejne kroki: przygotować i przetestować pierwszy skrypt (np. do CORE), potem kolejne.
