# DOAJ Pipeline Procedure (EvoLLM)

## Overview
Standardized procedure for generating OA article corpora for any research domain (e.g. altmed, climate) using DOAJ and the EvoLLM pipeline.

## Steps

1. **Fetch raw OA articles from DOAJ**
   - Script: `fetch_doaj_<domain>.py` (or adapt for domain)
   - Output: `data/raw/doaj_<domain>.jsonl`
   - Each record: article metadata, links to fulltext

2. **Filter for easy OA publisher candidates**
   - Script: `convert_to_easy_candidates_<domain>.py`
   - Input: `data/raw/doaj_<domain>.jsonl`
   - Output: `data/processed/doaj_<domain>_candidates_easy.jsonl`
   - Logic: select articles with fulltext HTML from easy_domains (springer.com, mdpi.com, plos.org, frontiersin.org, biomedcentral.com, hindawi.com)

3. **Batch extract main text from HTML**
   - Script: `batch_extract_easy_html_<domain>.py`
   - Input: `data/processed/doaj_<domain>_candidates_easy.jsonl`
   - Output: `data/processed/doaj_<domain>_structured.jsonl`
   - Removes bibliography, parses main text

4. **Validation & Analysis**
   - Check record count, text quality, fulltext coverage
   - Log issues, update pipeline as needed

## Example: altmed
- Step 1: `fetch_doaj_altmed.py` → `data/raw/doaj_altmed.jsonl`
- Step 2: `convert_to_easy_candidates.py` → `data/processed/doaj_altmed_candidates_easy.jsonl`
- Step 3: `batch_extract_easy_html.py` → `data/processed/doaj_altmed_structured.jsonl`

## Example: climate
- Step 1: `fetch_doaj_climate.py` (or adapt) → `data/raw/doaj_climate.jsonl`
- Step 2: `convert_to_easy_candidates_climate.py` → `data/processed/doaj_climate_candidates_easy.jsonl`
- Step 3: `batch_extract_easy_html_climate.py` → `data/processed/doaj_climate_structured.jsonl`

## Notes
- All scripts and outputs should be versioned and logged.
- Update this file for new domains or pipeline changes.
- Validate field mappings and domain filters for each new corpus.
