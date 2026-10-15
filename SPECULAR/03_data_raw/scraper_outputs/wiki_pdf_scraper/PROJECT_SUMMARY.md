# European Automotive Plant Panel — Project Summary

**Folder:** `~/Downloads/wiki:pdf scraper/`
**Summary generated:** 2026-08-30
**Scope of the dataset:** 43 German-carmaker vehicle plants in Europe, panel years **2006–2026**

---

## 1. What this project is

The goal is to build a **plant × year panel dataset** for the European plants of the
German carmakers (VW Group, Mercedes-Benz, BMW Group), covering three variables:

| Variable | Meaning |
|---|---|
| **employment** | headcount at the plant, with the *reference year* the figure actually refers to |
| **production** | annual vehicle output of the plant |
| **models** | which car models were built at the plant in each year (0/1 matrix + units where known) |

No single source covers all 43 plants for all 21 years, so the project is a **stack of
independent scrapers**, each attacking the same grid from a different source, plus
merge tools that consolidate them and gap-fillers that chase the remaining blanks.

The recurring engineering problem — and the thing most of the code is about — is
**not extraction but trust**: an LLM reading a web page will happily invent both a
headcount and the citation that supposedly supports it. Every later-generation script
therefore ends in a **programmatic verification stage** rather than a second LLM.

> Note on the folder name: "pdf scraper" is historical. There is one PDF here
> (`978-3-030-14799-0-114-126.pdf` — Potvin & Villemaire, *Robust Web Data Extraction
> Based on Unsupervised Visual Validation*), kept as methodological background. The
> actual pipelines scrape the Wayback Machine, Wikipedia revision history, and news
> search APIs — not PDFs.

---

## 2. The plant registry (the spine of everything)

**`plants of german carmakers eu.xlsx`** — sheet `automotive_employment_long_eu_2`,
1 032 rows × 22 cols. Every scraper loads this file; it defines the universe.

- **43 plants**, keyed by `plant_id` (e.g. `DE_VW_WOLFSBURG`, `HU_AUDI_GYOR`)
- **7 countries:** Germany (34 plants), Hungary (3), Poland (2), Belgium, Spain, Portugal, Slovakia (1 each)
- **5 brands:** vw, mercedes, bmw, audi, porsche — **3 groups:** Volkswagen Group, Mercedes-Benz, BMW Group
- Per plant-year it carries the Wayback registry URL, `url_kind`
  (`PLANT_SPECIFIC` / `GROUP_OVERVIEW` / `GROUP_LANDING`), archive timestamp,
  archive URL, and an optional `wikipedia_url` override column

**Plant type breakdown** (from `models_output_curated/plant_classification.csv`):

| Type | Count | Meaning for the panel |
|---|---|---|
| assembly | 27 | builds complete cars → models matrix applies |
| component / powertrain / battery | 14 | builds engines, axles, batteries → **zero models is real, not missing** |
| historic | 2 | pre-1991 sites (Audi Zwickau, BMW Eisenach) |

**Operational timelines** are hard-coded in the gap-fillers (`PLANT_OPERATIONAL`,
43 entries) so a blank cell can be classified rather than blindly chased:

- Closed / end-dated: Audi Brussels (→2025), VW Dresden (→2025), Audi Zwickau (→1991), BMW Eisenach (→1991)
- Late starts: BMW Irlbach-Straßkirchen (2023), BMW Debrecen (2025), VW Września (2016), Mercedes Kecskemét (2012), Mercedes Kamenz (2012)

---

## 3. The four generations of scrapers

```
GEN 1  Wayback Machine        scrape_european_2006_2026_v3_1_gemini.py
       (corporate plant pages through time)

GEN 2  Wikipedia              scrape_models_wikipedia_v1.py
       (current article,        scrape_wiki_history_employment_v1.py  (Claude Haiku)
        then revision history)  scrape_wiki_history_employment_v1_gemini.py  (Gemini)
                               scrape_wiki_full_v2_sonnet.py   (Claude)
                               scrape_wiki_full_v2_mistral.py  (Mistral)

GEN 3  Hand curation          curated_models_build.py
       + merging              merge_datasets.py, merge_press_into_master.py

GEN 4  Press / news search    fill_employment_gaps_press_tavily.py
       (verified gap-filling)  fill_established_plant_gaps.py
                               fill_gaps_tavily.py
                               fill_gaps_gdelt.py

POC    poc_models_per_plant_year.py  (one-plant demo of the models pipeline)
```

---

## 4. Scraper profiles

### 4.1 `scrape_european_2006_2026_v3_1_gemini.py` — Wayback employment scraper (2 238 lines)

The original engine. Walks the **Internet Archive** through time for each plant's
corporate page and extracts headcounts.

- **Source:** Wayback CDX API + Availability API; up to 5 snapshots per URL per year, 2006–2026
- **Input:** `Cleaned_Wayback_Registry.xlsx` (sheet `Cleaned Registry`)
- **Extraction:** brand-specific **regex packs** (strict + loose) for Mercedes, VW, Audi, BMW, Porsche + a generic pack, then an LLM fallback (`gemini-3.5-flash`) with a second LLM **validation** pass that classifies a number as *plant* vs *group total*
- **Guards:** `_BAD_NEAR` (rejects numbers next to "Fahrzeuge", "m²", "Umsatz"…), `_GROUP_TOTAL_NEAR`, `_STRONG_PLANT_ANCHOR_NEAR`, `LOOSE_SUSPECT_THRESHOLD = 50 000`, city-variant matching (`_CITY_VARIANTS`), stuck-value flagging
- **v3.1 fixes over v3:** CDX `host_prefix` queries on huge domains (audi.com, volkswagenag.com, porsche.com) were truncated at ~150 k rows and returned only the *oldest* snapshots — replaced with server-side **city-filtered prefix queries**. LLM hits now **must** carry a reference year, killing the "round-number-no-ref" hallucination (2000/2100/2200…) and the Audi Ingolstadt `2006 = 31337` leet artefact.
- **Concurrency / resilience:** 8 workers, CDX retry ×6 with backoff, pickle caches for CDX / page text / LLM, checkpoint every 50 rows with resume
- **Output:** `automotive_employment_long_european_2006_2026_v3_1.csv` (903 rows, 22 cols) + wide-by-snapshot-year and wide-by-reference-year variants
- **Result:** 236 of 903 attempted cells produced a headcount. Status split: `no_archive` 494, `not_found` 122, `found` 121, `found_llm` 100, `http_error` 23, `cdx_failed` 16, `found_stuck` 15, `cdx_empty` 12
- **Verdict:** the Wayback route is **archive-bound** — nearly half the grid has no snapshot at all. This is what motivated Gen 2.

### 4.2 `scrape_models_wikipedia_v1.py` — Wikipedia models + employment v1 (1 054 lines)

First move to Wikipedia, and the first script to target **models**, not just employment.

- **Resolution:** MediaWiki search per plant (no manual URL list), then `langlinks` to pull **en, de, fr, nl, es, it + the plant country's language** — different editions list different models, so they are unioned with per-language provenance
- **Extraction:** structured parser for the "models produced" list + infobox, then a loosened **Gemini** pass as fallback/corroboration; `_PERSON_RE` and `_MODEL_TOKEN_RE` filter out people's birth-death spans and non-model lines
- **Cross-check:** optionally corroborates spans against the Wayback snapshots the Gen-1 pipeline already fetched (`archive_url` column reused — no re-crawl)
- **Outputs:** `models_binary_matrix.csv` (full 0/1 grid with zeros, panel-ready), `models_production_units.csv` (annual units only — cumulative totals deliberately **not** spread across years), `models_long_evidence.csv`, `employment_from_wikipedia.csv`
- **Env:** `GEMINI_API_KEY`; 3 workers, 12 RPM cap

### 4.3 `scrape_wiki_history_employment_v1.py` / `…_v1_gemini.py` — Wikipedia as a time machine (739 / 743 lines)

Same idea as the Wayback scraper, but the time machine is **Wikipedia's own revision
history** — complete, fast, and not throttled by archive.org.

- For each (plant, language, year 2006–2026): fetch the article **as it existed at that year-end** via the revision API (`rvdir=older`, `rvstart={year}-12-31`); consecutive years sharing a revision are extracted once (dedup by `revid`)
- One LLM call per unique revision extracts headcount + reference year + any annual production figure
- Two interchangeable backends: **`_v1.py` = Claude Haiku** (`claude-haiku-4-5`, 50 RPM), **`_v1_gemini.py` = Gemini** (`gemini-3.1-flash-lite`, 12 RPM). Identical logic otherwise.
- Falls back to the Wayback Machine when an article/edition has no revisions
- `LLM_ENABLED=0` gives a **free regex-only dry run** (infobox employment)
- **Outputs** (`wiki_history_output/`): `employment_from_wikipedia_history.csv` (273 rows), `production_from_wikipedia_history.csv` (369 rows), `wiki_articles_resolved.csv` (42 plants resolved)

### 4.4 `scrape_wiki_full_v2_sonnet.py` / `scrape_wiki_full_v2_mistral.py` — Wikipedia v2, three deliverables in one run (932 / 977 lines)

The most complete Wikipedia scrapers. Same file, two LLM backends.

- **Regex-first to keep the paid API cheap.** Four deterministic extractors run before any LLM call: infobox employee field (wikitext), multilingual prose employee figure + year, production **table** rows from rendered HTML (`YEAR | NUMBER`), and structured model lists (`"2004–2009: Audi A3"`, brand-filtered)
- The LLM is then called only on a **sample of revisions per language** (`LLM_SAMPLE_MAX`, newest + oldest by default) to get the full model timeline and mop up prose
- **Article-quality gate:** `_acceptable_article()` uses `PLANT_WORDS` vs `VENUE_WORDS` to reject museums, forums, arenas and stadiums that share a plant's city name
- **Employment sanity:** `PLANT_EMP_MAX = 120 000`, plus an LLM `_VALIDATE_PROMPT` that answers one word — *plant*, *group* or *other*
- **Backends:** Sonnet file → `ANTHROPIC_API_KEY`, `claude-haiku-4-5`, 45 RPM. Mistral file → `MISTRAL_API_KEY`, `mistral-small-2506`, 250 RPM, and it **auto-names its output dir after the model** (`wiki_v2_output_<model-tag>/`) so runs don't overwrite each other
- **Outputs per run:** `models_binary_matrix.csv`, `models_long.csv`, `employment_from_wikipedia.csv`, `production_from_wikipedia.csv`, `wiki_articles_resolved.csv` (43/43 plants resolved in every run)

**Model comparison actually on disk** — this folder is a bake-off, and the numbers are the evidence:

| Run | Employment rows | Model rows | Production rows |
|---|---|---|---|
| `wiki_v2_output/sonnet/` | 134 | 253 | 124 |
| `wiki_v2_output/haihku 1/` | 49 | 270 | 132 |
| `wiki_v2_output/haiku LLM-only/` | 57 | 345 | 137 |
| `wiki_v2_output/haiku regex+llm/` | 135 | 338 | 137 |
| `wiki_v2_output/jlsjdf/` | 136 | 349 | 136 |
| `wiki_v2_output_mistral/` | 193 | 977 | 294 |
| `wiki_v2_output_mistral-small-latest/` | 164 | 675 | 563 |
| `wiki_v2_output_mistral-small-2506/2006-2026/` | 101 | 528 | 223 |
| `wiki_v2_output_mistral-small-2506/` (1950–2026 run) | 115 | 561 | 383 |

Two things this table shows: **regex+llm beats LLM-only** on employment (135 vs 57 rows on
the same Haiku model), and **higher row counts are not automatically better** — the Mistral
runs are far more prolific but were the ones that needed the extra
`verified_on_page` / `suspect` columns added to their employment schema.

### 4.5 `curated_models_build.py` — hand-verified models dataset (335 lines)

A no-API, no-network script: a **hand-built ground truth** for all 43 plants, compiled by
reading each plant's Wikipedia article(s) (EN + DE/FR) directly and supplementing with
documented production facts.

- Every model row carries a **confidence flag** — `High` (explicit Wikipedia statement / dedicated plant article), `Medium` (year boundary inferred from model-generation timing), `Low` (plausible, double-check) — plus a source note and an evidence snippet
- The 16 component / powertrain / battery / historic sites are listed separately in `NON_CAR_PLANTS` so their zero is recorded as **real, not missing**
- **Outputs** (`models_output_curated/`): binary matrix, production-units matrix, long evidence file (139 model rows each), `plant_classification.csv` (43 plants), `employment_from_wikipedia_curated.csv`

### 4.6 `merge_datasets.py` — curated ∪ scraped → master (152 lines)

Unions the curated dataset with a scraper's output into one master, tagging every row
with a `method` column.

- **Curated is authoritative** where a plant exists in both; the scraped rows are kept with `superseded = yes` so you can compare rather than silently lose them
- All 43 plants appear; component/historic sites get one explicit *"no passenger-car models"* row (`method = classification`)
- **Outputs** (`models_output_master/`): `master_models_long.csv` (224 rows: 139 curated + 69 scraped-superseded + 16 classification), `master_models_binary_matrix.csv` (155 authoritative rows), `master_employment.csv` (21 rows)

### 4.7 `merge_press_into_master.py` — fold verified press fills into the master (103 lines)

- Works with both the **grid master** (`.xlsx`, one row per plant-year → blanks are **filled in place**) and the older **long CSV** (new rows appended)
- An existing value is **never overwritten**; press-sourced fills are marked `low_confidence = "Y"`
- `DROP_SUSPECT=1` excludes rows flagged `suspect = cross_plant_repeat`
- Writes `<master>_with_press.<ext>` next to the master — non-destructive

### 4.8 `fill_employment_gaps_press_tavily.py` — press gap-filler, per plant-year (444 lines)

The first Gen-4 script, and the one presented in *Point Hebdo 9*.

**Pipeline per missing (plant, year):**
1. **Tavily search** with a query in the plant's local language (de/fr/es/hu/pl/sk/pt) → real result URLs + page text
2. **Mistral** (`mistral-small-2506`) returns strict JSON: `{employees, reference_year, exact_quote, source_url}`
3. **Programmatic verification** — deterministic, *not* a second LLM:
   - `exact_quote` is a literal substring of the fetched page text
   - the number actually appears inside `exact_quote` (separator-agnostic)
   - the plant city/name appears in or near the quote (not a group total)
   - `30 ≤ employees ≤ PLANT_EMP_MAX (120 000)`
   - `source_url` is one Tavily actually returned
   - `reference_year ∈ 2006–2026` **and** that cell is still blank

- **Cost control:** `MAX_TAVILY_CALLS = 900`, pickle caches for Tavily and LLM responses → re-runs resume for free
- **Outputs** (`press_gap_output/`): `press_gap_fills.csv` (75 rows) → `press_gap_fills_clean.csv` (62 rows after weak-source filtering), across **29 plants**, years 2006–2026, with source tiers: other 42, official_company 13, wikipedia 4, press 3

### 4.9 `fill_established_plant_gaps.py` — timeline-aware gap-filler (468 lines)

Adds the insight that **a blank is not always a gap**.

- Classifies every blank cell as `real_gap` (plant was operating → chase it), `pre_existence` (before the plant existed → legitimately empty) or `post_closure`
- On this master: **392 real gaps, 17 pre-existence, 31 post-closure** out of 440 blanks
- For real gaps only, searches press via Tavily and extracts **employment + production + models** with Mistral, each backed by a verbatim quote and URL, then verifies programmatically
- **Reuses the Tavily cache** from `press_gap_output/.cache` so overlapping searches cost nothing
- **Outputs** (`established_gap_output/`): `blank_classification.csv` (440), `established_gap_employment.csv` (52 rows / 15 plants / 2006–2026), `established_gap_production.csv` (17), `established_gap_models.csv` (165)

### 4.10 `fill_gaps_tavily.py` — per-plant multi-year Tavily filler (521 lines)

An efficiency rewrite of 4.8/4.9 against the **new grid master**
(`employment_merged_archive_final.xlsx`).

- Searches **once per plant**, not once per plant-year, and lets **one** Mistral call pull every year-tagged figure it can find → far fewer API calls
- Extracts employment + production + models in a single pass; Tavily returns article text inline (`raw_content`), so no separate page-fetch step
- `CLASSIFY_ONLY=1` writes just `blank_classification.csv` and **spends no API credit** — useful for auditing the grid
- **Outputs** (`tavily_gap_output/`): 400 blanks classified (352 real_gap / 17 pre_existence / 31 post_closure), 6 verified employment rows, 5 production rows, 22 model rows

### 4.11 `fill_gaps_gdelt.py` — GDELT Cloud sibling (609 lines)

Same verification contract, different discovery layer: **GDELT Cloud's clustered Stories API**.

- `GET /api/v2/stories?search=<plant> <city> employees production` → story cards with inline `top_articles`; `GDELT_DEEP=1` pulls more URLs per story
- GDELT returns **links, not body text**, so the script fetches and strips each article itself (`PAGE_MAX_CHARS = 6 000`), then makes one Mistral call per plant
- **Documented coverage caveat, in the script itself:** GDELT Cloud is a *recent-news* product. It fills ≈2023–2026 well and is weak for 2006–2015.
- **Result: 0 verified fills.** All three output CSVs are empty; only `blank_classification.csv` (400 rows) was produced. This is a **negative result worth keeping** — the source was tried, measured, and rejected.

### 4.12 `poc_models_per_plant_year.py` — proof of concept (168 lines)

A single-plant (Audi Brussels) demo written to show what a models-per-plant-per-year
extension of the employment pipeline would look like: source gathering → structured
LLM extraction into `(model_name, year_start, year_end, evidence, source)` → timeline
explosion into one row per year → cross-source validation against archived snippets,
with `DISCREPANCIES` recorded explicitly (e.g. Audi A1 end-of-production 2016 vs 2018).
Output: `models_per_plant_year_POC.xlsx`, three sheets.

---

## 5. The verification contract (the methodological core)

Every Gen-4 script ends with the **same six deterministic checks**, and the reason is
stated plainly in the code comments: *the dominant risk with press + LLM is fabricated
citations, and two LLMs validating each other is not enough.*

| # | Check | Rejects |
|---|---|---|
| 1 | `exact_quote` is a literal substring of the fetched page | invented quotes |
| 2 | the number appears inside that quote (separator-agnostic) | quote/number mismatch |
| 3 | plant city or name within ±120 chars of the quote | group-level or wrong-plant figures |
| 4 | `30 ≤ employees ≤ 120 000` | worldwide/group totals, typos |
| 5 | `reference_year ∈ [START_YEAR, END_YEAR]` | undated or out-of-window figures |
| 6 | `source_url` is one the search API actually returned | invented sources |

Plus two standing rules across all merge tools: **fill blanks only, never overwrite**,
and **every filled cell keeps its quote, its URL and a source tier**.

Source tiers used: `official_company` > `press` > `wikipedia` > `other`.

---

## 6. Where the data stands

From *Point Hebdo 9* (`Point_Hebdo_9.pptx`, the 12-slide FR presentation of the press method):

- The consolidated employment grid (plant × year, 2006–2026) was **~40 % filled — roughly 540 empty cells**
- ~540 missing cells attempted via local press → **~75 verified values, 62 after filtering weak sources, across 29–31 plants**
- API cost finding: paid (Claude Sonnet / Haiku) gives the best precision at telling plant from group and picking the right reference year; on the free tier **Mistral small-2506** was the usable option
- Next steps listed there: extend the press method to production and models, then move to **carmakers' annual and sustainability reports** as the next-best source

**Contribution by source, as it sits on disk:**

| Source | Verified employment rows |
|---|---|
| Wayback corporate pages (Gen 1) | 236 |
| Wikipedia revision history (Gen 2, best run) | 193 |
| Local press via Tavily (Gen 4) | 62 (clean) + 52 (established-plant) + 6 |
| GDELT Cloud | 0 |

---

## 7. File inventory

### Scripts (11 Python files, ~380 KB)

| File | Lines | Role |
|---|---|---|
| `scrape_european_2006_2026_v3_1_gemini.py` | 2 238 | Wayback employment scraper v3.1 |
| `scrape_models_wikipedia_v1.py` | 1 054 | Wikipedia models + employment v1 (Gemini) |
| `scrape_wiki_full_v2_mistral.py` | 977 | Wikipedia v2 — models/employment/production (Mistral) |
| `scrape_wiki_full_v2_sonnet.py` | 932 | Wikipedia v2 — same, Claude backend |
| `scrape_wiki_history_employment_v1_gemini.py` | 743 | Revision-history employment + production (Gemini) |
| `scrape_wiki_history_employment_v1.py` | 739 | Revision-history employment + production (Claude Haiku) |
| `fill_gaps_gdelt.py` | 609 | GDELT Cloud gap-filler (produced 0 fills) |
| `fill_gaps_tavily.py` | 521 | Per-plant multi-year Tavily gap-filler |
| `fill_established_plant_gaps.py` | 468 | Timeline-aware Tavily gap-filler |
| `fill_employment_gaps_press_tavily.py` | 444 | Per-cell press gap-filler (employment) |
| `curated_models_build.py` | 335 | Hand-verified models dataset, all 43 plants |
| `poc_models_per_plant_year.py` | 168 | Single-plant proof of concept |
| `merge_datasets.py` | 152 | curated ∪ scraped → master |
| `merge_press_into_master.py` | 103 | Fold press fills into the master grid |

### Inputs

| File | What it is |
|---|---|
| `plants of german carmakers eu.xlsx` | The plant registry — 43 plants, Wayback URLs, 1 032 rows |
| `automotive_employment_long_european_2006_2026_v3_1.csv` | Gen-1 output, and the baseline employment table the Wikipedia scrapers read |
| `978-3-030-14799-0-114-126.pdf` | Potvin & Villemaire, robust web extraction via visual validation (background reading) |
| `Point_Hebdo_9.pptx` + `_qa/` | Weekly presentation of the press-verification method (FR), with QA slide renders |

### Output directories

| Directory | Contents |
|---|---|
| `models_output_curated/` | Hand-verified models (139 rows) + 43-plant classification |
| `models_output_master/` | Merged curated + scraped master (224 long rows) |
| `wiki_history_output/` | Revision-history employment (273) + production (369) |
| `wiki_v2_output/` | Claude bake-off: `sonnet/`, `haihku 1/`, `haiku LLM-only/`, `haiku regex+llm/`, `jlsjdf/` |
| `wiki_v2_output_mistral/`, `…_mistral-small-latest/`, `…_mistral-small-2506/` | Mistral runs, incl. a 1950–2026 wide run |
| `press_gap_output/` | 75 raw / 62 clean verified press fills |
| `established_gap_output/` | 440 blanks classified; 52 employment + 17 production + 165 model fills |
| `tavily_gap_output/` | 400 blanks classified; 6 + 5 + 22 fills |
| `gdelt_gap_output/` | 400 blanks classified; **empty** fill files |

### Caches — the folder's disk hog

Each `*/.cache/` holds pickles of raw HTML, extracted text, revisions and LLM
responses so re-runs resume for free. They are large:

- `wiki_history_output/.cache/html.pkl` — **426 MB**
- `wiki_v2_output/.cache/html.pkl` — 244 MB
- `wiki_v2_output_mistral-small-2506/.cache/html.pkl` — 235 MB
- `wiki_v2_output_mistral-small-latest/.cache/html.pkl` — 227 MB
- `wiki_v2_output_mistral/.cache/html.pkl` — 220 MB
- Tavily caches: 71 MB (established) + 63 MB (tavily) + 50 MB (press)

**Total ≈ 1.7 GB of cache.** Safe to delete if you accept re-fetching; deleting them
does not touch any CSV output.

---

## 8. How to run

All scripts are standalone `python3 <file>` — no package, no CLI framework. Config is
via module constants and environment variables.

```bash
# Gen 1 — Wayback employment
export GEMINI_API_KEY=...
python3 scrape_european_2006_2026_v3_1_gemini.py

# Gen 2 — Wikipedia revision history (pick a backend)
export ANTHROPIC_API_KEY=sk-ant-...
python3 scrape_wiki_history_employment_v1.py
LLM_ENABLED=0 python3 scrape_wiki_history_employment_v1.py     # free regex-only dry run

# Gen 2 — Wikipedia v2, all three deliverables
export MISTRAL_API_KEY=...
python3 scrape_wiki_full_v2_mistral.py
LLM_SAMPLE_MAX=4 LLM_MAX_RPM=30 python3 scrape_wiki_full_v2_mistral.py

# Gen 3 — curated + merge (no network, no keys)
python3 curated_models_build.py
python3 merge_datasets.py

# Gen 4 — press gap-filling
export TAVILY_API_KEY=tvly-... MISTRAL_API_KEY=...
MASTER_CSV='.../employment_merged_archive_final.xlsx' python3 fill_gaps_tavily.py
CLASSIFY_ONLY=1 MASTER_CSV='...' python3 fill_gaps_tavily.py   # audit blanks, zero cost

# Fold verified fills back in (non-destructive)
MASTER_CSV='...' PRESS_CSV='press_gap_output/press_gap_fills_clean.csv' \
  python3 merge_press_into_master.py
```

**Environment variables in use**

| Variable | Used by |
|---|---|
| `GEMINI_API_KEY` | Gen-1 scraper, `scrape_models_wikipedia_v1.py`, `…_v1_gemini.py` |
| `ANTHROPIC_API_KEY` | `scrape_wiki_history_employment_v1.py`, `scrape_wiki_full_v2_sonnet.py` |
| `MISTRAL_API_KEY` | `scrape_wiki_full_v2_mistral.py`, all four gap-fillers |
| `TAVILY_API_KEY` | the three Tavily gap-fillers |
| `GDELT_API_KEY` | `fill_gaps_gdelt.py` |
| `LLM_ENABLED=0` | regex-only dry run, no key, no cost |
| `LLM_MODEL`, `LLM_MAX_RPM`, `LLM_SAMPLE_MAX` | model choice and throttling |
| `MAX_TAVILY_CALLS`, `MAX_GDELT_CALLS` | hard credit caps |
| `MASTER_CSV`, `PRESS_CSV`, `SCRAPED_LONG`, `SCRAPED_EMP` | point tools at your current master |
| `CLASSIFY_ONLY=1`, `DROP_SUSPECT=1`, `PLANT_EMP_MAX`, `SKIP_PLANTS` | behaviour switches |

Common dependencies: `requests`, `pandas`, `openpyxl`. No `requirements.txt` exists.

---

## 9. Known issues and caveats

1. **Hard-coded absolute paths.** Most scripts embed `/Users/majdshamashan/Downloads/…`
   or `/Users/majdshamashan/Desktop/cired_stage/…` as defaults. They will not run
   unchanged on another machine.
2. **Two referenced masters are not in this folder** — `employment_merged_comprehensive.csv`
   and `employment_merged_archive_final.xlsx` are the default `MASTER_CSV` values for the
   gap-fillers but live elsewhere. You must pass `MASTER_CSV=` explicitly.
3. **`scrape_european_2006_2026_v3_1_gemini.py` reads `Cleaned_Wayback_Registry.xlsx`**,
   also not in this folder, and writes to a Desktop path.
4. **Output directory naming is inconsistent.** `wiki_v2_output/` holds five hand-named
   subfolders (`haihku 1`, `jlsjdf`, `haiku regex+llm`…) that record experiments but do
   not say which script settings produced them. Only the Mistral variant auto-names by model.
5. **`models_binary_matrix1.csv` in `wiki_v2_output_mistral-small-2506/`** spans 1950–2026
   (82 columns), not the 2006–2026 panel window — a wider experimental run, not comparable
   to its siblings.
6. **GDELT produced nothing.** Keep the script for the record, but treat GDELT Cloud as
   unsuitable for pre-2020 plant data.
7. **No `README`, no tests, no `requirements.txt`, no version control** in this folder.
   All documentation lives in the module docstrings — which are, to be fair, unusually good.
8. **~1.7 GB of pickle caches** with no expiry policy (only the Gen-1 CDX cache has a
   30-day TTL).
9. **The models data has two competing truths** — the hand-curated 139 rows and the
   scraped 253–977 rows. `merge_datasets.py` resolves this by making curated authoritative,
   but only 27 assembly plants are curated, so scraped rows still carry the rest.

## 10. Suggested next steps

- Extract the shared machinery (rate gating, pickle caching, the six verification checks,
  `PLANT_OPERATIONAL`, `COUNTRY_LANG`) into a small common module — it is currently
  copy-pasted across four gap-fillers with small drifts.
- Move the hard-coded paths to a single `config.py` or `.env`.
- Write the two missing master files into this folder, or record where they live.
- Per the *Hebdo 9* roadmap: apply the verified-press method to **production and models**,
  then attack the remaining blanks with **carmakers' annual and sustainability reports**,
  which are dated, plant-level, and far less prone to group/plant confusion than press.
- Prune the caches, or move them out of the deliverable folder.
