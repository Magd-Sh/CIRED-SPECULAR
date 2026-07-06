# Methods — employment-merge pipeline

## 1. Objective

Produce **one employment figure per plant per year** (2006–2025) for 40 European
automotive plants, from several overlapping, noisy, partial scrapes. Each output
value keeps its source, extraction method, exact citation and URL.

## 2. Data sources

All raw inputs are frozen under `data/raw/` and are never modified. Two families:

**Archived scrapes** (a value keyed to the *snapshot* year it was captured):

| file | what it is | number extracted by |
|------|-----------|----------------------|
| `automotive_employment_long_european_2006_2026_v3_1.csv` (`base`) | web-archive scrape of plant pages | regex + LLM |
| `..._v3_1_filtered.csv`, `..._v3_1_gemini.csv` | variants of the same scrape | regex / LLM (Gemini) |
| `automotive_employment_long_eu_2003_2026_v3_2.csv` | extended scrape | regex + LLM |
| `automotive_employment_long_de_2006_2026_v3_1.csv`, `..._v2.csv` | Germany-only scrapes | regex / LLM |
| `automotive_plant_data_models_production_v3_2*.csv` | models/production scrape (employees column) | regex / Mistral LLM |
| `german_germany_employment.xlsx` | German plant spreadsheet | regex |

**Press / gap-fill** (contemporaneous figures):

| file | what it is |
|------|-----------|
| `press_gap_fills_v2.csv` | local-press search (Tavily) + LLM extraction |
| `established_gap_employment.csv` | press/official gap-fills |
| `backfill_...mistral_small.csv`, `wide_...mistral.csv` | Mistral re-extractions (a few individually-picked values) |

**Wikipedia** — a **single** file, `employment_from_wikipediaa.csv` (a value keyed
to its *reference* year). This deliberately replaces six earlier Wikipedia passes;
see §5.

> **Upstream note.** The *numbers themselves* were extracted from web pages by LLMs
> (Mistral / Haiku) in a separate scraping step. That step is **not** part of this
> repository and is **not** deterministic (see §6). This pipeline starts from the
> frozen CSVs those scrapers produced.

## 3. Pipeline (`src/pipeline.py`)

Deterministic, no LLM, driven entirely by `config/`.

1. **Load** every source into one long candidate table (plant, snapshot year,
   reference year, employees, source, family, extraction method, citation, URL).
2. **Contamination filters** — rule-based text checks (`contamination_rules.json`):
   - *cross-plant duplicate* (Wikipedia only): same citation on two plants → keep
     the one naming its own city;
   - *wrong-entity keyword* (names Kamenz/Accumotive, Hambach/Smart, Opel, Hannover…);
   - *city-mismatch* (Munich group figure mis-stamped on other BMW plants — see §5 caveat);
   - *production-units* (the value sits next to *Einheiten/units/Fahrzeuge/vehicles*);
   - *job-announcement* ("will create N jobs"); *historical* (*Jahrhundert/century*);
     *extraction artifact* (`N ~ 0`);
   - Wikipedia-specific: Porsche-AG group totals (Zuffenhausen infobox), tiny
     sub-units, and Września before it opened (2016).
   Every dropped row is logged to `outputs/contamination_removed.csv`.
3. **Outlier rule** — per plant with ≥3 values, flag a value that is >3.5× or
   <1/3.5× the plant median *and* more than 10,000 employees away from it. (The
   absolute floor avoids flagging real swings at small plants; its side effect is
   that small wrong values must be caught by the manual blocklist, step 4.)
4. **Manual blocklist** (`manual_blocks.csv`) — curated wrong values no automatic
   rule catches (two-plant combined totals, sub-units, boilerplate, regional
   figures), each with a reason.
5. **Consolidation** — one value per (plant, snapshot year): if ≥2 archived
   sources agree, take that; else the archived source with the best priority; else
   a press value. Non-archived sources never override an archived one.
6. **Curated + manual additions** fill still-empty cells (`curated_additions.csv`,
   `manual_additions.csv`).
7. **Wikipedia** (final file only) is placed on each figure's **reference year** and
   fills only cells still empty after the archived/press layers.
8. **Manual edits** (`manual_edits.csv`) — analyst deletions of whole plants / cells.
9. **Stability derivation** — a value is carried forward to a later year **only if a
   later snapshot reports the identical number** (evidence employment was unchanged).
   This produces `employment_observations_stability.csv` and the full 2006–2025
   time-series grid `employment_timeseries_full.csv`.

## 4. What was done by hand (and how it is now captured)

Human judgment was unavoidable in a few places. Rather than leave it in ad-hoc
edits, each decision is now a **versioned data file** with a reason, so it is
transparent and re-runnable:

| decision | file | rows |
|----------|------|------|
| curated wrong values to drop | `config/manual_blocks.csv` | 14 |
| plants / cells removed by the analyst | `config/manual_edits.csv` | 4 |
| press figures typed in by hand (no raw file) | `config/manual_additions.csv` | 16 |
| individually-picked values from the wide/backfill files | `config/curated_additions.csv` | 4 |

The analyst edits are: remove **BMW Berlin** (a motorcycle plant), **Irlbach-
Straßkirchen** and **Debrecen** (both too new to have operated over the period),
and drop **Sindelfingen 2011 = 37,000** (a Daimler-AG-wide figure).

## 5. The "single final Wikipedia" decision

Six earlier Wikipedia scrapes were consolidated into one final file. The pipeline
uses **only** that file. Consequence, quantified in §7: ~15 cells that the earlier
hand-finalised dataset had filled from the *older* Wikipedia files are not
reproduced, and a handful of AMG/Kamenz years differ where the old and new
Wikipedia gave slightly different numbers. This is a deliberate, documented choice,
not an error.

**Reproducibility caveat on the Munich rule.** The city-mismatch rule currently
lists a single city (Munich), because in this dataset only Munich's group figure
(7,800 / 6,000) was systematically mis-attributed to other BMW plants. That list
was found by inspecting *this* data, so it is reproducible on this data but not a
general algorithm. A city-agnostic version is possible but re-introduces false
positives (pages that merely mention a neighbouring city) and would need a reviewed
decision list. This is the one place where the rule encodes dataset-specific
judgment.

## 6. Reproducibility — honest status

**Reproducible (deterministic, re-runnable):** everything in `src/` + `config/`.
Given the frozen `data/raw/`, `python src/pipeline.py` reproduces the outputs
byte-for-byte, and every dropped or added value is logged with its reason.

**Not reproducible, and why:**

- **Upstream LLM extraction.** The raw numbers were pulled from web pages by
  LLMs (Mistral/Haiku), which are non-deterministic across runs and model
  versions. Re-scraping would yield different raw CSVs. We therefore *freeze* the
  raw inputs and treat them as the starting point.
- **The live web-searches** used during ad-hoc auditing (not part of this
  pipeline) reflect the web at a point in time.

**Made reproducible during this work:** the hand edits, the manual blocklist, the
hand-typed press figures and the picked additions — all previously done in Excel,
now versioned CSVs consumed by the pipeline (§4).

## 7. Verification

`python src/verify.py` compares the pipeline output to the analyst's hand-finalised
dataset:

- **362 / 381 cells (95.0%) match exactly.**
- 15 missing = old-Wikipedia cells intentionally excluded (§5).
- 4 value differences + 8 extra cells = the same old-vs-new-Wikipedia effect and
  its interaction with the historical refactor.

No difference is due to a code bug; all trace to the single documented "final wiki
only" decision.
