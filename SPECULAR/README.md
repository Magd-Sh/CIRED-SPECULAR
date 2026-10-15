# Mapping employment change in the European automotive industry

Data, code and outputs behind the master's thesis of the same name, written at
CIRED under the SPECULAR project, 2026.

**Author:** Magd Shamashan · **Supervisors:** Thomas Le Gallic, Augustin
Danneaux (CIRED)

---


## Changes of 7 October 2026

- **Model line-up corrections** in a versioned file,
  `01_code/analysis/lineup_corrections.csv`, applied when the line-ups are loaded
  (`foundation.apply_lineup_corrections`). Each rule drops or adds a model over a span
  of years, with its reason and a source. Cassino (historical Fiat models removed;
  Giulietta, Giulia, Stelvio, Grecale added), Dresden (Phaeton to 2016, e-Golf
  2017-2020, ID.3 2021-2025) and Audi Brussels (A1 and S1 limited to their Brussels
  years; Q4 e-tron removed, it is built at Zwickau).
- **Battery-electric matches restricted**: the e-Golf and the Kangoo Z.E. share
  their EEA names with combustion versions; their match now uses only the
  battery-electric registrations (`eea_crosswalk.FUEL_RESTRICTED`).
- **Early full hybrids** (2010-2012, filed as "electric") now go to the combustion
  class, like the full hybrids of later years; the hybrid class is mostly plug-in
  hybrids. Re-run of `eea_build_eu27.py` for 2010-2012.
- Figure 4.1 panel (b) title shortened; the electrification-and-mass scatter now
  labels plants by short names and reports r without its largest mover.
- Figure 4.12 (`fig03_powertrain_tracked`) redrawn in the format of Figure 4.10 for
  the vehicles of the tracked plants; new annex figure
  `fig03_powertrain_tracked_vs_fleet` sets them against the EU27 fleet.
- Wording of the nameplate split: "registrations shifting" towards heavier
  nameplates, not "buyers moving" (the data have no buyer identifier).
- The raw EEA files are on INARI in `03_data_raw/eea_registrations/`, the
  default input folder of `eea_build_eu27.py`.
- The seven images the thesis takes from exploration notebooks are shipped in
  `04_outputs/process/`.

## Changes of 6 October 2026

- **EEA data rebuilt on the full European files**: every make, the 27 member states
  of the European Union (UK, Iceland and Norway left out), final data 2010-2024 and
  provisional 2025. Built by `01_code/analysis/eea_build_eu27.py` from the raw yearly
  files (about 5 GB; on INARI in `03_data_raw/eea_registrations/`). It also writes two new
  inputs: `eea_mass_hist.csv` (mass distribution) and `eea_country_class.csv`.
- **Mass decomposition chained** (computed year by year and added up) instead of
  between the two end years; the endpoint figures are kept for comparison.
- Three new figures: `fig03_mass_percentiles`, `fig03_mass_by_nameplate`,
  `fig03_three_decompositions`.

## Changes of 4 October 2026

- **New imputation** (audited version): 34 plants instead of 35 (Audi Zwickau
  removed as a duplicate of Volkswagen Zwickau), 15 documented corrections of the
  source values applied first, flat carry at the edges, calibrated bands. Details
  and the audit in `01_code/imputation/README.md`.
- **Merge pipeline corrected**: the same corrections, plus four Porsche
  Zuffenhausen figures that are not plant headcounts, now live in
  `01_code/pipeline/config/source_corrections.csv`, and a bug in the stability
  derivation that could pair a value with another row's quotation is fixed
  (nine cells). Output: 353 values, 38 plants; 386 observations, 76 carried.
- Figures in `04_outputs/` regenerated with both.

## Run it

```bash
mamba env create -f environment.yml
mamba activate specular
python run_all.py
```

That regenerates every figure into `04_outputs/`. It takes well under a minute,
reads only `02_data_inputs/`, and needs no network and no API keys. If an input
is missing it says which one and stops rather than producing a half-finished
figure.

To rebuild the plant panel from the frozen raw scrapes first:

```bash
python run_merge.py      # stage 2, then run_all.py for the figures
```

That reads the sixteen source files in `01_code/pipeline/data/raw/` and
consolidates them. It is deterministic and needs no network either. It does not
scrape anything.

To check the setup without producing anything:

```bash
python 01_code/specular_paths.py
```

### Or work through it notebook by notebook

`run_all.py` is a batch job. The analysis was *written* as seven notebooks, and
they are the better way in if you want to follow the argument or change
something rather than just reproduce it:

```
01_code/analysis/
  00_foundation.ipynb                    sources, cleaning, the merged panel
  01_spatial_structural_change.ipynb     Chapter 4.1  where employment moved
  02_organisational_structural_change.ipynb  Chapter 4.2  inside the firm
  03_product.ipynb                       Chapter 4.3  model line-ups
  03_product_model_mix.ipynb             Chapter 4.3  mix and churn
  03b_vehicle_mass.ipynb                 Chapter 4.3  mass and powertrain

The three 03 notebooks all import `module03_product`, which now holds both the
model-line-up analysis and the vehicle-mass analysis. They were separate modules
(`module03_product` and `module03b_mass`) while the work was in progress; the
split was arbitrary and every figure in Section 4.3 drew on both, so they were
merged.
  04_macro_micro.ipynb                   Chapter 4.4  plants against regions
```

Each begins with a bootstrap cell that resolves the data paths, so they run from
wherever they are opened. `00_foundation` is the one to read first: everything
else imports from it.

---

## What is here

```
SPECULAR/
├── run_all.py          rebuilds every figure                  →  04_outputs/
├── run_merge.py        rebuilds the plant panel               →  01_code/pipeline/outputs/
├── environment.yml     the conda/mamba environment
│
├── 01_code/
│   ├── analysis/       the nine modules behind every figure in the thesis, and
│   │                   seven notebooks that work through them chapter by
│   │                   chapter. Start with foundation.py.
│   ├── pipeline/       the merge that turns many noisy scrapes into one value
│   │                   per plant and year. Has its own config/, src/, docs/.
│   ├── imputation/     how the gaps in the employment panel were filled. Holds
│   │                   the 50 draws and both runs; see the README inside it.
│   ├── scrapers/       the collection scripts. They need API keys and network
│   │                   access, and nothing here re-runs them.
│   ├── exploration/    the notebooks that tested each candidate data source
│   │                   before the analysis settled on the ones it uses. Their
│   │                   outputs are saved inside them.
│   └── specular_paths.py    every path in the project, resolved from this folder
│
├── 02_data_inputs/     what the analysis reads
│   ├── plants/         the plant panel and the completed employment grid
│   ├── eurostat/       regional employment, NACE 29 and 27
│   ├── urssaf/         French commune × APE employment
│   ├── eea/            vehicle mass and CO2, aggregated to make–model–year
│   ├── iea/            the facility map used as an external coverage check
│   └── geometry/       NUTS boundaries, so the maps draw without network access
│
├── 03_data_raw/        the bulky sources. None of it is needed to run the analysis.
│   ├── scraper_outputs/     the source files the merge consumes
│   ├── scraper_caches/      Wayback and LLM responses, about 5 GB. They make a
│   │                        re-scrape much cheaper and closer to reproducible.
│   ├── exploration_inputs/  what the exploration notebooks read
│   └── eea_registrations/   the raw EEA files
│
└── 04_outputs/         the 43 figures run_all.py produces. It can be deleted;
                        rebuilding takes about ninety seconds.
```

### Which command produces what

| Command | Reads | Produces |
|---|---|---|
| `python run_all.py` | `02_data_inputs/` only | 43 figures in `04_outputs/` |
| `python run_merge.py` | `01_code/pipeline/data/raw/` | 353 values and 386 observations in `01_code/pipeline/outputs/` |
| `01_code/imputation/step5_run_imputation.py` | `panel_all.csv` | the folder you name on the command line |
| the notebooks in `01_code/analysis/` | `02_data_inputs/` | figures in `04_outputs/` |
| the notebooks in `01_code/exploration/` | `03_data_raw/exploration_inputs/` | nothing; their results are already in the notebooks |
| the scripts in `01_code/scrapers/` | the internet | `03_data_raw/scraper_outputs/`, which should not be overwritten |

Nothing writes into `02_data_inputs/`. `run_merge.py` leaves its result in
`01_code/pipeline/outputs/` instead of replacing the panel the analysis reads,
so the two can be compared before either is adopted.

### The two side stages

`run_all.py` ends by listing nine images in the thesis that it does not build.
They are not lost; they belong to two stages that sit beside the main chain.

**`01_code/imputation/`** — the multiple imputation that filled the employment
panel. Its output *is* `02_data_inputs/plants/employment_completed_summary.csv`,
so this is the provenance of every shaded band in Chapter 4.2. It reproduces
offline from the saved panel:

```bash
cd 01_code/imputation
python step5_run_imputation.py panel_all.csv outputs_all 50 5
python step6_plots.py          panel_all.csv outputs_all
python step7_spaghetti.py                    outputs_all
```

The central estimate comes back exactly equal to the shipped file. The 50 draws
themselves are kept in `outputs_all/imputations/`. Read the README there before
changing anything: there are two runs, 26 assembly plants and 34 all-plants, and
the thesis uses the second.

**`01_code/exploration/`** — the notebooks that judged each candidate data
source before the analysis settled: the ERM reproduction, the IEA facility
counts, the Eurostat establishment counts, the URSSAF commune check. Seven
appendix figures come out of them. They are kept with their outputs, so they can
be read without being run, and their inputs are in
`03_data_raw/exploration_inputs/`. The README there maps each notebook to the
figure it produced.

## Where the paths live

Every path resolves from this folder through `01_code/specular_paths.py`. Move
the project anywhere and it still works. To point it somewhere else:

```bash
export SPECULAR_ROOT=/data/shared/SPECULAR
```

The scripts originally carried absolute paths from the laptop they were written
on. Those have been replaced; if you find one that was missed, it will announce
itself as a `FileNotFoundError` naming a `/Users/...` path.

---

## The three stages, and which of them reproduce

| | | |
|---|---|---|
| **1. Collection** | `01_code/scrapers/` | Needs keys and network. Not reproducible, by nature — see below |
| **2. Merge** | `run_merge.py` | Deterministic, offline |
| **2b. Imputation** | `01_code/imputation/` | Offline. The central estimate is deterministic; the interval is drawn |
| **3. Analysis** | `run_all.py` | Deterministic, offline |

**Stages 2 and 3 are deterministic.** Given the same inputs they give the same
output, every time. Together they are the chain that can be checked.

**Stage 1 is not, and cannot be.** The plant employment panel was built
by scraping web archives, Wikipedia and the local press, with a language model
reading pages that patterns could not. Two things make that irreproducible in
principle: the Internet Archive returns a different sample of snapshots on each
run, and the models are hosted services whose behaviour changes without notice.
Re-running the scrapers will recover *most* of the same cells, not the same
table.

That is why `03_data_raw/scraper_outputs/` is frozen and shipped. The merge that
runs on those files **is** deterministic, so the chain from raw scrape to final
panel can be checked even though the scrape itself cannot be repeated.

## Re-collecting, if you want to

```bash
cp .env.example .env      # then add your own keys
cd 01_code/scrapers
python scrape_european_2006_2026_v3_1_gemini.py
```

Four API keys may be needed — Mistral, Gemini, Anthropic and Tavily. **None is
stored in this archive**; each person supplies their own. The scrapers cost real
money to run and are rate-limited.

Three things worth knowing before you start:

- **Run each scraper more than once.** A single pass returns a sample of what is
  retrievable, not all of it, and a different sample each time. Three runs is a
  reasonable rule of thumb; the union of runs is what was used.
- **The caches in `03_data_raw/scraper_caches/` are worth keeping.** They hold
  Wayback CDX responses and LLM replies. With them, a re-run is far cheaper and
  much closer to reproducible. They are the largest thing here by a wide margin.
- **The models are pinned where possible.** `claude-haiku-4-5-20251001`,
  `mistral-small-2506`. The Gemini identifiers in the code no longer resolve in
  Google's published model list, which is itself the argument for freezing
  outputs rather than trusting a name to keep meaning the same thing.

---

## Reading the data

Two definitions cause more trouble than anything else here, and both are easy to
get wrong in a way that changes the headline numbers.

**Balanced panels.** "Balanced" means a unit appears in every year of the window
— *strict* balance. There is also an endpoint version, present in the first and
last year only, and the two differ materially: 68 regions against 75, European
growth +6.7% against +5.8%, a core losing 142,000 jobs against 149,000. The
thesis uses strict throughout. `run_all.py` builds it explicitly in
`strict_regional_panel()` rather than calling the convenience function, and says
why in a comment.

**Complete does not mean observed.** The 31-plant panel behind Section 4.2 has a
value in every year, but 49% of those cells are imputed rather than read from a
source, and only one plant is observed throughout. The shaded bands on those
figures are that uncertainty.

Sources disagree about plant headcount by a mean coefficient of variation of 36%
where they overlap. Changes are on firmer ground than levels.

---

## Provenance

Every figure in the panel carries its source and the exact quotation behind it.
The audit of a stratified sample, the scorecard of all sixteen source files, and
the four data routes that were tried and abandoned are written up in the thesis
appendices rather than repeated here.
