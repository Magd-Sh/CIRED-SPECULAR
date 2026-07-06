# Employment-merge pipeline

Reproducible consolidation of plant-level employment figures for European
automotive plants (Volkswagen Group, Mercedes-Benz, BMW Group), 2006–2025.

Many independent scrapes (web-archive pages, a German spreadsheet, local press,
and one Wikipedia pass) each provide noisy, partial employment numbers per plant
and year. This pipeline merges them into **one clean value per plant × year**,
with every value traceable to its source, citation and URL.

## Run

```bash
cd employment-pipeline
pip install -r requirements.txt
python src/pipeline.py     # raw data  ->  outputs/
python src/verify.py       # compare output to the analyst hand-final (optional)
python docs/make_flowchart.py   # regenerate docs/flowchart.png
```

## Layout

```
employment-pipeline/
├── data/
│   ├── raw/            frozen input scrapes (the pipeline never modifies these)
│   └── derived/        reference hand-final used only by verify.py
├── config/             every parameter, keyword list and manual decision (see below)
├── src/
│   ├── pipeline.py     end-to-end: raw -> outputs/employment_final.csv
│   └── verify.py       reproducibility report vs the hand-final
├── docs/
│   ├── METHODS.md      full methodology + reproducibility statement
│   ├── make_flowchart.py
│   └── flowchart.png
└── outputs/            generated: employment_final.csv, stability observations,
                        full time-series grid, contamination_removed.csv
```

## Config = the decisions, out in the open

Nothing is hidden in the code. Every choice lives in `config/`:

| file | what it holds |
|------|---------------|
| `source_registry.json`     | which raw files feed the merge, their family and priority |
| `parameters.json`          | period, outlier thresholds, consolidation order |
| `contamination_rules.json` | the text/keyword/regex rules that flag wrong values |
| `manual_blocks.csv`        | curated wrong values no automatic rule catches (with reasons) |
| `manual_additions.csv`     | analyst-entered press figures with no raw scrape file |
| `curated_additions.csv`    | individually justified picks from the wide/backfill files |
| `manual_edits.csv`         | analyst deletions of whole plants / single cells |

**See `docs/METHODS.md` for the methodology and an honest account of what is and
is not reproducible.**
