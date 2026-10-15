# Exploration notebooks — the checks on each data source

These are the notebooks that tested the candidate data sources before the
analysis settled on the ones it uses. They are not part of `run_all.py` and
nothing downstream imports them. They are here because several figures in the
thesis appendices come out of them, and because the appendices describe routes
that were tried and abandoned — this is the working evidence for that account.

They are kept **with their outputs**, so the figures and tables can be read
without running anything.

## Which notebook made which figure

| Notebook | Thesis figure | What it establishes |
|---|---|---|
| `erm_pavlinek_reproduction.ipynb` | `proc_pavlinek_fig6/7/8` | Reproduces Pavlínek's figures from the Eurofound ERM restructuring database, to see whether an event database could carry plant-level employment |
| `iea_facility_counts.ipynb` | `proc_iea_assembly`, `proc_iea_allcats` | Counts IEA facilities per region, first assembly only, then all seven categories. The order-of-magnitude jump between the two is the point |
| `eurostat_units_vs_employment.ipynb` | `proc_eurostat_units` | Eurostat establishment counts in NACE 29 against regional employment, for the six largest producers. These exceed the IEA totals, which closes the route from a regional figure to named plants |
| `urssaf_commune_check.ipynb` | `proc_commune_check` | Hand-collected plant employment against the URSSAF commune total for the same commune. This is what licenses reading a commune total as a plant figure where the commune holds one site |
| `urssaf_plant_matching.ipynb` | — | The earlier version of the same check: matching plants to communes and departments |
| `eurostat_nace27_vs_nace29.ipynb` | — | NACE 27 (electrical equipment) beside NACE 29, the two-sector question behind Figure C.2 |
| `eurostat_regional_first_look.ipynb` | — | First pass on Eurostat regional employment: countries, top regions, coverage |
| `eurostat_vs_production.ipynb` | — | Eurostat employment against OICA production, a consistency check on both |
| `source_reliability_check.ipynb` | — | Scorecard of the scraped sources: contamination, outliers, agreement |
| `employment_series_comparison.ipynb` | — | Early comparison of the scraped employment series against the hand-completed table |

## Running them

The inputs they read are in `../../03_data_raw/exploration_inputs/`, and the
paths inside the notebooks have been repointed there. Open a notebook from
**this** folder — the paths are relative to it:

```bash
cd 01_code/exploration && jupyter lab
```

Seven intermediate files that lived in the author's Downloads folder were not
kept, so a few cells will raise `FileNotFoundError`:

```
estimated_total_persons_employed_2021_2023.csv    eurostat_consolidated.xlsx
merged_sbs_motor_vehicle_nuts_employment_derived.xlsx
grouped_facilities_per_region.xlsx (the .csv version IS shipped)
top_communes.xlsx   urssaf_test.xlsx
etablissements-...-x-ape-last-test.xlsx
```

Each was itself produced by an earlier cell or an earlier notebook; the saved
outputs show what they contained. One more path is left pointing at the laptop,
`NUTS_RG_20M_2024_4326.geojson`, in the geopandas cell described below;
`../../02_data_inputs/geometry/` holds the 2021 vintage of the same boundaries
if you want to substitute it.

One cell of `iea_facility_counts.ipynb` joins IEA sites onto NUTS polygons and
needs **geopandas**, which is not in `environment.yml` (the analysis does
point-in-polygon by hand to avoid the dependency). Install it if you want that
cell: `mamba install -c conda-forge geopandas`. The counts it produces are also
shipped as `assembly_regional_counts.xlsx` and `regional_counts.xlsx`, so the
plotting cells run without it.

## A note on what these are

This folder is a record, not a pipeline. The notebooks were written while the
sources were still being judged, so they carry dead ends, duplicated cells and
French comments. They were renamed from the working names — `ERM_data`,
`IEA_json`, `eurostat_2`, `urssaf_data`, `Untitled-1` — and the large
interactive Plotly outputs were stripped, which took one file from 82 MB to
1.8 MB. Nothing else in them was changed.
