# Employment imputation - reproducible pipeline (final, audited version)

Completes the missing employment values of the plant x year panel (2006-2025) by
**multiple imputation** (M = 50), built around each plant's own trajectory, after a
**documented correction of source errors**, with **over-imputation** of low-reliability observed
values and **calibrated** uncertainty bands. Vehicle attributes come from the **EEA registration
data** (plus ADEME for the segment) and are built into the panels as the regression's explanatory
variables; they were tested as imputation covariates and did not improve the prediction, so they
are not used to fill employment.

Dependencies: `numpy`, `pandas`, `matplotlib`, `openpyxl`.

## Where this sits in SPECULAR

*(added when the pipeline was brought into the archive; everything else in this
file is the original documentation.)*

This is where `02_data_inputs/plants/employment_completed_summary.csv` comes
from. That file is the **34-plant run** (version of 4 October 2026, audited),
byte-identical to `outputs_all/employment_completed_summary.csv` here.
`foundation.load_employment_imputed()` reads it, and every shaded band in
Chapter 4.2 is its uncertainty.

- `run_all.py` of this folder is shipped as **`run_imputation_all.py`**, so it is
  not confused with the archive's own `run_all.py`.
- The source workbook it was built from is `source/employment_timeseries_full.xlsx`
  (the merge extract of 2 July 2026). The models workbook
  (`auto_models_production_consolidated_v5.xlsx`) was not kept, so the panels
  `panel_assembly.csv` / `panel_all.csv` are shipped and steps 4 onwards run:

```bash
cd 01_code/imputation
python step5_run_imputation.py panel_assembly.csv outputs 50
python step5_run_imputation.py panel_all.csv outputs_all 50 5
python audit_imputation.py source/employment_timeseries_full.xlsx panel_assembly.csv outputs
python audit_imputation.py source/employment_timeseries_full.xlsx panel_all.csv outputs_all 5
python step6_plots.py panel_assembly.csv outputs && python step7_spaghetti.py outputs
```

  Re-running step 5 reproduces the shipped summaries exactly (checked: every row,
  maximum difference 0). The audit gives 29/29 on the 26-plant run and 0 failures,
  1 expected warning on the 34-plant run.
- **Vintage.** The imputation was built on the 2 July extract of the merged data.
  The merge pipeline (`01_code/pipeline`, `run_merge.py`) was re-run on 7 September
  and corrected on 4 October; its output differs from the July extract in a few
  dozen cells, because sources were added and removed between the two runs. The
  15 corrections in `data_corrections.csv` are applied in both places; the merge
  adds four more, for two Porsche Zuffenhausen figures that are not plant headcounts
  and that the July extract did not contain.

## Run

```bash
python run_all.py ../uploads/employment_timeseries_full.xlsx \
                  ../uploads/auto_models_production_consolidated_v5.xlsx \
                  /path/to/EEA   50        # EEA folder only needed if eea_family_year.csv is absent
```
The chain ends with `audit_imputation.py` on both runs and stops if any check fails.

## Files

| File | Role |
|---|---|
| `eea_aggregate.py` | EEA registration files -> `eea_model_year.csv` (make x model x year) |
| `eea_match.py` | model-family matching -> `eea_family_year.csv`, `eea_match_report.csv` |
| `data_corrections.csv` | **documented corrections of source errors** (15 rules, reason + evidence) |
| `step1_2_build_panel.py` | builds `panel_assembly.csv` (26 plants) / `panel_all.csv` (40 plants): applies the corrections, reliability weights, covariates |
| `features.py` | active span + feature matrix |
| `impute_within.py` | **the imputation engine** (calibrated defaults at the top) |
| `emb.py` | Amelia MVN (EMB) engine - benchmark only |
| `step4_validate.py` | method choice: interpolation vs Amelia MVN vs retained engine |
| `step4b_calibrate.py`, `calib_check.py` | variant comparison; calibration of the uncertainty by cell type |
| `step5_run_imputation.py` | runs the imputation, writes the deliverables |
| `step6_plots.py`, `step7_spaghetti.py` | figures |
| `audit_imputation.py` | **audit** of the deliverables (29 checks) |
| `rubin.py` | Rubin's rules for the downstream regressions |
| `run_all.py` | orchestrator |
| `ba_extract.py`, `ba_wz29_state_quarter.csv` | BA state-level employment - tested, **not used** |

## Deliverables

**`outputs/` - 26 assembly plants (the research analysis)**; 504 active plant-years, 221 observed,
**56.2% imputed**.
- `imputations/imp_01..50.csv` - 50 completed panels: `employees_imputed`, `log_emp_imputed`,
  `was_observed`, `employees_recorded` (recorded value, NaN if missing), `emp_reliability`.
- `employment_completed_summary.csv` - one row per plant-year: `employees_est` (central estimate =
  recorded value if observed), `employees_lo95` / `employees_hi95` (2.5th-97.5th percentiles of
  the draws), `est_sd_log`, `status` (observed / imputed_internal / imputed_edge /
  imputed_low_coverage / inactive), `imputed`, `noisy_observed`, `emp_reliability`.
- `plots/employment_grid.png`, `plots/by_plant/*.png`, `plots/spaghetti_{VW,MERCEDES,BMW}.png`.

**`outputs_all/` - 34 plants** (40 minus 6 with < 5 observed years: Audi Zwickau, Eisenach,
Porsche Leipzig, Osnabrueck, Wrzesnia, Wackersdorf); 618 active plant-years, 302 observed,
**51.1% imputed**. Same structure. Component/historic plants have no vehicle size.

**Regression covariates:** `panel_assembly.csv` (`kerb_weight` = EEA mass, `ev_share`,
`seg_score`, `log_prod`, `regs_proxy`), joined on `plant_id, year` to each imputed panel; pool with
`rubin.rubin_pool`.

## Data corrections (`data_corrections.csv`)

Found by auditing the source snippets; applied when the panel is built (the source spreadsheet is
not modified):
- **Emden 2019** (6,000): attendees of an information meeting, not a headcount -> dropped.
- **Sindelfingen 2015, 2017, 2022** (35,000-37,000): whole-site totals incl. the R&D centre; the
  rest of the series is the production plant -> dropped.
- **BMW Leipzig 2006**: 4,100 included suppliers -> set to the 2,300 BMW staff stated in the same
  sentence. **2013-2014** (6,000 "Arbeitsplaetze auf dem Werksgelaende", incl. on-site suppliers)
  and **2019** (LLM "contextual inference", not sourced) -> dropped.
- **BMW Landshut 2017** (1,600): a 1987 historical figure -> dropped.
- **Mercedes Untertuerkheim 2014** (site total incl. headquarters) -> dropped; **2024** -> set to the
  13,500 production-plant figure stated in the same snippet.
- **Audi Zwickau** (historic site): all values duplicated VW Zwickau's -> dropped (plant leaves the panel).
- **VW Zwickau 2023-2025**: reference year shifted by one (full snippets dated 12/2022-12/2024) ->
  moved to 2022-2024.

Systematic checks that found nothing further: every explicit date in a snippet agrees with the
assigned year; no conflicting values for the same plant-year.

## Method

All on log-employment, plant by plant.
- **Central estimate**: linear interpolation in internal gaps; flat carry from the nearest
  observation at the edges; observed values kept as recorded.
- **Reliability of an observed value** (drives over-imputation): 1.0 minus 0.15 for Wikipedia
  (rounded figures), minus 0.10 per carried year, minus 0.10 per year for **stale** values (the same
  undated text re-used for a later year, i.e. an archived copy of an unchanged page); clipped to
  [0.3, 1]. 54 of 221 observed cells have reliability < 1.
- **Over-imputation**: such values are drawn around the recorded value with sd 0.3 x (1 - r) in
  logs; neighbouring gaps are filled from the perturbed anchors.
- **Draws**: Brownian bridge in internal gaps (zero at anchors, largest mid-gap); at edges a one-off
  level noise (nugget) plus a random walk. Scales in units of the robust yearly innovation scale
  s ~ 0.056 (log): `K_INT = 3.5`, `K_NUG = 3.5`, `K_EDGE = 1.75`.
- **Calibration** (`calib_check.py`, 10 masking seeds): 95% bands contain **94.5%** of hidden true
  values in internal gaps, **96.2%** at edges 1-2 years from an observation and **93.2%** further out.

Validation (`step4_validate.py`, reference mask): RMSE (log) 0.135 internal / 0.190 edge for the
retained engine = the best predictor found (Amelia MVN: 0.268 / 0.568).

## What was tested and NOT kept (masking validation)

| Candidate | Result |
|---|---|
| Amelia joint MVN (EMB), matrix completion | much worse |
| Edge path with common year effects + covariate changes | worse than flat carry (early edges 0.192 vs 0.178, late 0.167 vs 0.144) |
| Edge path with year effects only | worse than flat carry |
| EEA registration volume as output proxy | no gain; changes uncorrelated with employment (r = -0.13) |
| Covariate-shaped bridge inside gaps | no gain |
| Plant-specific uncertainty scales | would worsen coverage for smooth plants |
| BA state-level WZ 29 employment | plants are 5-30% of their state's industry, changes uncorrelated (r = -0.07), 2014+ only, Bremen suppressed |

**Consequence for the regression.** No covariate fills employment, so imputed cells cannot create a
spurious size-employment association; they can only dilute a real one. Report the complete-case
regression (observed cells only) alongside the Rubin-pooled one.

## Audit (`audit_imputation.py`)

29 checks: structure (one row per plant-year, identical rows across the 50 panels, nothing missing,
positive values); source fidelity (every recorded value traced to the source after the documented
corrections, and vice versa); observed values identical across panels unless low-reliability, and
over-imputed with the intended spread; summary = central estimate and percentiles of the panels;
status logic; inactive cells have no signal; plausibility; 1:1 join with the covariates.
Result: 26-plant run 0 fail / 0 warning; 34-plant run 0 fail / 1 warning (very wide bands on the
longest edges, e.g. Porsche Zuffenhausen after 2012 - expected).

## Known limitations

1. Long edges (many years from the nearest observation) are very uncertain by design.
2. EEA coverage starts in 2010 (BMW 2011, Mercedes 2012); earlier sizes are nearest-year values.
3. EEA mass is per model family, not plant-specific, and is inflated by batteries (consider footprint).
4. Definitions differ slightly across sources (with/without trainees); only clear-cut errors were corrected.
