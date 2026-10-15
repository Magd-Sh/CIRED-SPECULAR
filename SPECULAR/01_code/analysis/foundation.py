# -*- coding: utf-8 -*-
"""
================================================================================
 MODULE 00 — FOUNDATION DATA LAYER
 Thesis: "Cartographie des mutations de l'emploi dans l'industrie automobile
          européenne (2006-2025)"  ·  CIRED / SPECULAR
================================================================================

WHAT THIS MODULE DOES
---------------------
Loads, harmonises and validates every source used downstream, and returns
analysis-ready tidy tables. Nothing here computes results; it builds the
substrate that Modules 01+ (spatial shift-share, organisational, product/size,
macro-micro) all sit on.

  MICRO   plant panel        auto_plants_models_harmonised.xlsx (89 plants)
  MACRO   Eurostat SBS       C29 + C27, NUTS-2, persons employed & local units
          - 2008-2020  sbs_r_nuts06_r2  (legacy extract)
          - 2021-2023  sbs_r_nuts2021   (current extract)
  MACRO   URSSAF (France)    commune x APE, 2006-2025, C29 (2910/2920/2931/2932)
                             and C27 (ten 27xx classes)
  LINK    plant -> NUTS-2 crosswalk (built here, both NUTS vintages)

THREE METHODOLOGICAL RULES ENCODED HERE
---------------------------------------
1. FRANCE ALWAYS USES URSSAF.  Eurostat regional data is unreliable for France;
   every France figure is sourced from URSSAF. Enforced by `FRANCE_SOURCE`
   and by `eurostat_panel(drop_france=True)` being the default.

2. NUTS VINTAGES ARE RECONCILED EXPLICITLY, NOT ASSUMED.  The 2008-2020 extract
   contains BOTH legacy and current codes for regions that were recoded
   (e.g. Saxony DED1/DED3 legacy vs DED4/DED5 current; Brandenburg DE41+DE42
   vs DE40). We prefer the current code and fall back to the legacy code only
   where the current one is empty. Portugal is recoded at the seam
   (Palmela: PT17 -> PT1B). All of this is data-driven and reported.

3. NACE 27 IS READ AT TWO LEVELS.  The C27 aggregate (all electrical equipment)
   is the only level Eurostat offers at NUTS-2, but it is a weak proxy for
   electrification: it bundles batteries and motors with lighting, wiring and
   domestic appliances. URSSAF gives 4-digit APE for France, so we compute
   BOTH the Eurostat-comparable aggregate AND the narrow electrification
   classes (2711Z motors/generators + 2720Z batteries/accumulators). The France
   comparison bounds the bias in the European aggregate.

USAGE
-----
    import foundation as fd
    fd.CONFIG.data_dir = "/path/to/data"        # where the xlsx files live
    d = fd.build_all()                          # dict of tidy DataFrames
    fd.print_validation_report(d)

REQUIREMENTS
------------
    pandas, numpy, matplotlib            (core)
    plotly                               (choropleths)
    folium                               (plant explorer map)
    requests                             (GISCO NUTS geometries; optional)

Author: Magd Shamashan
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

__all__ = [
    "CONFIG", "build_all", "print_validation_report",
    "load_plants", "load_employment", "load_production", "load_model_year",
    "load_model_attributes", "eurostat_panel", "urssaf_panel", "regional_panel", "french_regional_totals",
    "flag_stale_employment", "KNOWN_CLOSURES",
    "DEPARTEMENT_TO_NUTS2", "FRENCH_NUTS2_NAMES",
    "plant_nuts_crosswalk", "validate_crosswalk",
    "plot_coverage_diagnostics", "plot_nuts_seam_diagnostic",
    "map_nuts_choropleth", "map_plants_folium",
]


# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================

@dataclass
class Config:
    """Paths and analysis-wide constants. Edit `data_dir` then call build_all()."""

    data_dir: str = "."
    out_dir: str = "./outputs"

    # --- filenames (override if yours differ) ---------------------------------
    f_plants: str = "auto_plants_models_harmonised.xlsx"
    f_eurostat_c29_old: str = "sbs_r_nuts06_r2__custom_21357611_spreadsheet.xlsx"
    f_eurostat_c29_new: str = "sbs_r_nuts2021__custom_21357526_spreadsheet.xlsx"
    f_eurostat_c27_old: str = "sbs_r_nuts06_r2__custom_21359815_spreadsheet.xlsx"
    f_eurostat_c27_new: str = "sbs_r_nuts2021__custom_21359779_spreadsheet.xlsx"
    f_urssaf_c29: str = "etablissements-et-effectifs-salaries-au-niveau-commune-x-ape-last.xlsx"
    f_urssaf_c27: str = "urssaf_nace_27.xlsx"
    f_employment_imputed: str = "employment_completed_summary.csv"

    # --- analysis window ------------------------------------------------------
    year_min: int = 2006
    year_max: int = 2025

    # --- methodological switches ---------------------------------------------
    # RULE 1: France is always URSSAF, never Eurostat.
    FRANCE_SOURCE: str = "urssaf"

    # RULE 3: NACE 27 read at two levels.
    APE_C27_ALL: tuple = ("2711Z", "2712Z", "2720Z", "2731Z", "2732Z",
                          "2733Z", "2740Z", "2751Z", "2752Z", "2790Z")
    APE_C27_NARROW: tuple = ("2711Z", "2720Z")   # motors/generators + batteries
    APE_C27_BATTERY: tuple = ("2720Z",)          # batteries/accumulators only
    APE_C29_ALL: tuple = ("2910Z", "2920Z", "2931Z", "2932Z")
    APE_C29_NARROW: tuple = ("2910Z",)           # motor vehicles proper

    # Eurostat special values
    NA_TOKENS: tuple = (":", ": ", "", "-")
    CONFIDENTIAL_FLAGS: tuple = ("c", "C")

    def path(self, filename: str) -> Path:
        """
        Resolve a data file by name.

        Looks directly under `data_dir` first, then one level of subfolders.
        The archived layout groups inputs by provenance (plants/, eurostat/,
        urssaf/, eea/, geometry/) while the code asks for them by bare filename,
        so a flat lookup alone would fail after the move. Falls back to the
        plain join so the error message still names the file it wanted.
        """
        base = Path(self.data_dir)
        direct = base / filename
        if direct.exists():
            return direct
        for sub in sorted(base.glob("*/")):
            cand = sub / filename
            if cand.exists():
                return cand
        return direct

    def out(self, filename: str = "") -> Path:
        """Path inside out_dir, creating the directory if needed."""
        p = Path(self.out_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p / filename if filename else p


CONFIG = Config()


# ==============================================================================
# 1. MICRO LAYER — the harmonised plant panel
# ==============================================================================

def _read_plant_sheet(sheet: str) -> pd.DataFrame:
    return pd.read_excel(CONFIG.path(CONFIG.f_plants), sheet_name=sheet)


def load_plants() -> pd.DataFrame:
    """tbl_plants — one row per plant (the plant registry)."""
    df = _read_plant_sheet("tbl_plants")
    df["plant_uid"] = df["plant_uid"].astype(str).str.strip()
    return df


def load_employment() -> pd.DataFrame:
    """
    tbl_employment — plant x year headcount.

    Note the two measurement bases, which must never be silently mixed:
      * plant_reported   : headcount for the plant itself (DE, ES, GB, HU, ...)
      * commune_all_APE  : URSSAF commune total across auto APE codes (FR)
    A boolean `is_plant_level` column is added to make this explicit downstream.
    """
    df = _read_plant_sheet("tbl_employment")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["employees"] = pd.to_numeric(df["employees"], errors="coerce")
    df["is_plant_level"] = df["measurement_basis"].eq("plant_reported")
    df["is_observed"] = df["employees_status"].eq("observed")
    return df[(df["year"] >= CONFIG.year_min) & (df["year"] <= CONFIG.year_max)]


def load_employment_imputed() -> pd.DataFrame:
    """
    The multiply-imputed employment panel (Amelia, M=50; Points Hebdo 11-13).

    Complete plant x year grid for 35 plants x 20 years (2006-2025), i.e. the
    German-block scrape universe (DE, plus BE/ES/HU/PL/PT/SK sites collected in
    the same pipeline) minus five plants too data-poor to impute:
    Eisenach, Wackersdorf, Porsche Leipzig, Osnabrück, Września.

    Columns of substance
    --------------------
      status          observed | imputed_internal | imputed_edge | inactive
      employees_obs   the observed value where one exists (NaN otherwise)
      employees_est   the point estimate to use (= obs where observed)
      est_sd_log      pooled SD on the LOG scale (Rubin); 0 for observed cells
      employees_lo95 / employees_hi95   95% interval from the M draws

    Two things this function enforces
    ---------------------------------
    1. `inactive` rows carry NaN, not zero, and must never be read as "zero
       employees" in an aggregate.
       NOTE ON WHAT `inactive` MEANS: empirically it marks years OUTSIDE each
       plant's observed span -- where the imputation declined to extrapolate
       beyond the evidence. It is a boundary of KNOWLEDGE, not of existence
       (VW Kassel is flagged inactive 2006-2018 yet has operated since 1958).
       Do not read plant openings or closures from it; use the registry's
       `classification == "historic"` or external verification instead.
    2. `plant_id` here is the LEGACY id. It is mapped to `plant_uid` through
       tbl_plant_crosswalk (e.g. PL_VWN_POZNAN -> PL_VW_POZNAN), so this panel
       joins cleanly onto the harmonised registry.

    Interpreting `est_sd_log`: it is a log-scale SD, so ~0.10 means roughly a
    +-10% one-sigma band. Edge imputations (start/end of the window, nothing to
    interpolate between) are systematically less certain than internal ones.
    """
    path = CONFIG.path(CONFIG.f_employment_imputed)
    df = pd.read_csv(path)

    # legacy id -> plant_uid
    xw = _read_plant_sheet("tbl_plant_crosswalk")[["legacy_plant_id", "plant_uid"]]
    mapping = dict(zip(xw["legacy_plant_id"].astype(str), xw["plant_uid"].astype(str)))
    df["plant_uid"] = df["plant_id"].astype(str).map(lambda x: mapping.get(x, x))

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    for c in ("employees_obs", "employees_est", "est_sd_log",
              "employees_lo95", "employees_hi95"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["is_inactive"] = df["status"].eq("inactive")
    df["is_imputed"] = df["imputed"].astype(bool) & ~df["is_inactive"]
    # guard: inactive cells must be empty
    assert df.loc[df["is_inactive"], "employees_est"].isna().all(), \
        "inactive plant-years must carry no estimate"
    return df


def load_production() -> pd.DataFrame:
    """tbl_production — plant x year output (vehicles/engines; see output_unit)."""
    df = _read_plant_sheet("tbl_production")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["production_value"] = pd.to_numeric(df["production_value"], errors="coerce")
    df["is_observed"] = df["production_status"].eq("observed")
    return df[(df["year"] >= CONFIG.year_min) & (df["year"] <= CONFIG.year_max)]


def load_model_year() -> pd.DataFrame:
    """tbl_model_year — plant x model x year presence (the model mix)."""
    df = _read_plant_sheet("tbl_model_year")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = apply_lineup_corrections(df)
    return df[(df["year"] >= CONFIG.year_min) & (df["year"] <= CONFIG.year_max)]


LINEUP_CORRECTIONS = Path(__file__).resolve().parent / "lineup_corrections.csv"


def apply_lineup_corrections(df: pd.DataFrame,
                             path: Path = LINEUP_CORRECTIONS) -> pd.DataFrame:
    """
    Correct model line-ups that the sources get wrong, from a versioned file.

    A Wikipedia model list is a history of the plant, not a list of what it
    builds this year, and the scraper stamped it on every year: Cassino kept the
    Fiat 126 and the Marea through 2025 and never recorded the Giulia, Stelvio
    or Grecale; Dresden kept the Phaeton after 2016 and never recorded the ID.3.
    Each rule drops a model over a span of years or adds one, with its reason
    and a source, so every intervention is visible and reversible (7 Oct 2026).
    """
    if not Path(path).exists():
        return df
    rules = pd.read_csv(path)
    n_drop = n_add = 0
    for r in rules.itertuples():
        span = df["year"].between(int(r.year_from), int(r.year_to))
        same = (df["plant_uid"] == r.plant_uid) & (df["model"] == r.model)
        if r.action == "drop":
            n_drop += int((same & span).sum())
            df = df[~(same & span)]
        elif r.action == "add":
            base = df[df["plant_uid"] == r.plant_uid]
            if base.empty:
                continue
            tmpl = base.iloc[[0]]
            have = set(df.loc[same, "year"].dropna().astype(int))
            new = []
            for y in range(int(r.year_from), int(r.year_to) + 1):
                if y in have:
                    continue
                row = tmpl.copy()
                row["year"] = y; row["model"] = r.model
                row["model_brand"] = r.model_brand
                for c in ("generation", "body_variant"):
                    if c in row:
                        row[c] = np.nan
                row["raw_model"] = f"{r.model_brand} {r.model}"
                row["model_surfaces"] = f"{r.model_brand} {r.model}"
                row["value_source"] = "lineup_correction"
                row["source_block"] = "lineup_corrections.csv"
                new.append(row)
            if new:
                n_add += len(new)
                df = pd.concat([df] + new, ignore_index=True)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    return df.reset_index(drop=True)


def load_model_attributes() -> pd.DataFrame:
    """
    tbl_model_attributes — ADEME attributes per canonical model
    (kerb weight, mass in running order, segment, body type, fuel, CO2, price).

    Adds:
      kerb_weight_mid_kg : midpoint of min/max, the size proxy used downstream
      is_electric        : energy_fuel indicates a full battery-electric variant
    """
    df = _read_plant_sheet("tbl_model_attributes")
    for c in ("kerb_weight_kg_min", "kerb_weight_kg_max",
              "mass_running_order_min", "mass_running_order_max",
              "vehicle_price_eur_min", "vehicle_price_eur_max"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["kerb_weight_mid_kg"] = df[["kerb_weight_kg_min", "kerb_weight_kg_max"]].mean(axis=1)
    fuel = df.get("energy_fuel", pd.Series(index=df.index, dtype=object)).astype(str).str.lower()
    df["is_electric"] = fuel.str.contains("electric|électri|elec", regex=True, na=False)
    return df


def attach_size_attributes(model_year: pd.DataFrame,
                           attrs: pd.DataFrame) -> pd.DataFrame:
    """
    Join ADEME attributes onto plant-model-years.

    Matching is on (model, model_brand) where available, falling back to model
    alone. Unmatched rows are KEPT with NaN attributes and flagged, because the
    share of the model mix that is attribute-covered is itself a result that
    must be reported (coverage is ~15% of plant-model-years and varies by
    country: FR/ES ~47%, DE ~8%).
    """
    a = attrs.copy()
    a["_m"] = a["model"].astype(str).str.strip().str.lower()
    a["_b"] = a.get("model_brand", pd.Series(index=a.index, dtype=object)).astype(str).str.strip().str.lower()

    my = model_year.copy()
    my["_m"] = my["model"].astype(str).str.strip().str.lower()
    my["_b"] = my.get("model_brand", pd.Series(index=my.index, dtype=object)).astype(str).str.strip().str.lower()

    cols = [c for c in ("kerb_weight_mid_kg", "market_segment", "body_type",
                        "energy_fuel", "is_electric", "vehicle_price_eur_min",
                        "co2_combined_mean_g_km") if c in a.columns]

    # pass 1: model + brand
    out = my.merge(a[["_m", "_b"] + cols], on=["_m", "_b"], how="left")
    # pass 2: model only, for rows still unmatched
    miss = out["kerb_weight_mid_kg"].isna()
    if miss.any():
        a2 = a.drop_duplicates("_m")[["_m"] + cols]
        fill = out.loc[miss, ["_m"]].merge(a2, on="_m", how="left")
        for c in cols:
            out.loc[miss, c] = fill[c].values

    out["has_size_attr"] = out["kerb_weight_mid_kg"].notna()
    return out.drop(columns=["_m", "_b"])


# --- stale / post-closure detection -------------------------------------------
#
# The source panel carries values forward past the point where they are known,
# and in several cases past the point where the plant existed. Module 04's
# ratio>1 test surfaced this: Ford Genk closed in December 2014 yet carries
# 4,300 employees to 2025, which made its region's tracked employment exceed the
# region's entire NACE C29 sector.
#
# Two independent checks are applied, and both FLAG rather than delete, so the
# analyst decides what to do and nothing changes silently.

# Verified last operating year. Only sites where the closure is well documented
# are listed; a plant absent from this table is not thereby asserted to be open.
KNOWN_CLOSURES: Dict[str, int] = {
    "BE_FORD_GENK": 2014,          # closed December 2014
    "GB_FORD_BRIDGEND": 2020,      # closed September 2020
    "RU_FORD_VSEVOLOZHSK": 2019,   # Ford ended passenger-car output in Russia
}

FROZEN_MIN_RUN = 5   # consecutive identical derived values to call a series frozen


def flag_stale_employment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add two quality flags to a plant-year employment frame.

      is_post_closure   the year is after a documented closure (KNOWN_CLOSURES)
      is_frozen_series  the value sits in a run of >= FROZEN_MIN_RUN consecutive
                        IDENTICAL non-observed values -- i.e. carried forward so
                        long it has stopped being evidence

    Frozen runs found in the current data (all non-observed):
        Toyota Onnaing   3,732  2011-2025   (15 years, plant still operating)
        Ford Genk        4,300  2015-2025   (closed 2014)
        Ford Saarlouis   6,800  2016-2025
        Ford Bridgend    1,980  2012-2025   (closed 2020)

    A frozen series is not necessarily wrong — a stable plant may genuinely hold
    headcount — but it carries no information about change, so any growth rate
    computed across it is an artefact of the source, not a measurement.
    """
    d = df.copy()
    d["is_post_closure"] = False
    for pid, last_year in KNOWN_CLOSURES.items():
        d.loc[(d["plant_uid"] == pid) & (d["year"] > last_year),
              "is_post_closure"] = True

    d["is_frozen_series"] = False
    # `urssaf_commune` counts as OBSERVED here: it is an administrative
    # measurement of the commune, not a value carried forward. A stable URSSAF
    # series is real stability (or a very small commune), which is a different
    # problem from a scraped figure that has simply stopped being updated.
    observed_kinds = {"observed", "observed_scraped", "urssaf_commune"}
    for pid, g in d.groupby("plant_uid", sort=False):
        g = g.sort_values("year")
        derived = g[~g["value_kind"].isin(observed_kinds)]
        if len(derived) < FROZEN_MIN_RUN:
            continue
        # longest run of identical consecutive derived values
        run_id = (derived["employees"] != derived["employees"].shift()).cumsum()
        for _, run in derived.groupby(run_id):
            if len(run) >= FROZEN_MIN_RUN:
                d.loc[run.index, "is_frozen_series"] = True

    d["is_suspect"] = d["is_post_closure"] | d["is_frozen_series"]
    return d


def employment_panel(prefer_imputed: bool = True,
                     include_inactive: bool = False,
                     drop_post_closure: bool = False,
                     drop_frozen: bool = False) -> pd.DataFrame:
    """
    THE analysis-ready employment series. Use this, not load_employment(),
    for every downstream figure.

    Resolution rule, per plant-year
    -------------------------------
      1. If the plant is in the imputed panel and `prefer_imputed`:
         take `employees_est` and its interval. Cells flagged `inactive` are
         dropped (the plant was not operating) unless `include_inactive`.
      2. Otherwise fall back to the source table: scraped plant headcount, or
         URSSAF commune employment for France.

    Output columns
    --------------
      employees, employees_lo95, employees_hi95, est_sd_log
      value_kind      observed | imputed_internal | imputed_edge |
                      observed_scraped | derived_scraped | urssaf_commune
      is_imputed      True only for genuinely imputed cells
      is_plant_level  False for URSSAF commune rows (different quantity!)
      in_imputed_universe  True for the 35 plants with a complete imputed
                           series. NOTE: this set is the GERMAN-GROUP universe —
                           Volkswagen (18), Mercedes-Benz (12) and BMW (5) only,
                           i.e. 28 plants in Germany plus 7 German-owned sites in
                           BE/ES/HU x2/PL/PT/SK. It contains no Renault,
                           Stellantis, Ford or Toyota plant, so "balanced" here
                           means "free of coverage artefacts", NOT "representative
                           of Europe".

    THE ASYMMETRY YOU MUST RESPECT
    ------------------------------
    Only 35 plants have a complete (imputed) series. Every other plant keeps
    its patchy observed record. So a naive sum over "all plants" mixes a
    complete panel with an incomplete one and will show spurious growth as
    coverage improves. For any aggregate or trend, either
      * restrict to `in_imputed_universe` (a consistent balanced panel), or
      * report coverage alongside the level,
    and never pool `is_plant_level=False` (France) with plant headcounts
    without saying so.
    """
    src = load_employment()
    plants = load_plants()[["plant_uid", "plant_name", "city", "country_iso2",
                            "group", "brand", "classification"]]

    frames = []
    imputed_uids: set = set()

    if prefer_imputed:
        try:
            imp = load_employment_imputed()
        except FileNotFoundError:
            warnings.warn("Imputed panel not found; falling back to observed only.")
            imp = None
        if imp is not None:
            imputed_uids = set(imp["plant_uid"].unique())
            keep = imp if include_inactive else imp[~imp["is_inactive"]]
            frames.append(pd.DataFrame({
                "plant_uid": keep["plant_uid"],
                "year": keep["year"],
                "employees": keep["employees_est"],
                "employees_lo95": keep["employees_lo95"],
                "employees_hi95": keep["employees_hi95"],
                "est_sd_log": keep["est_sd_log"],
                "value_kind": keep["status"],
                "is_imputed": keep["is_imputed"],
                "is_plant_level": True,
                "source_series": "imputed_panel",
            }))

    # everything not covered by the imputed panel
    rest = src[~src["plant_uid"].isin(imputed_uids)].copy()
    rest_kind = np.where(
        rest["measurement_basis"].eq("commune_all_APE"), "urssaf_commune",
        np.where(rest["employees_status"].eq("observed"),
                 "observed_scraped", "derived_scraped"))
    frames.append(pd.DataFrame({
        "plant_uid": rest["plant_uid"],
        "year": rest["year"],
        "employees": rest["employees"],
        "employees_lo95": np.nan,
        "employees_hi95": np.nan,
        "est_sd_log": np.nan,
        "value_kind": rest_kind,
        "is_imputed": False,
        "is_plant_level": rest["is_plant_level"].values,
        "source_series": "source_table",
    }))

    out = pd.concat(frames, ignore_index=True)
    out = out.merge(plants, on="plant_uid", how="left")
    out["in_imputed_universe"] = out["plant_uid"].isin(imputed_uids)
    out = out[out["employees"].notna() | out["value_kind"].eq("inactive")]
    out = out.sort_values(["plant_uid", "year"]).reset_index(drop=True)

    # quality flags — always computed, dropped only on request
    out = flag_stale_employment(out)
    if drop_post_closure:
        n = int(out["is_post_closure"].sum())
        out = out[~out["is_post_closure"]]
        if n:
            warnings.warn(f"dropped {n} post-closure plant-years "
                          f"({', '.join(KNOWN_CLOSURES)})")
    if drop_frozen:
        n = int(out["is_frozen_series"].sum())
        out = out[~out["is_frozen_series"]]
        if n:
            warnings.warn(f"dropped {n} frozen-series plant-years")
    return out.reset_index(drop=True)


def imputed_universe(d: Optional[Dict[str, pd.DataFrame]] = None) -> List[str]:
    """The 35 plant_uids with a complete imputed series (the balanced panel)."""
    imp = d["employment_imputed"] if d else load_employment_imputed()
    return sorted(imp["plant_uid"].unique())


# ==============================================================================
# 2. MACRO LAYER A — Eurostat SBS regional (NUTS-2)
# ==============================================================================

def _parse_eurostat_sheet(path: Path, sheet: str) -> pd.DataFrame:
    """
    Parse one Eurostat 'spreadsheet' export into tidy long form.

    The export is awkward in three specific ways, all handled here:
      * 8 metadata rows precede the header, and row 9 repeats 'GEO (Codes)'.
      * Each year column is followed by an unnamed FLAG column
        ('b' = break in series, 'c'/'C' = confidential, 'e' = estimate...).
      * ':' means "not available"; 'c' means "suppressed for confidentiality".
        These are analytically different and are preserved as separate flags,
        because suppression is systematically correlated with regions dominated
        by one or two large plants -- exactly our regions of interest.
    """
    raw = pd.read_excel(path, sheet_name=sheet, skiprows=8)
    raw = raw.rename(columns={raw.columns[0]: "geo_code", raw.columns[1]: "geo_label"})
    raw = raw[raw["geo_code"].notna() & (raw["geo_code"].astype(str) != "GEO (Codes)")]

    year_cols = [c for c in raw.columns if re.fullmatch(r"\d{4}(\.0)?", str(c))]
    records = []
    for _, row in raw.iterrows():
        for yc in year_cols:
            year = int(float(str(yc)))
            value_raw = row[yc]
            # the flag column is the one immediately to the right
            idx = list(raw.columns).index(yc)
            flag = row[raw.columns[idx + 1]] if idx + 1 < len(raw.columns) \
                and str(raw.columns[idx + 1]).startswith("Unnamed") else np.nan

            s = str(value_raw).strip()
            confidential = (str(flag).strip() in CONFIG.CONFIDENTIAL_FLAGS) or (s in ("c", "C"))
            if s in CONFIG.NA_TOKENS or s.lower() == "nan":
                value = np.nan
            else:
                value = pd.to_numeric(s.replace(" ", "").replace(",", ""), errors="coerce")

            records.append({
                "geo_code": str(row["geo_code"]).strip(),
                "geo_label": str(row["geo_label"]).strip(),
                "year": year,
                "value": value,
                "flag": (str(flag).strip() if pd.notna(flag) else ""),
                "is_confidential": bool(confidential),
                "is_missing": bool(pd.isna(value)),
            })

    df = pd.DataFrame.from_records(records)
    df["nuts_level"] = df["geo_code"].str.len().map({2: 0, 3: 1, 4: 2}).fillna(-1).astype(int)
    df["is_legacy_code"] = df["geo_label"].str.contains(r"NUTS 20\d\d", regex=True, na=False)
    return df


def eurostat_panel(nace: str = "C29",
                   indicator: str = "employment",
                   drop_france: Optional[bool] = None,
                   nuts_level: int = 2) -> pd.DataFrame:
    """
    Build a continuous 2008-2023 NUTS-2 panel by splicing the two extracts.

    Parameters
    ----------
    nace       : 'C29' (motor vehicles) or 'C27' (electrical equipment)
    indicator  : 'employment' (persons employed) or 'units' (local units)
    drop_france: default None -> follows RULE 1 (True when FRANCE_SOURCE=='urssaf').
                 France rows are removed so no figure ever mixes Eurostat FR
                 with URSSAF FR.
    nuts_level : 0 (country), 1, or 2 (default).

    Vintage reconciliation
    ----------------------
    The 2008-2020 extract contains legacy AND current codes for recoded regions
    (Eurostat labels legacy ones "(NUTS 2006)" etc.). We keep current codes and
    use legacy rows only to fill years where the current code has no value.
    The resulting `code_source` column records which was used, so the seam is
    auditable rather than hidden.
    """
    if drop_france is None:
        drop_france = (CONFIG.FRANCE_SOURCE == "urssaf")

    files = {
        ("C29", "old"): CONFIG.f_eurostat_c29_old, ("C29", "new"): CONFIG.f_eurostat_c29_new,
        ("C27", "old"): CONFIG.f_eurostat_c27_old, ("C27", "new"): CONFIG.f_eurostat_c27_new,
    }
    old = _parse_eurostat_sheet(CONFIG.path(files[(nace, "old")]), indicator)
    new = _parse_eurostat_sheet(CONFIG.path(files[(nace, "new")]), indicator)
    old["vintage"] = "2008-2020"
    new["vintage"] = "2021-2023"

    panel = pd.concat([old, new], ignore_index=True)
    panel = panel[panel["nuts_level"] == nuts_level].copy()

    # --- reconcile legacy vs current codes -----------------------------------
    # A (geo_code, year) can appear twice: once from each extract, or once under
    # a legacy label. Resolution order, applied deterministically:
    #   1. rows that actually carry a value beat rows that do not
    #   2. current codes beat legacy ("NUTS 2006"-labelled) codes
    # This keeps real observations and never lets an empty legacy row mask a
    # populated current one.
    panel["code_source"] = np.where(panel["is_legacy_code"], "legacy_code", "current_code")
    panel["_has_value"] = panel["value"].notna().astype(int)      # 1 better
    panel["_is_legacy"] = panel["is_legacy_code"].astype(int)     # 0 better
    panel = (panel
             .sort_values(["geo_code", "year", "_has_value", "_is_legacy"],
                          ascending=[True, True, False, True])
             .drop_duplicates(["geo_code", "year"], keep="first")
             .drop(columns=["_has_value", "_is_legacy"]))

    panel["country_iso2"] = panel["geo_code"].str[:2]
    panel["nace"] = nace
    panel["indicator"] = indicator

    if drop_france:
        panel = panel[panel["country_iso2"] != "FR"].copy()
        panel.attrs["france_excluded"] = True

    keep = ["geo_code", "geo_label", "country_iso2", "year", "value", "flag",
            "is_confidential", "is_missing", "vintage", "code_source",
            "nuts_level", "nace", "indicator"]
    return panel[keep].sort_values(["geo_code", "year"]).reset_index(drop=True)


# ==============================================================================
# 3. MACRO LAYER B — URSSAF (France)
# ==============================================================================

def _parse_urssaf(path: Path) -> pd.DataFrame:
    """
    Parse a URSSAF commune x APE export into tidy long form.

    Layout: one row per (commune, APE) with paired year columns
    "Nombre d'établissements YYYY" and "Effectifs salariés YYYY".
    Both files verified to have no duplicate (commune, APE) rows.
    """
    df = pd.read_excel(path)
    id_cols = {
        "region": "Région", "region_code": "Code région",
        "departement": "Département", "departement_code": "Code département",
        "zone_emploi": "Zone d'emploi", "zone_emploi_code": "Code zone d'emploi",
        "commune": "Nom commune", "commune_code": "Code commune",
        "ape": "Code APE", "ape_label": "APE",
    }
    present = {k: v for k, v in id_cols.items() if v in df.columns}

    emp_cols = [c for c in df.columns if str(c).startswith("Effectifs salariés")]
    est_cols = [c for c in df.columns if str(c).startswith("Nombre d'établissements")]

    recs = []
    for _, row in df.iterrows():
        base = {k: row[v] for k, v in present.items()}
        for ec in emp_cols:
            year = int(str(ec)[-4:])
            est_col = f"Nombre d'établissements {year}"
            recs.append({
                **base,
                "year": year,
                "employees": pd.to_numeric(row[ec], errors="coerce"),
                "n_establishments": pd.to_numeric(row[est_col], errors="coerce")
                if est_col in df.columns else np.nan,
            })
    out = pd.DataFrame.from_records(recs)
    out["departement_code"] = out["departement_code"].astype(str).str.strip()
    out["commune_code"] = out["commune_code"].astype(str).str.strip()
    return out[(out["year"] >= CONFIG.year_min) & (out["year"] <= CONFIG.year_max)]


def urssaf_panel(nace: str = "C29",
                 scope: str = "all",
                 level: str = "departement") -> pd.DataFrame:
    """
    France macro panel from URSSAF (RULE 1: France is always URSSAF).

    Parameters
    ----------
    nace  : 'C29' or 'C27'
    scope : which APE classes to sum (RULE 3)
              C29: 'all'     -> 2910Z+2920Z+2931Z+2932Z (Eurostat-comparable)
                   'narrow'  -> 2910Z only (motor vehicles proper)
              C27: 'all'     -> ten 27xx classes (Eurostat-comparable aggregate)
                   'narrow'  -> 2711Z + 2720Z (motors/generators + batteries)
                   'battery' -> 2720Z only
    level : 'commune' | 'departement' | 'region' | 'national'

    Why two scopes matter for C27: in France the C27 aggregate FALLS ~20%
    (2006-2023) while batteries RISE (+10%, and +120% from the 2015 trough to
    2025). The aggregate therefore masks -- even inverts -- the electrification
    signal. Reporting both bounds the bias in the Eurostat C27 aggregate that
    is all we can observe for the rest of Europe.
    """
    path = CONFIG.path(CONFIG.f_urssaf_c29 if nace == "C29" else CONFIG.f_urssaf_c27)
    df = _parse_urssaf(path)

    scopes = {
        ("C29", "all"): CONFIG.APE_C29_ALL, ("C29", "narrow"): CONFIG.APE_C29_NARROW,
        ("C27", "all"): CONFIG.APE_C27_ALL, ("C27", "narrow"): CONFIG.APE_C27_NARROW,
        ("C27", "battery"): CONFIG.APE_C27_BATTERY,
    }
    codes = scopes[(nace, scope)]
    df = df[df["ape"].isin(codes)].copy()

    group_keys = {
        "commune": ["commune_code", "commune", "departement_code", "departement",
                    "region_code", "region"],
        "departement": ["departement_code", "departement", "region_code", "region"],
        "region": ["region_code", "region"],
        "national": [],
    }[level]

    if group_keys:
        out = (df.groupby(group_keys + ["year"], dropna=False, as_index=False)
                 .agg(employees=("employees", "sum"),
                      n_establishments=("n_establishments", "sum"),
                      n_ape_classes=("ape", "nunique")))
    else:
        out = (df.groupby(["year"], as_index=False)
                 .agg(employees=("employees", "sum"),
                      n_establishments=("n_establishments", "sum"),
                      n_ape_classes=("ape", "nunique")))

    out["nace"], out["scope"], out["level"] = nace, scope, level
    out["source"] = "URSSAF"
    out["country_iso2"] = "FR"
    return out.sort_values(group_keys + ["year"] if group_keys else ["year"]).reset_index(drop=True)


# ==============================================================================
# 3b. FRANCE — département -> NUTS-2 (shared by Modules 01 and 04)
# ==============================================================================
#
# URSSAF reports by département (NUTS-3). Départements aggregate exactly into
# the 22 pre-2016 régions, which ARE French NUTS-2 — so France can join any
# NUTS-2 analysis with URSSAF as its denominator, never Eurostat (RULE 1).

DEPARTEMENT_TO_NUTS2: Dict[str, str] = {
    # FR10 Île-de-France
    **{d: "FR10" for d in ["75", "77", "78", "91", "92", "93", "94", "95"]},
    # FRB0 Centre-Val de Loire
    **{d: "FRB0" for d in ["18", "28", "36", "37", "41", "45"]},
    # FRC1 Bourgogne · FRC2 Franche-Comté
    **{d: "FRC1" for d in ["21", "58", "71", "89"]},
    **{d: "FRC2" for d in ["25", "39", "70", "90"]},
    # FRD1 Basse-Normandie · FRD2 Haute-Normandie
    **{d: "FRD1" for d in ["14", "50", "61"]},
    **{d: "FRD2" for d in ["27", "76"]},
    # FRE1 Nord-Pas-de-Calais · FRE2 Picardie
    **{d: "FRE1" for d in ["59", "62"]},
    **{d: "FRE2" for d in ["02", "60", "80"]},
    # FRF1 Alsace · FRF2 Champagne-Ardenne · FRF3 Lorraine
    **{d: "FRF1" for d in ["67", "68"]},
    **{d: "FRF2" for d in ["08", "10", "51", "52"]},
    **{d: "FRF3" for d in ["54", "55", "57", "88"]},
    # FRG0 Pays de la Loire · FRH0 Bretagne
    **{d: "FRG0" for d in ["44", "49", "53", "72", "85"]},
    **{d: "FRH0" for d in ["22", "29", "35", "56"]},
    # FRI1 Aquitaine · FRI2 Limousin · FRI3 Poitou-Charentes
    **{d: "FRI1" for d in ["24", "33", "40", "47", "64"]},
    **{d: "FRI2" for d in ["19", "23", "87"]},
    **{d: "FRI3" for d in ["16", "17", "79", "86"]},
    # FRJ1 Languedoc-Roussillon · FRJ2 Midi-Pyrénées
    **{d: "FRJ1" for d in ["11", "30", "34", "48", "66"]},
    **{d: "FRJ2" for d in ["09", "12", "31", "32", "46", "65", "81", "82"]},
    # FRK1 Auvergne · FRK2 Rhône-Alpes
    **{d: "FRK1" for d in ["03", "15", "43", "63"]},
    **{d: "FRK2" for d in ["01", "07", "26", "38", "42", "69", "73", "74"]},
    # FRL0 PACA · FRM0 Corse · FRY overseas
    **{d: "FRL0" for d in ["04", "05", "06", "13", "83", "84"]},
    **{d: "FRM0" for d in ["2A", "2B", "20"]},
    # overseas: NUTS-2 codes are 4 characters (FRY1..FRY5); FRY10 etc. are NUTS-3
    **{d: "FRY1" for d in ["971"]}, **{d: "FRY2" for d in ["972"]},
    **{d: "FRY3" for d in ["973"]}, **{d: "FRY4" for d in ["974"]},
    **{d: "FRY5" for d in ["976"]},
}


FRENCH_NUTS2_NAMES: Dict[str, str] = {
    "FR10": "Île-de-France", "FRB0": "Centre-Val de Loire", "FRC1": "Bourgogne",
    "FRC2": "Franche-Comté", "FRD1": "Basse-Normandie", "FRD2": "Haute-Normandie",
    "FRE1": "Nord-Pas-de-Calais", "FRE2": "Picardie", "FRF1": "Alsace",
    "FRF2": "Champagne-Ardenne", "FRF3": "Lorraine", "FRG0": "Pays de la Loire",
    "FRH0": "Bretagne", "FRI1": "Aquitaine", "FRI2": "Limousin",
    "FRI3": "Poitou-Charentes", "FRJ1": "Languedoc-Roussillon",
    "FRJ2": "Midi-Pyrénées", "FRK1": "Auvergne", "FRK2": "Rhône-Alpes",
    "FRL0": "Provence-Alpes-Côte d'Azur", "FRM0": "Corse",
    "FRY1": "Guadeloupe", "FRY2": "Martinique", "FRY3": "Guyane",
    "FRY4": "La Réunion", "FRY5": "Mayotte",
}



def french_regional_totals(nace: str = "C29", scope: str = "all",
                           level: int = 2) -> pd.DataFrame:
    """
    French regional automotive employment from URSSAF (RULE 1 preserved).

    level=2 : départements aggregated into the 22 pre-2016 régions (French NUTS-2)
    level=1 : URSSAF's 13 modern régions, which ARE French NUTS-1

    Returned with the same column names as `eurostat_panel`, so the two can be
    concatenated and mapped together — see `regional_panel`.
    """
    if level == 2:
        dep = urssaf_panel(nace, scope, "departement").copy()
        dep["geo_code"] = (dep["departement_code"].astype(str).str.strip()
                             .str.replace(r"\.0$", "", regex=True).str.zfill(2)
                             .map(DEPARTEMENT_TO_NUTS2))
    elif level == 1:
        reg = urssaf_panel(nace, scope, "region").copy()
        _R1 = {"11": "FR1", "24": "FRB", "27": "FRC", "28": "FRD", "32": "FRE",
               "44": "FRF", "52": "FRG", "53": "FRH", "75": "FRI", "76": "FRJ",
               "84": "FRK", "93": "FRL", "94": "FRM",
               "1": "FRY", "2": "FRY", "3": "FRY", "4": "FRY"}
        dep = reg
        dep["geo_code"] = dep["region_code"].astype(str).str.strip().map(_R1)
    else:
        raise ValueError("level must be 1 or 2")

    out = (dep.dropna(subset=["geo_code"])
              .groupby(["geo_code", "year"], as_index=False)
              .agg(value=("employees", "sum")))
    out["geo_label"] = out["geo_code"].map(FRENCH_NUTS2_NAMES).fillna(out["geo_code"])
    out["country_iso2"] = "FR"
    out["is_confidential"] = False
    out["is_missing"] = out["value"].isna()
    out["source"] = "urssaf"
    out["nace"], out["indicator"], out["nuts_level"] = nace, "employment", level
    return out


def regional_panel(nace: str = "C29", indicator: str = "employment",
                   level: int = 2, include_france: bool = True) -> pd.DataFrame:
    """
    THE complete European regional panel: Eurostat everywhere, URSSAF for France.

    Use this for maps and any figure that should show the whole of Europe.
    `eurostat_panel` alone leaves a France-shaped hole, because RULE 1 forbids
    using Eurostat's French figures — this fills it from the right source and
    marks every row with `source` so the two are never confused.
    """
    e = eurostat_panel(nace, indicator, drop_france=True, nuts_level=level).copy()
    e["source"] = "eurostat"
    if not include_france or indicator != "employment":
        return e
    fr = french_regional_totals(nace, "all", level)
    cols = [c for c in e.columns if c in fr.columns]
    return (pd.concat([e[cols], fr[cols]], ignore_index=True)
              .sort_values(["geo_code", "year"]).reset_index(drop=True))


# ==============================================================================
# 4. THE LINK — plant -> NUTS-2 crosswalk
# ==============================================================================
#
# Assignment is by plant municipality. Two NUTS codes are stored per plant
# because the 2008-2020 and 2021-2023 extracts use different classifications
# for some regions (verified against the actual code lists in the files):
#
#   Saxony      legacy DED1 Chemnitz / DED3 Leipzig  ->  current DED4 / DED5
#   Brandenburg legacy DE41 + DE42                   ->  current DE40
#   Portugal    PT17 Área Metropolitana de Lisboa    ->  PT1B Península de Setúbal
#   Hungary     HU10 Közép-Magyarország              ->  HU11 Budapest / HU12 Pest
#
# Two coverage facts, also verified against the files:
#   * The UK is present in the 2008-2020 extract only (absent 2021+).
#   * Turkey and Russia are absent from BOTH extracts -> no Eurostat macro link.
#
# `geo_confidence` flags how certain the municipality->region assignment is.
# THIS TABLE SHOULD BE REVIEWED BEFORE PUBLICATION; validate_crosswalk() checks
# every code against the codes actually present in the Eurostat extracts.
# Coordinates are municipality centroids (approximate, for mapping only).

_CROSSWALK_ROWS = [
    # plant_uid, nuts2_new(2021+), nuts2_old(2008-2020), lat, lon, confidence, note
    ("BE_AUDI_BRUSSELS",        "BE10", "BE10", 50.810,  4.318, "high", ""),
    ("BE_FORD_GENK",            "BE22", "BE22", 50.966,  5.501, "high", "closed 2014"),
    ("CZ_TOYOTA_KOLIN",         "CZ02", "CZ02", 50.028, 15.199, "high", "ex-TPCA"),

    ("DE_AUDI_INGOLSTADT",      "DE21", "DE21", 48.771, 11.425, "high", ""),
    ("DE_AUDI_NECKARSULM",      "DE11", "DE11", 49.192,  9.223, "high", ""),
    ("DE_AUDI_ZWICKAU",         "DED4", "DED4", 50.718, 12.496, "high", "historic; legacy code DED1"),
    ("DE_BMW_DINGOLFING",       "DE22", "DE22", 48.638, 12.492, "high", ""),
    ("DE_BMW_EISENACH",         "DEG0", "DEG0", 50.976, 10.316, "high", "historic"),
    ("DE_BMW_LANDSHUT",         "DE22", "DE22", 48.537, 12.152, "high", ""),
    ("DE_BMW_LEIPZIG",          "DED5", "DED5", 51.340, 12.375, "high", "legacy code DED3"),
    ("DE_BMW_MUNICH",           "DE21", "DE21", 48.137, 11.575, "high", ""),
    ("DE_BMW_REGENSBURG",       "DE23", "DE23", 49.020, 12.098, "high", ""),
    ("DE_BMW_WACKERSDORF",      "DE23", "DE23", 49.317, 12.199, "high", ""),
    ("DE_FORD_COLOGNE",         "DEA2", "DEA2", 50.938,  6.960, "high", ""),
    ("DE_FORD_SAARLOUIS",       "DEC0", "DEC0", 49.313,  6.751, "high", ""),
    ("DE_MERCEDES_AFFALTERBACH","DE11", "DE11", 48.940,  9.340, "high", "AMG"),
    ("DE_MERCEDES_BERLIN",      "DE30", "DE30", 52.520, 13.405, "high", ""),
    ("DE_MERCEDES_BREMEN",      "DE50", "DE50", 53.079,  8.802, "high", ""),
    ("DE_MERCEDES_DUSSELDORF",  "DEA1", "DEA1", 51.228,  6.782, "high", "Sprinter"),
    ("DE_MERCEDES_HAMBURG",     "DE60", "DE60", 53.551,  9.994, "high", ""),
    ("DE_MERCEDES_KAMENZ",      "DED2", "DED2", 51.270, 14.096, "high", "Accumotive batteries"),
    ("DE_MERCEDES_KOLLEDA",     "DEG0", "DEG0", 51.186, 11.242, "high", "engines"),
    ("DE_MERCEDES_LUDWIGSFELDE","DE40", "DE42", 52.299, 13.253, "high", "Brandenburg recoded DE42->DE40"),
    ("DE_MERCEDES_RASTATT",     "DE12", "DE12", 48.858,  8.204, "high", ""),
    ("DE_MERCEDES_SINDELFINGEN","DE11", "DE11", 48.713,  9.003, "high", ""),
    ("DE_MERCEDES_STUTTGART",   "DE11", "DE11", 48.778,  9.180, "high", "Untertürkheim"),
    ("DE_PORSCHE_LEIPZIG",      "DED5", "DED5", 51.412, 12.291, "high", "legacy code DED3"),
    ("DE_PORSCHE_ZUFFENHAUSEN", "DE11", "DE11", 48.832,  9.155, "high", ""),
    ("DE_VW_BRUNSWICK",         "DE91", "DE91", 52.269, 10.521, "high", ""),
    ("DE_VW_CHEMNITZ",          "DED4", "DED4", 50.827, 12.921, "high", "legacy code DED1"),
    ("DE_VW_DRESDEN",           "DED2", "DED2", 51.043, 13.750, "high", "Gläserne Manufaktur"),
    ("DE_VW_EMDEN",             "DE94", "DE94", 53.367,  7.206, "high", ""),
    ("DE_VW_KASSEL",            "DE73", "DE73", 51.253,  9.415, "high", "Baunatal"),
    ("DE_VW_OSNABRUECK",        "DE94", "DE94", 52.279,  8.047, "high", ""),
    ("DE_VW_SALZGITTER",        "DE91", "DE91", 52.154, 10.333, "high", "engines/battery"),
    ("DE_VW_WOLFSBURG",         "DE91", "DE91", 52.422, 10.786, "high", ""),
    ("DE_VW_ZWICKAU",           "DED4", "DED4", 50.718, 12.496, "high", "legacy code DED1"),

    ("ES_FORD_VALENCIA",        "ES52", "ES52", 39.280, -0.419, "high", "Almussafes"),
    ("ES_RENAULT_PALENCIA",     "ES41", "ES41", 42.010, -4.530, "high", ""),
    ("ES_RENAULT_SEVILLA",      "ES61", "ES61", 37.389, -5.985, "high", "gearboxes"),
    ("ES_RENAULT_VALLADOLID",   "ES41", "ES41", 41.652, -4.724, "high", ""),
    ("ES_STELLANTIS_MADRID",    "ES30", "ES30", 40.339, -3.703, "high", "Villaverde"),
    ("ES_STELLANTIS_VIGO",      "ES11", "ES11", 42.238, -8.724, "high", ""),
    ("ES_STELLANTIS_ZARAGOZA",  "ES24", "ES24", 41.793, -1.132, "high", "Figueruelas"),
    ("ES_VW_NAVARRA",           "ES22", "ES22", 42.812, -1.644, "high", "Pamplona/Landaben"),

    # France: NUTS codes given for completeness ONLY. RULE 1 -> France analysis
    # uses URSSAF at commune/departement level, never these.
    ("FR_INEOS_HAMBACH",        "FRF3", "FR41", 49.054,  7.053, "medium", "URSSAF used"),
    ("FR_RENAULT_BATILLY",      "FRF3", "FR41", 49.193,  5.940, "medium", "URSSAF used"),
    ("FR_RENAULT_CLEON",        "FRD2", "FR23", 49.313,  1.020, "medium", "URSSAF used"),
    ("FR_RENAULT_DIEPPE",       "FRD2", "FR23", 49.922,  1.078, "medium", "URSSAF used"),
    ("FR_RENAULT_DOUAI",        "FRE1", "FR30", 50.370,  3.080, "medium", "URSSAF used"),
    ("FR_RENAULT_FLINS",        "FR10", "FR10", 48.962,  1.869, "medium", "URSSAF used"),
    ("FR_RENAULT_GUYANCOURT",   "FR10", "FR10", 48.770,  2.073, "medium", "R&D; URSSAF used"),
    ("FR_RENAULT_LE_MANS",      "FRG0", "FR51", 48.004,  0.200, "medium", "URSSAF used"),
    ("FR_RENAULT_MAUBEUGE",     "FRE1", "FR30", 50.280,  3.970, "medium", "URSSAF used"),
    ("FR_RENAULT_MEYZIEU",      "FRK2", "FR71", 45.770,  5.000, "medium", "URSSAF used"),
    ("FR_RENAULT_RUITZ",        "FRE1", "FR30", 50.483,  2.573, "medium", "URSSAF used"),
    ("FR_RENAULT_SANDOUVILLE",  "FRD2", "FR23", 49.480,  0.280, "medium", "URSSAF used"),
    ("FR_STELLANTIS_CHARTRES_DE_BRETAGNE_RENNES", "FRH0", "FR52", 48.038, -1.700, "medium", "URSSAF used"),
    ("FR_STELLANTIS_CORMELLES_LE_ROYAL_CAEN",     "FRD1", "FR25", 49.153, -0.340, "medium", "URSSAF used"),
    ("FR_STELLANTIS_LIEU_SAINT_AMAND_HORDAIN",    "FRE1", "FR30", 50.300,  3.350, "medium", "URSSAF used"),
    ("FR_STELLANTIS_METZ",      "FRF3", "FR41", 49.120,  6.180, "medium", "URSSAF used"),
    ("FR_STELLANTIS_SAUSHEIM_MULHOUSE", "FRF1", "FR42", 47.790,  7.370, "medium", "URSSAF used"),
    ("FR_STELLANTIS_SOCHAUX",   "FRC2", "FR43", 47.510,  6.830, "medium", "URSSAF used"),
    ("FR_STELLANTIS_TREMERY",   "FRF3", "FR41", 49.240,  6.200, "medium", "engines; URSSAF used"),
    ("FR_STELLANTIS_TRITH_SAINT_LEGER_VALENCIENNES", "FRE1", "FR30", 50.330, 3.490, "medium", "URSSAF used"),
    ("FR_STELLANTIS_VESOUL",    "FRC2", "FR43", 47.620,  6.160, "medium", "logistics; URSSAF used"),
    ("FR_STELLANTIS_VILLERS_SEMEUSE_CHARLEVILLE_MEZIERES", "FRF2", "FR21", 49.750, 4.740, "medium", "URSSAF used"),
    ("FR_TOYOTA_ONNAING_VALENCIENNES", "FRE1", "FR30", 50.380, 3.600, "medium", "URSSAF used"),

    ("GB_FORD_BRIDGEND",        None,   "UKL1", 51.507, -3.577, "high", "UK absent from 2021+ extract"),
    ("GB_FORD_DAGENHAM",        None,   "UKI5", 51.539,  0.150, "high", "UK absent from 2021+ extract"),
    ("GB_TOYOTA_BURNASTON",     None,   "UKF1", 52.861, -1.531, "high", "UK absent from 2021+ extract"),
    ("GB_TOYOTA_DEESIDE",       None,   "UKL2", 53.218, -3.049, "high", "UK absent from 2021+ extract"),

    ("HU_AUDI_GYOR",            "HU22", "HU22", 47.689, 17.635, "high", ""),
    ("HU_MERCEDES_KECSKEMET",   "HU33", "HU33", 46.907, 19.692, "high", ""),

    ("IT_STELLANTIS_ATESSA",    "ITF1", "ITF1", 42.066, 14.445, "high", "Sevel Val di Sangro"),
    ("IT_STELLANTIS_CASSINO",   "ITI4", "ITI4", 41.492, 13.830, "high", ""),
    ("IT_STELLANTIS_MELFI",     "ITF5", "ITF5", 40.993, 15.653, "high", ""),
    ("IT_STELLANTIS_MIRAFIORI", "ITC1", "ITC1", 45.028,  7.622, "high", "Turin"),
    ("IT_STELLANTIS_POMIGLIANO","ITF3", "ITF3", 40.911, 14.391, "high", ""),

    ("PL_TOYOTA_WALBRZYCH",     "PL51", "PL51", 50.771, 16.285, "high", ""),
    ("PL_VW_POZNAN",            "PL41", "PL41", 52.407, 16.925, "high", ""),
    ("PL_VW_WRZESNIA",          "PL41", "PL41", 52.326, 17.566, "high", ""),

    ("PT_VW_PALMELA",           "PT1B", "PT17", 38.570, -8.900, "high", "Autoeuropa; PT recoded PT17->PT1B"),
    ("RO_FORD_CRAIOVA",         "RO41", "RO41", 44.319, 23.800, "high", ""),

    ("RU_FORD_VSEVOLOZHSK",     None,   None,   60.017, 30.652, "high", "outside Eurostat"),
    ("RU_TOYOTA_RUSSIA",        None,   None,   59.931, 30.341, "high", "outside Eurostat"),

    ("SK_VW_BRATISLAVA",        "SK01", "SK01", 48.148, 17.108, "high", ""),

    ("TR_FORD_OTOSAN_KOCAELI",  None,   None,   40.766, 29.941, "high", "TR absent from both extracts"),
    ("TR_TOYOTA_SAKARYA",       None,   None,   40.783, 30.403, "high", "TR absent from both extracts"),
]


def plant_nuts_crosswalk(plants: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Return the plant -> NUTS-2 crosswalk, joined to the plant registry."""
    cw = pd.DataFrame(_CROSSWALK_ROWS, columns=[
        "plant_uid", "nuts2_new", "nuts2_old", "lat", "lon", "geo_confidence", "geo_note"])
    if plants is None:
        plants = load_plants()
    out = plants[["plant_uid", "plant_name", "city", "country_iso2", "group",
                  "brand", "classification"]].merge(cw, on="plant_uid", how="left")
    out["has_eurostat_link"] = out["nuts2_new"].notna() | out["nuts2_old"].notna()
    out["uses_urssaf"] = out["country_iso2"].eq("FR") & (CONFIG.FRANCE_SOURCE == "urssaf")
    return out


def validate_crosswalk(cw: pd.DataFrame,
                       eurostat_old: pd.DataFrame,
                       eurostat_new: pd.DataFrame) -> pd.DataFrame:
    """
    Check every assigned NUTS code against the codes actually present in the
    Eurostat extracts. Returns one row per plant with explicit problem flags.
    Nothing is silently corrected.
    """
    codes_old = set(eurostat_old["geo_code"].unique())
    codes_new = set(eurostat_new["geo_code"].unique())

    rows = []
    for _, r in cw.iterrows():
        old_ok = (r["nuts2_old"] in codes_old) if pd.notna(r["nuts2_old"]) else None
        new_ok = (r["nuts2_new"] in codes_new) if pd.notna(r["nuts2_new"]) else None
        problems = []
        if pd.notna(r["nuts2_old"]) and old_ok is False:
            problems.append(f"nuts2_old {r['nuts2_old']} not in 2008-2020 extract")
        if pd.notna(r["nuts2_new"]) and new_ok is False:
            problems.append(f"nuts2_new {r['nuts2_new']} not in 2021-2023 extract")
        if not r["has_eurostat_link"] and not r["uses_urssaf"]:
            problems.append("no Eurostat link and not covered by URSSAF")
        rows.append({
            "plant_uid": r["plant_uid"], "country_iso2": r["country_iso2"],
            "city": r["city"], "nuts2_old": r["nuts2_old"], "nuts2_new": r["nuts2_new"],
            "old_code_found": old_ok, "new_code_found": new_ok,
            "geo_confidence": r["geo_confidence"],
            "n_problems": len(problems), "problems": "; ".join(problems),
        })
    return pd.DataFrame(rows)


# ==============================================================================
# 5. BUILD EVERYTHING
# ==============================================================================

def build_all(verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """Load and harmonise every source. Returns a dict of tidy DataFrames."""
    d: Dict[str, pd.DataFrame] = {}

    if verbose:
        print("Loading micro layer (plant panel) ...")
    d["plants"] = load_plants()
    d["employment"] = load_employment()
    try:
        d["employment_imputed"] = load_employment_imputed()
    except FileNotFoundError:
        warnings.warn(f"{CONFIG.f_employment_imputed} not found — "
                      "employment_panel() will fall back to observed values.")
    # the analysis-ready series every later module should use
    d["employment_panel"] = employment_panel(prefer_imputed=True)
    d["production"] = load_production()
    d["model_year"] = load_model_year()
    d["model_attributes"] = load_model_attributes()
    d["model_year_sized"] = attach_size_attributes(d["model_year"], d["model_attributes"])

    if verbose:
        print("Loading macro layer A (Eurostat SBS, NUTS-2) ...")
    d["eurostat_c29_emp"] = eurostat_panel("C29", "employment")
    d["eurostat_c29_units"] = eurostat_panel("C29", "units")
    d["eurostat_c27_emp"] = eurostat_panel("C27", "employment")
    d["eurostat_c29_emp_nuts0"] = eurostat_panel("C29", "employment", nuts_level=0)
    # complete European panel: Eurostat + URSSAF France — use this for maps
    d["regional_c29_nuts2"] = regional_panel("C29", "employment", level=2)
    d["regional_c29_nuts1"] = regional_panel("C29", "employment", level=1)

    if verbose:
        print("Loading macro layer B (URSSAF, France) ...")
    d["urssaf_c29_dep"] = urssaf_panel("C29", "all", "departement")
    d["urssaf_c29_commune"] = urssaf_panel("C29", "all", "commune")
    d["urssaf_c27_dep_all"] = urssaf_panel("C27", "all", "departement")
    d["urssaf_c27_dep_narrow"] = urssaf_panel("C27", "narrow", "departement")
    d["urssaf_c27_dep_battery"] = urssaf_panel("C27", "battery", "departement")
    d["urssaf_c27_nat_all"] = urssaf_panel("C27", "all", "national")
    d["urssaf_c27_nat_narrow"] = urssaf_panel("C27", "narrow", "national")
    d["urssaf_c27_nat_battery"] = urssaf_panel("C27", "battery", "national")

    if verbose:
        print("Building plant -> NUTS-2 crosswalk ...")
    d["crosswalk"] = plant_nuts_crosswalk(d["plants"])

    # validation needs the raw code universes
    old_codes = _parse_eurostat_sheet(CONFIG.path(CONFIG.f_eurostat_c29_old), "employment")
    new_codes = _parse_eurostat_sheet(CONFIG.path(CONFIG.f_eurostat_c29_new), "employment")
    d["crosswalk_validation"] = validate_crosswalk(d["crosswalk"], old_codes, new_codes)

    if verbose:
        print("Done.")
    return d


def print_validation_report(d: Dict[str, pd.DataFrame]) -> None:
    """Human-readable QA report — run this before trusting anything downstream."""
    line = "=" * 78
    print(line); print("FOUNDATION VALIDATION REPORT"); print(line)

    pl, emp, cw, val = d["plants"], d["employment"], d["crosswalk"], d["crosswalk_validation"]

    print(f"\n[1] PLANT PANEL")
    print(f"    plants ................ {pl['plant_uid'].nunique()}")
    print(f"    countries ............. {pl['country_iso2'].nunique()}  "
          f"({', '.join(sorted(pl['country_iso2'].unique()))})")
    print(f"    groups ................ {pl['group'].nunique()}")
    print(f"    classifications ....... {dict(pl['classification'].value_counts())}")

    print(f"\n[2] EMPLOYMENT (micro)")
    print(f"    plant-year rows ....... {len(emp)}")
    print(f"    plants with data ...... {emp['plant_uid'].nunique()}")
    print(f"    measurement bases ..... {dict(emp['measurement_basis'].value_counts())}")
    print(f"    status ................ {dict(emp['employees_status'].value_counts())}")
    print("    NOTE: plant_reported and commune_all_APE (FR) are different")
    print("          quantities and must not be pooled without a caveat.")

    if "employment_imputed" in d:
        imp, ep = d["employment_imputed"], d["employment_panel"]
        n_pl = imp["plant_uid"].nunique()
        print(f"\n[2b] IMPUTED EMPLOYMENT PANEL (Amelia, M=50)")
        print(f"    plants x years ........ {n_pl} x "
              f"{imp['year'].nunique()} = {len(imp)} cells (balanced grid)")
        print(f"    status ................ {dict(imp['status'].value_counts())}")
        obs = int((imp['status'] == 'observed').sum())
        act = len(imp) - int(imp['is_inactive'].sum())
        print(f"    observed / active ..... {obs}/{act} ({100*obs/act:.0f}%)  "
              f"-> {100*(1-obs/act):.0f}% imputed")
        for st in ("imputed_internal", "imputed_edge"):
            s = imp.loc[imp['status'] == st, 'est_sd_log']
            if len(s):
                print(f"    sd_log {st:17} mean {s.mean():.3f}  "
                      f"median {s.median():.3f}  max {s.max():.3f}")
        print("    'inactive' cells carry NaN (pre-existence / post-closure) —")
        print("    they are NOT zeros and must stay out of aggregates.")
        excluded = sorted(set(d['plants'].loc[d['plants'].get('source_block', '') == 'DE',
                                              'plant_uid']) - set(imp['plant_uid'])) \
            if 'source_block' in d['plants'] else []
        if excluded:
            print(f"    excluded (too data-poor): {', '.join(excluded)}")
        print(f"\n[2c] ANALYSIS-READY SERIES  (employment_panel)")
        print(f"    rows .................. {len(ep)}")
        print(f"    value_kind ............ {dict(ep['value_kind'].value_counts())}")
        print(f"    in imputed universe ... {ep['in_imputed_universe'].sum()} rows / "
              f"{ep.loc[ep['in_imputed_universe'],'plant_uid'].nunique()} plants (balanced)")
        print("    WARNING: plants outside that universe keep patchy observed data.")
        print("             Restrict to in_imputed_universe for trends/aggregates,")
        print("             or report coverage alongside levels.")

    if "is_suspect" in d["employment_panel"].columns:
        ep = d["employment_panel"]
        print(f"\n[2d] EMPLOYMENT QUALITY FLAGS")
        print(f"    post-closure rows ..... {int(ep['is_post_closure'].sum())}")
        print(f"    frozen-series rows .... {int(ep['is_frozen_series'].sum())}")
        print(f"    any suspect ........... {int(ep['is_suspect'].sum())} / {len(ep)}")
        sus = (ep[ep["is_suspect"]].groupby(["plant_uid", "plant_name"])
                 .agg(rows=("year", "size"), first=("year", "min"),
                      last=("year", "max"), value=("employees", "first")))
        if len(sus):
            print("    affected plants:")
            for (pid, name), r in sus.iterrows():
                print(f"       {name[:38]:38} {int(r['value']):>7,}  "
                      f"{int(r['first'])}-{int(r['last'])}  ({int(r['rows'])} yrs)")
        print("    -> values are FLAGGED, not removed. Pass drop_post_closure=True")
        print("       / drop_frozen=True to employment_panel() to exclude them.")

    print(f"\n[3] EUROSTAT (macro)")
    for k in ("eurostat_c29_emp", "eurostat_c27_emp"):
        e = d[k]
        n_sup = int(e["is_confidential"].sum()); n_mis = int(e["is_missing"].sum())
        print(f"    {k}: {len(e)} region-years, {e['geo_code'].nunique()} NUTS-2 regions, "
              f"{e['year'].min()}-{e['year'].max()}")
        print(f"       suppressed(confidential) {n_sup} ({100*n_sup/len(e):.1f}%) · "
              f"missing {n_mis} ({100*n_mis/len(e):.1f}%)")
        if e.attrs.get("france_excluded"):
            print("       France EXCLUDED (RULE 1: URSSAF is used for France)")

    print(f"\n[4] URSSAF (France macro) — the two-level C27 read")
    a = d["urssaf_c27_nat_all"].set_index("year")["employees"]
    n = d["urssaf_c27_nat_narrow"].set_index("year")["employees"]
    b = d["urssaf_c27_nat_battery"].set_index("year")["employees"]
    for y in (2006, 2015, 2023, 2025):
        if y in a.index:
            print(f"    {y}:  C27 all {a[y]:>9,.0f}   2711+2720 {n[y]:>8,.0f}   "
                  f"batteries {b[y]:>7,.0f}")
    if 2006 in a.index and 2023 in a.index:
        print(f"    2006->2023:  C27 all {100*(a[2023]-a[2006])/a[2006]:+.0f}%   "
              f"batteries {100*(b[2023]-b[2006])/b[2006]:+.0f}%")
        print("    -> the aggregate masks (here inverts) the electrification signal.")

    print(f"\n[5] CROSSWALK plant -> NUTS-2")
    print(f"    plants with Eurostat link .... {int(cw['has_eurostat_link'].sum())}/{len(cw)}")
    print(f"    France (URSSAF instead) ...... {int(cw['uses_urssaf'].sum())}")
    print(f"    confidence ................... {dict(cw['geo_confidence'].value_counts(dropna=False))}")
    bad = val[val["n_problems"] > 0]
    if len(bad):
        print(f"    !! {len(bad)} plant(s) with code problems:")
        for _, r in bad.iterrows():
            print(f"       - {r['plant_uid']:42} {r['problems']}")
    else:
        print("    all assigned NUTS codes found in the corresponding extract ✓")
    print("\n    ACTION: review crosswalk.csv before publication (assignments are")
    print("            municipality-based and should be verified independently).")
    print(line)


# ==============================================================================
# 6. DIAGNOSTIC FIGURES (assumption-checking, for the methods chapter)
# ==============================================================================

def plot_coverage_diagnostics(d: Dict[str, pd.DataFrame], save: Optional[str] = None):
    """
    Four-panel technical figure documenting what the data can and cannot support:
      (a) micro employment coverage by country x year
      (b) Eurostat suppression/missingness by year (the confidentiality problem)
      (c) size-attribute (ADEME) coverage of the model mix by country
      (d) France C27: aggregate vs narrow vs battery (the RULE 3 demonstration)
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Foundation diagnostics — what the data supports",
                 fontsize=14, fontweight="bold", x=0.02, ha="left")

    # (a) micro coverage
    emp = d["employment"]
    piv = (emp[emp["employees"].notna()]
           .pivot_table(index="country_iso2", columns="year",
                        values="plant_uid", aggfunc="nunique"))
    npl = d["plants"].groupby("country_iso2")["plant_uid"].nunique()
    share = piv.div(npl, axis=0) * 100
    ax = axes[0, 0]
    im = ax.imshow(share.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_yticks(range(len(share.index)))
    ax.set_yticklabels([f"{c} (n={npl[c]})" for c in share.index], fontsize=8)
    ax.set_xticks(range(len(share.columns)))
    ax.set_xticklabels(share.columns, rotation=90, fontsize=7)
    ax.set_title("(a) Micro: % of plants with employment, by country", loc="left", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.03).set_label("%", fontsize=8)

    # (b) Eurostat suppression
    ax = axes[0, 1]
    for key, lbl in (("eurostat_c29_emp", "C29"), ("eurostat_c27_emp", "C27")):
        e = d[key]
        g = e.groupby("year").agg(sup=("is_confidential", "mean"),
                                  mis=("is_missing", "mean")) * 100
        ax.plot(g.index, g["mis"], marker="o", ms=3, label=f"{lbl} missing")
        ax.plot(g.index, g["sup"], marker="s", ms=3, ls="--", label=f"{lbl} confidential")
    ax.set_title("(b) Eurostat NUTS-2: missing & suppressed cells", loc="left", fontsize=11)
    ax.set_ylabel("% of region-years"); ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=.3)

    # (c) ADEME size-attribute coverage
    ax = axes[1, 0]
    mys = d["model_year_sized"]
    cov = (mys.groupby("country_iso2")["has_size_attr"].mean() * 100).sort_values()
    ax.barh(cov.index, cov.values, color="#2E8B6F")
    ax.set_title("(c) Size attributes (ADEME): % of plant-model-years covered",
                 loc="left", fontsize=11)
    ax.set_xlabel("% covered"); ax.grid(axis="x", alpha=.3)
    ax.axvline(mys["has_size_attr"].mean() * 100, color="#C0563F", ls="--", lw=1,
               label=f"overall {mys['has_size_attr'].mean()*100:.0f}%")
    ax.legend(fontsize=8, frameon=False)

    # (d) France C27 two-level read
    ax = axes[1, 1]
    for key, lbl, col in (("urssaf_c27_nat_all", "C27 aggregate (Eurostat-comparable)", "#6E6E6E"),
                          ("urssaf_c27_nat_narrow", "2711Z+2720Z (motors+batteries)", "#2C6FB3"),
                          ("urssaf_c27_nat_battery", "2720Z batteries only", "#C6803D")):
        s = d[key].set_index("year")["employees"]
        ax.plot(s.index, s.values / 1000, marker="o", ms=3, label=lbl, color=col, lw=2)
    ax.set_title("(d) France C27: the aggregate hides the electrification signal",
                 loc="left", fontsize=11)
    ax.set_ylabel("employees (thousands)"); ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def plot_imputation_diagnostics(d: Dict[str, pd.DataFrame], save: Optional[str] = None):
    """
    Four-panel figure documenting the imputation — for the methods chapter.

      (a) status of every cell in the balanced grid, plant x year
      (b) imputation uncertainty (sd_log) by status: internal vs edge
      (c) share of the panel observed vs imputed, by year
      (d) example plant trajectories with 95% bands, so the reader sees where
          the series is evidence and where it is inference
    """
    import matplotlib.pyplot as plt

    if "employment_imputed" not in d:
        raise RuntimeError("No imputed panel loaded.")
    imp = d["employment_imputed"].copy()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Imputation diagnostics — employment panel (within-plant multiple imputation, M = 50)",
                 fontsize=14, fontweight="bold", x=0.02, ha="left")

    # (a) status grid
    ax = axes[0, 0]
    codes = {"observed": 3, "imputed_internal": 2, "imputed_edge": 1, "inactive": 0}
    grid = (imp.assign(c=imp["status"].map(codes))
              .pivot_table(index="plant_uid", columns="year", values="c", aggfunc="first"))
    order = imp[imp["status"] == "observed"].groupby("plant_uid").size().reindex(
        grid.index).fillna(0).sort_values(ascending=False).index
    grid = grid.reindex(order)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#F0F0F0", "#E9A23B", "#7FC3B0", "#14746F"])
    ax.imshow(grid.values, aspect="auto", cmap=cmap, vmin=0, vmax=3)
    ax.set_xticks(range(len(grid.columns)))
    ax.set_xticklabels(grid.columns, rotation=90, fontsize=7)
    ax.set_yticks([]); ax.set_ylabel(f"{len(grid)} plants (most observed first)")
    ax.set_title("(a) Cell status across the balanced grid", loc="left", fontsize=11)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#14746F", label="observed"),
                       Patch(color="#7FC3B0", label="imputed (internal)"),
                       Patch(color="#E9A23B", label="imputed (edge)"),
                       Patch(color="#F0F0F0", label="inactive (not imputed)")],
              fontsize=7.5, frameon=False, loc="upper center", ncol=4,
              bbox_to_anchor=(0.5, -0.09))

    # (b) uncertainty by status
    ax = axes[0, 1]
    data = [imp.loc[imp["status"] == s, "est_sd_log"].dropna()
            for s in ("imputed_internal", "imputed_edge")]
    bp = ax.boxplot(data, labels=["internal", "edge"], patch_artist=True, widths=.55)
    for patch, col in zip(bp["boxes"], ["#7FC3B0", "#E9A23B"]):
        patch.set_facecolor(col)
    ax.set_title("(b) Imputation uncertainty by type", loc="left", fontsize=11)
    ax.set_ylabel("SD of the draws, log scale")
    ax.grid(axis="y", alpha=.3)
    ax.text(0.02, 0.96, "edge cells are less certain:\nnothing to interpolate between",
            transform=ax.transAxes, ha="left", va="top", fontsize=8, color="#6E6E6E")

    # (c) observed share by year
    ax = axes[1, 0]
    act = imp[~imp["is_inactive"]]
    g = act.groupby("year").agg(obs=("status", lambda s: (s == "observed").mean() * 100),
                                n=("status", "size"))
    ax.bar(g.index, g["obs"], color="#14746F", width=.7)
    ax.set_title("(c) Share of active cells that are observed", loc="left", fontsize=11)
    ax.set_ylabel("% observed"); ax.set_ylim(0, 100); ax.grid(axis="y", alpha=.3)
    ax.axhline(50, color="#C0563F", ls="--", lw=1)
    ax.text(g.index.min(), 52, "50%", fontsize=8, color="#C0563F")
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # (d) example trajectories with bands
    ax = axes[1, 1]
    n_imp = imp[imp["is_imputed"]].groupby("plant_uid").size().sort_values(ascending=False)
    picks = list(n_imp.head(3).index) + list(n_imp.tail(1).index)
    cols = ["#14746F", "#C6803D", "#2C6FB3", "#7A5C99"]
    for pid, col in zip(picks, cols):
        g = imp[(imp["plant_uid"] == pid) & ~imp["is_inactive"]].sort_values("year")
        ax.plot(g["year"], g["employees_est"] / 1000, color=col, lw=1.8,
                label=f"{pid} ({int((g['status']!='observed').sum())} imp.)")
        ax.fill_between(g["year"], g["employees_lo95"] / 1000, g["employees_hi95"] / 1000,
                        color=col, alpha=.18, lw=0)
        o = g[g["status"] == "observed"]
        ax.scatter(o["year"], o["employees_est"] / 1000, color=col, s=16, zorder=3)
    ax.set_title("(d) Trajectories with 95% bands (dots = observed)", loc="left", fontsize=11)
    ax.set_ylabel("employees (thousands)")
    ax.legend(fontsize=7.5, frameon=False); ax.grid(alpha=.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def plot_nuts_seam_diagnostic(d: Dict[str, pd.DataFrame], save: Optional[str] = None):
    """
    Technical figure documenting the 2020/2021 vintage seam: for the regions
    hosting our plants, plot the spliced series and mark the seam, so any
    discontinuity introduced by the recoding is visible rather than assumed away.
    """
    import matplotlib.pyplot as plt

    e = d["eurostat_c29_emp"]
    cw = d["crosswalk"]
    regions = sorted(set(cw["nuts2_new"].dropna()) | set(cw["nuts2_old"].dropna()))
    sub = e[e["geo_code"].isin(regions) & e["value"].notna()]
    top = (sub.groupby("geo_code")["value"].mean().sort_values(ascending=False).head(12).index)

    fig, ax = plt.subplots(figsize=(11, 6))
    for code in top:
        g = sub[sub["geo_code"] == code].sort_values("year")
        ax.plot(g["year"], g["value"] / 1000, marker="o", ms=3, lw=1.4, label=code)
    ax.axvline(2020.5, color="#C0563F", ls="--", lw=1.5)
    ax.text(2020.6, ax.get_ylim()[1] * 0.95, "NUTS vintage seam\n(2008-2020 | 2021-2023)",
            fontsize=8, color="#C0563F", va="top")
    ax.set_title("NUTS vintage seam check — C29 employment, plant-hosting regions",
                 loc="left", fontsize=12, fontweight="bold")
    ax.set_ylabel("persons employed (thousands)")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.grid(alpha=.3)
    fig.tight_layout()
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


# ==============================================================================
# 7. MAPS
# ==============================================================================

GISCO_NUTS_GEOJSON = (
    "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/"
    "NUTS_RG_20M_2021_4326_LEVL_{level}.geojson"
)


def load_nuts_geometry(level: int = 2, cache: Optional[str] = None) -> Optional[dict]:
    """
    Fetch (and cache) Eurostat GISCO NUTS geometries for choropleths.

    `level` MUST match the level of the panel being mapped, because the
    geometry files are one-level-per-file and `NUTS_ID` only contains codes of
    that level. Mapping a NUTS-1 panel (DE1, FRF, CZ0) onto NUTS-2 geometry
    silently produces an empty map: the basemap draws, no region is coloured.

    Returns None if offline — the caller should degrade gracefully.
    """
    cache = cache or f"nuts{level}_2021.geojson"
    # Look in the shipped geometry folder before anything else: these files are
    # archived with the project so the analysis runs offline. out_dir is only a
    # download cache, and on a fresh checkout it is empty.
    for cand in (Path(CONFIG.data_dir) / "geometry" / cache,
                 Path(CONFIG.data_dir) / cache,
                 Path(CONFIG.out_dir) / cache):
        if cand.exists():
            return json.loads(cand.read_text())
    p = Path(CONFIG.out_dir) / cache
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        import requests
        r = requests.get(GISCO_NUTS_GEOJSON.format(level=level), timeout=60)
        r.raise_for_status()
        p.write_text(r.text)
        return r.json()
    except Exception as exc:                                    # pragma: no cover
        warnings.warn(f"Could not fetch NUTS-{level} geometry ({exc}). "
                      f"Download manually to {p}.")
        return None


def _norm_dep_name(s: str) -> str:
    """Fold a département name to a comparable key: no accents, no separators."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


# Départements whose URSSAF spelling differs from the GISCO NAME_LATN beyond
# what accent/separator folding fixes. Kept explicit so the difference is
# auditable rather than silently absorbed by fuzzy matching.
_DEP_NAME_ALIASES = {
    "cotesdarmor": "cotesdarmor", "valdoise": "valdoise",
    "lareunion": "reunion", "reunion": "lareunion",
    "territoiredebelfort": "territoiredebelfort",
}


def french_departement_to_nuts3(geo3: Optional[dict] = None) -> Dict[str, str]:
    """
    Map URSSAF département codes ("59", "2A", "971") to NUTS-3 codes ("FRE11").

    THIS CANNOT BE DONE BY STRING SURGERY ON THE CODE. French NUTS-3 codes are
    sequential within their NUTS-2 parent — FR101 is Paris (75), FR102 is
    Seine-et-Marne (77) — so the digits carry no relation to the département
    number. Taking the last two characters of the NUTS code silently matches
    almost nothing, and the resulting map draws a blank France.

    The join is therefore made on the département NAME, which both sources
    carry, after folding away accents and separators. Unmatched départements
    are reported rather than dropped quietly.
    """
    geo3 = geo3 if geo3 is not None else load_nuts_geometry(level=3)
    if geo3 is None:
        return {}

    by_name: Dict[str, str] = {}
    for f in geo3.get("features", []):
        p = f.get("properties", {})
        nid = str(p.get("NUTS_ID", ""))
        if not nid.startswith("FR") or len(nid) != 5:
            continue
        for key in ("NAME_LATN", "NUTS_NAME", "NAME_FREN"):
            if p.get(key):
                by_name.setdefault(_norm_dep_name(p[key]), nid)

    urssaf = pd.read_excel(CONFIG.path(CONFIG.f_urssaf_c29))
    deps = (urssaf[["Code département", "Département"]].drop_duplicates()
            .dropna())
    out, missing = {}, []
    for code, name in zip(deps["Code département"], deps["Département"]):
        code = str(code).strip().replace(".0", "").upper()
        code = code.zfill(2) if code.isdigit() and len(code) < 2 else code
        k = _norm_dep_name(name)
        nid = by_name.get(k) or by_name.get(_DEP_NAME_ALIASES.get(k, ""))
        if nid:
            out[code] = nid
        else:
            missing.append(f"{code} {name}")
    if missing:
        warnings.warn(f"{len(missing)} département(s) could not be matched to "
                      f"NUTS-3 geometry and will not be drawn: {missing[:8]}")
    return out


def infer_nuts_level(codes: Iterable[str]) -> int:
    """Infer the NUTS level from code lengths (2=country, 3=NUTS-1, 4=NUTS-2)."""
    lens = pd.Series([len(str(c)) for c in codes]).value_counts()
    return int({2: 0, 3: 1, 4: 2}.get(int(lens.idxmax()), 2))


def _warn_unmatched(codes, geo: dict, level: int) -> None:
    """Loudly report codes with no geometry — the usual cause of a blank map."""
    ids = {f["properties"]["NUTS_ID"] for f in geo.get("features", [])}
    missing = sorted(set(map(str, codes)) - ids)
    if missing:
        warnings.warn(
            f"{len(missing)} region code(s) have no NUTS-{level} geometry and will "
            f"not be drawn, e.g. {missing[:6]}. Check that `level` matches the panel."
        )


def map_nuts_choropleth(panel: pd.DataFrame, year: int,
                        title: str = "", save: Optional[str] = None,
                        level: Optional[int] = None,
                        french_nuts3: bool = True,
                        country_borders: bool = True):
    """
    Interactive Plotly choropleth of a NUTS indicator for one year.
    Suppressed/missing regions are left blank (not zero) by construction.

    `level` is inferred from the panel's code lengths when not given; it must
    match the geometry, or the map renders empty.
    """
    import plotly.express as px

    df = panel[(panel["year"] == year) & panel["value"].notna()].copy()
    if "source" in df.columns and df["source"].nunique() > 1:
        df["source_note"] = df["source"]
    if level is None:
        level = infer_nuts_level(df["geo_code"])
    geo = load_nuts_geometry(level=level)
    if geo is None:
        raise RuntimeError("NUTS geometry unavailable; see load_nuts_geometry().")

    # FRANCE AT DEPARTEMENT LEVEL.
    # French employment is measured from URSSAF, which reports at commune level
    # and therefore supports a finer geography than the NUTS-2 used elsewhere.
    # The French NUTS-2 rows are swapped for departement rows and the two
    # geometries are merged into one collection, so a single choropleth call
    # draws the map at mixed resolution.
    if french_nuts3 and level == 2:
        geo3 = load_nuts_geometry(level=3)
        if geo3 is not None:
            dep = urssaf_panel("C29", "all", "departement")
            dep = dep[dep["year"] == year]
            dep_vals = dict(zip(dep["departement_code"].astype(str).str.strip()
                                .str.replace(r"\.0$", "", regex=True).str.zfill(2),
                                dep["employees"]))
            dep2nuts = french_departement_to_nuts3(geo3)
            names = {str(f["properties"]["NUTS_ID"]): f["properties"].get("NAME_LATN", "")
                     for f in geo3["features"]}
            rows = [{"geo_code": nid, "geo_label": names.get(nid, nid),
                     "year": year, "value": dep_vals[dc],
                     **({"source": "urssaf"} if "source" in df.columns else {})}
                    for dc, nid in dep2nuts.items() if dc in dep_vals]
            if rows:
                df = pd.concat([df[~df["geo_code"].astype(str).str.startswith("FR")],
                                pd.DataFrame(rows)], ignore_index=True)
                keep2 = [f for f in geo["features"]
                         if not str(f["properties"]["NUTS_ID"]).startswith("FR")]
                keep3 = [f for f in geo3["features"]
                         if str(f["properties"]["NUTS_ID"]).startswith("FR")]
                geo = {"type": "FeatureCollection", "features": keep2 + keep3}

    _warn_unmatched(df["geo_code"], geo, level)

    common = dict(
        geojson=geo, locations="geo_code", featureidkey="properties.NUTS_ID",
        color="value", color_continuous_scale="YlGnBu",
        hover_name="geo_label",
        hover_data={"value": ":,.0f", "geo_code": True, "year": True,
                    **({"source": True} if "source" in df.columns else {})},
        zoom=3.1, center={"lat": 50.0, "lon": 10.0}, opacity=0.78,
        labels={"value": "persons employed"},
        title=title or f"NUTS-2 employment, {year}",
    )
    # Plotly renamed the tile-map API: choropleth_mapbox (<6) -> choropleth_map (>=6).
    # Support both so this runs on whatever version is installed.
    if hasattr(px, "choropleth_map"):
        fig = px.choropleth_map(df, map_style="carto-positron", **common)
    else:                                                    # pragma: no cover
        fig = px.choropleth_mapbox(df, mapbox_style="carto-positron", **common)

    # National borders, heavier than the regional outlines, so that a country
    # boundary can be told from an internal one.
    if country_borders:
        geo0 = load_nuts_geometry(level=0)
        if geo0 is not None:
            lons, lats = [], []
            for f in geo0["features"]:
                g = f.get("geometry") or {}
                polys = ([g.get("coordinates")] if g.get("type") == "Polygon"
                         else g.get("coordinates") or [])
                for poly in polys:
                    for ring in poly:
                        for x, y in ring:
                            lons.append(x); lats.append(y)
                        lons.append(None); lats.append(None)
            if hasattr(px, "choropleth_map"):
                import plotly.graph_objects as go
                fig.add_trace(go.Scattermap(
                    lon=lons, lat=lats, mode="lines", hoverinfo="skip",
                    showlegend=False, line=dict(width=1.6, color="#1a1a1a")))
            else:                                            # pragma: no cover
                import plotly.graph_objects as go
                fig.add_trace(go.Scattermapbox(
                    lon=lons, lat=lats, mode="lines", hoverinfo="skip",
                    showlegend=False, line=dict(width=1.6, color="#1a1a1a")))

    fig.update_layout(margin=dict(l=0, r=0, t=45, b=0), height=720)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save)
    return fig


def map_plants_folium(crosswalk: pd.DataFrame,
                      employment: Optional[pd.DataFrame] = None,
                      year: Optional[int] = None,
                      save: Optional[str] = None):
    """
    Interactive Folium map of the plant universe — the visual check on the
    crosswalk. Marker size ~ employment (if supplied), colour = group,
    popup carries the NUTS assignment so errors are easy to spot.
    """
    import folium

    cw = crosswalk.dropna(subset=["lat", "lon"]).copy()
    if employment is not None and year is not None:
        e = employment[(employment["year"] == year)
                       & employment["employees"].notna()].copy()
        # `measurement_basis` lives on the RAW employment table; the assembled
        # panel exposes the same information as `is_plant_level` instead.
        # Accept either, so this works whichever is handed in.
        if "measurement_basis" not in e.columns:
            if "is_plant_level" in e.columns:
                e["measurement_basis"] = np.where(
                    e["is_plant_level"], "plant headcount",
                    "commune total (URSSAF)")
            else:
                e["measurement_basis"] = ""
        e = e[["plant_uid", "employees", "measurement_basis"]].drop_duplicates(
            "plant_uid", keep="first")
        cw = cw.merge(e, on="plant_uid", how="left")
    else:
        cw["employees"], cw["measurement_basis"] = np.nan, ""

    colours = {"Volkswagen Group": "#2E8B6F", "BMW Group": "#2C6FB3",
               "Mercedes-Benz": "#6A6A6A", "Renault": "#C6803D",
               "Stellantis": "#7A5C99", "Ford": "#30609F",
               "Toyota": "#B0483C", "Ineos": "#444444"}

    m = folium.Map(location=[50.0, 9.0], zoom_start=4, tiles="cartodbpositron")
    for _, r in cw.iterrows():
        emp_val = r.get("employees")
        radius = 5 if pd.isna(emp_val) else float(np.clip(np.sqrt(emp_val) / 8, 4, 26))
        nuts = r["nuts2_new"] if pd.notna(r["nuts2_new"]) else r["nuts2_old"]
        popup = folium.Popup(html=(
            f"<b>{r['plant_name']}</b><br>{r['city']}, {r['country_iso2']}<br>"
            f"group: {r['group']}<br>class: {r['classification']}<br>"
            f"NUTS-2: {nuts if pd.notna(nuts) else '— (outside Eurostat)'}<br>"
            f"employees: {'n/a' if pd.isna(emp_val) else f'{emp_val:,.0f}'}"
            f" <i>{r.get('measurement_basis','')}</i><br>"
            f"geo confidence: {r['geo_confidence']}"), max_width=320)
        folium.CircleMarker(
            location=[r["lat"], r["lon"]], radius=radius,
            color=colours.get(r["group"], "#333333"), fill=True,
            fill_color=colours.get(r["group"], "#333333"),
            fill_opacity=0.75, weight=1.4, popup=popup,
            tooltip=f"{r['plant_name']} ({r['country_iso2']})",
        ).add_to(m)

    legend = "".join(
        f"<div><span style='display:inline-block;width:10px;height:10px;"
        f"background:{c};border-radius:50%;margin-right:6px'></span>{g}</div>"
        for g, c in colours.items())
    # National borders, so the plant cloud can be read against countries.
    geo0 = load_nuts_geometry(level=0)
    if geo0 is not None:
        folium.GeoJson(geo0, name="country borders",
                       style_function=lambda x: {"fillOpacity": 0,
                                                 "color": "#1a1a1a",
                                                 "weight": 2.1, "opacity": .9}
                       ).add_to(m)

    m.get_root().html.add_child(folium.Element(
        f"<div style='position:fixed;bottom:22px;left:22px;z-index:9999;"
        f"background:white;padding:10px 12px;border:1px solid #ccc;"
        f"border-radius:4px;font:11px sans-serif'><b>Group</b>{legend}</div>"))

    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        m.save(save)
    return m


# ==============================================================================
# 8. EXPORT
# ==============================================================================

def export_all(d: Dict[str, pd.DataFrame], out_dir: Optional[str] = None) -> None:
    """Write every tidy table to CSV (and the crosswalk for manual review)."""
    out = Path(out_dir or CONFIG.out_dir); out.mkdir(parents=True, exist_ok=True)
    for name, df in d.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(out / f"{name}.csv", index=False)
    print(f"Wrote {len(d)} tables to {out}")


if __name__ == "__main__":
    CONFIG.data_dir = os.environ.get("DATA_DIR", ".")
    CONFIG.out_dir = os.environ.get("OUT_DIR", "./outputs")
    data = build_all()
    print_validation_report(data)
    export_all(data)
