# -*- coding: utf-8 -*-
"""
================================================================================
 MODULE 01 — SPATIAL STRUCTURAL CHANGE
 Where did European automotive employment go, 2008-2023?
================================================================================

THE ARGUMENT THIS MODULE SERVES
-------------------------------
Two literatures frame the chapter:

* **Pavlínek (2019)**, *Restructuring and internationalization of the European
  automotive industry* (J. Econ. Geography). Uses Harvey's uneven development
  and the *spatio-temporal fix* to explain a changing geography: firms pursue
  spatial profit-seeking, so the industry grows in **integrated peripheries**
  (regions adjacent to the core, absorbed into macro-regional production
  networks) while **restructuring in the traditional core**. His evidence is
  2,124 restructuring *events*, 2005-2016, plus interviews.

* **Lefèvre et al. (2022)**, *Global socio-economic and climate change
  mitigation scenarios through the lens of structural change*. Structural
  change = the evolving **composition** of output and employment, not the
  aggregate. Transitions should be read through composition, not totals.

**Our contribution here:** Pavlínek's claim is tested on *events*; we test it on
**stocks** — actual regional employment levels across the whole sector, with an
explicit decomposition. Shift-share is the natural instrument: it separates
"this region shrank because the whole industry shrank" from "this region shrank
relative to the industry", which is exactly the core-vs-periphery question.

WHAT IS COMPUTED
----------------
1. Bloc trajectories: employment and *shares* for core / southern (old) periphery
   / eastern (new) periphery  -> the headline reallocation.
2. Shift-share, two variants:
     - one-sector (C29 only): EU-growth component + differential (competitive) shift
     - two-sector (C29 + C27): adds an INDUSTRY-MIX component, which here carries
       real meaning -- C27 (electrical equipment) is the electrification-adjacent
       sector, so "mix" separates regions hurt by being in vehicles from regions
       underperforming within vehicles.
3. Concentration: HHI, top-k shares, number of regions holding 50% of employment.
4. Average establishment size (employment / local units) -- consolidation vs
   fragmentation, a structural indicator Eurostat supports directly.
5. Maps: level and change choropleths.

THREE METHODOLOGICAL COMMITMENTS
--------------------------------
1. **Shift-share requires a balanced set.** A region missing either endpoint is
   dropped, not zero-filled; `balance_report()` shows what each candidate window
   costs. Never run this on the raw panel.
2. **France follows RULE 1 (URSSAF, never Eurostat) and is INCLUDED.** It can
   be spliced at either level: NUTS-1 via URSSAF's 13 modern régions, NUTS-2 by
   aggregating départements into the 22 pre-2016 régions
   (fd.DEPARTEMENT_TO_NUTS2). NUTS-1 stays the headline (better balanced);
   NUTS-2 is the robustness check. Measurement bases differ (URSSAF *effectifs
   salariés* vs Eurostat *persons employed*) and every row carries `source`.
3. **Suppression is not random.** Eurostat withholds cells for confidentiality
   precisely in regions dominated by one or two large plants -- our regions of
   interest. Coverage is reported alongside every result.

USAGE
-----
    import foundation as fd, module01_spatial as m1
    fd.CONFIG.data_dir = './data'
    d  = fd.build_all()
    p  = m1.european_panel(level=1, france='urssaf')     # France included
    ss = m1.shift_share_2sector(p2_c29, p2_c27, 2013, 2019)

Author: Magd Shamashan
"""

from __future__ import annotations

import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import foundation as fd

__all__ = [
    "BLOCS", "BLOC_ORDER", "BLOC_COLOURS", "assign_bloc",
    "european_panel", "balance_report", "balanced_subset",
    "bloc_trajectories", "shift_share_1sector", "shift_share_2sector",
    "concentration", "unit_size",
    "fig_bloc_trajectories", "fig_shift_share", "fig_shift_share_explained",
    "fig_shift_share_2sector", "fig_concentration", "fig_hhi", "fig_concentration_detail",
    "whole_country_units", "panel_description", "fig_lorenz", "region_label",
    "NUTS1_NAMES", "map_battery_france", "map_employment_europe",
    "concentration_by_bloc", "lorenz_points",
    "fig_unit_size", "fig_france_urssaf", "map_change_choropleth",
]


# ==============================================================================
# 1. BLOC CLASSIFICATION  (Pavlínek-informed; change it if your reading differs)
# ==============================================================================
#
# Pavlínek distinguishes the traditional *core*, older *integrated peripheries*
# in southern/western Europe (Spain, Portugal), and the newer *East European
# integrated periphery* that absorbed much of the industry's growth after 2004.
# Assignment is at country level: simple, transparent, and revisable. A finer
# treatment (e.g. splitting northern from southern Italy) is possible at NUTS
# level but introduces judgement we would rather keep explicit.

BLOCS: Dict[str, str] = {
    # --- traditional core -----------------------------------------------------
    "DE": "core", "FR": "core", "IT": "core", "UK": "core", "GB": "core",
    "SE": "core", "BE": "core", "AT": "core", "NL": "core", "LU": "core",
    "DK": "core", "FI": "core", "IE": "core", "NO": "core", "CH": "core",
    "IS": "core",
    # --- older / southern integrated periphery -------------------------------
    "ES": "southern_periphery", "PT": "southern_periphery",
    "EL": "southern_periphery", "GR": "southern_periphery",
    # --- eastern integrated periphery (Pavlínek's central object) -------------
    "PL": "eastern_periphery", "CZ": "eastern_periphery", "SK": "eastern_periphery",
    "HU": "eastern_periphery", "RO": "eastern_periphery", "SI": "eastern_periphery",
    "BG": "eastern_periphery", "HR": "eastern_periphery", "EE": "eastern_periphery",
    "LV": "eastern_periphery", "LT": "eastern_periphery",
    # --- outside the EU frame / negligible ------------------------------------
    "TR": "other", "RS": "other", "MK": "other", "ME": "other", "AL": "other",
    "CY": "other", "MT": "other", "RU": "other",
}

BLOC_ORDER = ["core", "southern_periphery", "eastern_periphery", "other"]
BLOC_LABELS = {
    "core": "Core (DE, FR, IT, UK, SE, BE, AT…)",
    "southern_periphery": "Southern / older integrated periphery (ES, PT)",
    "eastern_periphery": "Eastern integrated periphery (PL, CZ, SK, HU, RO…)",
    "other": "Other / outside frame",
}
BLOC_COLOURS = {
    "core": "#14746F",
    "southern_periphery": "#C6803D",
    "eastern_periphery": "#2C6FB3",
    "other": "#AFAFAF",
}


def assign_bloc(iso2: str) -> str:
    return BLOCS.get(str(iso2).upper(), "other")


# ==============================================================================
# 1b. READABLE REGION NAMES
# ==============================================================================
#
# Eurostat labels NUTS-1 regions in the national language ("Česko", "Sachsen",
# "Makroregion południowo-zachodni") and the URSSAF-injected French rows carry
# no label at all. Charts built on raw codes ("CZ0", "DEA", "PL5") are unreadable
# to anyone outside the project, so every figure uses this lookup instead.

NUTS1_NAMES: Dict[str, str] = {
    # Germany
    "DE1": "Baden-Württemberg", "DE2": "Bavaria", "DE3": "Berlin",
    "DE4": "Brandenburg", "DE5": "Bremen", "DE6": "Hamburg", "DE7": "Hesse",
    "DE8": "Mecklenburg-Vorpommern", "DE9": "Lower Saxony",
    "DEA": "North Rhine-Westphalia", "DEB": "Rhineland-Palatinate",
    "DEC": "Saarland", "DED": "Saxony", "DEE": "Saxony-Anhalt",
    "DEF": "Schleswig-Holstein", "DEG": "Thuringia",
    # France (URSSAF)
    "FR1": "Île-de-France", "FRB": "Centre-Val de Loire",
    "FRC": "Bourgogne-Franche-Comté", "FRD": "Normandy",
    "FRE": "Hauts-de-France", "FRF": "Grand Est", "FRG": "Pays de la Loire",
    "FRH": "Brittany", "FRI": "Nouvelle-Aquitaine", "FRJ": "Occitanie",
    "FRK": "Auvergne-Rhône-Alpes", "FRL": "Provence-Alpes-Côte d'Azur",
    "FRM": "Corsica", "FRY": "French overseas",
    # Central & eastern Europe
    "CZ0": "Czechia", "CZZ": "Czechia (extra-regio)", "SK0": "Slovakia",
    "HU1": "Central Hungary", "HU2": "Transdanubia", "HU3": "Great Plain & North",
    "PL2": "Poland — South", "PL4": "Poland — North-West",
    "PL5": "Poland — South-West", "PL6": "Poland — North",
    "PL7": "Poland — Central", "PL8": "Poland — East", "PL9": "Warsaw region",
    "RO1": "Romania — Macroregion 1", "RO2": "Romania — Macroregion 2",
    "RO3": "Romania — Macroregion 3", "RO4": "Romania — Macroregion 4",
    "SI0": "Slovenia", "BG3": "Bulgaria — North", "BG4": "Bulgaria — South",
    "EE0": "Estonia", "LT0": "Lithuania", "LV0": "Latvia", "HR0": "Croatia",
    # Southern Europe
    "ES1": "Spain — North-West", "ES2": "Spain — North-East",
    "ES3": "Madrid", "ES4": "Spain — Centre", "ES5": "Spain — East",
    "ES6": "Spain — South", "ES7": "Canary Islands",
    "PT1": "Portugal (mainland)", "PT2": "Azores", "PT3": "Madeira",
    "EL3": "Attica", "EL4": "Aegean & Crete", "EL5": "Northern Greece",
    "EL6": "Central Greece",
    "ITC": "Italy — North-West", "ITF": "Italy — South",
    "ITG": "Italy — Islands", "ITH": "Italy — North-East",
    "ITI": "Italy — Centre",
    # Rest
    "AT1": "Eastern Austria", "AT2": "Southern Austria", "AT3": "Western Austria",
    "BE1": "Brussels", "BE2": "Flanders", "BE3": "Wallonia",
    "NL1": "Netherlands — North", "NL2": "Netherlands — East",
    "NL3": "Netherlands — West", "NL4": "Netherlands — South",
    "SE1": "Eastern Sweden", "SE2": "Southern Sweden", "SE3": "Northern Sweden",
    "DK0": "Denmark", "FI1": "Finland", "IE0": "Ireland", "LU0": "Luxembourg",
    "NO0": "Norway", "CY0": "Cyprus", "MT0": "Malta",
}


def region_label(code: str, with_country: bool = True, maxlen: int = 34,
                 whole_country: Optional[set] = None) -> str:
    """
    Human-readable region name, falling back to the code if unknown.

    `whole_country` — codes that cover an entire national territory get a
    trailing dagger, so a reader can see at once that the unit is a country
    rather than a region of one (see `whole_country_units`).
    """
    code = str(code)
    mark = "\u2020" if whole_country and code in whole_country else ""
    name = NUTS1_NAMES.get(code) or fd.FRENCH_NUTS2_NAMES.get(code) or code
    if with_country and name != code and len(code) >= 2:
        iso = code[:2]
        # don't tag country onto a name that already is the country
        if name.lower() not in {"czechia", "slovakia", "slovenia", "estonia",
                                "lithuania", "latvia", "croatia", "denmark",
                                "finland", "ireland", "luxembourg", "norway",
                                "cyprus", "malta"}:
            name = f"{name} ({iso})"
    name = name if len(name) <= maxlen else name[: maxlen - 1] + "…"
    return name + mark


# --- URSSAF région (INSEE code) -> French NUTS-1 -------------------------------
# Verified against the codes present in the Eurostat extracts.
URSSAF_REGION_TO_NUTS1: Dict[str, str] = {
    "11": "FR1",   # Île-de-France
    "24": "FRB",   # Centre-Val de Loire
    "27": "FRC",   # Bourgogne-Franche-Comté
    "28": "FRD",   # Normandie
    "32": "FRE",   # Hauts-de-France
    "44": "FRF",   # Grand Est
    "52": "FRG",   # Pays de la Loire
    "53": "FRH",   # Bretagne
    "75": "FRI",   # Nouvelle-Aquitaine
    "76": "FRJ",   # Occitanie
    "84": "FRK",   # Auvergne-Rhône-Alpes
    "93": "FRL",   # Provence-Alpes-Côte d'Azur
    "94": "FRM",   # Corse
    "1": "FRY", "2": "FRY", "3": "FRY", "4": "FRY",   # DOM
}


# ==============================================================================
# 2. THE ANALYSIS PANEL
# ==============================================================================

def european_panel(nace: str = "C29",
                   indicator: str = "employment",
                   level: int = 2,
                   france: str = "exclude",
                   drop_other_bloc: bool = False) -> pd.DataFrame:
    """
    Build the European regional panel for spatial analysis.

    Parameters
    ----------
    level   : 1 (NUTS-1, coarser, better balanced) or 2 (NUTS-2, finer).
              France can now be spliced at EITHER level: at NUTS-1 via URSSAF's
              13 modern régions, at NUTS-2 by aggregating départements into the
              22 pre-2016 régions (fd.DEPARTEMENT_TO_NUTS2).
    france  : 'exclude'  -> EU-minus-France (France reported apart)
              'urssaf'   -> splice France in from URSSAF (RULE 1 preserved)

    WHICH LEVEL TO USE
    ------------------
    NUTS-1 is the headline: over 2012-2023 it balances to ~76 regions covering
    92%/97% of the endpoints. NUTS-2 gives finer geography but WORSE balance
    (~160 regions, 89%/82%). Use NUTS-2 as a robustness check, not a
    replacement -- more regions is not the same as more coverage.

    Returns columns
    ---------------
      geo_code, geo_label, country_iso2, bloc, year, value,
      source ('eurostat' | 'urssaf'), is_confidential, is_missing

    Caveat carried in the data
    --------------------------
    Where France is spliced in, `source='urssaf'` marks rows measured as
    *effectifs salariés* (salaried employment) rather than Eurostat's *persons
    employed*. The two are close for manufacturing but not identical; any figure
    mixing them must say so.
    """
    if france == "urssaf" and level not in (1, 2):
        raise ValueError("france='urssaf' supports level 1 or 2 only.")

    e = fd.eurostat_panel(nace=nace, indicator=indicator,
                          drop_france=True, nuts_level=level).copy()
    e["source"] = "eurostat"

    frames = [e[["geo_code", "geo_label", "country_iso2", "year", "value",
                 "is_confidential", "is_missing", "source"]]]

    if france == "urssaf":
        scope = "all"          # all auto APE codes -> comparable to NACE C29
        if level == 1:
            # URSSAF's 13 modern régions ARE French NUTS-1
            u = fd.urssaf_panel(nace=nace, scope=scope, level="region").copy()
            u["geo_code"] = (u["region_code"].astype(str).str.strip()
                               .map(URSSAF_REGION_TO_NUTS1))
        else:
            # NUTS-2: aggregate départements into the 22 pre-2016 régions
            u = fd.urssaf_panel(nace=nace, scope=scope, level="departement").copy()
            u["geo_code"] = (u["departement_code"].astype(str).str.strip()
                               .str.replace(r"\.0$", "", regex=True).str.zfill(2)
                               .map(fd.DEPARTEMENT_TO_NUTS2))
            u["region"] = u["geo_code"].map(fd.FRENCH_NUTS2_NAMES)
        u = u.dropna(subset=["geo_code"])
        u = (u.groupby(["geo_code", "year"], as_index=False)
               .agg(value=("employees", "sum"), geo_label=("region", "first")))
        u["country_iso2"] = "FR"
        u["is_confidential"], u["is_missing"] = False, u["value"].isna()
        u["source"] = "urssaf"
        frames.append(u[["geo_code", "geo_label", "country_iso2", "year", "value",
                         "is_confidential", "is_missing", "source"]])

    panel = pd.concat(frames, ignore_index=True)

    # Drop Eurostat's EXTRA-REGIO codes (CZZ/CZZZ, BEZ/BEZZ, ...). These are the
    # residual "not assigned to any region" bucket, not places. They carry
    # almost nothing (max 29 employees anywhere, 0.00% of the 2023 total) but
    # they inflate region counts and, worse, make a single-region country look
    # like a two-region one, which silently defeats `whole_country_units`.
    extra = panel["geo_code"].astype(str).str[2:].str.startswith("Z")
    if extra.any():
        panel = panel[~extra]

    panel["bloc"] = panel["country_iso2"].map(assign_bloc)
    if drop_other_bloc:
        panel = panel[panel["bloc"] != "other"]
    panel = panel.sort_values(["geo_code", "year"]).reset_index(drop=True)
    # Provenance travels WITH the panel so every figure can name itself exactly
    # instead of saying "balanced panel" and leaving the reader to guess which.
    panel.attrs.update(nace=nace, indicator=indicator, level=int(level),
                       france=france)
    return panel


# ==============================================================================
# 2b. SELF-DESCRIPTION — so no figure ever just says "balanced panel"
# ==============================================================================

def whole_country_units(panel: pd.DataFrame) -> set:
    """
    Region codes that ARE their entire country in this panel.

    This matters for comparability and it is not cosmetic. At NUTS-1, Czechia
    (CZ0) and Slovakia (SK0) are single regions covering the whole national
    territory, while Germany is split into sixteen Länder. Ranking "Czechia"
    against "North Rhine-Westphalia" therefore compares a country with a region
    of another country, and the country will tend to look larger for purely
    administrative reasons. Detected from the data rather than hard-coded, so it
    stays correct if the level or the country set changes.
    """
    n = panel.groupby("country_iso2")["geo_code"].nunique()
    solo = set(n[n == 1].index)
    return set(panel.loc[panel["country_iso2"].isin(solo), "geo_code"].unique())


def panel_description(panel: pd.DataFrame, y0: Optional[int] = None,
                      y1: Optional[int] = None,
                      n_regions: Optional[int] = None) -> str:
    """One sentence naming exactly which panel a figure was computed on."""
    a = panel.attrs
    lvl = a.get("level", "?")
    nace = a.get("nace", "?")
    ind = a.get("indicator", "employment")
    fr = a.get("france", "?")
    n = n_regions if n_regions is not None else panel["geo_code"].nunique()
    # "observed in both endpoints" describes ENDPOINT balance. The headline
    # figures use STRICT balance -- a value in every year of the window -- which
    # is a different and smaller set (68 regions, not 75). Say which one.
    strict = a.get("balance") == "strict"
    if y0 is None:
        win = ""
    elif strict:
        win = f" observed in every year from {y0} to {y1}"
    else:
        win = f" observed in both {y0} and {y1}"
    src = {"urssaf": "France measured from URSSAF, all other countries from Eurostat",
           "exclude": "France excluded (Eurostat's French regional data are not used)"
           }.get(fr, f"France: {fr}")
    return (f"Balanced panel: {n} NUTS-{lvl} regions with NACE {nace} {ind}"
            f"{win}. {src}.")


# ==============================================================================
# 3. BALANCING — never skip this
# ==============================================================================

def balance_report(panel: pd.DataFrame,
                   windows: Iterable[Tuple[int, int]] = (
                       (2008, 2019), (2008, 2023), (2012, 2019),
                       (2013, 2019), (2012, 2023), (2021, 2023))
                   ) -> pd.DataFrame:
    """
    For each candidate window: how many regions survive balancing, and what
    share of each endpoint's total they represent.

    Use this to CHOOSE the window before computing anything. A long window with
    60% coverage tells you less than a short one with 95%.
    """
    piv = panel.pivot_table(index="geo_code", columns="year",
                            values="value", aggfunc="first")
    rows = []
    for y0, y1 in windows:
        if y0 not in piv.columns or y1 not in piv.columns:
            continue
        ok = piv[[y0, y1]].dropna()
        rows.append({
            "window": f"{y0}-{y1}",
            "y0": y0, "y1": y1,
            "n_regions_balanced": len(ok),
            "n_regions_total": int(piv[[y0, y1]].notna().any(axis=1).sum()),
            "pct_of_y0_total": 100 * ok[y0].sum() / piv[y0].sum(),
            "pct_of_y1_total": 100 * ok[y1].sum() / piv[y1].sum(),
            "balanced_y0": ok[y0].sum(),
            "balanced_y1": ok[y1].sum(),
            "growth_pct": 100 * (ok[y1].sum() - ok[y0].sum()) / ok[y0].sum(),
        })
    return pd.DataFrame(rows)


def balanced_subset(panel: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    """Regions observed at BOTH endpoints, in long form (only y0 and y1 rows)."""
    piv = panel.pivot_table(index="geo_code", columns="year",
                            values="value", aggfunc="first")
    keep = piv[[y0, y1]].dropna().index
    out = panel[panel["geo_code"].isin(keep) & panel["year"].isin([y0, y1])]
    return out.copy()


# ==============================================================================
# 4. STRUCTURAL-CHANGE MEASURES
# ==============================================================================

def bloc_trajectories(panel: pd.DataFrame,
                      balanced: bool = True,
                      y0: Optional[int] = None,
                      y1: Optional[int] = None) -> pd.DataFrame:
    """
    Employment and SHARE by bloc, per year — the headline Pavlínek result.

    `balanced=True` restricts to regions present in every year of the window, so
    the share series is not contaminated by regions entering and leaving.
    """
    p = panel.dropna(subset=["value"]).copy()
    if y0 is not None:
        p = p[(p["year"] >= y0) & (p["year"] <= y1)]
    if balanced:
        yrs = sorted(p["year"].unique())
        counts = p.groupby("geo_code")["year"].nunique()
        keep = counts[counts == len(yrs)].index
        p = p[p["geo_code"].isin(keep)]

    g = (p.groupby(["year", "bloc"], as_index=False)["value"].sum()
           .rename(columns={"value": "employment"}))
    tot = g.groupby("year")["employment"].transform("sum")
    g["share_pct"] = 100 * g["employment"] / tot
    base = g[g["year"] == g["year"].min()].set_index("bloc")["employment"]
    g["index_100"] = g.apply(
        lambda r: 100 * r["employment"] / base.get(r["bloc"], np.nan), axis=1)
    g.attrs["n_regions"] = p["geo_code"].nunique()
    g.attrs["balanced"] = balanced
    return g


def shift_share_1sector(panel: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    """
    Classic two-component shift-share on a single sector.

        ΔE_i  =  E_i0 · g        +      E_i0 · (g_i − g)
                 ^^^^^^^^^^             ^^^^^^^^^^^^^^^^^
                 EU-growth               DIFFERENTIAL (competitive) shift
                 component

    `g` is EU-wide growth over the balanced set; `g_i` the region's own growth.

    Reading it: the differential shift is the part of a region's change that is
    *not* explained by the industry's overall trajectory. Positive = the region
    outperformed the European industry; negative = it lost ground relative to it.
    That is precisely Pavlínek's core-vs-periphery question, measured on stocks.
    """
    b = balanced_subset(panel, y0, y1)
    w = b.pivot_table(index=["geo_code", "country_iso2", "bloc"],
                      columns="year", values="value", aggfunc="first").reset_index()
    w = w.rename(columns={y0: "E0", y1: "E1"})

    E0, E1 = w["E0"].sum(), w["E1"].sum()
    g = (E1 - E0) / E0

    w["actual_change"] = w["E1"] - w["E0"]
    w["growth_rate"] = (w["E1"] - w["E0"]) / w["E0"]
    w["eu_growth_component"] = w["E0"] * g
    w["differential_shift"] = w["actual_change"] - w["eu_growth_component"]
    w["differential_pct_of_E0"] = 100 * w["differential_shift"] / w["E0"]

    w.attrs.update({"y0": y0, "y1": y1, "eu_growth": g,
                    "E0_total": E0, "E1_total": E1, "n_regions": len(w)})
    return w.sort_values("differential_shift", ascending=False).reset_index(drop=True)


def shift_share_2sector(panel_c29: pd.DataFrame, panel_c27: pd.DataFrame,
                        y0: int, y1: int) -> pd.DataFrame:
    """
    Three-component shift-share over two sectors: C29 (motor vehicles) and
    C27 (electrical equipment).

        ΔE_ij = E_ij0·G  +  E_ij0·(G_j − G)  +  E_ij0·(g_ij − G_j)
                 national      INDUSTRY MIX        COMPETITIVE
                 growth

    Why two sectors, and why these two
    ----------------------------------
    With one sector the mix component does not exist. Adding C27 gives it
    meaning that is substantive rather than technical: C27 is where batteries
    and electric motors sit (alongside lighting, wiring and appliances -- see
    the C27 caveat in Module 00). So the MIX component answers "did this region
    suffer because it was concentrated in vehicles rather than electrical
    equipment?" while the COMPETITIVE component answers "did it underperform
    within its own sector?".

    Returns one row per (region, sector) plus a region-level total.
    """
    frames = []
    for nace, p in (("C29", panel_c29), ("C27", panel_c27)):
        b = balanced_subset(p, y0, y1)
        w = b.pivot_table(index=["geo_code", "country_iso2", "bloc"],
                          columns="year", values="value", aggfunc="first").reset_index()
        w = w.rename(columns={y0: "E0", y1: "E1"})
        w["nace"] = nace
        frames.append(w)
    df = pd.concat(frames, ignore_index=True).dropna(subset=["E0", "E1"])

    # keep only regions present in BOTH sectors, so components are comparable
    both = (df.groupby("geo_code")["nace"].nunique() == 2)
    df = df[df["geo_code"].isin(both[both].index)].copy()

    G = (df["E1"].sum() - df["E0"].sum()) / df["E0"].sum()          # all sectors
    sect = df.groupby("nace").agg(E0=("E0", "sum"), E1=("E1", "sum"))
    Gj = ((sect["E1"] - sect["E0"]) / sect["E0"]).to_dict()          # per sector

    df["g_ij"] = (df["E1"] - df["E0"]) / df["E0"]
    df["G_j"] = df["nace"].map(Gj)
    df["actual_change"] = df["E1"] - df["E0"]
    df["national_growth"] = df["E0"] * G
    df["industry_mix"] = df["E0"] * (df["G_j"] - G)
    df["competitive"] = df["E0"] * (df["g_ij"] - df["G_j"])

    df.attrs.update({"y0": y0, "y1": y1, "G_total": G, "G_by_sector": Gj,
                     "n_regions": df["geo_code"].nunique()})
    return df.reset_index(drop=True)


def lorenz_points(panel: pd.DataFrame, year: int,
                  bloc: Optional[str] = None) -> pd.DataFrame:
    """
    Points of the Lorenz curve for one year: cumulative share of REGIONS against
    cumulative share of EMPLOYMENT, regions ordered smallest first.

    Read it as: "the smallest x% of regions hold y% of European automotive
    employment". The further the curve bows away from the 45° line, the more
    concentrated the industry. The Gini is twice the area between them.
    """
    d = panel[(panel["year"] == year) & panel["value"].notna()]
    if bloc:
        d = d[d["bloc"] == bloc]
    v = np.sort(d["value"].values)
    if len(v) == 0 or v.sum() == 0:
        return pd.DataFrame(columns=["p_regions", "p_employment"])
    cum = np.concatenate([[0], np.cumsum(v) / v.sum()])
    p = np.concatenate([[0], np.arange(1, len(v) + 1) / len(v)])
    n = len(v)
    gini = (2 * np.sum((np.arange(1, n + 1)) * v) / (n * v.sum())) - (n + 1) / n
    out = pd.DataFrame({"p_regions": p, "p_employment": cum})
    out.attrs.update({"gini": float(gini), "year": year,
                      "bloc": bloc or "all", "n_regions": n})
    return out


def concentration_by_bloc(panel: pd.DataFrame, y0: int = 2012,
                          y1: int = 2023) -> pd.DataFrame:
    """
    Concentration measured WITHIN each bloc, so the core and the peripheries can
    be compared. A bloc whose HHI falls is spreading work across more of its own
    regions; one whose HHI rises is consolidating into fewer.
    """
    frames = []
    for b in [x for x in BLOC_ORDER if x in set(panel["bloc"])]:
        sub = panel[panel["bloc"] == b]
        c = concentration(sub, y0, y1)
        if len(c):
            c["bloc"] = b
            frames.append(c)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def concentration(panel: pd.DataFrame, y0: int = 2012, y1: int = 2023,
                  balanced: bool = True) -> pd.DataFrame:
    """
    How concentrated is European automotive employment, and does it disperse
    over time? (Dispersal toward peripheries is Pavlínek's expectation.)

      hhi          Herfindahl over regional shares, x10,000
      top5 / top10 share of the largest regions, %
      n_for_50pct  how many regions hold half of employment
      gini         inequality of the regional distribution

    ALWAYS pass an explicit window. Balancing over the panel's full year range
    is a trap: Eurostat runs 2008-2023 while spliced URSSAF France runs
    2006-2025, so "present in every year" silently collapses the set to the
    French regions alone. `y0`/`y1` bound the window before balancing.
    """
    p = panel.dropna(subset=["value"]).copy()
    p = p[(p["year"] >= y0) & (p["year"] <= y1)]
    if balanced:
        yrs = sorted(p["year"].unique())
        counts = p.groupby("geo_code")["year"].nunique()
        p = p[p["geo_code"].isin(counts[counts == len(yrs)].index)]
        if p["geo_code"].nunique() < 20:
            warnings.warn(
                f"Only {p['geo_code'].nunique()} regions survive balancing over "
                f"{y0}-{y1}. Check the window against balance_report().")

    rows = []
    for year, g in p.groupby("year"):
        v = g["value"].sort_values(ascending=False).values
        tot = v.sum()
        if tot <= 0:
            continue
        sh = v / tot
        cum = np.cumsum(sh)
        n50 = int(np.searchsorted(cum, 0.50) + 1)
        # Gini on the regional distribution
        s = np.sort(v)
        n = len(s)
        gini = (2 * np.sum((np.arange(1, n + 1)) * s) / (n * s.sum())) - (n + 1) / n
        rows.append({
            "year": year, "n_regions": n,
            "hhi": float((sh ** 2).sum() * 10_000),
            "top5_pct": float(100 * sh[:5].sum()),
            "top10_pct": float(100 * sh[:10].sum()),
            "n_for_50pct": n50,
            "gini": float(gini),
            "total_employment": float(tot),
        })
    return pd.DataFrame(rows)


def unit_size(panel_emp: pd.DataFrame, panel_units: pd.DataFrame) -> pd.DataFrame:
    """
    Average establishment size = persons employed / local units, by region-year.

    A structural indicator Eurostat supports directly: rising average size means
    consolidation into large plants; falling size means fragmentation toward a
    supplier-dense tissue. Useful for contrasting core (large assembly) with
    periphery (supplier networks).
    """
    e = panel_emp[["geo_code", "country_iso2", "bloc", "year", "value"]].rename(
        columns={"value": "employment"})
    u = panel_units[["geo_code", "year", "value"]].rename(columns={"value": "units"})
    m = e.merge(u, on=["geo_code", "year"], how="inner")
    m = m[(m["units"] > 0) & m["employment"].notna()]
    m["avg_unit_size"] = m["employment"] / m["units"]
    return m


# ==============================================================================
# 5. FIGURES
# ==============================================================================

def _style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.grid": True, "grid.color": "#E4E9E8", "grid.linewidth": .7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#666666", "figure.facecolor": "white",
    })


def fig_bloc_trajectories(traj: pd.DataFrame, save: Optional[str] = None,
                          title_note: str = ""):
    """
    THE headline figure: levels, shares and indexed trajectories by bloc.
    Panel (b) — shares — is the structural-change statement in Lefèvre's sense.
    """
    import matplotlib.pyplot as plt
    _style()
    blocs = [b for b in BLOC_ORDER if b in traj["bloc"].unique()]
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.4))
    fig.suptitle("Spatial structural change in European automotive employment"
                 + (f" — {title_note}" if title_note else ""),
                 fontsize=14, fontweight="bold", x=0.02, ha="left")

    ax = axes[0]
    for b in blocs:
        g = traj[traj["bloc"] == b].sort_values("year")
        ax.plot(g["year"], g["employment"] / 1000, lw=2.2, marker="o", ms=3,
                color=BLOC_COLOURS[b], label=b.replace("_", " "))
    ax.set_title("(a) Employment level", loc="left")
    ax.set_ylabel("persons employed (thousands)")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    piv = traj.pivot(index="year", columns="bloc", values="share_pct")[blocs]
    ax.stackplot(piv.index, *[piv[b] for b in blocs],
                 colors=[BLOC_COLOURS[b] for b in blocs],
                 labels=[b.replace("_", " ") for b in blocs], alpha=.92)
    ax.set_title("(b) Share of European total", loc="left")
    ax.set_ylabel("% of total"); ax.set_ylim(0, 100)
    ax.legend(fontsize=8, frameon=False, loc="lower left")

    ax = axes[2]
    for b in blocs:
        g = traj[traj["bloc"] == b].sort_values("year")
        ax.plot(g["year"], g["index_100"], lw=2.2, marker="o", ms=3,
                color=BLOC_COLOURS[b], label=b.replace("_", " "))
    ax.axhline(100, color="#999999", ls="--", lw=1)
    ax.set_title(f"(c) Indexed ({int(traj['year'].min())} = 100)", loc="left")
    ax.set_ylabel("index")

    n = traj.attrs.get("n_regions")
    _foot(fig, f"The same {n} regions appear in every year, so a line cannot "
             "move because the sample moved. Figures are Eurostat structural "
             "business statistics, except for France, which is measured from "
             "URSSAF administrative records \u2014 a different basis, and not "
             "exactly the same quantity.", y=-0.01)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def fig_shift_share_explained(ss: pd.DataFrame, top_n: int = 14,
                              panel: Optional[pd.DataFrame] = None,
                              save: Optional[str] = None):
    """
    The shift-share result, told in plain language for a reader who has never
    met the method.

    Panel (a) works one region at a time: the bar shows how many jobs the region
    WOULD have had if it had simply grown at the European rate, against how many
    it actually had. The gap between them is the whole point of the method, and
    it is labelled in words rather than left as "differential shift".

    Panel (b) then shows that gap for every region, in jobs, with real names.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    _style()
    y0, y1 = ss.attrs["y0"], ss.attrs["y1"]
    g = ss.attrs["eu_growth"]

    # Units that ARE a whole country get a dagger: at NUTS-1 "Czechia" is the
    # entire country while "North Rhine-Westphalia" is one Land of sixteen, and
    # a reader ranking them side by side deserves to be told.
    solo = whole_country_units(panel) if panel is not None else set()

    fig = plt.figure(figsize=(16, 7.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.32)
    fig.suptitle("Regional employment change relative to the European trend, "
                 f"{y0}\u2013{y1}", fontsize=15, fontweight="bold", x=0.02,
                 ha="left")
    fig.text(0.02, 0.905,
             f"European automotive employment grew {100*g:+.1f}% over this period. "
             "A region that grew faster gained ground; one that grew slower lost it, "
             "even if its employment rose.",
             fontsize=10, color="#444444")

    # ---- (a) the method, on four illustrative regions ----------------------
    ax = fig.add_subplot(gs[0])
    show = pd.concat([ss.head(2), ss.tail(2)]).iloc[::-1]
    ypos = np.arange(len(show)) * 1.0
    h = 0.34
    for i, r in enumerate(show.itertuples()):
        exp = r.E0 * (1 + g)
        ax.barh(ypos[i] + h/2, r.E0 / 1000, h, color="#CBD8D7",
                label="start" if i == 0 else None)
        ax.barh(ypos[i] + h/2, (exp - r.E0) / 1000, h, left=r.E0 / 1000,
                color="#8FB8B4", label="if it grew at the EU rate" if i == 0 else None)
        ax.barh(ypos[i] - h/2, r.E1 / 1000, h, color="#14746F",
                label="actual" if i == 0 else None)
        gap = (r.E1 - exp) / 1000
        ax.annotate(f"{gap:+.0f}k vs EU pace", (max(r.E1, exp) / 1000, ypos[i]),
                    xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=9, fontweight="bold",
                    color="#14746F" if gap >= 0 else "#C0563F")
    ax.set_yticks(ypos)
    ax.set_yticklabels([region_label(r.geo_code, whole_country=solo)
                        for r in show.itertuples()], fontsize=9)
    ax.set_xlabel("employees (thousands)")
    ax.set_title("(a) The measure worked through — two gainers, two losers",
                 loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_xlim(0, ax.get_xlim()[1] * 1.28)

    # ---- (b) the gap, for every region -------------------------------------
    ax = fig.add_subplot(gs[1])
    sel = pd.concat([ss.head(top_n // 2), ss.tail(top_n // 2)]).sort_values("differential_shift")
    cols = [BLOC_COLOURS.get(b, "#888") for b in sel["bloc"]]
    bars = ax.barh(range(len(sel)), sel["differential_shift"] / 1000, color=cols)
    ax.set_yticks(range(len(sel)))
    ax.set_yticklabels([region_label(c, whole_country=solo)
                        for c in sel["geo_code"]], fontsize=9)
    ax.axvline(0, color="#333333", lw=1.2)
    for b, v in zip(bars, sel["differential_shift"] / 1000):
        ax.annotate(f"{v:+.0f}k", (v, b.get_y() + b.get_height()/2),
                    xytext=(6 if v >= 0 else -6, 0), textcoords="offset points",
                    va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
    ax.set_title("(b) Jobs gained or lost RELATIVE to the European trend", loc="left")
    ax.set_xlabel("thousands of jobs vs the EU pace")
    ax.set_xlim(ax.get_xlim()[0] * 1.18, ax.get_xlim()[1] * 1.18)
    ax.legend(handles=[Patch(color=BLOC_COLOURS[b], label=b.replace("_", " "))
                       for b in BLOC_ORDER if b in set(sel["bloc"])],
              fontsize=9, frameon=False, loc="lower right")

    note = (panel_description(panel, y0, y1, ss.attrs["n_regions"])
            if panel is not None else
            f"Balanced panel: {ss.attrs['n_regions']} regions observed in "
            f"every year from {y0} to {y1}. France measured from URSSAF, all "
            "other countries from Eurostat.")
    if solo & set(sel["geo_code"]) or solo & set(show["geo_code"]):
        note += ("\n\u2020 marks a unit that is an entire country rather than "
                 "a region of one. At this level of the European regional "
                 "classification some smaller states are a single region, so "
                 "they are not comparable in size with, say, one German Land "
                 "among sixteen.")
    _foot(fig, note, y=0.005)
    fig.subplots_adjust(top=0.86, bottom=0.10)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def fig_shift_share(ss: pd.DataFrame, top_n: int = 20, save: Optional[str] = None):
    """
    Differential shift by region (one-sector). Positive = outperformed the
    European industry; negative = lost ground relative to it.

    Kept for continuity; `fig_shift_share_explained` is the reader-facing version.
    """
    import matplotlib.pyplot as plt
    _style()
    y0, y1 = ss.attrs["y0"], ss.attrs["y1"]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    fig.suptitle(f"Shift-share decomposition, {y0}–{y1}", fontsize=14,
                 fontweight="bold", x=0.02, ha="left")

    # (a) winners and losers
    ax = axes[0]
    sel = pd.concat([ss.head(top_n // 2), ss.tail(top_n // 2)])
    sel = sel.sort_values("differential_shift")
    cols = [BLOC_COLOURS[b] for b in sel["bloc"]]
    ax.barh(range(len(sel)), sel["differential_shift"] / 1000, color=cols)
    ax.set_yticks(range(len(sel)))
    ax.set_yticklabels([region_label(r.geo_code) for r in sel.itertuples()], fontsize=8)
    ax.axvline(0, color="#333333", lw=1)
    ax.set_title("(a) Differential shift — biggest gainers and losers", loc="left")
    ax.set_xlabel("thousands of jobs relative to EU trend")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLOC_COLOURS[b], label=b.replace("_", " "))
                       for b in BLOC_ORDER if b in set(sel["bloc"])],
              fontsize=8, frameon=False, loc="lower right")

    # (b) by bloc
    ax = axes[1]
    agg = (ss.groupby("bloc")
             .agg(differential=("differential_shift", "sum"),
                  eu_component=("eu_growth_component", "sum"),
                  actual=("actual_change", "sum"))
             .reindex([b for b in BLOC_ORDER if b in set(ss["bloc"])]))
    x = np.arange(len(agg)); w = .27
    ax.bar(x - w, agg["eu_component"] / 1000, w, label="EU-growth component", color="#AFC8C6")
    ax.bar(x, agg["differential"] / 1000, w, label="differential shift",
           color=[BLOC_COLOURS[b] for b in agg.index])
    ax.bar(x + w, agg["actual"] / 1000, w, label="actual change", color="#444444")
    ax.set_xticks(x); ax.set_xticklabels([b.replace("_", "\n") for b in agg.index], fontsize=8)
    ax.axhline(0, color="#333333", lw=1)
    ax.set_title("(b) Aggregated by bloc", loc="left")
    ax.set_ylabel("thousands of jobs"); ax.legend(fontsize=8, frameon=False)

    _foot(fig, f"The same {ss.attrs['n_regions']} regions appear at both ends "
               "of the period. European automotive employment as a whole grew "
               f"{100*ss.attrs['eu_growth']:+.1f}% over these years, and that is "
               "the pace each region is being measured against.", y=-0.01)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def fig_shift_share_2sector(ss2: pd.DataFrame, panel: Optional[pd.DataFrame] = None,
                            save: Optional[str] = None):
    """
    Two-sector shift-share by bloc, one panel per bloc.

    Each bloc's change splits into three parts that add up exactly:

      * national growth  — what if it had simply grown with the whole economy?
      * sector effect    — was it hurt just by being in vehicles rather than
                           electrical equipment? (the industry-mix term)
      * regional effect  — did it do better or worse AT that sector than
                           everyone else? (the competitive term)

    The net change is drawn as a diamond rather than a fourth bar, because the
    three bars already sum to it: shift-share is an accounting identity, and a
    fourth bar would invite the reader to add it in twice.
    """
    import matplotlib.pyplot as plt
    _style()
    y0, y1 = ss2.attrs["y0"], ss2.attrs["y1"]
    agg = (ss2.groupby(["bloc", "nace"])[["national_growth", "industry_mix",
                                          "competitive", "actual_change"]]
             .sum().reset_index())
    blocs = [b for b in BLOC_ORDER if b in set(agg["bloc"])]

    fig, axes = plt.subplots(1, len(blocs), figsize=(4.7 * len(blocs), 5.6),
                             sharey=True)
    if len(blocs) == 1:
        axes = [axes]
    fig.suptitle("Two-sector shift-share of regional employment by bloc, "
                 f"vehicles (C29) and electrical equipment (C27), {y0}\u2013{y1}",
                 fontsize=13.5, fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, .895,
             "Each bloc's change split into the part explained by overall growth, "
             "the part explained by which sector it was in, and the part "
             "explained by how it did within that sector.",
             fontsize=9.6, color="#4A4A4A", ha="left")

    comps = [("national_growth", "if it had grown with the whole economy",
              "#AFC8C6"),
             ("industry_mix", "sector effect \u2014 being in this sector rather "
              "than the other", "#C6803D"),
             ("competitive", "regional effect \u2014 doing better or worse "
              "within the sector", "#14746F")]

    for ax, b in zip(axes, blocs):
        sub = agg[agg["bloc"] == b].set_index("nace")
        x = np.arange(len(sub)); w = .25
        for k, (col, lab, colour) in enumerate(comps):
            ax.bar(x + (k - 1) * w, sub[col] / 1000, w, label=lab, color=colour)

        net = sub["actual_change"] / 1000
        ax.scatter(x, net, marker="D", s=48, color="#111111", zorder=5,
                   label="net change (= the three added together)")
        for xi, ni in zip(x, net):
            ax.hlines(ni, xi - 1.5 * w, xi + 1.5 * w, color="#111111",
                      lw=1.1, ls=":", zorder=4)
            ax.annotate(f"{ni:+,.0f}k", (xi, ni), textcoords="offset points",
                        xytext=(0, 10 if ni >= 0 else -17), ha="center",
                        fontsize=9, fontweight="bold", color="#111111")

        ax.axhline(0, color="#333333", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(sub.index)
        ax.set_title(b.replace("_", " "), loc="left", fontsize=11.5)
        ax.margins(y=.16)
        if b == blocs[0]:
            ax.set_ylabel("thousands of jobs")
            handles, labels = ax.get_legend_handles_labels()

    c29 = ss2[ss2["nace"] == "C29"].groupby("bloc")[["industry_mix",
                                                     "competitive"]].sum()
    verdict = ""
    if len(c29):
        if c29["competitive"].abs().max() > 2 * c29["industry_mix"].abs().max():
            verdict = ("For vehicles the sector effect is small and negative "
                       "everywhere, while the regional effect is large and "
                       "points in opposite directions: where production "
                       "happened mattered far more than which of the two "
                       "sectors a region sat in.  ")
    desc = (panel_description(panel, y0, y1, ss2.attrs.get("n_regions"))
            if panel is not None else
            f"Balanced panel: {ss2.attrs.get('n_regions', '?')} regions, "
            f"{y0}\u2013{y1}.")
    fig.legend(handles, labels, fontsize=8.8, frameon=False, ncol=2,
               loc="upper left", bbox_to_anchor=(0.02, 0.885))
    _foot(fig, verdict + desc + "  The three components sum exactly to the net "
                "change \u2014 shift-share is an accounting identity, not a "
                "model, and carries no causal claim.")
    fig.subplots_adjust(top=.745, bottom=.10)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def fig_hhi(conc: pd.DataFrame, conc_bloc: Optional[pd.DataFrame] = None,
            panel: Optional[pd.DataFrame] = None, save: Optional[str] = None):
    """
    Section 6, first figure: the Herfindahl index, overall and within blocs.

    (a) the whole industry — one line, with an arrow saying which direction
        means what, because HHI is not self-explanatory.
    (b) the same index computed WITHIN each bloc. This asks a different and
        sharper question: not "is Europe concentrating?" but "is the core
        thinning out while the east builds up its own hubs?"
    """
    import matplotlib.pyplot as plt
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.6))
    fig.suptitle("Geographic concentration of automotive employment, "
                 "overall and within blocs",
                 fontsize=14.5, fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, 0.895,
             "Herfindahl index = sum of squared regional shares. It rises when "
             "employment gathers into fewer regions.",
             fontsize=9.7, color="#4A4A4A", ha="left")

    # ---------------------------------------------------------------- (a)
    ax = axes[0]
    conc = conc.sort_values("year")
    ax.plot(conc["year"], conc["hhi"], "-o", color="#14746F", lw=2.3, ms=5)
    # label only the endpoints: twelve value labels is noise, two is a reading
    for pos, ha, dx, dy in ((0, "left", 2, 12), (len(conc) - 1, "left", 8, 0)):
        r = conc.iloc[pos]
        ax.annotate(f"{r['hhi']:,.0f}", (r["year"], r["hhi"]),
                    textcoords="offset points", xytext=(dx, dy), ha=ha,
                    va="center", fontsize=10.5, fontweight="bold",
                    color="#14746F")
    lo, hi = conc["hhi"].min(), conc["hhi"].max()
    pad = max((hi - lo) * 1.9, hi * .05)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_ylabel("Herfindahl index"); ax.set_xlabel("")
    yrs = list(conc["year"])
    ax.set_xticks(yrs[::2] if len(yrs) > 7 else yrs)
    ax.set_title("(a) The whole industry", loc="left", fontsize=11.5)

    # direction arrows, in a reserved margin so they never sit on the data
    x0 = conc["year"].min() - 1.30
    ax.set_xlim(x0 - .35, conc["year"].max() + 1.2)
    span = (hi + pad) - (lo - pad)
    mid = (lo + hi) / 2
    ax.annotate("", xy=(x0, mid + span * .30), xytext=(x0, mid + span * .04),
                arrowprops=dict(arrowstyle="-|>", color="#B03A2E", lw=2.1))
    ax.text(x0 + .22, mid + span * .30, "MORE\nconcentrated", fontsize=8.8,
            color="#B03A2E", ha="left", va="center", fontweight="bold")
    ax.annotate("", xy=(x0, mid - span * .30), xytext=(x0, mid - span * .04),
                arrowprops=dict(arrowstyle="-|>", color="#14746F", lw=2.1))
    ax.text(x0 + .22, mid - span * .30, "MORE\ndispersed", fontsize=8.8,
            color="#14746F", ha="left", va="center", fontweight="bold")

    d = conc["hhi"].iloc[-1] - conc["hhi"].iloc[0]
    ax.text(.98, .04,
            f"{d:+,.0f} points overall \u2014 "
            + ("bunching up" if d > 0 else "spreading out"),
            transform=ax.transAxes, ha="right", fontsize=9.6,
            fontweight="bold", color="#B03A2E" if d > 0 else "#14746F")

    # ---------------------------------------------------------------- (b)
    ax = axes[1]
    if conc_bloc is not None and len(conc_bloc):
        # 'other' is the residual bucket, and at NUTS-1 it is often a single
        # region, which forces HHI to its 10,000 ceiling and flattens every
        # real line against the axis. It is a bookkeeping category, not a bloc.
        cbl = conc_bloc[conc_bloc["bloc"] != "other"]
        shown = [x for x in BLOC_ORDER if x != "other" and x in set(cbl["bloc"])]
        ends = {}
        for b in shown:
            s = cbl[cbl["bloc"] == b].sort_values("year")
            ax.plot(s["year"], s["hhi"], "-o", color=BLOC_COLOURS[b], lw=2.2,
                    ms=4.5, label=b.replace("_", " "))
            ends[b] = (s["year"].iloc[-1], s["hhi"].iloc[0], s["hhi"].iloc[-1])

        # nudge end labels apart when two blocs finish close together
        order = sorted(ends, key=lambda b: ends[b][2])
        prev = None
        span = (max(e[2] for e in ends.values())
                - min(e[2] for e in ends.values())) or 1.0
        for b in order:
            xe, v0, v1 = ends[b]
            yy = v1
            if prev is not None and yy - prev < .085 * span:
                yy = prev + .085 * span
            prev = yy
            ax.annotate(f"{v1:,.0f} ({v1 - v0:+,.0f})", (xe, yy),
                        textcoords="offset points", xytext=(9, 0), fontsize=9.2,
                        fontweight="bold", color=BLOC_COLOURS[b], va="center")
        byrs = sorted(cbl["year"].unique())
        ax.set_xticks(byrs[::2] if len(byrs) > 7 else byrs)
        ax.set_xlim(min(byrs) - .3, max(byrs) + 3.4)
        ax.legend(fontsize=9.2, frameon=False, loc="upper right")
        ax.set_ylabel("Herfindahl index within the bloc")
        ax.set_title("(b) Within each bloc, computed separately", loc="left",
                     fontsize=11.5)
        ax.margins(y=.20)
    else:
        ax.axis("off")
        ax.text(.5, .5, "pass conc_bloc=concentration_by_bloc(panel, y0, y1)",
                ha="center", va="center", fontsize=10, color="#8A8A8A")

    foot = ("Panel (b): each line uses only the regions of its own bloc, so it "
            "measures internal spread, not the bloc's size. The residual "
            "\u2018other\u2019 bloc is omitted \u2014 it is a bookkeeping "
            "category, and at this level often a single region, which pins its "
            "index at the 10,000 ceiling.")
    if panel is not None:
        n = int(conc["n_regions"].iloc[-1]) if "n_regions" in conc.columns else None
        foot = panel_description(panel, int(conc["year"].min()),
                                 int(conc["year"].max()), n) + "\n" + foot
    _foot(fig, foot, y=-0.005)
    fig.subplots_adjust(top=.835, bottom=.155, wspace=.30)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def fig_concentration_detail(conc: pd.DataFrame,
                             panel: Optional[pd.DataFrame] = None,
                             save: Optional[str] = None):
    """
    Section 6, second figure: the two readings that need no index at all.

    (a) the share of European employment held by the largest regions, against
        reference thresholds a reader already understands (a half, a third).
    (b) how many regions it takes to account for half of European employment
        — the same statement, turned inside out.
    """
    import matplotlib.pyplot as plt
    _style()
    c = conc.sort_values("year")
    yrs = c["year"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.4))
    fig.suptitle("Concentration measured in shares and region counts",
                 fontsize=14.5, fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, .895,
             "Concentration expressed in shares and counts rather than a "
             "squared-share index.", fontsize=9.7, color="#4A4A4A", ha="left")

    # ------------------------------------------------------------------- (a)
    ax = axes[0]
    series = [("top10_pct", "top 10 regions", "#14746F"),
              ("top5_pct", "top 5 regions", "#7FB2AE")]
    series = [s for s in series if s[0] in c.columns]
    for col, lab, colour in series:
        ax.plot(yrs, c[col], "-o", color=colour, lw=2.4, ms=5, label=lab, zorder=3)
        v0, v1 = c[col].iloc[0], c[col].iloc[-1]
        # start label sits ABOVE its point (to the left it collides with the
        # y-tick labels); end label carries the change, so no corner caption
        ax.annotate(f"{v0:.1f}%", (yrs[0], v0), textcoords="offset points",
                    xytext=(2, 11), ha="left", fontsize=9.6, fontweight="bold",
                    color=colour, zorder=4)
        ax.annotate(f"{v1:.1f}%  ({v1 - v0:+.1f} pp)", (yrs[-1], v1),
                    textcoords="offset points", xytext=(9, 0), ha="left",
                    va="center", fontsize=9.6, fontweight="bold", color=colour,
                    zorder=4)
    ymin = min(c[s[0]].min() for s in series)
    ymax = max(c[s[0]].max() for s in series)
    for thr, lab, colour in ((50, "half the industry", "#B03A2E"),
                             (33.3, "one third", "#C6803D"),
                             (25, "one quarter", "#9AA0A6")):
        if ymin - 6 <= thr <= ymax + 6:
            ax.axhline(thr, color=colour, lw=1.2, ls="--", zorder=2)
            ax.text(yrs[0], thr + .35, lab, fontsize=8.8, color=colour,
                    ha="left", va="bottom", style="italic", zorder=4)
    ax.set_ylim(ymin - 4.5, ymax + 5.5)
    ax.set_xlim(yrs[0] - 1.0, yrs[-1] + 4.6)
    _int_years(ax, yrs)
    ax.set_ylabel("% of European employment")
    ax.set_title("(a) Share held by the largest regions", loc="left", fontsize=11.5)
    ax.legend(fontsize=9.2, frameon=False, loc="center left")

    # ------------------------------------------------------------------- (b)
    ax = axes[1]
    if "n_for_50pct" in c.columns:
        n = c["n_for_50pct"].to_numpy()
        ax.step(yrs, n, where="mid", color="#4E6E8E", lw=2.6, zorder=3)
        ax.scatter(yrs, n, color="#4E6E8E", s=42, zorder=4)
        seen = set()
        for xi, v in zip(yrs, n):
            if v not in seen or xi in (yrs[0], yrs[-1]):
                ax.annotate(f"{v:.0f}", (xi, v), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=11,
                            fontweight="bold", color="#4E6E8E")
                seen.add(v)
        ax.set_ylim(n.min() - 1.4, n.max() + 1.4)
        ax.set_yticks(range(int(n.min()), int(n.max()) + 1))
        ax.set_xlim(yrs[0] - .6, yrs[-1] + .6)
        ax.set_ylabel("number of regions")
        ax.set_title("(b) Regions needed to reach half of European employment",
                     loc="left", fontsize=11.5)
        chg = n[-1] - n[0]
        ax.text(.5, .045,
                ("unchanged over the window" if chg == 0 else
                 f"{chg:+.0f} region(s) \u2014 "
                 + ("more spread out" if chg > 0 else "more concentrated")),
                transform=ax.transAxes, ha="center", fontsize=9.8,
                fontweight="bold",
                color="#6E6E6E" if chg == 0 else ("#14746F" if chg > 0 else "#B03A2E"))
        _int_years(ax, yrs)
    else:
        ax.axis("off")

    foot = ("Panel (b): a step upward means the industry needs MORE places to "
            "make up half of itself \u2014 i.e. it is spreading out.")
    if panel is not None:
        n = int(c["n_regions"].iloc[-1]) if "n_regions" in c.columns else None
        foot = panel_description(panel, int(c["year"].min()),
                                 int(c["year"].max()), n) + "\n" + foot
    _foot(fig, foot, y=-0.01)
    fig.subplots_adjust(top=.835, bottom=.14, wspace=.26)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def lab_top(col: str) -> str:
    """'top10_pct' -> 'top 10'."""
    import re as _re
    m = _re.search(r"top(\d+)", col)
    return f"top {m.group(1)}" if m else col


def fig_concentration(conc: pd.DataFrame, conc_bloc: Optional[pd.DataFrame] = None,
                      panel: Optional[pd.DataFrame] = None,
                      save: Optional[str] = None):
    """
    Deprecated wrapper. Section 6 is now two figures: `fig_hhi` (index, overall
    and by bloc) and `fig_concentration_detail` (shares and region counts).
    """
    warnings.warn("fig_concentration is superseded by fig_hhi + "
                  "fig_concentration_detail; returning fig_hhi only.",
                  DeprecationWarning, stacklevel=2)
    return fig_hhi(conc, conc_bloc, panel=panel, save=save)


def fig_lorenz(panel: pd.DataFrame, years: Tuple[int, ...] = (2012, 2023),
               save: Optional[str] = None):
    """
    Lorenz curves and Gini — how unequally employment is spread across regions.

    Reads directly: "the smallest 60% of regions hold only x% of European
    automotive employment". The 45-degree line is perfect equality; the further
    the curve bows away from it, the more the industry sits in a few places.
    """
    import matplotlib.pyplot as plt
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    fig.suptitle("Inequality of automotive employment across regions "
                 "(Lorenz curve and Gini)",
                 fontsize=14.5, fontweight="bold", x=0.02, ha="left")

    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls="--", color="#999999", lw=1.3, label="perfect equality")
    cols = ["#8FB8B4", "#14746F", "#0B3B39"]
    for y, c in zip(years, cols):
        lz = lorenz_points(panel, y)
        if not len(lz):
            continue
        ax.plot(lz["p_regions"], lz["p_employment"], lw=2.6, color=c,
                label=f"{y}  (Gini {lz.attrs['gini']:.3f})")
        ax.fill_between(lz["p_regions"], lz["p_employment"], lz["p_regions"],
                        color=c, alpha=.10, lw=0)
    ax.set_xlabel("cumulative share of regions (smallest first)")
    ax.set_ylabel("cumulative share of employment")
    ax.set_title("(a) Lorenz curve — all regions", loc="left")
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.annotate("the further the curve bows\naway from this line, the more\n"
                "concentrated the industry",
                xy=(.55, .18), fontsize=8.6, color="#6E6E6E")

    ax = axes[1]
    rows = []
    for b in [x for x in BLOC_ORDER if x in set(panel["bloc"])]:
        for y in years:
            lz = lorenz_points(panel, y, bloc=b)
            if len(lz):
                rows.append({"bloc": b, "year": y, "gini": lz.attrs["gini"],
                             "n": lz.attrs["n_regions"]})
    g = pd.DataFrame(rows)
    if len(g):
        blocs = [x for x in BLOC_ORDER if x in set(g["bloc"])]
        x = np.arange(len(blocs)); w = 0.36
        for k, y in enumerate(years):
            vals = [g[(g.bloc == b) & (g.year == y)]["gini"].iloc[0] for b in blocs]
            bars = ax.bar(x + (k - (len(years) - 1) / 2) * w, vals, w, label=str(y),
                          color=cols[k])
            for bb, v in zip(bars, vals):
                ax.annotate(f"{v:.2f}", (bb.get_x() + bb.get_width() / 2, v),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([b.replace("_", " ") + f"\n(n={g[g.bloc==b]['n'].iloc[0]})"
                            for b in blocs], fontsize=9)
        ax.set_ylabel("Gini coefficient")
        ax.set_title("(b) Inequality WITHIN each bloc", loc="left")
        ax.legend(fontsize=9, frameon=False)
        ax.set_ylim(0, max(g["gini"]) * 1.28)

    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    _foot(fig, "Gini 0 = every region has equal employment; 1 = one region "
               "has all of it.  " + panel_description(panel), y=-0.02)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def fig_unit_size(us: pd.DataFrame, save: Optional[str] = None):
    """Average establishment size by bloc — consolidation vs supplier tissue."""
    import matplotlib.pyplot as plt
    _style()
    g = (us.groupby(["year", "bloc"])
           .apply(lambda x: x["employment"].sum() / x["units"].sum(),
                  include_groups=False)
           .reset_index(name="avg_size"))
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for b in [b for b in BLOC_ORDER if b in set(g["bloc"])]:
        s = g[g["bloc"] == b].sort_values("year")
        ax.plot(s["year"], s["avg_size"], lw=2.2, marker="o", ms=3.5,
                color=BLOC_COLOURS[b], label=b.replace("_", " "))
    ax.set_title("Average establishment size (persons per local unit), NACE C29",
                 loc="left", fontsize=12, fontweight="bold")
    ax.set_ylabel("employees per local unit"); ax.legend(fontsize=8, frameon=False)
    _foot(fig, "A rising line means employment is consolidating into larger "
               "establishments; a falling one means it is fragmenting towards a "
               "denser tissue of smaller suppliers.", y=-0.02)
    fig.tight_layout()
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


# Explanatory notes used to be printed under each figure. They now live in
# the LaTeX captions, which lets the chart itself print larger. The text is
# still passed to _foot and kept on the figure as `fig._foot_text`; set this
# to True to draw it again.
SHOW_FOOTNOTES = False


def _foot(fig, text: str, width: int = 110, y: float = -0.01,
          size: float = 8.5) -> None:
    """
    Figure footnote, hard-wrapped.

    `bbox_inches="tight"` grows the canvas to fit the widest artist, so a long
    unwrapped caption silently stretches the saved PNG to several times its
    intended width. Wrapping is what keeps the figure the size it looks.
    """
    fig._foot_text = str(text)
    if not SHOW_FOOTNOTES:
        return
    import textwrap
    lines = []
    for para in str(text).split("\n"):
        lines.extend(textwrap.wrap(para, width=width) or [""])
    fig.text(0.02, y, "\n".join(lines), fontsize=size, color="#5A5A5A",
             ha="left", va="top")


def _int_years(ax, years) -> None:
    """Year axes should show years, not 2007.5."""
    y = sorted(int(v) for v in years)
    step = 1 if len(y) <= 8 else (2 if len(y) <= 16 else 3)
    ax.set_xticks(y[::step])
    ax.set_xticklabels([str(v) for v in y[::step]])


def fig_france_urssaf(save: Optional[str] = None, level: str = "region"):
    """
    France on its own terms (RULE 1): URSSAF C29 by région, plus the C27
    two-level read that bounds the electrification proxy.
    """
    import matplotlib.pyplot as plt
    _style()
    c29 = fd.urssaf_panel("C29", "all", level)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    fig.suptitle("France (URSSAF) \u2014 automotive employment (NACE C29) "
                 "and the electrical sector (NACE C27)",
                 fontsize=14, fontweight="bold", x=0.02, ha="left")

    ax = axes[0]
    piv = c29.pivot_table(index="year", columns="region", values="employees", aggfunc="sum")
    top = piv.mean().sort_values(ascending=False).head(6).index
    for r in top:
        ax.plot(piv.index, piv[r] / 1000, lw=2, marker="o", ms=3, label=str(r)[:28])
    ax.set_title("(a) Automotive employment (NACE C29), six largest régions",
                 loc="left")
    ax.set_ylabel("salaried employment (thousands)")
    ax.legend(fontsize=7.5, frameon=False)
    _int_years(ax, piv.index)

    ax = axes[1]
    # The C27 aggregate (~110-140k) is 20x the battery series and flattened it,
    # so it is reported in the caption instead of plotted.
    agg = fd.urssaf_panel("C27", "all", "national").set_index("year")["employees"]
    for scope, lbl, col in (("narrow", "2711Z + 2720Z  motors + batteries", "#2C6FB3"),
                            ("battery", "2720Z  batteries only", "#C6803D")):
        ser = fd.urssaf_panel("C27", scope, "national").set_index("year")["employees"]
        ax.plot(ser.index, ser.values / 1000, lw=2.6, marker="o", ms=4, label=lbl, color=col)
    bat = fd.urssaf_panel("C27", "battery", "national").set_index("year")["employees"]
    lo_yr = int(bat.idxmin())
    ax.annotate(f"trough {lo_yr}", (lo_yr, bat.loc[lo_yr] / 1000), xytext=(6, -14),
                textcoords="offset points", fontsize=8, color="#C6803D")
    ax.set_title("(b) Electrification-relevant classes within NACE C27",
                 loc="left")
    ax.set_ylabel("salaried employment (thousands)")
    ax.legend(fontsize=8.5, frameon=False)
    _int_years(ax, bat.index)
    ax.margins(y=.14)
    a0, a1 = int(agg.index.min()), int(agg.index.max())
    _foot(fig,
             f"Panel (b): the whole NACE C27 aggregate runs "
             f"{agg.loc[a0]/1000:.0f}k \u2192 {agg.loc[a1]/1000:.0f}k "
             f"({100*(agg.loc[a1]-agg.loc[a0])/agg.loc[a0]:+.0f}%) over the same "
             "years. It FALLS while batteries rise, and at roughly twenty times "
             "their scale it would flatten them off the axis, so it is reported "
             "here rather than plotted.", y=-0.02)

    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def map_change_choropleth(panel: pd.DataFrame, y0: int, y1: int,
                          relative: bool = True, save: Optional[str] = None,
                          title: str = "", level: Optional[int] = None):
    """
    Interactive choropleth of change between two years (diverging scale).
    Only balanced regions are shown; the rest stay blank, never zero.
    """
    import plotly.express as px

    b = balanced_subset(panel, y0, y1)
    w = b.pivot_table(index=["geo_code", "geo_label", "bloc"],
                      columns="year", values="value", aggfunc="first").reset_index()

    # geometry MUST match the panel's NUTS level, or nothing is drawn
    if level is None:
        level = fd.infer_nuts_level(w["geo_code"])
    geo = fd.load_nuts_geometry(level=level)
    if geo is None:
        raise RuntimeError("NUTS geometry unavailable; see fd.load_nuts_geometry().")
    fd._warn_unmatched(w["geo_code"], geo, level)
    w["change"] = w[y1] - w[y0]
    w["change_pct"] = 100 * w["change"] / w[y0]
    col = "change_pct" if relative else "change"
    lim = float(np.nanpercentile(np.abs(w[col]), 95))

    common = dict(
        geojson=geo, locations="geo_code", featureidkey="properties.NUTS_ID",
        color=col, color_continuous_scale="RdBu", range_color=(-lim, lim),
        hover_name="geo_label",
        hover_data={"change": ":,.0f", "change_pct": ":.1f", "bloc": True},
        zoom=3.1, center={"lat": 50.0, "lon": 10.0}, opacity=0.8,
        labels={col: "% change" if relative else "change (persons)"},
        title=title or f"Change in C29 employment, {y0}–{y1}",
    )
    if hasattr(px, "choropleth_map"):
        fig = px.choropleth_map(w, map_style="carto-positron", **common)
    else:                                                       # pragma: no cover
        fig = px.choropleth_mapbox(w, mapbox_style="carto-positron", **common)
    fig.update_layout(margin=dict(l=0, r=0, t=45, b=0), height=720)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save)
    return fig


# ==============================================================================
# 6. MAPS
# ==============================================================================
#
# Three commitments, all requested and all visible in the output:
#
#   1. FRANCE IS ALWAYS DRAWN FROM URSSAF AT NUTS-3 (DÉPARTEMENT) LEVEL.
#      Départements are the finest level URSSAF supports and the level at which
#      French automotive employment is actually observed. This makes the French
#      part of every map finer-grained than the NUTS-2 used elsewhere, so France
#      gets its own colour ramp and the legend states the difference. The two
#      shadings are never presented as one measurement.
#
#   2. COUNTRY BORDERS ARE DRAWN THICKER THAN REGION BORDERS, so a national
#      boundary is distinguishable from an internal NUTS boundary at a glance.
#
#   3. EVERY MAP CARRIES A LEGEND that says what is shaded, from which source,
#      at which level, and what the blanks mean.

# Commune coordinates for the French battery map. URSSAF gives INSEE commune
# codes but no geometry, so positions for the labelled communes are supplied
# here. Only communes in this table get a bubble; ALL communes are counted in
# the département shading, so nothing is lost from the map itself.
_COMMUNE_LATLON: Dict[str, Tuple[float, float]] = {
    "Billy-Berclau": (50.5147, 2.8544), "Bruges": (44.8756, -0.6122),
    "Bordeaux": (44.8378, -0.5792), "Arras": (50.2910, 2.7778),
    "Poitiers": (46.5802, 0.3404), "Grenoble": (45.1885, 5.7245),
    "Nersac": (45.6167, 0.0667), "Ergué-Gabéric": (47.9833, -4.0333),
    "Bourbourg": (50.9500, 2.1936), "Lille": (50.6292, 3.0573),
    "Lambres-lez-Douai": (50.3567, 3.0700), "Sélestat": (48.2594, 7.4544),
    "Bourges": (47.0810, 2.3987), "Chasseneuil-du-Poitou": (46.6500, 0.3667),
    "Gennevilliers": (48.9333, 2.3000), "Dardilly": (45.8000, 4.7500),
    "Sarreguemines": (49.1108, 7.0678), "Levallois-Perret": (48.8944, 2.2872),
    "Rueil-Malmaison": (48.8760, 2.1810), "Aix-en-Provence": (43.5297, 5.4474),
    "Bouguenais": (47.1667, -1.6167), "Lempdes": (45.7833, 3.1833),
    "Genas": (45.7333, 5.0000), "Amiens": (49.8942, 2.2957),
    "Monteux": (44.0333, 4.9833), "Courbevoie": (48.8975, 2.2564),
    "Dunkerque": (51.0344, 2.3768), "Aiton": (45.5500, 6.2833),
    "Vieux-Thann": (47.8167, 7.0833), "Guebwiller": (47.9078, 7.2117),
}

# Battery sites identified from PUBLIC KNOWLEDGE, not from the data. APE 2720Z
# cannot distinguish cell chemistries, so this table is what lets the map say
# anything about EV relevance at all — and it is labelled as external evidence
# everywhere it is used. `kind` drives the symbol:
#   'traction'  — lithium-ion cells for vehicle traction (the EV signal proper)
#   'liion_ind' — lithium-ion, but industrial/defence rather than automotive
_BATTERY_SITES: Dict[str, Tuple[str, str]] = {
    "Billy-Berclau":     ("traction",  "ACC gigafactory (Stellantis · Mercedes · TotalEnergies)"),
    "Bourbourg":         ("traction",  "Verkor gigafactory, Dunkirk"),
    "Lambres-lez-Douai": ("traction",  "Envision AESC, Renault ElectriCity"),
    "Grenoble":          ("traction",  "Verkor headquarters and pilot line"),
    "Bruges":            ("liion_ind", "Saft (TotalEnergies) — industrial Li-ion"),
    "Bordeaux":          ("liion_ind", "Saft (TotalEnergies) — historic site"),
    "Dunkerque":         ("traction",  "Verkor / ProLogium area"),
}


def _country_border_geojson(fallback_geo: Optional[dict] = None) -> Optional[dict]:
    """
    Country outlines for the heavy border layer.

    Prefers the real NUTS-0 file. If that is unavailable (offline, no cache),
    it dissolves country outlines out of `fallback_geo` by keeping only the
    polygon edges that appear ONCE: an edge shared by two adjacent regions of
    the same country appears twice and is dropped, leaving the national
    outline. GISCO geometry is topologically clean, so the shared vertices
    match exactly and this works without shapely.
    """
    geo0 = fd.load_nuts_geometry(level=0)
    if geo0 is not None:
        return geo0
    if fallback_geo is None:
        return None

    def rings(geom):
        if geom is None:
            return []
        t, c = geom.get("type"), geom.get("coordinates") or []
        if t == "Polygon":
            return list(c)
        if t == "MultiPolygon":
            return [r for poly in c for r in poly]
        return []

    by_country: Dict[str, Counter] = defaultdict(Counter)
    for f in fallback_geo.get("features", []):
        cc = str(f["properties"].get("NUTS_ID", ""))[:2]
        for ring in rings(f.get("geometry")):
            pts = [(round(x, 6), round(y, 6)) for x, y in ring]
            for a, b in zip(pts, pts[1:]):
                by_country[cc][(a, b) if a <= b else (b, a)] += 1

    feats = []
    for cc, edges in by_country.items():
        segs = [[list(a), list(b)] for (a, b), n in edges.items() if n == 1]
        if segs:
            feats.append({"type": "Feature", "properties": {"CNTR": cc},
                          "geometry": {"type": "MultiLineString", "coordinates": segs}})
    if not feats:
        return None
    warnings.warn("NUTS-0 geometry unavailable; country borders dissolved from "
                  "the region geometry. Outlines are correct but unsmoothed.")
    return {"type": "FeatureCollection", "features": feats}


def _add_country_borders(m, fallback_geo: Optional[dict] = None) -> bool:
    """Heavy national outlines — commitment (2). Returns True if drawn."""
    import folium
    geo = _country_border_geojson(fallback_geo)
    if geo is None:
        warnings.warn("No country-border layer could be built; national and "
                      "regional boundaries will look identical on this map.")
        return False
    folium.GeoJson(geo, name="country borders (heavy)",
                   style_function=lambda f: {"fillOpacity": 0, "color": "#1a1a1a",
                                             "weight": 2.1, "opacity": .9}).add_to(m)
    return True


def _legend(m, html: str) -> None:
    import folium
    m.get_root().html.add_child(folium.Element(
        "<div style='position:fixed;bottom:22px;left:22px;z-index:9999;background:"
        "rgba(255,255,255,.96);padding:11px 13px;border:1px solid #b9b9b9;"
        "border-radius:5px;font:11.5px/1.55 -apple-system,Segoe UI,sans-serif;"
        "max-width:315px;box-shadow:0 1px 5px rgba(0,0,0,.18)'>" + html + "</div>"))


def _sw(colour: str, border: str = "") -> str:
    b = f";border:1px solid {border}" if border else ""
    return (f"<span style='display:inline-block;width:13px;height:11px;"
            f"background:{colour};vertical-align:-1px{b}'></span>")


# ------------------------------------------------------------------ Europe map

def _with_values(geo: dict, values: Dict[str, float], key: str = "employment",
                 unit: str = "employees", keep=None) -> dict:
    """
    Copy a GeoJSON and write a formatted value into each feature's properties.

    Folium's GeoJsonTooltip can only show what is IN the properties, so the
    number has to be attached to the geometry before the layer is built —
    otherwise hovering shows the region name and nothing else.
    """
    feats = []
    for f in geo.get("features", []):
        nid = str(f["properties"].get("NUTS_ID", ""))
        if keep is not None and not keep(nid):
            continue
        v = values.get(nid)
        p = dict(f["properties"])
        p[key] = (f"{v:,.0f} {unit}" if v is not None and pd.notna(v)
                  else "no value (missing or confidential)")
        feats.append({"type": "Feature", "properties": p,
                      "geometry": f.get("geometry")})
    return {"type": "FeatureCollection", "features": feats}


def map_employment_europe(year: int = 2023, nace: str = "C29",
                          show_plants: bool = True,
                          save: Optional[str] = None):
    """
    Regional automotive employment across Europe, at mixed and labelled
    resolution: NUTS-2 from Eurostat everywhere except France, NUTS-3
    départements from URSSAF for France (RULE 1 — Eurostat's French regional
    figures are not used anywhere in Module 01).
    """
    import folium
    import branca.colormap as cm

    geo2 = fd.load_nuts_geometry(level=2)
    if geo2 is None:
        raise RuntimeError("NUTS-2 geometry unavailable — see fd.load_nuts_geometry(). "
                           "Download it once while online; it is then cached.")
    geo3 = fd.load_nuts_geometry(level=3)

    e = fd.eurostat_panel(nace, "employment", nuts_level=2)
    vals2 = (e[(e["year"] == year) & e["value"].notna()]
             .set_index("geo_code")["value"].to_dict())
    dropped_fr = sum(1 for k in vals2 if str(k).startswith("FR"))
    vals2 = {k: v for k, v in vals2.items() if not str(k).startswith("FR")}

    dep = fd.urssaf_panel(nace, "all", "departement")
    dep = dep[dep["year"] == year]
    dep_vals = dict(zip(dep["departement_code"].astype(str).str.strip()
                        .str.replace(r"\.0$", "", regex=True).str.zfill(2),
                        dep["employees"]))

    dep2nuts = fd.french_departement_to_nuts3(geo3) if geo3 is not None else {}
    # NUTS-3 id -> employment, via the NAME-based crosswalk (never code surgery)
    nuts3_vals = {nid: dep_vals[dc] for dc, nid in dep2nuts.items()
                  if dc in dep_vals and pd.notna(dep_vals[dc])}

    fr_fallback = geo3 is None or not nuts3_vals
    if fr_fallback:                       # keep France on the map at NUTS-2
        warnings.warn("NUTS-3 geometry unavailable — France falls back to NUTS-2 "
                      "(still URSSAF, so RULE 1 holds, but coarser than intended).")
        fr2 = fd.french_regional_totals(nace, "all", level=2)
        fr2 = fr2[fr2["year"] == year]
        vals2.update(dict(zip(fr2["geo_code"], fr2["value"])))

    allv = [v for v in list(vals2.values()) + list(nuts3_vals.values())
            if pd.notna(v)]
    vmax = float(np.nanpercentile(allv, 97)) if allv else 1.0
    scale2 = cm.linear.YlGnBu_09.scale(0, vmax)
    scale3 = cm.linear.BuPu_09.scale(0, max(vmax * .5, 1))

    m = folium.Map(location=[50.2, 9.5], zoom_start=4, tiles="cartodbpositron")

    def style2(feat):
        v = vals2.get(feat["properties"]["NUTS_ID"])
        has = v is not None and pd.notna(v)
        return {"fillColor": scale2(v) if has else "#f0f0f0",
                "fillOpacity": .85 if has else .30,
                "color": "#8e959c", "weight": .45}

    folium.GeoJson(_with_values(geo2, vals2), style_function=style2,
                   name=f"NUTS-2 · {nace} employment (Eurostat)",
                   highlight_function=lambda f: {"weight": 1.8, "color": "#111111"},
                   tooltip=folium.GeoJsonTooltip(
                       fields=["NAME_LATN", "NUTS_ID", "employment"],
                       aliases=["region", "NUTS-2", f"{nace} employment {year}"],
                       sticky=True)).add_to(m)

    n_fr = 0
    if not fr_fallback:
        fr3 = _with_values(geo3, nuts3_vals,
                           keep=lambda nid: nid.startswith("FR"))

        def style3(feat):
            v = nuts3_vals.get(str(feat["properties"]["NUTS_ID"]))
            has = v is not None and pd.notna(v) and v > 0
            return {"fillColor": scale3(v) if has else "#f7f7f7",
                    "fillOpacity": .88 if has else .32,
                    "color": "#a99ac4", "weight": .45}

        n_fr = sum(1 for v in nuts3_vals.values() if pd.notna(v) and v > 0)
        folium.GeoJson(fr3, style_function=style3,
                       name="France · NUTS-3 départements (URSSAF)",
                       highlight_function=lambda f: {"weight": 1.8,
                                                     "color": "#111111"},
                       tooltip=folium.GeoJsonTooltip(
                           fields=["NAME_LATN", "NUTS_ID", "employment"],
                           aliases=["département", "NUTS-3",
                                    f"{nace} employment {year} (URSSAF)"],
                           sticky=True)).add_to(m)

    borders = _add_country_borders(m, fallback_geo=geo2)

    n_plants = 0
    if show_plants:
        cw = fd.plant_nuts_crosswalk().dropna(subset=["lat", "lon"])
        ep = fd.employment_panel()
        ep = ep[(ep["year"] == year) & ep["employees"].notna()]
        emp_by = dict(zip(ep["plant_uid"], ep["employees"]))
        for _, r in cw.iterrows():
            emp = emp_by.get(r["plant_uid"])
            if emp is None or not np.isfinite(emp):
                continue
            n_plants += 1
            folium.CircleMarker(
                [r["lat"], r["lon"]],
                radius=float(np.clip(np.sqrt(float(emp)) / 11, 3, 15)),
                color="#111111", weight=1.1, fill=True, fill_color="#ffffff",
                fill_opacity=.85,
                tooltip=f"{r['plant_name']} — {float(emp):,.0f}",
                popup=folium.Popup(html=(
                    f"<b>{r['plant_name']}</b><br>{r['city']}, {r['country_iso2']}"
                    f"<br>group: {r['group']}<br>employees: {float(emp):,.0f}"),
                    max_width=300)).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    _legend(m, (
        f"<b>Automotive employment (NACE {nace}), {year}</b><br>"
        f"{_sw('#225ea8')} rest of Europe — <b>NUTS-2</b>, Eurostat<br>"
        f"{_sw('#8c6bb1')} France — <b>NUTS-3 départements</b>, URSSAF"
        f"{' <i>(NUTS-2 fallback)</i>' if fr_fallback else ''}<br>"
        f"{_sw('#f0f0f0', '#ccc')} no value — missing or confidential<br>"
        "<span style='display:inline-block;width:11px;height:11px;border:1.5px "
        "solid #111;border-radius:50%;background:#fff;vertical-align:-1px'></span> "
        "tracked plant, area &prop; employment<br>"
        "<hr style='margin:6px 0;border:0;border-top:1px solid #ddd'>"
        + ("<b>Thick lines are country borders</b>; thin lines are regions.<br>"
           if borders else "")
        + "<i>France is shown at a finer level and from a different source, so "
          "its shading is not directly comparable with the rest of the map. "
          "Eurostat's French regional data are not used anywhere in this study.</i>"))

    print(f"Europe map · {len(vals2)} NUTS-2 regions shaded from Eurostat"
          + (f" ({dropped_fr} French Eurostat regions dropped here by RULE 1)"
             if dropped_fr else " (RULE 1: French Eurostat data already excluded "
             "upstream by fd.eurostat_panel)"))
    print(f"           · {n_fr} French départements shaded from URSSAF")
    print(f"           · {n_plants} tracked plants placed"
          f" · country borders: {'yes' if borders else 'NO'}")
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        m.save(save)
    return m


# ----------------------------------------------------------- France batteries

def map_battery_france(year: int = 2024, ref_year: int = 2019,
                       save: Optional[str] = None):
    """
    Battery manufacturing in France — URSSAF APE 2720Z.

    WHAT THIS MAP CAN AND CANNOT SAY
    --------------------------------
    APE 2720Z is "fabrication de piles et d'accumulateurs électriques". It does
    NOT separate lithium-ion traction cells from lead-acid starter batteries or
    industrial storage, so the activity class ALONE cannot identify EV
    production. Read on its own, a 2720Z map is a battery map, not an EV map.

    Two things make it informative anyway:

      * GROWTH SINCE `ref_year`. The class was flat and mature before 2019;
        the post-2019 increase is overwhelmingly the traction-cell build-out,
        because that is what was actually being built. Communes appearing from
        nothing are new plants, not expanded lead-acid lines.
      * NAMED SITES. `_BATTERY_SITES` identifies the known cell plants from
        public knowledge and distinguishes automotive traction cells from
        industrial lithium-ion. That evidence is external to the data and is
        labelled as such in the legend and in every popup.

    A European equivalent is NOT possible: Eurostat regional SBS publishes C27
    only as an aggregate, with no four-digit breakdown, so no other country can
    be resolved to battery activity at all.
    """
    import folium

    raw = pd.read_excel(fd.CONFIG.path(fd.CONFIG.f_urssaf_c27))
    b = raw[raw["Code APE"] == "2720Z"].copy()
    c_now, c_ref = f"Effectifs salariés {year}", f"Effectifs salariés {ref_year}"
    for c in (c_now, c_ref):
        if c not in b.columns:
            raise KeyError(f"{c!r} not in the URSSAF C27 file — check the year.")
        b[c] = pd.to_numeric(b[c], errors="coerce").fillna(0)
    b["growth"] = b[c_now] - b[c_ref]
    b["dep"] = b["Code département"].astype(str).str.strip().str.zfill(2)
    live = b[b[c_now] > 0].sort_values(c_now, ascending=False)

    m = folium.Map(location=[46.7, 2.4], zoom_start=6, tiles="cartodbpositron")

    geo3 = fd.load_nuts_geometry(level=3)
    dep_tot = live.groupby("dep")[c_now].sum().to_dict()
    n_dep = 0
    if geo3 is not None:
        dep2nuts = fd.french_departement_to_nuts3(geo3)
        nuts3_tot = {nid: dep_tot[dc] for dc, nid in dep2nuts.items()
                     if dc in dep_tot}
        vmax = max(nuts3_tot.values()) if nuts3_tot else 1.0

        def style(feat):
            nid = str(feat["properties"]["NUTS_ID"])
            if not nid.startswith("FR"):
                return {"fillOpacity": 0, "weight": 0, "color": "#00000000"}
            v = nuts3_tot.get(nid, 0.0)
            return {"fillColor": "#6A4C93",
                    "fillOpacity": .07 + .63 * min(v / vmax, 1.0) if v else .05,
                    "color": "#ada0c4", "weight": .5}

        n_dep = sum(1 for v in nuts3_tot.values() if v > 0)
        folium.GeoJson(_with_values(geo3, nuts3_tot,
                                    keep=lambda nid: nid.startswith("FR")),
                       style_function=style,
                       name="battery employment by département (URSSAF NUTS-3)",
                       highlight_function=lambda f: {"weight": 1.8,
                                                     "color": "#111111"},
                       tooltip=folium.GeoJsonTooltip(
                           fields=["NAME_LATN", "NUTS_ID", "employment"],
                           aliases=["département", "NUTS-3",
                                    f"battery employment {year}"],
                           sticky=True)).add_to(m)
    else:
        warnings.warn("NUTS-3 geometry unavailable — the département choropleth "
                      "is omitted; only commune bubbles are drawn.")

    _add_country_borders(m, fallback_geo=geo3)

    KIND_COLOUR = {"traction": "#0B6E4F", "liion_ind": "#1F6FB2"}
    plotted = 0.0
    for _, r in live.iterrows():
        town = str(r["Nom commune"])
        if town not in _COMMUNE_LATLON:
            continue
        lat, lon = _COMMUNE_LATLON[town]
        emp, grow = float(r[c_now]), float(r["growth"])
        plotted += emp
        kind, note = _BATTERY_SITES.get(town, (None, None))
        new_site = r[c_ref] == 0 and emp > 0
        fill = KIND_COLOUR.get(kind, "#C6803D" if grow > 0 else "#9AA0A6")
        folium.CircleMarker(
            [lat, lon], radius=float(np.clip(np.sqrt(emp) / 2.6, 4, 22)),
            color="#111111" if kind else "#666666",
            weight=2.4 if kind else 1.0,
            fill=True, fill_color=fill, fill_opacity=.85,
            tooltip=f"{town} — {emp:,.0f} employees ({grow:+,.0f} since {ref_year})",
            popup=folium.Popup(html=(
                f"<b>{town}</b> — {r['Département']}<br>"
                f"battery employment {year}: <b>{emp:,.0f}</b><br>"
                f"{ref_year}: {float(r[c_ref]):,.0f} &nbsp;&rarr;&nbsp; "
                f"change <b>{grow:+,.0f}</b>"
                + ("<br><span style='color:#0B6E4F'><b>site new since "
                   f"{ref_year}</b></span>" if new_site else "")
                + (f"<hr style='margin:5px 0'><b>{note}</b><br>"
                   "<small>identified from public knowledge, <i>not</i> from "
                   "APE 2720Z</small>" if kind else
                   "<hr style='margin:5px 0'><small>APE 2720Z does not "
                   "distinguish EV cells from lead-acid or industrial "
                   "batteries</small>")), max_width=340)).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    tot_now, tot_ref = live[c_now].sum(), b[c_ref].sum()
    pct = 100 * tot_now / tot_ref - 100 if tot_ref else float("nan")
    _legend(m, (
        f"<b>Battery manufacturing in France, {year}</b><br>"
        "URSSAF, APE <b>2720Z</b> — <i>piles et accumulateurs électriques</i><br>"
        f"{_sw('#6A4C93')} département total (NUTS-3 shading)<br>"
        "<b>bubble = commune</b>, area &prop; employment:<br>"
        f"&nbsp;{_sw('#0B6E4F')} known <b>traction-cell</b> plant<br>"
        f"&nbsp;{_sw('#1F6FB2')} known industrial Li-ion plant<br>"
        f"&nbsp;{_sw('#C6803D')} grew since {ref_year}, type unidentified<br>"
        f"&nbsp;{_sw('#9AA0A6')} flat or declining<br>"
        "<hr style='margin:6px 0;border:0;border-top:1px solid #ddd'>"
        f"National total <b>{tot_ref:,.0f}</b> ({ref_year}) &rarr; "
        f"<b>{tot_now:,.0f}</b> ({year}), <b>{pct:+.0f}%</b>.<br>"
        "<i>2720Z covers all batteries including lead-acid, so the class alone "
        "cannot identify EV production. The growth is the signal: the named "
        "cell plants are identified from public knowledge, not from the data.</i>"))

    print(f"France battery map · {len(live)} communes with employment in {year}; "
          f"{n_dep} départements shaded")
    print(f"                   · bubbles cover {plotted:,.0f} of {tot_now:,.0f} "
          f"employees ({100*plotted/tot_now:.0f}% of the national total)")
    print(f"                   · national {tot_ref:,.0f} ({ref_year}) -> "
          f"{tot_now:,.0f} ({year}), {pct:+.0f}%")
    new = live[(live[c_ref] == 0) & (live[c_now] > 100)]
    if len(new):
        print(f"                   · {len(new)} sites new since {ref_year} with "
              f">100 employees: "
              + ", ".join(f"{t} ({v:,.0f})" for t, v
                          in zip(new['Nom commune'], new[c_now])))
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        m.save(save)
    return m
