# -*- coding: utf-8 -*-
"""
================================================================================
 MODULE 03 — PRODUCT STRUCTURAL CHANGE (part 1: model mix and electrification)
================================================================================

WHY THIS MODULE IS SPLIT
------------------------
The thesis's distinctive contribution is bringing VEHICLE SIZE into a structural
-change reading of automotive employment. That requires per-model mass, and the
attribute source currently in the dataset cannot support it:

    tbl_ademe_variants holds 1,890 rows / 80 distinct (brand, model) pairs
    across 18 brands. For Mercedes it contains only E 200/300/400/E 53 AMG;
    for Audi only A3, e-tron GT, Q3; for VW only Caddy and Touran.

Consequences, measured:
  * 15% of plant-model-years carry attributes; only 12% of the model mix in the
    balanced German employment universe.
  * The match rate is severely uneven by brand -- Mercedes 1/137, VW 3/39,
    Audi 8/53, against Porsche 12/22 and Renault 18/37.
  * The unmatched set contains the most-produced cars in Europe: Golf, C-Class,
    Passat, Polo, Tiguan, Sprinter, Cayenne, A1, TT.

A mean kerb weight computed on that sample would track WHICH BRANDS HAPPENED TO
MATCH rather than the production mix; it could show weight falling for a plant
that merely switched from an unmatched Golf to a matched Polo. So mass-based
analysis is deferred to Module 03b, once the EEA CO2 monitoring database
(Regulation (EU) 2019/631 -- EU-wide, per-registration, mass in running order,
fuel type, make and commercial name, 2010 onward) replaces the ADEME extract.

WHAT THIS MODULE DOES INSTEAD
-----------------------------
Everything that does NOT depend on mass, and that the model panel supports at
near-full coverage:

  1. Model mix per plant-year: portfolio size, churn (models added/dropped),
     renewal rate, model tenure -- the product-cycle dynamics of each plant.
  2. Multi-sourcing: how many plants build a given model, and how that changes.
  3. ELECTRIFICATION, from nameplates rather than attributes. Dedicated BEV
     nameplates (ID.3, i4, EQS, e-tron, Mégane E-Tech...) are identifiable by
     name at ~full coverage, where ADEME's fuel field reaches only 15%. The
     classifier is hand-verified and validated against the ADEME subset.
  4. The link to employment, descriptively, on the balanced universe.
  5. An attribute-readiness diagnostic -- the figure that justifies the switch
     to EEA, and belongs in the methods chapter.

Author: Magd Shamashan
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import foundation as fd
import eea_crosswalk as xw

import module02_organisational as m2

__all__ = [
    "BEV_PATTERNS",
    "CO2_FIRST_YEAR",
    "MIN_BEV_REGS",
    "NOT_BEV_PATTERNS",
    "attribute_readiness",
    "balance_report",
    "balanced_plants",
    "classify_bev",
    "coverage_report",
    "electrification_by",
    "electrification_vs_mass",
    "employment_vs_mix",
    "exact_mass_plants",
    "fig_attribute_readiness",
    "fig_churn",
    "fig_electrification",
    "fig_electrification_vs_mass",
    "fig_employment_vs_mix",
    "fig_mass_by_bloc",
    "fig_mass_trend",
    "fig_model_mix",
    "fig_multi_sourcing",
    "fig_plant_co2",
    "fig_plant_mass",
    "fig_plant_trajectories",
    "fig_powertrain",
    "fig_weighting_sensitivity",
    "mass_by_bloc",
    "mass_trend",
    "model_churn",
    "model_panel",
    "model_tenure",
    "multi_sourcing",
    "plant_co2",
    "plant_electrification",
    "plant_mass",
    "plant_model_mass",
    "plant_model_mix",
    "plant_trajectories",
    "powertrain_decomposition",
    "validate_bev_classifier",
    "weighting_sensitivity",
]





# ==============================================================================
# 1. BEV CLASSIFIER — nameplate-based, hand-verified
# ==============================================================================
#
# Dedicated battery-electric NAMEPLATES are recognisable by name, which gives
# near-complete coverage where the attribute route gives 15%. Every candidate
# name in the panel was inspected by hand; the traps that motivate the
# exclusion list are real and present in this data:
#
#   "C-Class EQ"              -> EQ Boost MILD HYBRID, not a BEV
#   "GLC mit EQ-Technologie"  -> EQ hybrid technology, not a BEV
#   "Punto EVO"               -> EVO = evoluzione, nothing to do with electric
#   "Macan / Macan S / GTS"   -> ICE; only "Macan 4 Electric" is battery-electric
#   "Electric Vehicle Drive Modules" -> a COMPONENT, not a car
#
# This identifies BEV NAMEPLATES. It is a different quantity from ADEME's
# `energy_fuel`, which is model-level and multi-valued ("Electric; Diesel"
# means the model has both variants). See validate_bev_classifier().

BEV_PATTERNS: Tuple[str, ...] = (
    r"\be-?tron\b",                    # Audi e-tron, Q4/Q8 e-tron, RS e-tron GT
    r"\bID\.?\s?[3-9]\b",              # VW ID.3, ID.4, ID.5, ID.7
    r"\bID\.?\s?Buzz\b",
    r"\bEQ[ABCESV]\b",                 # Mercedes EQA/EQB/EQC/EQE/EQS/EQV
    r"\bi[3-7]\b", r"\biX\b", r"\biX[1-3]\b",   # BMW i3, i4, i5, i7, iX, iX1
    r"\be-Golf\b", r"\be-Up\b", r"\be-Crafter\b",
    r"\bTaycan\b", r"\bMacan\s+\d*\s*Electric\b",
    r"\bZo[ée]\b", r"\bSpring\b", r"\bKangoo\s+Z\.?\s?E\.?", r"\bMaster\s+Z\.?\s?E\.?",
    r"\bE-?Tech\s+Electric\b", r"\bM[ée]gane\s+E-?Tech\b", r"\bSc[ée]nic\s+E-?Tech\b",
    r"\bBorn\b", r"\bEnyaq\b",
    r"\bbZ4X\b", r"\bMach-?E\b",
    r"\bCapri\s+EV\b", r"\bExplorer\s+EV\b",
    r"\bLeaf\b", r"\bAriya\b",
    # --- added after a check against the plant panel: BEV nameplates the
    # --- earlier list missed, each verified against its launch year.
    r"\b500\s?e\b",                    # Fiat 500e (Mirafiori, 2020-)
    r"\be-?Sprinter\b",                # Mercedes eSprinter van
    r"\be-?Crafter\b", r"\be-?TGE\b",  # VW e-Crafter / MAN e-TGE (Września)
    r"SUV\s+[ée]lectrique", r"^[ée]lectrique$",   # Audi Brussels e-tron, recorded in French
)

NOT_BEV_PATTERNS: Tuple[str, ...] = (
    r"EQ[- ]?Technologie", r"EQ\s*Boost", r"C-Class\s+EQ", r"\bEQ\b(?![ABCESV])",
    r"\bEVO\b",                        # Punto EVO
    r"Drive\s+Modules", r"Components?$", r"\bAxles?\b",
    r"Hybrid(?!.*\bElectric\b)", r"Hybride",
    r"\d\s?50\s+e\b",                  # A 250 e / B 250 e / CLA 250 e: plug-in hybrids
    r"\b4xe\b",                        # Jeep Compass / Renegade 4xe: plug-in hybrids
    r"E-?PERFORMANCE",                 # AMG GT / SL 63 S: plug-in hybrids
    r"Rechargeable",                   # Prius rechargeable: plug-in hybrid
)


def classify_bev(model: str) -> bool:
    """True if the nameplate is a dedicated battery-electric model."""
    s = str(model)
    if any(re.search(p, s, re.I) for p in NOT_BEV_PATTERNS):
        return False
    return any(re.search(p, s, re.I) for p in BEV_PATTERNS)


# --- plausibility guard -------------------------------------------------------
# Earliest year each BEV nameplate could plausibly be in series production
# (start of production, not concept or announcement). This exists because the
# source data contains back-projection errors: the press gap-filler assigned a
# plant's CURRENT model list to earlier years, putting the Mercedes EQA at
# Rastatt in 2006-2012 when it launched in 2021, and the BMW iX1 in 2013 when
# it launched in 2022. Without this guard the early BEV share is fictitious.
BEV_FIRST_YEAR: Dict[str, int] = {
    "i3": 2013, "i3 120 Ah": 2018, "i3s": 2017,
    "e-Golf": 2014, "e-Up": 2013,
    "e-tron": 2018, "Q8 e-tron": 2022, "SQ8 e-tron": 2022,
    "SQ8 e-tron Quattro": 2022, "e-tron GT": 2021, "RS e-tron GT": 2021,
    "S e-tron GT": 2024, "Q4 e-tron": 2021, "Q4 e-tron Quattro": 2021,
    "ID.3": 2019, "ID.4": 2020, "ID.5": 2021, "ID. Buzz": 2022,
    "EQA": 2021, "EQB": 2021, "EQC": 2019, "EQE": 2022, "EQS": 2021, "EQV": 2020,
    "i4": 2021, "i4 M50": 2021, "i4 eDrive35": 2022, "i4 eDrive40": 2021,
    "i4 xDrive40": 2022, "i5": 2023, "i7": 2022,
    "iX": 2021, "iX Series": 2021, "iX1": 2022, "iX3": 2020,
    "Taycan": 2019, "Macan 4 Electric": 2024,
    "Born": 2021, "Enyaq": 2020, "Mégane E-Tech": 2022, "Scénic E-Tech": 2024,
    "Zoe": 2012, "Zoé": 2012, "Spring": 2021, "bZ4X": 2022,
    "Kangoo Z.E.": 2011, "Master Z.E.": 2018,
    "500e": 2020, "E-Sprinter": 2019, "eCrafter": 2018, "e-TGE": 2018,
    "électrique": 2018,
    "Capri EV": 2024, "Explorer EV": 2024, "Mustang Mach-E": 2020,
    "Leaf": 2010, "Ariya": 2022,
}


def flag_implausible_bev(mp: pd.DataFrame) -> pd.DataFrame:
    """
    Add `bev_implausible`: a BEV record dated before that nameplate could exist.

    These are source-data back-projection errors, not real production. They are
    FLAGGED rather than silently deleted, and `model_panel()` sets `is_bev` to
    False for them so electrification series are not inflated at the start of
    the window.
    """
    first = mp["model"].map(BEV_FIRST_YEAR)
    mp = mp.copy()
    mp["bev_first_year_known"] = first
    mp["bev_implausible"] = mp["is_bev"] & first.notna() & (mp["year"] < first)
    return mp


def validate_bev_classifier(verbose: bool = True) -> pd.DataFrame:
    """
    Check the nameplate classifier against the ADEME subset that has a fuel field.

    Only PRECISION is checkable, and only in one direction: any nameplate we
    call BEV should, where ADEME covers it, report an Electric variant. Recall
    cannot be assessed -- ADEME's coverage is exactly the gap this classifier
    exists to close.

    Note the concepts differ: ours is "dedicated BEV nameplate", ADEME's
    `energy_fuel` is "fuels available for this model", so a model listed
    "Electric; Diesel" is a nameplate sold in both forms.
    """
    attr = fd.load_model_attributes()
    a = attr[["model", "model_brand", "energy_fuel"]].dropna(subset=["energy_fuel"]).copy()
    a["ademe_has_electric"] = a["energy_fuel"].astype(str).str.contains("Electric", case=False)
    a["we_call_bev"] = a["model"].map(classify_bev)

    checked = a[a["we_call_bev"]]
    agree = int((checked["ademe_has_electric"]).sum())
    if verbose:
        print(f"nameplates we call BEV that ADEME also covers : {len(checked)}")
        print(f"  of which ADEME reports an Electric variant  : {agree}")
        if len(checked):
            print(f"  precision on the overlap                   : {100*agree/len(checked):.0f}%")
        disagree = checked[~checked["ademe_has_electric"]]
        if len(disagree):
            print("  DISAGREEMENTS (inspect):")
            for _, r in disagree.iterrows():
                print(f"    {r['model_brand']} {r['model']}: ADEME says '{r['energy_fuel']}'")
    return a


# ==============================================================================
# 2. THE MODEL PANEL
# ==============================================================================

def model_panel(scope: Sequence[str] = m2.DEFAULT_SCOPE,
                balanced_only: bool = False,
                window: Optional[Tuple[int, int]] = None) -> pd.DataFrame:
    """
    plant x model x year, with BEV flag and plant metadata attached.

    `balanced_only` restricts to the 35-plant imputed universe (for figures that
    are put beside employment); otherwise all plants are kept, since the model
    panel does not suffer the employment panel's coverage asymmetry.
    """
    my = fd.load_model_year()
    pl = fd.load_plants()[["plant_uid", "plant_name", "city", "country_iso2",
                           "group", "brand", "classification"]]
    df = my.drop(columns=[c for c in ("plant_name", "city", "country_iso2",
                                      "group", "brand", "classification")
                          if c in my.columns]).merge(pl, on="plant_uid", how="left")
    if scope:
        df = df[df["classification"].isin(list(scope))]
    if balanced_only:
        keep = set(fd.employment_panel().query("in_imputed_universe")["plant_uid"])
        df = df[df["plant_uid"].isin(keep)]
    if window:
        df = df[(df["year"] >= window[0]) & (df["year"] <= window[1])]
    df["is_bev"] = df["model"].map(classify_bev)
    df = flag_implausible_bev(df)
    n_bad = int(df["bev_implausible"].sum())
    if n_bad:
        warnings.warn(
            f"{n_bad} BEV record(s) pre-date their nameplate's launch "
            f"(source back-projection, e.g. EQA at Rastatt 2006-2012). "
            f"is_bev set to False for these; see `bev_implausible`.")
        df.loc[df["bev_implausible"], "is_bev"] = False
    return df.reset_index(drop=True)


def plant_model_mix(mp: pd.DataFrame,
                    portfolio_scope: Sequence[str] = ("assembly",)) -> pd.DataFrame:
    """
    Portfolio size and BEV content per plant-year.

    WHY COMPONENT PLANTS ARE EXCLUDED BY DEFAULT
    --------------------------------------------
    For an engine or component plant, the model list is not a portfolio — it is
    the set of vehicles the plant's parts go INTO. Mercedes-AMG Affalterbach
    (an engine builder) carries 71 "models", including 'C 63 AMG', 'C36 AMG',
    'CL 55 AMG', 'CLK 63 AMG'…: engine variants of cars assembled elsewhere.
    Counting that as a portfolio put a 71-model, 2,600-employee plant in the
    same scatter as Wolfsburg, and inflated the Mercedes group mean to 11.9
    models against Renault's 2.4 — a difference in bookkeeping, not in industry.

    Restricting to assembly plants closes most of that gap (Mercedes 6.3, VW
    6.7, BMW 5.9) and caps the maximum at a plausible 26 (Audi Neckarsulm).

    RESIDUAL CAVEAT — cross-brand counts are still not strictly comparable.
    Naming granularity differs between sources: some brands appear at nameplate
    level, others at trim/engine level ('A 45', 'A 45 S 4MATIC+', 'RS 3').
    Treat portfolio SIZE as indicative and prefer WITHIN-plant change over time
    to BETWEEN-brand comparison.
    """
    d = mp
    if portfolio_scope:
        excluded = sorted(set(mp["classification"].dropna()) - set(portfolio_scope))
        d = mp[mp["classification"].isin(list(portfolio_scope))]
        if excluded:
            warnings.warn(
                f"plant_model_mix: portfolio metrics restricted to "
                f"{tuple(portfolio_scope)}; excluded {excluded} because their "
                f"model lists are applications, not portfolios "
                f"(e.g. AMG Affalterbach carries 71 engine variants).")
    g = (d.groupby(["plant_uid", "plant_name", "country_iso2", "group",
                    "classification", "year"], as_index=False)
           .agg(n_models=("model", "nunique"),
                n_bev=("is_bev", "sum")))
    g["bev_share"] = np.where(g["n_models"] > 0, g["n_bev"] / g["n_models"], np.nan)
    g.attrs["portfolio_scope"] = tuple(portfolio_scope) if portfolio_scope else ("all",)
    return g


def model_churn(mp: pd.DataFrame) -> pd.DataFrame:
    """
    Models entering and leaving each plant, year over year.

    entries / exits are counted against the previous year's portfolio;
    `renewal_rate` = entries / portfolio size, a product-cycle intensity measure.
    The first year of each plant's series has no predecessor and is dropped.
    """
    sets = (mp.groupby(["plant_uid", "year"])["model"]
              .apply(lambda s: frozenset(s.dropna())).reset_index(name="models"))
    rows = []
    for pid, g in sets.groupby("plant_uid"):
        g = g.sort_values("year")
        prev_year, prev = None, None
        for _, r in g.iterrows():
            if prev is not None and r["year"] == prev_year + 1:
                ent, ex = r["models"] - prev, prev - r["models"]
                rows.append({
                    "plant_uid": pid, "year": int(r["year"]),
                    "n_models": len(r["models"]),
                    "entries": len(ent), "exits": len(ex),
                    "net": len(ent) - len(ex),
                    "renewal_rate": len(ent) / len(r["models"]) if r["models"] else np.nan,
                    "entered": "; ".join(sorted(ent)) if ent else "",
                    "exited": "; ".join(sorted(ex)) if ex else "",
                })
            prev, prev_year = r["models"], int(r["year"])
    out = pd.DataFrame(rows)
    meta = mp[["plant_uid", "plant_name", "country_iso2", "group",
               "classification"]].drop_duplicates("plant_uid")
    return out.merge(meta, on="plant_uid", how="left")


def model_tenure(mp: pd.DataFrame) -> pd.DataFrame:
    """How long each model stays at each plant (span within the window)."""
    t = (mp.groupby(["plant_uid", "plant_name", "country_iso2", "group",
                     "model", "is_bev"], as_index=False)
           .agg(first_year=("year", "min"), last_year=("year", "max"),
                n_years=("year", "nunique")))
    yr_min, yr_max = int(mp["year"].min()), int(mp["year"].max())
    t["left_censored"] = t["first_year"] <= yr_min
    t["right_censored"] = t["last_year"] >= yr_max
    t["censored"] = t["left_censored"] | t["right_censored"]
    return t


def multi_sourcing(mp: pd.DataFrame) -> pd.DataFrame:
    """
    How many plants build each model, per year — a footprint-strategy indicator.
    Rising multi-sourcing means a model is produced in more locations at once
    (flexibility, or a transfer in progress).
    """
    g = (mp.groupby(["model", "year"], as_index=False)
           .agg(n_plants=("plant_uid", "nunique"),
                countries=("country_iso2", lambda s: sorted(set(s))),
                is_bev=("is_bev", "first")))
    g["n_countries"] = g["countries"].map(len)
    return g


def electrification_by(mp: pd.DataFrame, by: str = "country_iso2") -> pd.DataFrame:
    """BEV share of the model portfolio over time, grouped by `by`."""
    g = (mp.groupby([by, "year"], as_index=False)
           .agg(n_models=("model", "nunique"),
                n_bev=("is_bev", "sum"),
                n_plants=("plant_uid", "nunique")))
    g["bev_share_pct"] = 100 * g["n_bev"] / g["n_models"]
    return g


def employment_vs_mix(window: Tuple[int, int] = (2012, 2023)) -> pd.DataFrame:
    """
    Put the model mix beside employment, on the strictly balanced universe.

    DESCRIPTIVE ONLY. These are associations between a plant's portfolio and its
    headcount; nothing here identifies a causal effect, and the thesis does not
    claim one.
    """
    emp = m2.analysis_universe(window=window, strict=True)
    mp = model_panel(balanced_only=True, window=window)
    # assembly-only portfolio (see plant_model_mix): component plants' model
    # lists are applications, not portfolios, so they get no n_models here
    mix = plant_model_mix(mp, portfolio_scope=("assembly",))
    m = emp.merge(
        mix[["plant_uid", "year", "n_models", "n_bev", "bev_share"]],
        on=["plant_uid", "year"], how="left")
    m["emp_per_model"] = m["employees"] / m["n_models"]
    return m


# ==============================================================================
# 3. THE ATTRIBUTE-READINESS DIAGNOSTIC (methods chapter)
# ==============================================================================

def attribute_readiness() -> Dict[str, pd.DataFrame]:
    """
    Quantify why the ADEME extract cannot carry a mass-based analysis, and how
    much a replacement source needs to cover. Returns three frames:

      by_brand   match rate per brand (the bias)
      by_country share of each country's model mix that is attribute-covered
      frontier   cumulative share of plant-model-years covered by the top-N
                 most frequent models -- i.e. the size of the task
    """
    my = fd.load_model_year()
    cov = pd.read_excel(fd.CONFIG.path(fd.CONFIG.f_plants),
                        sheet_name="tbl_ademe_coverage")

    by_brand = (cov.groupby("model_brand")["matched"]
                  .agg(matched="sum", models="count").reset_index())
    by_brand["match_rate_pct"] = 100 * by_brand["matched"] / by_brand["models"]
    by_brand = by_brand.sort_values("models", ascending=False)

    mys = fd.build_all(verbose=False)["model_year_sized"]
    by_country = (mys.groupby("country_iso2")["has_size_attr"]
                    .agg(covered="sum", total="size").reset_index())
    by_country["cov_pct"] = 100 * by_country["covered"] / by_country["total"]
    by_country = by_country.sort_values("cov_pct", ascending=False)

    freq = (my.groupby(["model_brand", "model"]).size()
              .sort_values(ascending=False).rename("pmy").reset_index())
    matched = set(map(tuple, cov[cov["matched"] == True]
                      [["model_brand", "model"]].dropna().values))
    freq["matched"] = [tuple(r) in matched for r in freq[["model_brand", "model"]].values]
    freq["cum_pct"] = 100 * freq["pmy"].cumsum() / freq["pmy"].sum()
    freq["rank"] = np.arange(1, len(freq) + 1)

    return {"by_brand": by_brand, "by_country": by_country, "frontier": freq}


# ==============================================================================
# 4. FIGURES
# ==============================================================================

def _style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.grid": True, "grid.color": "#E4E9E8", "grid.linewidth": .7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#666666", "figure.facecolor": "white",
    })


def _save(fig, save):
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


# Explanatory notes used to be printed under each figure. They now live in
# the LaTeX captions, which lets the chart itself print larger. The text is
# still passed to _foot and kept on the figure as `fig._foot_text`; set this
# to True to draw it again.
SHOW_FOOTNOTES = False


def _foot(fig, text: str, width: int = 110, y: float = -0.02,
          size: float = 8.0) -> None:
    """
    Figure footnote, hard-wrapped to about two lines.

    `bbox_inches="tight"` grows the canvas to fit the widest artist. An
    unwrapped one-line note therefore sets the image width, and the plot itself
    is squeezed into a short, very wide strip that prints small on the page.
    Wrapping keeps the figure close to its intended proportions.
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


def fig_model_mix(mix: pd.DataFrame, save: Optional[str] = None):
    """Portfolio size over time — overall, by group, and its distribution."""
    import matplotlib.pyplot as plt
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.3))
    fig.suptitle("Model portfolio per plant", fontsize=14, fontweight="bold",
                 x=0.02, ha="left")

    ax = axes[0]
    g = mix.groupby("year").agg(mean_n=("n_models", "mean"),
                                median_n=("n_models", "median"),
                                plants=("plant_uid", "nunique"))
    ax.plot(g.index, g["mean_n"], lw=2.2, marker="o", ms=3, color="#14746F", label="mean")
    ax.plot(g.index, g["median_n"], lw=2.2, marker="s", ms=3, color="#2C6FB3", label="median")
    ax.set_title("(a) Models per plant", loc="left")
    ax.set_ylabel("models"); ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    for grp, s in mix.groupby("group"):
        gg = s.groupby("year")["n_models"].mean()
        ax.plot(gg.index, gg.values, lw=2, marker="o", ms=2.6,
                color=m2.GROUP_COLOURS.get(grp, "#888888"), label=grp)
    ax.set_title("(b) Mean portfolio size by group", loc="left")
    ax.set_ylabel("models per plant"); ax.legend(fontsize=7.5, frameon=False, ncol=2)

    ax = axes[2]
    yrs = sorted(mix["year"].unique())
    pick = [yrs[0], yrs[len(yrs) // 2], yrs[-1]]
    for y, c in zip(pick, ["#AFC8C6", "#2C6FB3", "#14746F"]):
        s = mix[mix["year"] == y]["n_models"]
        ax.hist(s, bins=range(0, int(mix["n_models"].max()) + 2), alpha=.55,
                color=c, label=str(y))
    ax.set_title("(c) Distribution of portfolio size", loc="left")
    ax.set_xlabel("models per plant"); ax.set_ylabel("plants")
    ax.legend(fontsize=8, frameon=False)
    _foot(fig, "Assembly plants only. For a component site the recorded model list is "
             "the set of vehicles its parts end up in rather than a portfolio it "
             "builds \u2014 AMG Affalterbach, for instance, lists 71 engine "
             "variants. Note also that brands name their models at different "
             "levels of detail, so a change within one plant over time is more "
             "reliable than a comparison of counts between brands.", y=-0.02)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    return _save(fig, save)


def fig_churn(ch: pd.DataFrame, save: Optional[str] = None):
    """Model entries and exits — the product-cycle pulse of the network."""
    import matplotlib.pyplot as plt
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model churn across the plant network", fontsize=14,
                 fontweight="bold", x=0.02, ha="left")
    ax = axes[0]
    g = ch.groupby("year").agg(entries=("entries", "sum"), exits=("exits", "sum"))
    ax.bar(g.index - .2, g["entries"], .4, label="models added", color="#14746F")
    ax.bar(g.index + .2, -g["exits"], .4, label="models dropped", color="#C0563F")
    ax.axhline(0, color="#333333", lw=1)
    ax.set_title("(a) Model entries and exits per year", loc="left")
    ax.set_ylabel("count across all plants"); ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    g2 = ch.groupby("year")["renewal_rate"].mean() * 100
    ax.plot(g2.index, g2.values, lw=2.4, marker="o", ms=3.5, color="#14746F")
    ax.set_title("(b) Mean renewal rate (entries ÷ portfolio)", loc="left")
    ax.set_ylabel("% of portfolio new that year")
    _foot(fig, "A model counts as present in a year when it appears in that plant's "
             "sources for that year. Some of the churn shown here therefore "
             "reflects how completely the sources covered the plant in each "
             "year, not only genuine changes to what it built.", y=-0.02)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    return _save(fig, save)


def fig_electrification(mp: pd.DataFrame, save: Optional[str] = None):
    """The BEV transition, from nameplates — near-full coverage."""
    import matplotlib.pyplot as plt
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.3))
    fig.suptitle("Electrification of the model mix (dedicated BEV nameplates)",
                 fontsize=14, fontweight="bold", x=0.02, ha="left")

    ax = axes[0]
    g = electrification_by(mp, "classification").groupby("year").agg(
        n_models=("n_models", "sum"), n_bev=("n_bev", "sum"))
    g["share"] = 100 * g["n_bev"] / g["n_models"]
    ax.bar(g.index, g["n_bev"], color="#14746F", width=.7)
    ax.set_title("(a) Distinct BEV nameplates in production", loc="left")
    ax.set_ylabel("BEV model-plant records")

    ax = axes[1]
    ax.plot(g.index, g["share"], lw=2.4, marker="o", ms=3.5, color="#14746F")
    ax.set_title("(b) BEV share of the model mix", loc="left")
    ax.set_ylabel("% of plant-model records")

    ax = axes[2]
    eb = electrification_by(mp, "group")
    for grp, s in eb.groupby("group"):
        s = s.sort_values("year")
        ax.plot(s["year"], s["bev_share_pct"], lw=2, marker="o", ms=2.6,
                color=m2.GROUP_COLOURS.get(grp, "#888888"), label=grp)
    ax.set_title("(c) BEV share by group", loc="left")
    ax.set_ylabel("% of the group's model mix")
    ax.legend(fontsize=7.5, frameon=False, ncol=2)

    from matplotlib.ticker import MaxNLocator
    for a in axes:
        a.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))

    _foot(fig, "Battery-electric here means a dedicated electric nameplate \u2014 ID.3, "
             "i4, EQS, e-tron, Zo\u00e9, Kangoo Z.E., Fiat 500e, M\u00e9gane E-Tech and "
             "the like. Mild and plug-in hybrids such as the C-Class EQ, the Jeep 4xe "
             "or the plug-in Prius are deliberately excluded, so this measures full "
             "electrification rather than any degree of it. Toyota reads zero "
             "throughout because its European sites in this panel build the Yaris, "
             "Corolla, Auris, Aygo and C-HR, none of which is a dedicated electric "
             "nameplate; the bZ4X is not assembled in Europe.", y=-0.02)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    return _save(fig, save)


def fig_multi_sourcing(ms: pd.DataFrame, save: Optional[str] = None):
    """Are models built in more places at once over time?"""
    import matplotlib.pyplot as plt
    _style()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    g = ms.groupby("year").agg(mean_plants=("n_plants", "mean"),
                               pct_multi=("n_plants", lambda s: 100 * (s > 1).mean()))
    ax.plot(g.index, g["mean_plants"], lw=2.4, marker="o", ms=3.5, color="#14746F",
            label="mean plants per model")
    ax2 = ax.twinx()
    ax2.plot(g.index, g["pct_multi"], lw=2.2, marker="s", ms=3.5, color="#C6803D",
             label="% of models built in >1 plant")
    ax2.grid(False)
    ax.set_ylabel("plants per model", color="#14746F")
    ax2.set_ylabel("% multi-sourced", color="#C6803D")
    # `_style` hides the right spine, but the twin axis still drew its tick
    # MARKS. With no spine behind them each mark sat as a short dash just left
    # of its label, so "13" read as "-13" and the percentages looked negative.
    # Drop the marks and put the unit in the label, which removes the ambiguity
    # rather than relying on the reader to notice the axis title.
    from matplotlib.ticker import FuncFormatter
    ax2.tick_params(axis="y", length=0, colors="#C6803D")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.tick_params(axis="y", colors="#14746F")
    _int_years(ax, g.index)
    ax.set_title("Multi-sourcing of models across plants", loc="left",
                 fontsize=13, fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    return _save(fig, save)


def fig_employment_vs_mix(ev: pd.DataFrame, save: Optional[str] = None):
    """Employment beside the model portfolio — associations, not effects."""
    import matplotlib.pyplot as plt
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Employment and the model mix (balanced universe) — descriptive",
                 fontsize=14, fontweight="bold", x=0.02, ha="left")
    d = ev.dropna(subset=["n_models", "employees"])

    ax = axes[0]
    for grp, s in d.groupby("group"):
        ax.scatter(s["n_models"], s["employees"] / 1000, s=13, alpha=.55,
                   color=m2.GROUP_COLOURS.get(grp, "#888888"), label=grp)
    ax.set_title("(a) Plant-years: portfolio size vs employment", loc="left")
    ax.set_xlabel("models produced"); ax.set_ylabel("employees (thousands)")
    ax.legend(fontsize=7.5, frameon=False)

    ax = axes[1]
    g = d.groupby("year").agg(emp_per_model=("emp_per_model", "median"))
    ax.plot(g.index, g["emp_per_model"], lw=2.4, marker="o", ms=3.5, color="#14746F")
    ax.set_title("(b) Median employees per model produced", loc="left")
    ax.set_ylabel("employees per model")

    ax = axes[2]
    has_bev = d[d["n_bev"] > 0].groupby("year")["employees"].sum() / 1000
    no_bev = d[d["n_bev"] == 0].groupby("year")["employees"].sum() / 1000
    ax.plot(has_bev.index, has_bev.values, lw=2.4, marker="o", ms=3.5,
            color="#14746F", label="plants building a BEV")
    ax.plot(no_bev.index, no_bev.values, lw=2.4, marker="s", ms=3.5,
            color="#C6803D", label="plants building no BEV")
    ax.set_title("(c) Employment by BEV presence", loc="left")
    ax.set_ylabel("employees (thousands)"); ax.legend(fontsize=8, frameon=False)
    _foot(fig, "Panel (c) compares two groups whose membership changes over time: a "
             "plant moves from one to the other the moment it starts building an "
             "electric model. The gap between the lines therefore describes "
             "which plants are in which group, and should not be read as the "
             "effect of electrification on employment.", y=-0.02)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    return _save(fig, save)


def fig_attribute_readiness(ar: Dict[str, pd.DataFrame], save: Optional[str] = None):
    """The methods-chapter figure justifying the move to the EEA database."""
    import matplotlib.pyplot as plt
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Coverage of the ADEME vehicle attribute data",
                 fontsize=14, fontweight="bold", x=0.02, ha="left")

    ax = axes[0]
    b = ar["by_brand"][ar["by_brand"]["models"] >= 5].sort_values("match_rate_pct")
    cols = ["#C0563F" if v < 20 else "#E9A23B" if v < 50 else "#14746F"
            for v in b["match_rate_pct"]]
    ax.barh(b["model_brand"], b["match_rate_pct"], color=cols)
    ax.set_title("(a) Attribute match rate by brand", loc="left")
    ax.set_xlabel("% of the brand's models matched")
    for i, (_, r) in enumerate(b.iterrows()):
        ax.text(r["match_rate_pct"] + 1, i, f"{int(r['matched'])}/{int(r['models'])}",
                va="center", fontsize=7.5, color="#444444")

    ax = axes[1]
    c = ar["by_country"]
    ax.barh(c["country_iso2"], c["cov_pct"], color="#2C6FB3")
    ax.axvline(15, color="#C0563F", ls="--", lw=1.2)
    ax.text(15.6, 0.2, "overall 15%", fontsize=8, color="#C0563F")
    ax.set_title("(b) Coverage of the model mix by country", loc="left")
    ax.set_xlabel("% of plant-model-years with attributes")

    ax = axes[2]
    f = ar["frontier"]
    ax.plot(f["rank"], f["cum_pct"], lw=2.4, color="#14746F")
    for k, col in ((150, "#E9A23B"), (200, "#C6803D"), (300, "#C0563F")):
        if k <= len(f):
            y = f.iloc[k - 1]["cum_pct"]
            ax.plot([k, k], [0, y], ls=":", color=col, lw=1.2)
            ax.plot([0, k], [y, y], ls=":", color=col, lw=1.2)
            ax.annotate(f"top {k} → {y:.0f}%", (k, y), textcoords="offset points",
                        xytext=(6, -12), fontsize=8, color=col, fontweight="bold")
    ax.scatter([94], [f[f["matched"]]["pmy"].sum() / f["pmy"].sum() * 100],
               color="#C0563F", zorder=5, s=40)
    ax.annotate("current: 94 models → 15%", (94, 15), textcoords="offset points",
                xytext=(10, 8), fontsize=8, color="#C0563F", fontweight="bold")
    ax.set_title("(c) How many models a source must cover", loc="left")
    ax.set_xlabel("models ranked by frequency"); ax.set_ylabel("% of plant-model-years covered")

    _foot(fig, "The vehicle attributes available from ADEME cover only a small and "
             "unrepresentative part of the fleet: the unmatched models include "
             "the most-produced cars in Europe \u2014 Golf, C-Class, Passat, "
             "Polo, Tiguan, Sprinter and Cayenne. Vehicle mass is therefore "
             "taken from the EEA CO2 monitoring database (Regulation (EU) "
             "2019/631) instead, which covers the whole fleet from 2010.", y=-0.02)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    return _save(fig, save)


# ==============================================================================
#  VEHICLE MASS AND POWERTRAIN  (was module03b_mass.py)
# ------------------------------------------------------------------------------
#  Merged into this module because the split was arbitrary: 03 described what a
#  plant builds from the model names, 03b described the same vehicles' physical
#  characteristics from the registration database, and every figure in Section
#  4.3 draws on both. `_style` and `_foot` above are shared; the m3. prefix that
#  03b used to reach back into this file is gone.
# ==============================================================================
# Below this many registrations a yearly mass mean is noise, not a measurement.
# The BEV series reads 2,253 kg in 2010 on a handful of cars; plotted unguarded
# it invents a dramatic early decline that is pure small-sample wobble.
MIN_BEV_REGS = 5_000

_BLOC = None  # filled lazily from module01


# ==============================================================================
# 1. PLANT-LEVEL MASS
# ==============================================================================

def plant_model_mass(mp: Optional[pd.DataFrame] = None,
                     cw: Optional[pd.DataFrame] = None,
                     assembly_only: bool = True,
                     drop_post_closure: bool = True,
                     by_year: bool = False) -> pd.DataFrame:
    """
    plant x year x model, with the model's mass and its European registrations.

    Component, engine and battery plants are dropped by default: they have no
    vehicle portfolio, so a "mass of what this plant builds" is undefined for
    them rather than zero.

    POST-CLOSURE ROWS ARE DROPPED. Model presence is a Wikipedia-snapshot fact,
    and an article's model list does not disappear when the plant does: Ford
    Genk shut in December 2014 and still carries model rows through 2025. Left
    in, it contributes a "mass trajectory" for eleven years in which it built
    nothing, and it survives panel balancing precisely because the stale rows
    are complete. `fd.KNOWN_CLOSURES` is the same list Module 00 uses on the
    employment series.
    """
    mp = model_panel() if mp is None else mp
    if assembly_only:
        mp = mp[(mp["classification"] == "assembly")
                & (mp["output_unit"] == "vehicles")]
    # Electric records dated before the nameplate could exist (the launch-year
    # guard, `flag_implausible_bev`) are dropped here too, not only kept out of
    # the electric share: otherwise a model listed a year early still moves the
    # plant's mass (Cologne's 2023 Explorer EV). Added 4 Oct 2026.
    if "bev_implausible" in mp.columns:
        mp = mp[~mp["bev_implausible"].fillna(False)]
    if drop_post_closure:
        before = len(mp)
        mask = pd.Series(False, index=mp.index)
        for uid, yr in fd.KNOWN_CLOSURES.items():
            mask |= (mp["plant_uid"] == uid) & (mp["year"] > yr)
        if mask.any():
            warnings.warn(
                f"dropped {int(mask.sum())} model-year row(s) recorded AFTER a "
                f"known plant closure ({sorted(mp.loc[mask, 'plant_uid'].unique())}). "
                "Pass drop_post_closure=False to keep them.")
            mp = mp[~mask]
    out = xw.attach_mass(mp, cw, by_year=by_year)
    keep = ["plant_uid", "plant_name", "city", "country", "country_iso2",
            "group", "brand", "year", "model", "mass_kg", "eea_regs",
            "match_rule", "bev_share", "is_bev", "co2_g_km", "co2_coverage"]
    return out[[c for c in keep if c in out.columns]].copy()


def plant_mass(pmm: Optional[pd.DataFrame] = None,
               weighting: str = "equal",
               min_models: int = 1) -> pd.DataFrame:
    """
    Average mass of the vehicles a plant builds, per year.

    weighting : 'equal'          — every model counts once
                'registrations'  — models weighted by European registrations

    Returns one row per plant-year with the mean, the number of models it rests
    on, and the share of that plant's models that carry a mass at all. A plant
    whose portfolio is only half matched gets a mean computed on that half, and
    `coverage` says so rather than hiding it.
    """
    pmm = plant_model_mass() if pmm is None else pmm
    if weighting not in ("equal", "registrations"):
        raise ValueError("weighting must be 'equal' or 'registrations'")

    g = pmm.copy()
    g["has_mass"] = g["mass_kg"].notna()
    # Count DISTINCT models, not rows. A plant can list the same model twice in
    # a year (two source snapshots, two body variants), and summing the boolean
    # counted those twice -- which pushed `coverage` above 1.0 and would have
    # over-weighted duplicated models in the mean.
    tot = (g.groupby(["plant_uid", "year"])
             .agg(n_models_listed=("model", "nunique")).reset_index())
    massed = (g[g["has_mass"]].groupby(["plant_uid", "year"])["model"]
                .nunique().rename("n_models_massed").reset_index())
    tot = tot.merge(massed, on=["plant_uid", "year"], how="left")
    tot["n_models_massed"] = tot["n_models_massed"].fillna(0).astype(int)

    m = (g[g["has_mass"]]
         .drop_duplicates(subset=["plant_uid", "year", "model"]).copy())
    if weighting == "equal":
        m["w"] = 1.0
    else:
        # Registrations as a proxy for relative volume inside the plant.
        # Missing or zero falls back to 1 so a model is never silently dropped.
        m["w"] = pd.to_numeric(m["eea_regs"], errors="coerce").fillna(0.0)
        m.loc[m["w"] <= 0, "w"] = 1.0

    m["mw"] = m["mass_kg"] * m["w"]
    agg = (m.groupby(["plant_uid", "plant_name", "country_iso2", "group",
                      "year"], as_index=False)
             .agg(mass_w=("mw", "sum"), w=("w", "sum"),
                  mass_min=("mass_kg", "min"), mass_max=("mass_kg", "max")))
    agg["mean_mass_kg"] = agg["mass_w"] / agg["w"]
    agg = agg.drop(columns=["mass_w", "w"])

    out = agg.merge(tot, on=["plant_uid", "year"], how="left")
    out["coverage"] = out["n_models_massed"] / out["n_models_listed"]
    out["weighting"] = weighting
    out = out[out["n_models_massed"] >= min_models]
    out.attrs["weighting"] = weighting
    return out.sort_values(["plant_uid", "year"]).reset_index(drop=True)


def exact_mass_plants(pmm: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Plant-years where the portfolio is a SINGLE model, so the plant's average
    mass is known without any weighting assumption at all.

    These are the anchor points. They are not a test of the weighting schemes —
    with one model both schemes give the same answer — but they are the rows
    where the level is certain, and a sensible estimate should agree with them.
    """
    pmm = plant_model_mass() if pmm is None else pmm
    n = pmm.groupby(["plant_uid", "year"])["model"].nunique().rename("n_models")
    one = pmm.merge(n, on=["plant_uid", "year"])
    one = one[(one["n_models"] == 1) & one["mass_kg"].notna()]

    prod = pd.read_excel(fd.CONFIG.path(fd.CONFIG.f_plants),
                         sheet_name="tbl_production")
    prod = prod[(prod["output_unit"] == "vehicles")
                & prod["production_value"].notna()]
    out = one.merge(prod[["plant_uid", "year", "production_value"]],
                    on=["plant_uid", "year"], how="left")
    return out.sort_values(["plant_uid", "year"]).reset_index(drop=True)


def weighting_sensitivity(pmm: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    How much does the within-plant weighting assumption actually change things?

    Returns plant-year rows with both estimates and their gap. If the gap is
    small the assumption is harmless and can be stated in a footnote; if it is
    large it belongs in the text as a stated limit on what the module can claim.
    """
    pmm = plant_model_mass() if pmm is None else pmm
    a = plant_mass(pmm, "equal")[["plant_uid", "plant_name", "year",
                                  "mean_mass_kg", "n_models_massed"]]
    b = plant_mass(pmm, "registrations")[["plant_uid", "year", "mean_mass_kg"]]
    m = a.merge(b, on=["plant_uid", "year"], suffixes=("_equal", "_regs"))
    m["gap_kg"] = m["mean_mass_kg_regs"] - m["mean_mass_kg_equal"]
    m["gap_pct"] = 100 * m["gap_kg"] / m["mean_mass_kg_equal"]
    return m


# ==============================================================================
# 2. THE EUROPEAN TREND
# ==============================================================================

def balance_report(pmm: Optional[pd.DataFrame] = None,
                   plant_weight: str = "production",
                   windows: Sequence[Tuple[int, int]] = (
                       (2006, 2025), (2010, 2024), (2012, 2023),
                       (2013, 2023), (2014, 2023), (2015, 2022))
                   ) -> pd.DataFrame:
    """
    How many plants survive balancing over each candidate window.

    THIS IS NOT OPTIONAL. The raw production-weighted series rests on 5 plants
    in 2006, 22 in 2016 and 12 in 2025. A "trend" over that is mostly a record
    of which plants happened to publish an output figure, and in Module 02 the
    same artefact manufactured two findings that did not survive balancing.
    Choose the window on this table before computing anything.
    """
    pmm = plant_model_mass() if pmm is None else pmm
    pm = plant_mass(pmm, "equal")
    if plant_weight == "production":
        prod = pd.read_excel(fd.CONFIG.path(fd.CONFIG.f_plants),
                             sheet_name="tbl_production")
        prod = prod[(prod["output_unit"] == "vehicles")
                    & prod["production_value"].notna()]
        pm = pm.merge(prod[["plant_uid", "year", "production_value"]],
                      on=["plant_uid", "year"], how="inner")

    rows = []
    for y0, y1 in windows:
        s = pm[(pm["year"] >= y0) & (pm["year"] <= y1)]
        yrs = sorted(s["year"].unique())
        n = s.groupby("plant_uid")["year"].nunique()
        keep = n[n == len(yrs)].index
        rows.append(dict(window=f"{y0}-{y1}", y0=y0, y1=y1,
                         n_years=len(yrs),
                         n_plants_any=s["plant_uid"].nunique(),
                         n_plants_balanced=len(keep),
                         pct_kept=100 * len(keep) / max(s["plant_uid"].nunique(), 1)))
    return pd.DataFrame(rows)


def balanced_plants(pm: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    """Restrict to plants observed in EVERY year of the window."""
    s = pm[(pm["year"] >= y0) & (pm["year"] <= y1)].copy()
    yrs = sorted(s["year"].unique())
    n = s.groupby("plant_uid")["year"].nunique()
    keep = n[n == len(yrs)].index
    out = s[s["plant_uid"].isin(keep)].copy()
    out.attrs.update(y0=y0, y1=y1, n_plants=len(keep))
    return out


def mass_trend(pmm: Optional[pd.DataFrame] = None,
               weighting: str = "equal",
               plant_weight: str = "production",
               window: Optional[Tuple[int, int]] = None) -> pd.DataFrame:
    """
    Mean mass of the European plant portfolio, by year.

    plant_weight : 'production' — plants weighted by their vehicle output, so a
                                  600k-unit plant counts more than a 20k one
                   'equal'      — every plant counts once

    Production is observed for only part of the panel, so the production-weighted
    series necessarily rests on fewer plants. Both are returned side by side and
    `n_plants` reports the difference rather than burying it.
    """
    pmm = plant_model_mass() if pmm is None else pmm
    pm = plant_mass(pmm, weighting)

    if plant_weight == "production":
        prod = pd.read_excel(fd.CONFIG.path(fd.CONFIG.f_plants),
                             sheet_name="tbl_production")
        prod = prod[(prod["output_unit"] == "vehicles")
                    & prod["production_value"].notna()]
        pm = pm.merge(prod[["plant_uid", "year", "production_value"]],
                      on=["plant_uid", "year"], how="left")
        pm["w"] = pm["production_value"]
        pm = pm[pm["w"].notna() & (pm["w"] > 0)]
    elif plant_weight == "equal":
        pm["w"] = 1.0
    else:
        raise ValueError("plant_weight must be 'production' or 'equal'")

    # Balance BEFORE aggregating: otherwise the series tracks which plants
    # published an output figure that year rather than what they built.
    if window is not None:
        pm = balanced_plants(pm, *window)

    pm["mw"] = pm["mean_mass_kg"] * pm["w"]
    t = (pm.groupby("year", as_index=False)
           .agg(mass_w=("mw", "sum"), w=("w", "sum"),
                n_plants=("plant_uid", "nunique")))
    t["mean_mass_kg"] = t["mass_w"] / t["w"]
    t = t.drop(columns=["mass_w"]).rename(columns={"w": "total_weight"})
    t.attrs.update(weighting=weighting, plant_weight=plant_weight,
                   window=window, balanced=window is not None)
    return t


def powertrain_decomposition(min_regs: int = MIN_BEV_REGS) -> pd.DataFrame:
    """
    Split the rise in European mean mass into a WITHIN and a BETWEEN part.

        mean_t = SUM_f  share_ft * mass_ft

    within  — what mass would have done if the powertrain mix had never moved
              (cars of each type getting heavier)
    between — what it would have done if each type's mass had never moved but
              the mix shifted (electrification)

    THE BASELINE PROBLEM. Battery cars have no 2010 mass to be measured against
    -- essentially none were sold. Filling that gap with the overall 2010 mean
    gives every class the SAME baseline, and the between term then collapses to
    zero by construction: the composition can shift all it likes and the mean
    cannot move if every class weighs the same. The first version of this
    function did exactly that and drew a flat green line at zero.

    The fix is a symmetric (Bennet) decomposition, which weights each
    class's mass change by its AVERAGE share and each class's share change by
    its AVERAGE mass, so a class that appears mid-period still contributes:

        within(y)  = SUM_f  s_bar_f * (m_f(y) - m_f(first observed))
        between(y) = SUM_f  m_bar_f * (s_f(y) - s_f(y0))

    `interaction` carries whatever the two terms do not explain; if it is large
    relative to the total the decomposition should not be trusted.
    """
    f = pd.read_csv(fd.CONFIG.path("eea_model_mass_by_fuel.csv"))
    f["mw"] = f["mass_mean_kg"] * f["registrations"]
    g = (f.groupby(["year", "fuel_class"], as_index=False)
           .agg(regs=("registrations", "sum"), mw=("mw", "sum")))
    g["mass"] = g["mw"] / g["regs"]
    g.loc[g["regs"] < min_regs, "mass"] = np.nan
    tot = g.groupby("year")["regs"].transform("sum")
    g["share"] = g["regs"] / tot

    w = g.pivot(index="year", columns="fuel_class", values="share").fillna(0.0)
    m = g.pivot(index="year", columns="fuel_class", values="mass")
    years = list(w.index)
    y0, y1 = years[0], years[-1]

    R_ = g.pivot(index="year", columns="fuel_class", values="regs").fillna(0.0)
    M_ = g.pivot(index="year", columns="fuel_class", values="mass")
    d = _chained(R_, M_, y0, y1, series=True)
    d.attrs["endpoint"] = _bennet(R_, M_, y0, y1)
    d.attrs["y0"], d.attrs["y1"] = y0, y1
    return d


def mass_by_bloc(pmm: Optional[pd.DataFrame] = None,
                 weighting: str = "equal",
                 window: Optional[Tuple[int, int]] = (2012, 2023)) -> pd.DataFrame:
    """
    Mean mass by Pavlínek bloc — does the core build heavier cars than the
    periphery? This is the module's link back to Module 01: if the core keeps
    the large, high-value vehicles while the periphery takes the small ones,
    that is functional specialisation, not just relocation.
    """
    global _BLOC
    if _BLOC is None:
        import module01_spatial as m1
        _BLOC = m1.assign_bloc
    pmm = plant_model_mass() if pmm is None else pmm
    pm = plant_mass(pmm, weighting)
    if window is not None:
        pm = balanced_plants(pm, *window)
    pm["bloc"] = pm["country_iso2"].map(_BLOC)
    pm = pm[pm["bloc"] != "other"]
    g = (pm.groupby(["year", "bloc"], as_index=False)
           .agg(mean_mass_kg=("mean_mass_kg", "mean"),
                median_mass_kg=("mean_mass_kg", "median"),
                n_plants=("plant_uid", "nunique")))
    g.attrs["window"] = window
    return g


def fig_mass_by_bloc(bloc: Optional[pd.DataFrame] = None,
                     pmm: Optional[pd.DataFrame] = None,
                     window: Tuple[int, int] = (2012, 2023),
                     save: Optional[str] = None):
    """Average vehicle mass over time, core against the two peripheries."""
    import matplotlib.pyplot as plt
    _style()
    bloc = mass_by_bloc(pmm, window=window) if bloc is None else bloc
    import module01_spatial as m1
    order = [b for b in m1.BLOC_ORDER if b != "other" and b in set(bloc["bloc"])]
    y0, y1 = window

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.9),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    fig.suptitle("Average vehicle mass in the core and the two peripheries",
                 fontsize=14.5, fontweight="bold", x=0.02, ha="left")

    ax = axes[0]
    for b in order:
        s = bloc[bloc["bloc"] == b].sort_values("year")
        ax.plot(s["year"], s["mean_mass_kg"], "-o", color=m1.BLOC_COLOURS[b],
                lw=2.5, ms=5, label=f"{b.replace('_', ' ')} "
                                    f"({int(s['n_plants'].iloc[0])} plants)")
    ends = {b: bloc[bloc.bloc == b].sort_values("year")["mean_mass_kg"].iloc[-1]
            for b in order}
    span = max(ends.values()) - min(ends.values()) or 1
    prev = None
    for b in sorted(order, key=lambda k: ends[k]):
        s = bloc[bloc["bloc"] == b].sort_values("year")
        v0, v1 = s["mean_mass_kg"].iloc[0], s["mean_mass_kg"].iloc[-1]
        yy = v1 if prev is None or v1 - prev > .16 * span else prev + .16 * span
        prev = yy
        ax.annotate(f"{v1:,.0f} kg ({v1 - v0:+,.0f})", (s["year"].iloc[-1], yy),
                    xytext=(9, 0), textcoords="offset points", va="center",
                    fontsize=9.4, fontweight="bold", color=m1.BLOC_COLOURS[b])
    ax.set_ylabel("kilograms")
    ax.set_xlim(bloc["year"].min() - .5, bloc["year"].max() + 3.8)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - .22 * (hi - lo), hi)
    ax.legend(fontsize=9, frameon=False, loc="lower left")
    ax.set_title(f"(a) Average mass, {y0}\u2013{y1}", loc="left")
    _int_years(ax, bloc["year"].unique())

    ax = axes[1]
    x = np.arange(len(order)); w = .36
    first = [bloc[(bloc.bloc == b) & (bloc.year == y0)]["mean_mass_kg"].iloc[0]
             for b in order]
    last = [bloc[(bloc.bloc == b) & (bloc.year == y1)]["mean_mass_kg"].iloc[0]
            for b in order]
    ax.bar(x - w / 2, first, w, color="#CBD8D7", label=str(y0))
    ax.bar(x + w / 2, last, w,
           color=[m1.BLOC_COLOURS[b] for b in order], label=str(y1))
    for xi, (a, z) in enumerate(zip(first, last)):
        for off, v in ((-w / 2, a), (w / 2, z)):
            ax.annotate(f"{v:,.0f}", (xi + off, v), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=9,
                        fontweight="bold", color="#333333")
        ax.annotate(f"{z - a:+,.0f} kg", (xi, max(a, z)), xytext=(0, 20),
                    textcoords="offset points", ha="center", fontsize=10,
                    fontweight="bold",
                    color="#14746F" if z >= a else "#B03A2E")
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("_", "\n") for b in order], fontsize=9.5)
    ax.set_ylabel("kilograms"); ax.legend(fontsize=9, frameon=False)
    ax.margins(y=.24)
    ax.set_title(f"(b) Start and end of the period", loc="left")

    ns = ", ".join(f"{b.replace('_', ' ')} "
                   f"{int(bloc[bloc.bloc == b]['n_plants'].iloc[0])}"
                   for b in order)
    _foot(fig,
          f"Each line is the average, across the plants of a bloc, of the mass "
          f"of the vehicles those plants build. The panel is balanced: the same "
          f"plants appear in every year of {y0}\u2013{y1}, so a line cannot "
          f"move because the sample moved. Plants per bloc: {ns}. The two "
          f"peripheral blocs rest on a handful of plants each, so a single "
          f"plant changing its model line-up shifts the line; read the levels "
          f"as indicative and the direction as the finding.", y=-0.02)
    fig.subplots_adjust(top=.885, bottom=.19, wspace=.28)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def coverage_report(pmm: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """What share of the plant portfolio carries a mass, by year."""
    pmm = plant_model_mass() if pmm is None else pmm
    g = (pmm.assign(has=pmm["mass_kg"].notna())
            .groupby("year", as_index=False)
            .agg(model_rows=("model", "size"), with_mass=("has", "sum"),
                 plants=("plant_uid", "nunique")))
    g["pct"] = 100 * g["with_mass"] / g["model_rows"]
    return g


# ==============================================================================
# 3. FIGURES
# ==============================================================================


def _int_years(ax, years) -> None:
    """Year axes should show years, not 2027.5."""
    y = sorted(int(v) for v in years)
    step = 1 if len(y) <= 8 else (2 if len(y) <= 16 else 3)
    ax.set_xticks(y[::step]); ax.set_xticklabels([str(v) for v in y[::step]])



def fig_plant_mass(pm: Optional[pd.DataFrame] = None, year: int = 2023,
                   top_n: int = 18, save: Optional[str] = None):
    """Average mass of the vehicles each plant builds, in one year."""
    import matplotlib.pyplot as plt
    _style()
    pm = plant_mass() if pm is None else pm
    d = pm[(pm["year"] == year) & pm["mean_mass_kg"].notna()].copy()
    d = d.sort_values("mean_mass_kg", ascending=False)
    sel = pd.concat([d.head(top_n // 2), d.tail(top_n // 2)]).iloc[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.0),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    fig.suptitle(f"Average mass of the vehicles built at each plant, {year}",
                 fontsize=14.5, fontweight="bold", x=0.02, ha="left")

    ax = axes[0]
    cols = ["#14746F" if v >= d["mean_mass_kg"].median() else "#7FB2AE"
            for v in sel["mean_mass_kg"]]
    ax.barh(range(len(sel)), sel["mean_mass_kg"], color=cols)
    ax.set_yticks(range(len(sel)))
    ax.set_yticklabels([f"{n[:30]} ({c})" for n, c
                        in zip(sel["plant_name"], sel["country_iso2"])],
                       fontsize=8.6)
    for i, (v, n) in enumerate(zip(sel["mean_mass_kg"], sel["n_models_massed"])):
        ax.annotate(f"{v:,.0f} kg  ({n} model{'s' if n != 1 else ''})",
                    (v, i), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.4)
    ax.axvline(d["mean_mass_kg"].median(), color="#B03A2E", lw=1.2, ls="--")
    ax.annotate(f"median {d['mean_mass_kg'].median():,.0f} kg",
                (d["mean_mass_kg"].median(), len(sel) - .4), fontsize=8.6,
                color="#B03A2E", ha="center")
    ax.set_xlabel("kilograms"); ax.set_xlim(0, d["mean_mass_kg"].max() * 1.38)
    ax.set_title(f"(a) Heaviest and lightest {top_n} plants", loc="left")

    ax = axes[1]
    ax.hist(d["mean_mass_kg"], bins=16, color="#4E6E8E", edgecolor="white")
    ax.axvline(d["mean_mass_kg"].mean(), color="#B03A2E", lw=1.6)
    ax.annotate(f"mean {d['mean_mass_kg'].mean():,.0f} kg",
                (d["mean_mass_kg"].mean(), ax.get_ylim()[1] * .93),
                xytext=(7, 0), textcoords="offset points", fontsize=9,
                color="#B03A2E", fontweight="bold")
    ax.set_xlabel("kilograms"); ax.set_ylabel("number of plants")
    ax.set_title(f"(b) Distribution across {len(d)} plants", loc="left")

    _foot(fig, f"Assembly plants building vehicles in {year}. A plant's figure "
               "is the average mass of the models it is recorded as building, "
               "taken from European registration data collected under Regulation "
               "(EU) 2019/631. How many of each model a plant built is not "
               "recorded anywhere in the sources, so the average treats every "
               "model in the plant's line-up alike. The chart on the sensitivity "
               "of this choice shows how much difference the assumption makes.",
          y=-0.02)
    fig.subplots_adjust(top=.90, bottom=.13, wspace=.42)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def fig_mass_trend(window: Tuple[int, int] = (2012, 2023),
                   save: Optional[str] = None):
    """
    The European trend, on a BALANCED panel, with the weighting assumption
    drawn as a band.

    Panel (b) exists because the obvious alternative -- weighting plants by
    their output -- cannot be balanced: only five plants report a production
    figure in every year of any usable window, against thirty-one that report a
    model list. The unbalanced production-weighted series is drawn there purely
    so the reader can see why it is not the headline.
    """
    import matplotlib.pyplot as plt
    _style()
    pmm = plant_model_mass()
    a = mass_trend(pmm, "equal", "equal", window=window)
    b = mass_trend(pmm, "registrations", "equal", window=window)
    raw = mass_trend(pmm, "equal", "equal")
    prod = mass_trend(pmm, "equal", "production")

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.8))
    fig.suptitle("Average mass of vehicles built in European plants, "
                 f"{window[0]}\u2013{window[1]}",
                 fontsize=14.5, fontweight="bold", x=0.02, ha="left")

    ax = axes[0]
    m = a.merge(b[["year", "mean_mass_kg"]], on="year", suffixes=("_eq", "_rg"))
    ax.fill_between(m["year"], m["mean_mass_kg_eq"], m["mean_mass_kg_rg"],
                    color="#14746F", alpha=.16,
                    label="range spanned by the two within-plant weightings")
    ax.plot(a["year"], a["mean_mass_kg"], "-o", color="#14746F", lw=2.6, ms=5,
            label="equal weight within plant")
    ax.plot(b["year"], b["mean_mass_kg"], "-o", color="#C6803D", lw=2.0, ms=4,
            ls="--", label="weighted by European registrations")
    for pos, dx, ha in ((0, -8, "right"), (len(a) - 1, 9, "left")):
        r = a.iloc[pos]
        ax.annotate(f"{r['mean_mass_kg']:,.0f} kg",
                    (r["year"], r["mean_mass_kg"]), xytext=(dx, 7),
                    textcoords="offset points", ha=ha, fontsize=10,
                    fontweight="bold", color="#14746F")
    d = a["mean_mass_kg"].iloc[-1] - a["mean_mass_kg"].iloc[0]
    ax.annotate(f"{d:+,.0f} kg  ({100*d/a['mean_mass_kg'].iloc[0]:+.1f}%)",
                xy=(.03, .90), xycoords="axes fraction", fontsize=11,
                fontweight="bold", color="#14746F")
    ax.set_ylabel("kilograms")
    ax.set_xlim(a["year"].min() - .6, a["year"].max() + 1.6)
    ax.set_ylim(top=ax.get_ylim()[1] + .18 * (ax.get_ylim()[1] - ax.get_ylim()[0]))
    ax.legend(fontsize=8.8, frameon=False, loc="upper left",
              bbox_to_anchor=(0.0, 0.88))
    _int_years(ax, a["year"])
    ax.set_title(f"(a) Balanced panel \u2014 the same "
                 f"{int(a['n_plants'].iloc[0])} plants every year", loc="left")

    ax = axes[1]
    ax.plot(raw["year"], raw["mean_mass_kg"], "-o", color="#9AA0A6", lw=1.8,
            ms=3.5, label="all plants, unbalanced")
    ax.plot(prod["year"], prod["mean_mass_kg"], "-o", color="#B03A2E", lw=1.8,
            ms=3.5, ls="--", label="weighted by plant output, unbalanced")
    ax.plot(a["year"], a["mean_mass_kg"], "-o", color="#14746F", lw=2.8, ms=5,
            label=f"balanced panel ({int(a['n_plants'].iloc[0])} plants)")
    ax2 = ax.twinx()
    ax2.bar(prod["year"], prod["n_plants"], color="#EBD5D2", alpha=.75,
            zorder=0, width=.7)
    ax2.set_ylabel("plants reporting output", color="#B08A85", fontsize=9)
    ax2.grid(False); ax2.set_ylim(0, prod["n_plants"].max() * 3.6)
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
    ax.set_ylabel("kilograms")
    ax.legend(fontsize=8.6, frameon=False, loc="upper left")
    ax.set_title("(b) Balanced panel compared with the unbalanced series", loc="left")
    _int_years(ax, raw["year"])

    _foot(fig,
          "Panel (a) uses only plants observed in every year of the period, so "
          "the line cannot move simply because the sample changed. The shaded "
          "band is not a confidence interval: it is the full range between the "
          "two reasonable ways of averaging the models inside a plant, and it "
          "exists because how many of each model a plant built is not recorded "
          "in the sources.  Panel (b) explains why this panel is the one to "
          "trust. The pale bars count how many plants report an output figure "
          "each year. Weighting plants by output would be preferable in "
          "principle, but only five plants report output in every year of any "
          "usable period, against thirty that report a model line-up. The "
          "output-weighted line is shown to make that visible, not to be read "
          "as a trend.", y=-0.02)
    fig.subplots_adjust(top=.885, bottom=.19, wspace=.30)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def plant_trajectories(pmm: Optional[pd.DataFrame] = None,
                       window: Tuple[int, int] = (2012, 2023),
                       weighting: str = "equal") -> pd.DataFrame:
    """
    Each plant's average mass in the first and last year of the window, on the
    balanced panel, with the change between them.

    This is the per-plant view across TIME, as distinct from `fig_plant_mass`,
    which is a single-year cross-section. A plant that switches from a small
    hatchback to a large electric SUV shows up here and nowhere else.
    """
    pmm = plant_model_mass() if pmm is None else pmm
    pm = plant_mass(pmm, weighting)
    b = balanced_plants(pm, *window)
    y0, y1 = window
    w = (b.pivot_table(index=["plant_uid", "plant_name", "country_iso2", "group"],
                       columns="year", values="mean_mass_kg")
          .reset_index())
    w = w[["plant_uid", "plant_name", "country_iso2", "group", y0, y1]].dropna()
    w.columns = ["plant_uid", "plant_name", "country_iso2", "group",
                 "mass_y0", "mass_y1"]
    w["change_kg"] = w["mass_y1"] - w["mass_y0"]
    w["change_pct"] = 100 * w["change_kg"] / w["mass_y0"]
    w.attrs.update(y0=y0, y1=y1, weighting=weighting, n_plants=len(w))
    return w.sort_values("change_kg", ascending=False).reset_index(drop=True)


def fig_plant_trajectories(pmm: Optional[pd.DataFrame] = None,
                           window: Tuple[int, int] = (2012, 2023),
                           label_n: int = 5,
                           save: Optional[str] = None):
    """
    Average mass per plant, across years.

    (a) every plant's path, with the biggest movers named
    (b) the change per plant, which is where the product strategy is visible
    """
    import matplotlib.pyplot as plt
    _style()
    pmm = plant_model_mass() if pmm is None else pmm
    pm = plant_mass(pmm, "equal")
    b = balanced_plants(pm, *window)
    tr = plant_trajectories(pmm, window)
    y0, y1 = window

    fig, axes = plt.subplots(1, 2, figsize=(15.4, 6.6),
                             gridspec_kw={"width_ratios": [1, 1.15]})
    fig.suptitle("Average mass of the vehicles each plant builds, "
                 f"{y0}\u2013{y1}",
                 fontsize=14.5, fontweight="bold", x=0.02, ha="left")

    # ------------------------------------------------------------------ (a)
    ax = axes[0]
    movers = set(tr.head(label_n)["plant_uid"]) | set(tr.tail(2)["plant_uid"])
    for uid, s in b.groupby("plant_uid"):
        s = s.sort_values("year")
        if uid in movers:
            continue
        ax.plot(s["year"], s["mean_mass_kg"], "-", color="#C4CDCC", lw=1.1,
                zorder=1)
    cmap = plt.get_cmap("tab10")
    for i, uid in enumerate(sorted(movers, key=lambda u: -float(
            tr.loc[tr.plant_uid == u, "change_kg"].iloc[0]))):
        s = b[b["plant_uid"] == uid].sort_values("year")
        row = tr[tr.plant_uid == uid].iloc[0]
        ax.plot(s["year"], s["mean_mass_kg"], "-o", color=cmap(i % 10), lw=2.2,
                ms=3.5, zorder=3,
                label=f"{row['plant_name'][:26]} ({row['country_iso2']}) "
                      f"{row['change_kg']:+,.0f} kg")
    ax.set_ylabel("kilograms")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + .30 * (hi - lo))   # headroom so the legend clears the lines
    ax.legend(fontsize=7.8, frameon=False, loc="upper left")
    ax.set_title(f"(a) All {tr.attrs['n_plants']} balanced plants; "
                 "biggest movers named", loc="left")
    _int_years(ax, b["year"].unique())

    # ------------------------------------------------------------------ (b)
    ax = axes[1]
    t = tr.sort_values("change_kg")
    cols = ["#14746F" if v >= 0 else "#B03A2E" for v in t["change_kg"]]
    ax.barh(range(len(t)), t["change_kg"], color=cols)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels([f"{n[:30]} ({c})" for n, c
                        in zip(t["plant_name"], t["country_iso2"])], fontsize=7.4)
    ax.axvline(0, color="#333333", lw=1.2)
    for i, v in enumerate(t["change_kg"]):
        ax.annotate(f"{v:+,.0f}", (v, i), xytext=(5 if v >= 0 else -5, 0),
                    textcoords="offset points", va="center",
                    ha="left" if v >= 0 else "right", fontsize=7.2)
    rng = t["change_kg"].max() - t["change_kg"].min()
    ax.set_xlim(t["change_kg"].min() - .18 * rng, t["change_kg"].max() + .18 * rng)
    ax.set_xlabel(f"change in average mass, {y0}\u2013{y1} (kg)")
    up = int((tr["change_kg"] > 0).sum())
    ax.set_title(f"(b) {up} of {len(tr)} plants build heavier cars than in {y0}"
                 f" \u2014 median {tr['change_kg'].median():+,.0f} kg", loc="left")

    _foot(fig,
          f"The same {tr.attrs['n_plants']} assembly plants appear in every "
          f"year from {y0} to {y1}, and every model in a plant's line-up counts "
          "alike. The large movements are changes of product rather than gradual "
          "drift: Audi Brussels moved from the A1 to the Q8 e-tron, Zwickau from "
          "the Golf to the ID.3 and ID.4, Cologne from the Fiesta to an electric "
          "SUV. Where a plant has closed, the years after closure are excluded, "
          "because the underlying sources continue to list its models long "
          "afterwards.", y=-0.02)
    fig.subplots_adjust(top=.895, bottom=.15, wspace=.34)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def fig_powertrain(dec: Optional[pd.DataFrame] = None,
                   save: Optional[str] = None):
    """Cars getting bigger, versus the powertrain mix changing."""
    import matplotlib.pyplot as plt
    _style()
    dec = powertrain_decomposition() if dec is None else dec
    f = pd.read_csv(fd.CONFIG.path("eea_model_mass_by_fuel.csv"))
    f["mw"] = f["mass_mean_kg"] * f["registrations"]
    g = (f.groupby(["year", "fuel_class"], as_index=False)
           .agg(regs=("registrations", "sum"), mw=("mw", "sum")))
    g["mass"] = g["mw"] / g["regs"]
    g.loc[g["regs"] < MIN_BEV_REGS, "mass"] = np.nan
    tot = g.groupby("year")["regs"].transform("sum")
    g["share"] = 100 * g["regs"] / tot

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.8))
    fig.suptitle(f"Vehicle mass by powertrain, {int(g['year'].min())}–{int(g['year'].max())}",
                 fontsize=14.5, fontweight="bold", x=0.02, ha="left")

    COL = {"ICE": "#8A7A6D", "BEV": "#0B6E4F", "HEV": "#2C6FB3"}
    LBL = {"ICE": "internal combustion", "BEV": "battery electric",
           "HEV": "hybrid"}
    ax = axes[0]
    ends = {}
    for k in ("ICE", "HEV", "BEV"):
        s = g[(g["fuel_class"] == k) & g["mass"].notna()].sort_values("year")
        if not len(s):
            continue
        ax.plot(s["year"], s["mass"], "-o", color=COL[k], lw=2.4, ms=4.5,
                label=LBL[k])
        ends[k] = (s["year"].iloc[-1], s["mass"].iloc[-1])
    # BEV and hybrid finish ~9 kg apart; without nudging, the two end labels
    # print on top of each other
    span = g["mass"].max() - g["mass"].min()
    prev = None
    for k, (xe, ye) in sorted(ends.items(), key=lambda kv: kv[1][1]):
        yy = ye if prev is None or ye - prev > .07 * span else prev + .07 * span
        prev = yy
        ax.annotate(f"{ye:,.0f} kg", (xe, yy), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9.6,
                    fontweight="bold", color=COL[k])
    ice = g[(g.fuel_class == "ICE") & g["mass"].notna()].sort_values("year")
    d_ice = ice["mass"].iloc[-1] - ice["mass"].iloc[0]
    ax.set_ylabel("kilograms")
    ax.set_xlim(g["year"].min() - .5, g["year"].max() + 4.2)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.set_title("(a) Mass within each powertrain", loc="left")
    ax.annotate(f"combustion cars alone: {d_ice:+,.0f} kg",
                xy=(.97, .05), xycoords="axes fraction", ha="right",
                fontsize=9.8, fontweight="bold", color="#8A7A6D")
    _int_years(ax, g["year"].unique())

    ax = axes[1]
    w = dec.set_index("year")
    # Deliberately OFF the powertrain palette of panel (a). Reusing brown and
    # green here would invite the reader to read the within-effect as "the
    # combustion line" and the between-effect as "the electric line", which is
    # wrong: the within-effect contains the mass gains of all three powertrains.
    C_WITHIN, C_BETWEEN, C_TOTAL = "#7B3294", "#D2760A", "#111111"
    ax.plot(w.index, w["within_effect"], "-o", color=C_WITHIN, lw=2.4, ms=4.5,
            label="every powertrain getting heavier (within)")
    ax.plot(w.index, w["between_effect"], "-o", color=C_BETWEEN, lw=2.4, ms=4.5,
            label="market shifting between powertrains (between)")
    ax.plot(w.index, w["total_change"], "-", color=C_TOTAL, lw=2.0, ls=":",
            label="total change")
    for lab, col in (("within_effect", C_WITHIN), ("between_effect", C_BETWEEN),
                     ("total_change", C_TOTAL)):
        ax.annotate(f"{w[lab].iloc[-1]:+,.0f} kg",
                    (w.index[-1], w[lab].iloc[-1]), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=9.4,
                    fontweight="bold", color=col)
    ax.axhline(0, color="#333333", lw=1)
    ax.set_ylabel(f"kg relative to {dec.attrs['y0']}")
    ax.set_xlim(w.index.min() - .5, w.index.max() + 4.2)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.set_title(f"(b) Decomposition of the change since {dec.attrs['y0']}", loc="left")
    r = dec.iloc[-1]
    # the decomposition is exact (Bennet), so there is no interaction to report
    _int_years(ax, w.index)

    # No in-image note. The full explanation lives in the LaTeX caption, which
    # keeps the plotting area as large as the text block allows.
    fig.subplots_adjust(top=.87, bottom=.10, wspace=.26)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def fig_weighting_sensitivity(sens: Optional[pd.DataFrame] = None,
                              save: Optional[str] = None):
    """How much the unknown within-plant mix could be moving the answer."""
    import matplotlib.pyplot as plt
    _style()
    sens = weighting_sensitivity() if sens is None else sens
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.4))
    fig.suptitle("Sensitivity to the unknown within-plant model mix",
                 fontsize=14.5, fontweight="bold", x=0.02, ha="left")

    ax = axes[0]
    ax.hist(sens["gap_pct"], bins=30, color="#4E6E8E", edgecolor="white")
    ax.axvline(0, color="#333333", lw=1.2)
    med = sens["gap_pct"].median()
    ax.axvline(med, color="#B03A2E", lw=1.6)
    ax.annotate(f"median {med:+.1f}%", (med, ax.get_ylim()[1] * .92),
                xytext=(7, 0), textcoords="offset points", fontsize=9.4,
                fontweight="bold", color="#B03A2E")
    ax.set_xlabel("difference when weighting by registrations instead of equally (%)")
    ax.set_ylabel("plant-years")
    ax.set_title("(a) Most plant-years barely move", loc="left")

    ax = axes[1]
    ax.scatter(sens["n_models_massed"], sens["gap_pct"].abs(), s=16,
               color="#14746F", alpha=.45)
    ax.set_xlabel("models in the plant's portfolio")
    ax.set_ylabel("absolute difference (%)")
    ax.set_title("(b) The assumption only bites on mixed plants", loc="left")
    one = sens[sens["n_models_massed"] == 1]
    ax.annotate(f"{len(one)} single-model plant-years:\nno assumption needed, "
                "gap is exactly 0",
                xy=(1, 0), xytext=(.30, .80), textcoords="axes fraction",
                fontsize=8.8, color="#5A5A5A",
                arrowprops=dict(arrowstyle="->", color="#9AA0A6", lw=1))

    _foot(fig, "Each point is one plant in one year. How many of each model a "
               "plant built is not recorded in any available source, so the two "
               "calculations bracket what the truth could be: one treats every "
               "model in a plant's line-up alike, the other assumes a plant "
               "builds more of the models that sell more across Europe. Neither "
               "is right, and the distance between them is the size of the "
               "uncertainty this missing figure introduces.", y=-0.03)
    fig.subplots_adjust(top=.865, bottom=.17, wspace=.26)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


# ==============================================================================
# 4. EMISSIONS AND ELECTRIFICATION PER PLANT
# ==============================================================================
#
# CO2 IS ONLY USABLE FROM 2019. Ewltp replaced NEDC across 2017-2019, so the
# field is empty before 2018 and thin in it. The aggregator now returns NaN
# rather than 0 where nothing was measured -- the earlier version divided a
# NaN-summed numerator by TOTAL registrations and produced a confident 0.0 g/km
# for 2010-2017, which reads as "zero emissions" when it means "not measured".
CO2_FIRST_YEAR = 2019


def plant_co2(pmm: Optional[pd.DataFrame] = None,
              weighting: str = "equal",
              first_year: int = CO2_FIRST_YEAR,
              min_coverage: float = 0.5) -> pd.DataFrame:
    """
    Average tailpipe CO2 of the vehicles a plant builds, per year (2019+).

    `min_coverage` drops plant-years where less than that share of the models
    carry a CO2 reading, because a mean over a third of a portfolio is not a
    portfolio mean.

    WHAT THIS IS NOT. Tailpipe CO2 under Reg. EU 2019/631, not life-cycle
    emissions. A battery car is 0 g/km here by regulatory definition; the
    emissions of building its battery, and of the electricity it runs on, are
    outside the measure entirely. A plant that switches to electric therefore
    goes to zero on this chart while its actual footprint does not.
    """
    pmm = plant_model_mass(by_year=True) if pmm is None else pmm
    if "co2_g_km" not in pmm.columns:
        raise KeyError("pass a panel built with plant_model_mass(by_year=True)")
    g = pmm[(pmm["year"] >= first_year) & pmm["co2_g_km"].notna()].copy()
    g = g.drop_duplicates(subset=["plant_uid", "year", "model"])

    listed = (pmm[pmm["year"] >= first_year]
              .groupby(["plant_uid", "year"])["model"].nunique()
              .rename("n_models_listed").reset_index())

    g["w"] = (1.0 if weighting == "equal"
              else pd.to_numeric(g["eea_regs"], errors="coerce").fillna(1.0).clip(lower=1))
    g["cw"] = g["co2_g_km"] * g["w"]
    agg = (g.groupby(["plant_uid", "plant_name", "country_iso2", "group",
                      "year"], as_index=False)
             .agg(cw=("cw", "sum"), w=("w", "sum"),
                  n_models_co2=("model", "nunique")))
    agg["mean_co2_g_km"] = agg["cw"] / agg["w"]
    agg = agg.drop(columns=["cw", "w"]).merge(listed, on=["plant_uid", "year"],
                                              how="left")
    agg["coverage"] = agg["n_models_co2"] / agg["n_models_listed"]
    # A plant reading exactly 0 g/km has an all-electric RECORDED portfolio.
    # Sometimes that is true; often it means the model list caught only the new
    # electric models. BMW Munich reads 0.0 for 2023 because its recorded list
    # that year is four i4 variants, though it plainly also built the 3-series.
    # The flag says "check this", it does not delete the row.
    agg["all_electric_recorded"] = agg["mean_co2_g_km"] <= 0.5
    out = agg[agg["coverage"] >= min_coverage].copy()
    n_zero = int(out["all_electric_recorded"].sum())
    if n_zero:
        warnings.warn(
            f"{n_zero} plant-year(s) read 0 g/km because every model in their "
            "RECORDED portfolio is electric. Verify against the plant's actual "
            "line-up before quoting: the model list comes from article "
            "snapshots and can miss combustion models still in production.")
    out.attrs.update(weighting=weighting, first_year=first_year,
                     min_coverage=min_coverage)
    return out.sort_values(["plant_uid", "year"]).reset_index(drop=True)


def plant_electrification(pmm: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Share of a plant's portfolio that is battery-electric, per year, alongside
    its average mass -- the two measures side by side on the same rows.

    Two independent readings of electrification are kept, deliberately:

        bev_nameplate  our own classifier (module03.classify_bev), which reads
                       the model NAME and is available for the whole period
        bev_registered EEA's registration-weighted BEV share of that model,
                       which is measured rather than inferred but only exists
                       where the model matched

    Where they disagree, the disagreement is the finding.
    """
    pmm = plant_model_mass(by_year=True) if pmm is None else pmm
    g = pmm.drop_duplicates(subset=["plant_uid", "year", "model"]).copy()
    agg = (g.groupby(["plant_uid", "plant_name", "country_iso2", "group",
                      "year"], as_index=False)
             .agg(n_models=("model", "nunique"),
                  bev_nameplate=("is_bev", "mean"),
                  bev_registered=("bev_share", "mean"),
                  mean_mass_kg=("mass_kg", "mean")))
    for c in ("bev_nameplate", "bev_registered"):
        agg[c] = 100 * agg[c]
    return agg.sort_values(["plant_uid", "year"]).reset_index(drop=True)


def electrification_vs_mass(pmm: Optional[pd.DataFrame] = None,
                            window: Tuple[int, int] = (2012, 2023)
                            ) -> pd.DataFrame:
    """
    Did the plants that electrified also get heavier?

    One row per plant on the balanced panel, with the change in BEV share and
    the change in average mass over the window. DESCRIPTIVE ONLY: this is a
    cross-plant association, not an effect. Thirty plants, no controls, and the
    two quantities are measured from the same model list, so a plant adding an
    electric SUV moves both by construction.
    """
    pmm = plant_model_mass(by_year=True) if pmm is None else pmm
    el = plant_electrification(pmm)
    pm = plant_mass(pmm, "equal")
    b = balanced_plants(pm, *window)
    keep = set(b["plant_uid"])
    y0, y1 = window

    e = el[el["plant_uid"].isin(keep) & el["year"].isin([y0, y1])]
    w = e.pivot_table(index=["plant_uid", "plant_name", "country_iso2", "group"],
                      columns="year",
                      values=["bev_nameplate", "mean_mass_kg"]).reset_index()
    w.columns = ["_".join(str(x) for x in c if x != "") for c in w.columns]
    ren = {f"bev_nameplate_{y0}": "bev_y0", f"bev_nameplate_{y1}": "bev_y1",
           f"mean_mass_kg_{y0}": "mass_y0", f"mean_mass_kg_{y1}": "mass_y1"}
    w = w.rename(columns=ren).dropna(subset=list(ren.values()))
    w["d_bev_pp"] = w["bev_y1"] - w["bev_y0"]
    w["d_mass_kg"] = w["mass_y1"] - w["mass_y0"]
    w.attrs.update(y0=y0, y1=y1, n_plants=len(w))
    if len(w) > 2:
        w.attrs["pearson_r"] = float(w["d_bev_pp"].corr(w["d_mass_kg"]))
        # Pearson on ranks == Spearman, and avoids a scipy dependency
        w.attrs["spearman_r"] = float(
            w["d_bev_pp"].rank().corr(w["d_mass_kg"].rank()))
    return w.sort_values("d_mass_kg", ascending=False).reset_index(drop=True)


def fig_plant_co2(co2: Optional[pd.DataFrame] = None,
                  year: int = 2023, top_n: int = 18,
                  save: Optional[str] = None):
    """Tailpipe CO2 of each plant's portfolio, and how it has moved."""
    import matplotlib.pyplot as plt
    _style()
    co2 = plant_co2() if co2 is None else co2
    d = co2[co2["year"] == year].sort_values("mean_co2_g_km", ascending=False)
    sel = pd.concat([d.head(top_n // 2), d.tail(top_n // 2)]).iloc[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(15.4, 6.4),
                             gridspec_kw={"width_ratios": [1.3, 1]})
    fig.suptitle(f"Tailpipe CO2 of the vehicles built at each plant, {year}",
                 fontsize=14.5, fontweight="bold", x=0.02, ha="left")

    ax = axes[0]
    cols = ["#B03A2E" if v > 100 else "#C6803D" if v > 20 else "#0B6E4F"
            for v in sel["mean_co2_g_km"]]
    n_zero = int(sel.get("all_electric_recorded",
                         pd.Series(False, index=sel.index)).sum())
    ax.barh(range(len(sel)), sel["mean_co2_g_km"], color=cols)
    ax.set_yticks(range(len(sel)))
    ax.set_yticklabels([f"{n[:30]} ({c})" for n, c
                        in zip(sel["plant_name"], sel["country_iso2"])],
                       fontsize=8.4)
    xmax = max(d["mean_co2_g_km"].max() * 1.20, 10)

    # A plant at 0 g/km draws no bar at all, so the bottom of the chart read as
    # a row of empty labels and looked like a plotting fault rather than a
    # finding. Give each zero an open stub, which says "measured, and it is
    # zero" rather than "nothing here".
    flags = sel.get("all_electric_recorded", pd.Series(False, index=sel.index))
    zero_stub = xmax * 0.012
    for i, v in enumerate(sel["mean_co2_g_km"]):
        if v <= 0.5:
            ax.barh(i, zero_stub, facecolor="none", edgecolor="#0B6E4F",
                    linewidth=1.1, hatch="////", zorder=3)

    for i, (v, f) in enumerate(zip(sel["mean_co2_g_km"], flags)):
        lab = "0  \u26a0  all recorded models electric" if v <= 0.5 \
              else f"{v:,.0f}" + ("  \u26a0" if f else "")
        ax.annotate(lab, (max(v, zero_stub), i),
                    xytext=(5, 0), textcoords="offset points", va="center",
                    fontsize=8.6, color="#0B6E4F" if v <= 0.5
                    else ("#B03A2E" if f else "#333333"))
    ax.set_xlabel("g CO2 / km (WLTP, tailpipe)")
    ax.set_xlim(0, xmax)
    ax.set_title(f"(a) Highest and lowest {top_n} plants", loc="left")

    ax = axes[1]
    # A mean on its own says nothing about whether the plants are moving
    # together or the average is being pulled by a few of them, which is the
    # live question here because the zeros sit inside it. Show the spread.
    t = (co2.groupby("year", as_index=False)
           .agg(mean_co2=("mean_co2_g_km", "mean"),
                med_co2=("mean_co2_g_km", "median"),
                p25=("mean_co2_g_km", lambda s: s.quantile(.25)),
                p75=("mean_co2_g_km", lambda s: s.quantile(.75)),
                lo=("mean_co2_g_km", "min"),
                hi=("mean_co2_g_km", "max"),
                n_plants=("plant_uid", "nunique")))
    ax.fill_between(t["year"], t["lo"], t["hi"], color="#B03A2E", alpha=.08,
                    lw=0, label="full range across plants")
    ax.fill_between(t["year"], t["p25"], t["p75"], color="#B03A2E", alpha=.20,
                    lw=0, label="middle half of plants")
    ax.plot(t["year"], t["med_co2"], "--", color="#7B241C", lw=1.8,
            label="median plant")
    ax.plot(t["year"], t["mean_co2"], "-o", color="#B03A2E", lw=2.6, ms=5,
            label="mean across plants")
    for pos, dx, ha in ((0, -8, "right"), (len(t) - 1, 8, "left")):
        r = t.iloc[pos]
        ax.annotate(f"{r['mean_co2']:,.0f}", (r["year"], r["mean_co2"]),
                    xytext=(dx, 6), textcoords="offset points", ha=ha,
                    fontsize=10, fontweight="bold", color="#B03A2E")
    ax.set_ylabel("g CO2 / km across plants")
    ax.set_ylim(bottom=0)
    ax.set_xlim(t["year"].min() - .6, t["year"].max() + 1.2)
    ax.set_title(f"(b) Since {co2.attrs.get('first_year', CO2_FIRST_YEAR)}, "
                 f"{int(t['n_plants'].min())}–{int(t['n_plants'].max())} "
                 "plants a year", loc="left", fontsize=10.5)
    ax.legend(fontsize=7.8, frameon=False, loc="lower left", ncol=2,
              handlelength=1.6, columnspacing=1.0)
    _int_years(ax, t["year"])

    _foot(fig,
          "These are tailpipe emissions as measured for Regulation (EU) "
          "2019/631, not life-cycle emissions. A battery-electric car counts as "
          "0 g/km by definition: the emissions of manufacturing its battery and "
          "of generating the electricity it runs on fall outside the measure "
          "entirely. A plant that switches to electric models therefore drops to "
          "zero on this chart while its real footprint does not.  The series "
          f"begins in {co2.attrs.get('first_year', CO2_FIRST_YEAR)} because the "
          "European test procedure changed between 2017 and 2019 and no "
          "comparable figure exists before then; plant-years where fewer than "
          "half the models carry a reading are left out.  A warning sign marks "
          "plants where every model on record is electric, which is why they "
          "read zero. Some genuinely build only electric vehicles; others simply "
          "have a recorded line-up that captured the new electric models and "
          "missed the rest. BMW Munich, for instance, is listed with four i4 "
          "variants for 2023 and no 3-series. Treat an individual plant at zero "
          "as a prompt to check the line-up rather than as a finding.  Panel (b) "
          "shows the mean with the spread behind it: the middle half of plants "
          "and the full range. The mean sits below the median throughout, "
          "because the plants reading zero pull it down.", y=-0.02)
    fig.subplots_adjust(top=.895, bottom=.19, wspace=.36)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def fig_electrification_vs_mass(ev: Optional[pd.DataFrame] = None,
                                save: Optional[str] = None):
    """Did the plants that electrified also get heavier?"""
    import matplotlib.pyplot as plt
    _style()
    ev = electrification_vs_mass() if ev is None else ev
    y0, y1 = ev.attrs["y0"], ev.attrs["y1"]

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.0))
    fig.suptitle(f"Electrification and vehicle mass at plant level, {y0}\u2013{y1}",
                 fontsize=14.5, fontweight="bold", x=0.02, ha="left")

    ax = axes[0]
    ax.scatter(ev["d_bev_pp"], ev["d_mass_kg"], s=54, color="#14746F",
               alpha=.75, edgecolor="white", zorder=3)
    ax.axhline(0, color="#333333", lw=1); ax.axvline(0, color="#333333", lw=1)
    big = ev.reindex(ev["d_mass_kg"].abs().sort_values(ascending=False).index).head(5)
    xr = ev["d_bev_pp"].max()

    def _short(uid):
        t = str(uid).split("_")[1:]
        keep = {"VW": "Volkswagen", "BMW": "BMW"}
        out = " ".join(keep.get(w, w.title()) for w in t)
        return out.replace("Kecskemet", "Kecskemét").replace("Gyor", "Győr")

    flip = False      # points stacked near zero alternate sides, so labels never overlap
    for r in big.sort_values("d_mass_kg", ascending=False).itertuples():
        if r.d_bev_pp > 0.75 * xr:
            left = True                    # keep labels inside the axes
        elif r.d_bev_pp < 0.1 * xr:
            left, flip = flip, not flip
        else:
            left = False
        ax.annotate(f"{_short(r.plant_uid)} ({r.country_iso2})",
                    (r.d_bev_pp, r.d_mass_kg), xytext=(-6 if left else 6, 6),
                    textcoords="offset points", fontsize=7.8, color="#444444",
                    ha="right" if left else "left")
    if len(ev) > 2 and ev["d_bev_pp"].std() > 0:
        z = np.polyfit(ev["d_bev_pp"], ev["d_mass_kg"], 1)
        xs = np.linspace(ev["d_bev_pp"].min(), ev["d_bev_pp"].max(), 20)
        ax.plot(xs, np.polyval(z, xs), color="#B03A2E", lw=1.5, ls="--",
                label=f"least squares (r = {ev.attrs.get('pearson_r', float('nan')):.2f})")
        # one plant can carry r on its own; say what it is without the largest mover
        top = ev["d_mass_kg"].abs().idxmax()
        rest = ev.drop(index=top)
        if len(rest) > 2:
            r_wo = rest["d_bev_pp"].corr(rest["d_mass_kg"])
            ax.plot([], [], " ", label=f"r = {r_wo:.2f} without {_short(ev.loc[top, 'plant_uid'])}")
        ax.legend(fontsize=8.8, frameon=False, loc="upper left")
    ax.set_xlabel(f"change in BEV share of the portfolio, {y0}\u2013{y1} (pp)")
    ax.set_ylabel(f"change in average mass ({y0}\u2013{y1}, kg)")
    ax.set_title(f"(a) One point per plant ({ev.attrs['n_plants']} plants)",
                 loc="left")

    ax = axes[1]
    grp = ev.copy()
    grp["band"] = np.where(grp["d_bev_pp"] <= 0, "no change",
                    np.where(grp["d_bev_pp"] < 25, "up to +25 pp", "+25 pp or more"))
    order = ["no change", "up to +25 pp", "+25 pp or more"]
    order = [b for b in order if (grp["band"] == b).any()]
    data = [grp.loc[grp["band"] == b, "d_mass_kg"].values for b in order]
    bp = ax.boxplot(data, labels=order, patch_artist=True, widths=.55)
    for p, c in zip(bp["boxes"], ["#CBD8D7", "#7FB2AE", "#14746F"]):
        p.set_facecolor(c); p.set_edgecolor("#444444")
    for med in bp["medians"]:
        med.set_color("#B03A2E"); med.set_linewidth(2)
    # headroom above the highest point, so the labels never sit on a marker
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.16 * (hi - lo))
    for i, b in enumerate(order, start=1):
        v = grp.loc[grp["band"] == b, "d_mass_kg"]
        ax.annotate(f"n={len(v)}\nmedian {v.median():+,.0f} kg",
                    (i, ax.get_ylim()[1]), xytext=(0, -6),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=8.6, color="#444444")
    ax.axhline(0, color="#333333", lw=1)
    ax.set_ylabel(f"change in average mass ({y0}\u2013{y1}, kg)")
    ax.set_title("(b) Grouped by how much the plant electrified", loc="left")

    _foot(fig,
          "This is an association across plants, and it should not be read as "
          f"an effect. It rests on {ev.attrs['n_plants']} plants with nothing "
          "held constant, and both axes are built from the same list of models, "
          "so a plant that adds a large electric model necessarily moves along "
          "both at once. Electric share is identified from the model name, which "
          "makes it available for the whole period; the registration data "
          "measures the same thing directly but only for later years.",
          y=-0.02)
    fig.subplots_adjust(top=.885, bottom=.20, wspace=.28)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


# ==============================================================================
#  THE SAME DECOMPOSITION, RESTRICTED TO MODELS BUILT AT TRACKED PLANTS
# ------------------------------------------------------------------------------
#  `powertrain_decomposition` above runs on every car in the registration
#  extract: 96.5m registrations, 255 makes, no reference to the plant panel.
#  The extract covers ELEVEN national markets -- BE, CZ, DE, ES, FR, HU, IT, PL,
#  PT, RO, SK (DE and ES missing in 2012) -- of which ten host a tracked plant;
#  Turkey and the UK host plants but are absent. It is also restricted to the
#  makes of the seven groups in the panel (Mercedes-Benz only from 2020).
#  That is its strength (it inherits none of the panel's coverage limits) and
#  its weakness (it says nothing about the plants this study is actually about).
#
#  The functions below re-run it on the subset of EEA registrations whose model
#  is built at one of the tracked assembly plants, which turns a fleet result
#  into a statement about the vehicles these particular factories put on the
#  market.
#
#  TWO THINGS TO KNOW BEFORE READING THE OUTPUT.
#
#  1. COVERAGE, NOT MARKET, DRIVES THE UNBALANCED VERSION. Matched registrations
#     swing 2.2m (2010) -> 1.7m (2012) -> 5.0m (2017) -> 1.8m (2025). The market
#     did not do that; the recorded line-ups did, for the same reason recorded
#     churn is an upper bound on real product replacement. Left uncorrected the
#     between-term partly measures the scraper. `balanced=True` keeps only models
#     observed in at least `min_years` years and is the version worth reporting.
#
#  2. THIS IS "MODELS THESE PLANTS BUILD", NOT "WHAT THESE PLANTS PRODUCED".
#     The panel records which models a plant builds, never how many, so each
#     model enters weighted by its EUROPEAN REGISTRATIONS. A model built at
#     several tracked plants (12% of model-years, up to four plants) is counted
#     ONCE, not once per plant: de-duplicating on (brand, model, year) is what
#     stops multi-sourced models being double weighted.
# ==============================================================================

TRACKED_MIN_YEARS: int = 12


def _fuel_class_panel(mp: pd.DataFrame) -> pd.DataFrame:
    """
    Registrations and mean mass by year and powertrain class, for the models in
    `mp`. The crosswalk is run once per fuel class, because the by-fuel EEA
    aggregate carries an extra key that `build_crosswalk` does not know about.
    """
    import eea_crosswalk as xw
    bf = pd.read_csv(fd.CONFIG.path("eea_model_mass_by_fuel.csv"))
    bf["bev_share"] = np.where(bf["fuel_class"].eq("BEV"), 1.0, 0.0)
    out = []
    for cls in ("ICE", "BEV", "HEV"):
        cw = xw.build_crosswalk(mp, eea=bf[bf["fuel_class"] == cls].copy(),
                                by_year=True)
        j = mp.merge(cw[["brand", "model", "year", "mass_kg", "registrations"]],
                     on=["brand", "model", "year"], how="inner")
        j = j.dropna(subset=["mass_kg", "registrations"])
        # one row per model-year: see note 2 above
        u = j.drop_duplicates(subset=["brand", "model", "year"])
        g = (u.groupby("year")
               .apply(lambda d: pd.Series({
                   "regs": d["registrations"].sum(),
                   "mass": np.average(d["mass_kg"], weights=d["registrations"])}))
               .reset_index())
        g["fuel_class"] = cls
        out.append(g)
    return pd.concat(out, ignore_index=True)


def _bennet(R: pd.DataFrame, M: pd.DataFrame, y0: int, y1: int) -> dict:
    """
    Symmetric (Bennet) decomposition of the change in mean mass between two
    years, the one written in Section B.4. Per powertrain class c:

        s1*m1 - s0*m0 = 1/2 (s0+s1)(m1-m0)  +  1/2 (m0+m1)(s1-s0)
                           within               between

    It is exact: the interaction term is zero by construction. A class with no
    measurable mass in y0 (battery cars in 2010) takes its first observed mass
    as m0; its y0 share is ~0, so that choice moves the split, not the total.
    Until 4 Oct 2026 the weights were averages over ALL years of the window,
    which left a residual of about 22 kg (reviewer, row 56).
    """
    M = M.mask(R < MIN_BEV_REGS)
    S = R.div(R.sum(axis=1), axis=0)
    yrs = [y for y in S.index if y0 <= y <= y1]
    first = M.loc[yrs].apply(
        lambda c: c.dropna().iloc[0] if c.notna().any() else np.nan)
    m0 = M.loc[yrs[0]].fillna(first); m1 = M.loc[yrs[-1]].fillna(first)
    s0 = S.loc[yrs[0]]; s1 = S.loc[yrs[-1]]
    within = float((0.5 * (s0 + s1) * (m1 - m0)).dropna().sum())
    between = float((0.5 * (m0 + m1) * (s1 - s0)).dropna().sum())
    total = float((s1 * m1).dropna().sum() - (s0 * m0).dropna().sum())
    return {"within": within, "between": between,
            "interaction": total - within - between, "total": total}


_tornqvist = _bennet     # old name, kept so nothing else breaks


def _chained(R: pd.DataFrame, M: pd.DataFrame, y0: int, y1: int,
             series: bool = False):
    """
    The same split, computed year by year and added up (a chain-linked
    decomposition). Each pair of consecutive years is split exactly with
    `_bennet`; the cumulated terms add up to the total change.

    WHY CHAIN. The endpoint split values the whole rise in a class's share at the
    average of its first and last masses. Battery cars are the case where that
    matters: their first measurable mass (2011, small city cars around 1,265 kg)
    is far below their mass when most of the share was gained (2019-2025, around
    1,800-1,950 kg), so the endpoint version understates the shift between
    powertrains. Chaining weights each year's change in share by that year's
    masses. Adopted 6 Oct 2026; the endpoint figures are kept in the attrs.
    """
    M = M.mask(R < MIN_BEV_REGS)
    yrs = [y for y in R.index if y0 <= y <= y1]
    M = M.loc[yrs].bfill().ffill()          # a class not yet measurable takes its first mass
    S = R.loc[yrs].div(R.loc[yrs].sum(axis=1), axis=0)
    rows, w, b = [dict(year=yrs[0], within_effect=0.0, between_effect=0.0,
                       total_change=0.0, interaction=0.0)], 0.0, 0.0
    for a, c in zip(yrs[:-1], yrs[1:]):
        w += float((0.5 * (S.loc[a] + S.loc[c]) * (M.loc[c] - M.loc[a])).sum())
        b += float((0.5 * (M.loc[a] + M.loc[c]) * (S.loc[c] - S.loc[a])).sum())
        rows.append(dict(year=c, within_effect=w, between_effect=b,
                         total_change=w + b, interaction=0.0))
    if series:
        return pd.DataFrame(rows)
    return {"within": w, "between": b, "interaction": 0.0, "total": w + b}



def tracked_model_panel(balanced: bool = True,
                        min_years: int = TRACKED_MIN_YEARS) -> pd.DataFrame:
    """
    Assembly-plant model rows, optionally restricted to well-covered PLANTS.

    BALANCE ON PLANTS, NOT ON MODELS -- this is not a free choice.
    The first version of this function kept models observed in at least
    `min_years` years, which is the obvious way to stabilise a panel and is
    wrong here. Electric nameplates are young by construction: the ID.3, the i4
    and the EQ range have existed for three to six years, so a twelve-year
    filter removes essentially every one of them. The surviving "electric"
    registrations were long-lived nameplates that later gained a battery
    variant, and the subset reached a 2025 battery share of 6.8% against 12.9%
    in the fleet. The between-term then measured the filter rather than the
    market, and the within/between ratio came out at 1.02 instead of 1.57.

    Balancing on plants keeps the site fixed and lets its line-up change, which
    is what the question actually requires. It reaches a 2025 battery share of
    12.8%, against 12.9% in the fleet, so the subset is representative on the
    dimension the decomposition turns on.
    """
    mp = model_panel()
    mp = mp[(mp["classification"] == "assembly")
            & (mp["output_unit"] == "vehicles")]
    if balanced:
        cnt = mp.groupby("plant_uid")["year"].nunique()
        mp = mp[mp["plant_uid"].isin(cnt[cnt >= min_years].index)]
    return mp


def tracked_powertrain_decomposition(y0: int = 2012, y1: int = 2025,
                                     balanced: bool = True,
                                     min_years: int = TRACKED_MIN_YEARS
                                     ) -> pd.DataFrame:
    """Year-by-year within/between/interaction for the tracked-plant subset."""
    mp = tracked_model_panel(balanced=balanced, min_years=min_years)
    p = _fuel_class_panel(mp)
    R = p.pivot(index="year", columns="fuel_class", values="regs").fillna(0.0)
    M = p.pivot(index="year", columns="fuel_class", values="mass")
    out = _chained(R, M, y0, y1, series=True)
    out.attrs["endpoint"] = _bennet(R, M, y0, y1)
    S_ = R.div(R.sum(axis=1), axis=0)
    out.attrs["bev_share_end"] = float(S_.loc[y1].get("BEV", 0.0))
    out.attrs["y0"], out.attrs["y1"] = y0, y1
    out.attrs["n_models"] = mp[["brand", "model"]].drop_duplicates().shape[0]
    out.attrs["n_plants"] = mp["plant_uid"].nunique()
    return out


def powertrain_universe_comparison(full: bool = False) -> pd.DataFrame:
    """
    The decomposition on the whole fleet and on the tracked plants, over the same
    years, so the two ratios can be read against each other.

    `full=True` adds the 2010 start and the unbalanced plant subset, which are
    reported in the annex as sensitivity rather than in the figure.
    """
    bf = pd.read_csv(fd.CONFIG.path("eea_model_mass_by_fuel.csv"))
    bf["mw"] = bf["mass_mean_kg"] * bf["registrations"]
    g = (bf.groupby(["year", "fuel_class"], as_index=False)
           .agg(regs=("registrations", "sum"), mw=("mw", "sum")))
    g["mass"] = g["mw"] / g["regs"]
    Rf = g.pivot(index="year", columns="fuel_class", values="regs").fillna(0.0)
    Mf = g.pivot(index="year", columns="fuel_class", values="mass")
    Sf = Rf.div(Rf.sum(axis=1), axis=0)

    rows = []
    starts = [2010, 2012] if full else [2012]
    for a in starts:
        rows.append(dict(universe=f"Every car registered\nin the EU27, {a}–2025",
                         tracked=False, **_chained(Rf, Mf, a, 2025),
                         bev_2025=float(Sf["BEV"][2025])))
    subsets = [(True, "balanced")] + ([(False, "unbalanced")] if full else [])
    for bal, tag in subsets:
        p = _fuel_class_panel(tracked_model_panel(balanced=bal))
        R = p.pivot(index="year", columns="fuel_class", values="regs").fillna(0.0)
        M = p.pivot(index="year", columns="fuel_class", values="mass")
        S = R.div(R.sum(axis=1), axis=0)
        lab = ("Only what the tracked\nplants build, 2012–2025" if not full
               else f"Tracked plants, {tag}, 2012–2025")
        rows.append(dict(universe=lab, tracked=True,
                         **_chained(R, M, 2012, 2025),
                         bev_2025=float(S["BEV"][2025])))
    out = pd.DataFrame(rows)
    out["ratio"] = out["within"] / out["between"]
    return out


def _class_mass_frame(R: pd.DataFrame, M: pd.DataFrame) -> pd.DataFrame:
    """Long year x class frame with registrations, mass (thin cells blanked) and share."""
    g = (R.stack().rename("regs").to_frame()
           .join(M.stack().rename("mass")).reset_index())
    g.columns = ["year", "fuel_class", "regs", "mass"]
    g.loc[g["regs"] < MIN_BEV_REGS, "mass"] = np.nan
    g["share"] = 100 * g["regs"] / g.groupby("year")["regs"].transform("sum")
    return g


def _fleet_class_matrices():
    bf = pd.read_csv(fd.CONFIG.path("eea_model_mass_by_fuel.csv"))
    bf["mw"] = bf["mass_mean_kg"] * bf["registrations"]
    g = (bf.groupby(["year", "fuel_class"], as_index=False)
           .agg(regs=("registrations", "sum"), mw=("mw", "sum")))
    g["mass"] = g["mw"] / g["regs"]
    return (g.pivot(index="year", columns="fuel_class", values="regs").fillna(0.0),
            g.pivot(index="year", columns="fuel_class", values="mass"))


def _tracked_class_matrices():
    p = _fuel_class_panel(tracked_model_panel(balanced=True))
    return (p.pivot(index="year", columns="fuel_class", values="regs").fillna(0.0),
            p.pivot(index="year", columns="fuel_class", values="mass"))


_PT_COL = {"ICE": "#8A7A6D", "BEV": "#0B6E4F", "HEV": "#2C6FB3"}
_PT_LBL = {"ICE": "internal combustion", "BEV": "battery electric", "HEV": "hybrid"}
_C_WITHIN, _C_BETWEEN, _C_TOTAL = "#7B3294", "#D2760A", "#111111"


def fig_powertrain_tracked(save: Optional[str] = None):
    """
    Figure 4.10 for the vehicles of the tracked plants (changed 7 Oct 2026).

    Panel (a): the mass within each powertrain of the models built at the
    tracked assembly plants, weighted by their EU27 registrations. Panel (b):
    the chained decomposition of the change since 2012 for the same vehicles.
    The fleet comparison and the within/between ratios are in the annex figure
    `fig_powertrain_tracked_vs_fleet`.
    """
    import matplotlib.pyplot as plt
    _style()
    dec = tracked_powertrain_decomposition()
    y0, y1 = dec.attrs["y0"], dec.attrs["y1"]
    R, M = _tracked_class_matrices()
    g = _class_mass_frame(R.loc[y0:y1], M.loc[y0:y1])

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.8))
    fig.suptitle(f"Vehicle mass by powertrain, vehicles of the tracked plants, {y0}–{y1}",
                 fontsize=14.5, fontweight="bold", x=0.02, ha="left")
    ax = axes[0]
    ends = {}
    for k in ("ICE", "HEV", "BEV"):
        s = g[(g["fuel_class"] == k) & g["mass"].notna()].sort_values("year")
        if not len(s):
            continue
        ax.plot(s["year"], s["mass"], "-o", color=_PT_COL[k], lw=2.4, ms=4.5,
                label=_PT_LBL[k])
        ends[k] = (s["year"].iloc[-1], s["mass"].iloc[-1])
    span = g["mass"].max() - g["mass"].min(); prev = None
    for k, (xe, ye) in sorted(ends.items(), key=lambda kv: kv[1][1]):
        yy = ye if prev is None or ye - prev > .07 * span else prev + .07 * span
        prev = yy
        ax.annotate(f"{ye:,.0f} kg", (xe, yy), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9.6,
                    fontweight="bold", color=_PT_COL[k])
    ice = g[(g.fuel_class == "ICE") & g["mass"].notna()].sort_values("year")
    ax.annotate(f"combustion cars alone: {ice['mass'].iloc[-1] - ice['mass'].iloc[0]:+,.0f} kg",
                xy=(.97, .05), xycoords="axes fraction", ha="right",
                fontsize=9.8, fontweight="bold", color=_PT_COL["ICE"])
    ax.set_ylabel("kilograms")
    ax.set_xlim(y0 - .5, y1 + 3.6)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.set_title("(a) Mass within each powertrain", loc="left")
    _int_years(ax, range(y0, y1 + 1))

    ax = axes[1]
    w = dec.set_index("year")
    ax.plot(w.index, w["within_effect"], "-o", color=_C_WITHIN, lw=2.4, ms=4.5,
            label="every powertrain getting heavier (within)")
    ax.plot(w.index, w["between_effect"], "-o", color=_C_BETWEEN, lw=2.4, ms=4.5,
            label="market shifting between powertrains (between)")
    ax.plot(w.index, w["total_change"], "-", color=_C_TOTAL, lw=2.0, ls=":",
            label="total change")
    for lab, col in (("within_effect", _C_WITHIN), ("between_effect", _C_BETWEEN),
                     ("total_change", _C_TOTAL)):
        ax.annotate(f"{w[lab].iloc[-1]:+,.0f} kg", (w.index[-1], w[lab].iloc[-1]),
                    xytext=(7, 0), textcoords="offset points", va="center",
                    fontsize=9.4, fontweight="bold", color=col)
    ax.axhline(0, color="#333333", lw=1)
    ax.set_ylabel(f"kg relative to {y0}")
    ax.set_xlim(w.index.min() - .5, w.index.max() + 3.6)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.set_title(f"(b) Decomposition of the change since {y0}", loc="left")
    Rf, _ = _fleet_class_matrices()
    bev_fleet = float(Rf.loc[y1, "BEV"] / Rf.loc[y1].sum())
    ax.annotate(f"{dec.attrs['n_plants']} plants reporting in at\nleast "
                f"{TRACKED_MIN_YEARS} years, {dec.attrs['n_models']} models;\n"
                f"battery share {100*dec.attrs['bev_share_end']:.1f}% in {y1},\n"
                f"against {100*bev_fleet:.1f}% in the EU27",
                xy=(.02, .50), xycoords="axes fraction", ha="left", va="bottom",
                fontsize=8.4, color="#5A5A5A", linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                          edgecolor="#CCCCCC", linewidth=.6, alpha=.96))
    _int_years(ax, w.index)
    fig.subplots_adjust(top=.87, bottom=.10, wspace=.26)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def fig_powertrain_tracked_vs_fleet(save: Optional[str] = None):
    """
    Annex figure (added 7 Oct 2026): the tracked plants (solid) against every
    car registered in the EU27 (dashed), over the same years. (a) mass within
    each powertrain; (b) the chained decomposition since 2012. The ratio panel
    was dropped on 7 Oct 2026; the ratios are given in the text.
    """
    import matplotlib.pyplot as plt
    _style()
    dec = tracked_powertrain_decomposition()
    y0, y1 = dec.attrs["y0"], dec.attrs["y1"]
    Rt, Mt = _tracked_class_matrices(); Rf, Mf = _fleet_class_matrices()
    gt = _class_mass_frame(Rt.loc[y0:y1], Mt.loc[y0:y1])
    gf = _class_mass_frame(Rf.loc[y0:y1], Mf.loc[y0:y1])
    fl = _chained(Rf, Mf, y0, y1, series=True).set_index("year")
    tr = dec.set_index("year")

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.9))
    fig.suptitle(f"The tracked plants against the EU27 fleet, {y0}–{y1}",
                 fontsize=14.5, fontweight="bold", x=0.02, ha="left")
    ax = axes[0]
    for k in ("ICE", "HEV", "BEV"):
        s = gt[(gt.fuel_class == k) & gt["mass"].notna()].sort_values("year")
        t = gf[(gf.fuel_class == k) & gf["mass"].notna()].sort_values("year")
        ax.plot(s["year"], s["mass"], "-o", color=_PT_COL[k], lw=2.3, ms=3.8,
                label=f"{_PT_LBL[k]}")
        ax.plot(t["year"], t["mass"], "--", color=_PT_COL[k], lw=1.5, alpha=.85)
    ax.plot([], [], "-", color="#555555", lw=2.3, label="tracked plants")
    ax.plot([], [], "--", color="#555555", lw=1.5, label="EU27 fleet")
    ax.set_ylabel("kilograms"); ax.legend(fontsize=8.4, frameon=False, loc="upper left")
    ax.set_title("(a) Mass within each powertrain", loc="left")
    _int_years(ax, range(y0, y1 + 1))

    ax = axes[1]
    for col, c in (("within_effect", _C_WITHIN), ("between_effect", _C_BETWEEN),
                   ("total_change", _C_TOTAL)):
        ax.plot(tr.index, tr[col], "-o" if col != "total_change" else ":",
                color=c, lw=2.3, ms=3.8)
        ax.plot(fl.index, fl[col], "--", color=c, lw=1.5, alpha=.85)
        ax.annotate(f"{tr[col].iloc[-1]:+,.0f}", (tr.index[-1], tr[col].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, fontweight="bold", color=c)
        ax.annotate(f"{fl[col].iloc[-1]:+,.0f}", (fl.index[-1], fl[col].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=8.6, color=c, alpha=.85)
    for lab, c, ls in (("within", _C_WITHIN, "-"), ("between", _C_BETWEEN, "-"),
                       ("total change", _C_TOTAL, ":")):
        ax.plot([], [], ls, color=c, lw=2.3, label=lab)
    ax.plot([], [], "-", color="#555555", lw=2.3, label="tracked plants")
    ax.plot([], [], "--", color="#555555", lw=1.5, label="EU27 fleet")
    ax.axhline(0, color="#333333", lw=1)
    ax.set_ylabel(f"kg relative to {y0}"); ax.set_xlim(y0 - .5, y1 + 2.4)
    ax.legend(fontsize=8.4, frameon=False, loc="upper left")
    ax.set_title(f"(b) Decomposition of the change since {y0}", loc="left")
    _int_years(ax, tr.index)

    fig.subplots_adjust(top=.86, bottom=.10, wspace=.24)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


# ==============================================================================
#  ADDED 6 OCT 2026 -- THREE VIEWS OF THE RISE IN MASS, ON THE FULL EU27 DATA
# ------------------------------------------------------------------------------
#  Inputs built by `eea_build_eu27.py` from the full EEA files:
#    eea_mass_hist.csv     year x powertrain x 10-kg bin, registrations (EU27)
#    eea_country_class.csv country x seven-groups flag x year x powertrain
#  and the usual eea_model_mass_by_fuel.csv (now EU27, every make).
# ==============================================================================

_MAKE_ALIAS = {"VOLKSWAGEN": "VW", "VW": "VW", "MERCEDES": "MERCEDES", "CITRO": "CITROEN",
               "AUTOMOBILESCITRO": "CITROEN", "AUTOMOBILESPEUGEOT": "PEUGEOT",
               "KODA": "SKODA", "DSAUTOMOBILES": "DS"}
_GENERIC = {"ID", "MODEL", "SERIE", "SERIES", "CLASS", "CLASSE", "KLASSE"}


def _make_key(s) -> str:
    import re
    n = re.sub(r"[^A-Z]", "", str(s).upper())
    for k, v in sorted(_MAKE_ALIAS.items(), key=lambda x: -len(x[0])):
        if n.startswith(k):
            return v
    return n[:10]


def _nameplate(cname, make_key) -> str:
    """Model family: the first word of the cleaned EEA name (two words when the
    first is a generic or a short code, e.g. 'ID 3', 'MODEL 3', 'C 220')."""
    import eea_crosswalk as xw
    t = [x for x in xw.eea_stem(str(cname)).split() if x != make_key] or [""]
    return " ".join(t[:2]) if (len(t[0]) <= 2 or t[0] in _GENERIC) else t[0]


def nameplate_decomposition() -> pd.DataFrame:
    """
    The rise in average mass split three ways, year by year and added up:

      within_nameplate     the same nameplate in the same powertrain getting
                           heavier (new generations, heavier versions)
      between_nameplates   registrations shifting between nameplates of the
                           same powertrain (towards SUVs and larger cars), and
                           nameplates entering and leaving the market
      between_powertrains  the shift between combustion, hybrid and electric

    Each pair of years is split exactly; the three terms add up to the change in
    the average. A nameplate is a make plus model family (see `_nameplate`); in
    every pair of years the nameplates present in both cover 96-99% of
    registrations.
    """
    f = pd.read_csv(fd.CONFIG.path("eea_model_mass_by_fuel.csv"))
    f = f.dropna(subset=["eea_make", "eea_model"])
    f["fuel_class"] = f["fuel_class"].replace({"PHEV": "HEV"})
    f["mk"] = f["eea_make"].map({m: _make_key(m) for m in f["eea_make"].unique()})
    pairs = f[["mk", "eea_model"]].drop_duplicates()
    pairs["plate"] = [_nameplate(c, m) for c, m in zip(pairs["eea_model"], pairs["mk"])]
    f = f.merge(pairs, on=["mk", "eea_model"])
    f["mw"] = f["mass_mean_kg"] * f["registrations"]
    u = f.groupby(["year", "mk", "plate", "fuel_class"], as_index=False).agg(
        r=("registrations", "sum"), mw=("mw", "sum"))
    yrs = sorted(u["year"].unique())
    rows = [dict(year=yrs[0], total=0.0, within_nameplate=0.0, between_nameplates=0.0,
                 between_powertrains=0.0, continuing_share=1.0)]
    cum = dict(total=0.0, within_nameplate=0.0, between_nameplates=0.0, between_powertrains=0.0)
    key = ["mk", "plate", "fuel_class"]
    for y0, y1 in zip(yrs[:-1], yrs[1:]):
        A = u[u["year"] == y0].set_index(key); B = u[u["year"] == y1].set_index(key)
        wA, wB = A["r"] / A["r"].sum(), B["r"] / B["r"].sum()
        mA, mB = A["mw"] / A["r"], B["mw"] / B["r"]
        tot = B["mw"].sum() / B["r"].sum() - A["mw"].sum() / A["r"].sum()
        com = A.index.intersection(B.index)
        within = float((0.5 * (wA[com] + wB[com]) * (mB[com] - mA[com])).sum())
        cA = A.groupby("fuel_class")[["r", "mw"]].sum(); cB = B.groupby("fuel_class")[["r", "mw"]].sum()
        cls = cA.index.union(cB.index)
        sA = (cA["r"] / cA["r"].sum()).reindex(cls, fill_value=0)
        sB = (cB["r"] / cB["r"].sum()).reindex(cls, fill_value=0)
        kA = (cA["mw"] / cA["r"]).reindex(cls); kB = (cB["mw"] / cB["r"]).reindex(cls)
        kA, kB = kA.fillna(kB), kB.fillna(kA)
        bpt = float((0.5 * (kA + kB) * (sB - sA)).sum())
        for k, v in (("total", tot), ("within_nameplate", within),
                     ("between_nameplates", tot - within - bpt), ("between_powertrains", bpt)):
            cum[k] += v
        rows.append(dict(year=int(y1), **cum,
                         continuing_share=float(0.5 * (wA[com].sum() + wB[com].sum()))))
    d = pd.DataFrame(rows)
    # the best-selling combustion nameplates sold in both end years
    v = u[u["fuel_class"] == "ICE"].groupby(["year", "mk", "plate"], as_index=False)[["r", "mw"]].sum()
    v["m"] = v["mw"] / v["r"]
    e0 = v[v["year"] == yrs[0]].set_index(["mk", "plate"]); e1 = v[v["year"] == yrs[-1]].set_index(["mk", "plate"])
    both = e0[e0["r"] >= 20000].index.intersection(e1[e1["r"] >= 20000].index)
    ex = pd.DataFrame({"r": e0["r"][both] + e1["r"][both], "m0": e0["m"][both],
                       "m1": e1["m"][both]}).sort_values("r", ascending=False).head(14)
    d.attrs.update(y0=int(yrs[0]), y1=int(yrs[-1]), examples=ex)
    return d


def fig_mass_by_nameplate(save: Optional[str] = None):
    import matplotlib.pyplot as plt
    _style()
    d = nameplate_decomposition().set_index("year"); ex = d.attrs["examples"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.suptitle(f"Change in the average mass of new cars by source, EU27, "
                 f"{d.attrs['y0']}–{d.attrs['y1']}", fontsize=14, fontweight="bold", x=0.02, ha="left")
    ax = axes[0]
    parts = {"within_nameplate": ("#7B3294", "the same nameplate getting heavier"),
             "between_nameplates": ("#C6803D", "registrations shifting to heavier nameplates (same powertrain)"),
             "between_powertrains": ("#2C6FB3", "shift between powertrains")}
    pos = np.zeros(len(d)); neg = np.zeros(len(d))
    for k, (col, lab) in parts.items():
        v = d[k].to_numpy(); base = np.where(v >= 0, pos, neg)
        ax.bar(d.index, v, bottom=base, color=col, width=.75, label=lab)
        pos += np.where(v >= 0, v, 0); neg += np.where(v < 0, v, 0)
        ax.annotate(f"{v[-1]:+,.0f}", (d.index[-1] + .45, v[-1]), fontsize=9.4,
                    fontweight="bold", color=col, va="center")
    ax.plot(d.index, d["total"], ":", color="#111111", lw=2, marker="o", ms=3, label="total change")
    ax.annotate(f"total {d['total'].iloc[-1]:+,.0f} kg", (d.index[-1], d["total"].iloc[-1]),
                xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9.6, fontweight="bold")
    ax.axhline(0, color="#333333", lw=1)
    ax.set_ylabel(f"kg relative to {d.attrs['y0']}, added up year by year")
    ax.set_xlim(d.index.min() - .6, d.index.max() + 1.6)
    ax.legend(fontsize=8.8, frameon=False, loc="upper left")
    ax.set_title(f"(a) Contribution of each source, cumulated since {d.attrs['y0']}", loc="left")
    _int_years(ax, d.index)
    ax = axes[1]; y = np.arange(len(ex))[::-1]
    for yi, (idx, r) in zip(y, ex.iterrows()):
        ax.plot([r.m0, r.m1], [yi, yi], color="#BBBBBB", lw=2, zorder=1)
        ax.scatter(r.m0, yi, color="#9AA5AD", s=34, zorder=2)
        ax.scatter(r.m1, yi, color="#7B3294", s=34, zorder=2)
        ax.annotate(f"{r.m1 - r.m0:+.0f}", (max(r.m0, r.m1), yi), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.6)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{p.title().replace('C 3', 'C3')} ({'VW' if m == 'VW' else m.title()})"
                        for m, p in ex.index], fontsize=8.8)
    ax.scatter([], [], color="#9AA5AD", label=str(d.attrs["y0"]))
    ax.scatter([], [], color="#7B3294", label=str(d.attrs["y1"]))
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, .78)); ax.set_xlabel("average mass of new registrations (kg)")
    ax.grid(axis="y", visible=False)
    ax.set_title("(b) Best-selling nameplates, combustion versions", loc="left")
    fig.subplots_adjust(top=.86, bottom=.10, wspace=.30)
    return _save(fig, save)


def mass_percentiles() -> Dict[str, pd.DataFrame]:
    h = pd.read_csv(fd.CONFIG.path("eea_mass_hist.csv"))
    t = h.groupby(["year", "bin_lo"])["registrations"].sum().unstack(fill_value=0).sort_index()
    cum = t.cumsum(axis=1).div(t.sum(axis=1), axis=0)
    pct = pd.DataFrame({q: cum.apply(lambda r: r.index[np.searchsorted(r.values, q)] + 5, axis=1)
                        for q in (.1, .25, .5, .75, .9)})
    above = pd.DataFrame({k: 100 * t.loc[:, t.columns >= k].sum(axis=1) / t.sum(axis=1)
                          for k in (1500, 1800, 2000)})
    return {"hist": t, "pct": pct, "above": above}


def fig_mass_percentiles(save: Optional[str] = None):
    import matplotlib.pyplot as plt
    _style()
    m = mass_percentiles(); P, A, H = m["pct"], m["above"], m["hist"]; yrs = P.index
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    fig.suptitle(f"The mass of new cars, EU27, {yrs.min()}–{yrs.max()}", fontsize=14,
                 fontweight="bold", x=0.02, ha="left")
    ax = axes[0]; C = "#14746F"
    ax.fill_between(yrs, P[.1], P[.9], color=C, alpha=.13, lw=0, label="10th–90th percentile")
    ax.fill_between(yrs, P[.25], P[.75], color=C, alpha=.28, lw=0, label="25th–75th percentile")
    ax.plot(yrs, P[.5], color=C, lw=2.4, marker="o", ms=3, label="median")
    for q, lab in ((.1, "10th"), (.5, "median"), (.9, "90th")):
        ax.annotate(f"{lab} {P[q].iloc[-1]:,.0f}", (yrs[-1], P[q].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.6)
        ax.annotate(f"{P[q].iloc[0]:,.0f}", (yrs[0], P[q].iloc[0]), xytext=(-6, 0),
                    textcoords="offset points", va="center", ha="right", fontsize=8.6)
    ax.set_xlim(yrs.min() - 2.6, yrs.max() + 2.8); ax.set_ylabel("kilograms")
    ax.legend(frameon=False, fontsize=8.6, loc="upper left")
    ax.set_title("(a) Spread of new-car mass", loc="left"); ax.set_xticks(range(int(yrs.min()), int(yrs.max()) + 1, 3))
    ax = axes[1]
    for y, col in ((yrs.min(), "#9AA5AD"), (2018, "#C6803D"), (yrs.max(), C)):
        row = H.loc[y]; c = row.index + 5
        dens = np.convolve(100 * row / row.sum(), np.ones(9) / 9, mode="same")
        ax.plot(c, dens, color=col, lw=2, label=str(y)); ax.fill_between(c, dens, color=col, alpha=.08)
    ax.set_xlim(700, 2800); ax.set_xlabel("kilograms"); ax.set_ylabel("% of registrations per 10 kg")
    ax.legend(frameon=False); ax.set_title("(b) Distribution in 2010, 2018 and 2025", loc="left")
    ax = axes[2]
    for k, col in zip(A.columns, ("#9AA5AD", "#C6803D", "#7B3294")):
        ax.plot(yrs, A[k], color=col, lw=2.2, marker="o", ms=3, label=f"above {k:,} kg")
        ax.annotate(f"{A[k].iloc[-1]:.0f}%", (yrs[-1], A[k].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=col)
        ax.annotate(f"{A[k].iloc[0]:.0f}%", (yrs[0], A[k].iloc[0]), xytext=(-6, 0),
                    textcoords="offset points", va="center", ha="right", fontsize=9, color=col)
    ax.set_xlim(yrs.min() - 2.1, yrs.max() + 2.2); ax.set_ylabel("% of new cars")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("(c) Share of new cars above fixed weights", loc="left"); ax.set_xticks(range(int(yrs.min()), int(yrs.max()) + 1, 3))
    fig.subplots_adjust(top=.85, bottom=.12, wspace=.36)
    return _save(fig, save)


ELEVEN_MARKETS = ("BE", "CZ", "DE", "ES", "FR", "HU", "IT", "PL", "PT", "RO", "SK")
EU27_CODES = ("AT BE BG CY CZ DE DK EE ES FI FR GR HR HU IE IT LT LU LV MT NL PL PT "
              "RO SE SI SK").split()


def three_population_decomposition() -> pd.DataFrame:
    """The powertrain split on three populations and two start years, chained
    (the method of the main text) and at the endpoints (for comparison)."""
    c = pd.read_csv(fd.CONFIG.path("eea_country_class.csv"))
    c["fuel_class"] = c["fuel_class"].replace({"PHEV": "HEV"})

    def mats(d):
        g = d.groupby(["year", "fuel_class"])[["registrations", "mass_w"]].sum().reset_index()
        g["mass"] = g["mass_w"] / g["registrations"]
        return (g.pivot(index="year", columns="fuel_class", values="registrations").fillna(0.0),
                g.pivot(index="year", columns="fuel_class", values="mass"))
    pops = {"EU27, every car": c[c["country"].isin(EU27_CODES)],
            "Seven groups' brands,\n11 markets": c[c["country"].isin(ELEVEN_MARKETS) & c["group7"]]}
    rows = []
    for lab, d in pops.items():
        R, M = mats(d)
        for y0 in (2010, 2012):
            ch, ep = _chained(R, M, y0, 2025), _bennet(R, M, y0, 2025)
            rows.append(dict(population=lab, y0=y0, **ch, ep_within=ep["within"],
                             ep_between=ep["between"]))
    p = _fuel_class_panel(tracked_model_panel(balanced=True))
    R = p.pivot(index="year", columns="fuel_class", values="regs").fillna(0.0)
    M = p.pivot(index="year", columns="fuel_class", values="mass")
    ch, ep = _chained(R, M, 2012, 2025), _bennet(R, M, 2012, 2025)
    rows.append(dict(population="Vehicles of the\ntracked plants", y0=2012, **ch,
                     ep_within=ep["within"], ep_between=ep["between"]))
    return pd.DataFrame(rows)


def three_population_mass_series() -> pd.DataFrame:
    """Average mass of a newly registered car, by year, for the populations of
    panel (a) of the three-decompositions figure."""
    c = pd.read_csv(fd.CONFIG.path("eea_country_class.csv"))
    pops = {"EU27, all brands": c["country"].isin(EU27_CODES),
            "11 markets, seven groups' brands": c["country"].isin(ELEVEN_MARKETS) & c["group7"],
            "11 markets, all brands": c["country"].isin(ELEVEN_MARKETS)}
    out = {}
    for lab, m in pops.items():
        g = c[m].groupby("year")[["registrations", "mass_w"]].sum()
        out[lab] = g["mass_w"] / g["registrations"]
    return pd.DataFrame(out)


def fig_three_decompositions(save: Optional[str] = None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    _style()
    t = three_population_decomposition()
    ms = three_population_mass_series()
    order = [("Vehicles of the\ntracked plants", 2012), ("Seven groups' brands,\n11 markets", 2012),
             ("EU27, every car", 2012), ("Seven groups' brands,\n11 markets", 2010),
             ("EU27, every car", 2010)]
    fig, (axa, ax) = plt.subplots(1, 2, figsize=(17, 6.6),
                                  gridspec_kw={"width_ratios": [1, 1.25]})
    fig.suptitle("The powertrain split on three populations and two start years",
                 fontsize=14, fontweight="bold", x=0.02, ha="left")

    sty = {"EU27, all brands": dict(color="#14746F", lw=2.4, ls="-"),
           "11 markets, seven groups' brands": dict(color="#C6803D", lw=2.4, ls="-"),
           "11 markets, all brands": dict(color="#8E979E", lw=1.6, ls="--")}
    for lab, st in sty.items():
        axa.plot(ms.index, ms[lab], marker="o", ms=3.5, label=lab, **st)
    axa.set_ylabel("kilograms"); axa.legend(frameon=False, fontsize=9.5, loc="upper left")
    _int_years(axa, pd.Series(ms.index))
    axa.set_title("(a) Average mass of a newly registered car", loc="left")

    y = np.arange(len(order))[::-1]; h = .36
    for yi, (pop, y0) in zip(y, order):
        r = t[(t["population"] == pop) & (t["y0"] == y0)].iloc[0]
        ax.barh(yi + h/2, r["within"], h, color="#7B3294"); ax.barh(yi - h/2, r["between"], h, color="#D2760A")
        ax.plot([r["ep_within"]], [yi + h/2], "|", color="black", ms=14, mew=2)
        ax.plot([r["ep_between"]], [yi - h/2], "|", color="black", ms=14, mew=2)
        ax.annotate(f"{r['within']:.0f}", (r["within"], yi + h/2), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=9, color="#7B3294", fontweight="bold")
        ax.annotate(f"{r['between']:.0f}", (r["between"], yi - h/2), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=9, color="#D2760A", fontweight="bold")
        ax.annotate(f"total {r['total']:.0f} kg\nwithin {100*r['within']/r['total']:.0f}%"
                    f" (endpoints {100*r['ep_within']/r['total']:.0f}%)",
                    (215, yi), va="center", fontsize=8.8, color="#444444")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{p}, {y0}–2025" if "tracked" not in p else f"{p}, 2012–2025" for p, y0 in order], fontsize=9)
    ax.axhline(1.5, color="#999999", ls=":", lw=1)
    ax.set_xlim(0, 300); ax.set_xlabel("kilograms"); ax.grid(axis="y", visible=False)
    ax.set_title("(b) Within and between, by population and start year", loc="left")
    ax.legend(handles=[Patch(color="#7B3294", label="every powertrain getting heavier (within)"),
                       Patch(color="#D2760A", label="shift between powertrains (between)"),
                       Line2D([], [], marker="|", ls="", color="black", ms=12, mew=2,
                              label="same term, endpoint method")],
              frameon=False, fontsize=8.8, loc="upper center", bbox_to_anchor=(.4, -.09), ncol=3)
    fig.subplots_adjust(top=.87, bottom=.17, left=.05, right=.98, wspace=.36)
    return _save(fig, save)
