# -*- coding: utf-8 -*-
"""
================================================================================
 MODULE 04 — MACRO-MICRO INTEGRATION
 How much of a region's automotive employment IS the plants we track?
================================================================================

WHAT THIS MODULE IS FOR
-----------------------
Modules 01 and 02 ran in parallel worlds: exhaustive regional statistics on one
side, a hand-built plant panel on the other. This module joins them through the
plant -> NUTS-2 crosswalk and asks the question neither can answer alone:

    for each region and year,
    tracked plant employment  /  regional NACE C29 employment  =  ?

That ratio does three jobs at once:

  1. **Representativeness.** It says how much of the thesis's micro panel
     stands for the sector in each region — the honest answer to "is your
     41-plant panel representative?"
  2. **A substantive result.** NACE C29 covers vehicles, bodies AND parts, so
     the RESIDUAL (region minus tracked plants) measures the non-assembler
     tissue: the supplier ecosystem around each assembly site. Its trajectory is
     a finding, not a nuisance.
  3. **A data-quality detector.** A ratio above 1 is impossible if everything is
     right, so it flags stale values, mis-assigned regions or measurement
     mismatch. It works: it caught Ford Genk, which closed in December 2014 yet
     carries 4,300 employees through 2025 in the source panel (see
     `implausible_ratios`).

FOUR THINGS THAT ARE NOT ERRORS
-------------------------------
1. **Ratios well below 1 are expected.** NACE C29 is the whole sector; an
     assembly plant is a part of it. The median across our regions is ~0.41.
2. **Measurement bases differ.** Plant headcount is employer-reported; Eurostat
     reports "persons employed in local units". Close, not identical.
3. **Suppression is not random, and here it is measurable.** Regions hosting a
     tracked plant have LOWER missingness (11% vs 28% — they are big enough to
     report) but HIGHER confidentiality suppression (8.6% vs 6.6% — one plant
     dominates, so the cell is withheld). Both directions matter.
4. **France IS included** (RULE 1 preserved: its denominator is URSSAF, never
     Eurostat). Départements aggregate into the 22 pre-2016 régions = French
     NUTS-2, so France appears in every figure and on the map, shaded a
     different hue. Its numerator is the plant's COMMUNE, not a plant headcount
     — see `numerator_basis`.

Author: Magd Shamashan
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import foundation as fd
import module01_spatial as m1
import module02_organisational as m2

__all__ = [
    "plant_region_panel", "coverage_ratio_summary", "supplier_residual",
    "plant_vs_region", "tracking_series", "fig_plant_vs_region",
    "fig_tracking_examples", "MIN_TRACK_YEARS",
    "micro_macro_divergence", "suppression_diagnostic", "implausible_ratios", "year_coverage",
    "france_macro_micro", "french_region_totals",
        "fig_coverage_ratio", "fig_supplier_residual", "fig_divergence",
    "fig_suppression", "fig_france_macro_micro", "map_plants_over_regions",
]


# ==============================================================================
# 0. FRANCE ON THE SAME MAP — département -> NUTS-2
# ==============================================================================
#
# France can join the European analysis WITHOUT breaking RULE 1: the denominator
# is still URSSAF, never Eurostat. URSSAF reports by département (= NUTS-3), and
# French NUTS-2 is the 22 pre-2016 régions, so départements aggregate cleanly
# into exactly the codes the Eurostat panel and the GISCO geometry use.
#
# Two honesty requirements come with this, both carried in the data:
#   * `region_source` marks every row 'eurostat' or 'urssaf'.
#   * The French NUMERATOR is the plant's COMMUNE (all auto APE codes), not a
#     plant headcount, because the reported plant figure is a single static
#     value and cannot form a series. The substitution is defensible -- the
#     commune/plant ratio has median 0.96 -- but it is a commune, and the
#     column is named accordingly.

def french_region_totals(scope: str = "all") -> pd.DataFrame:
    """
    French NUTS-2 automotive employment from URSSAF (RULE 1 preserved).

    URSSAF départements are aggregated to the 22 pre-2016 régions, which ARE
    French NUTS-2 — so the result drops straight into the European panel and
    onto the NUTS-2 geometry, with no Eurostat figure involved.
    """
    dep = fd.urssaf_panel("C29", scope, "departement").copy()
    dep["departement_code"] = (dep["departement_code"].astype(str).str.strip()
                                 .str.replace(r"\.0$", "", regex=True).str.zfill(2))
    dep["nuts2"] = dep["departement_code"].map(fd.DEPARTEMENT_TO_NUTS2)
    unmapped = dep.loc[dep["nuts2"].isna(), "departement_code"].unique()
    if len(unmapped):
        warnings.warn(f"départements without a NUTS-2 mapping: {sorted(unmapped)[:10]}")
    out = (dep.dropna(subset=["nuts2"])
              .groupby(["nuts2", "year"], as_index=False)
              .agg(region_emp=("employees", "sum"),
                   n_departements=("departement_code", "nunique")))
    out["country_iso2"] = "FR"
    out["region_source"] = "urssaf"
    return out


# ==============================================================================
# 1. ATTRIBUTION — join the two worlds
# ==============================================================================

def plant_region_panel(nace: str = "C29",
                       level: int = 2,
                       scope: Sequence[str] = ("assembly", "component"),
                       exclude_france: bool = False,
                       drop_suspect: bool = False) -> pd.DataFrame:
    """
    One row per (region, year): tracked plant employment beside the regional
    total, with the ratio and the residual.

    `scope` decides which plants count toward the numerator. Default is
    assembly + component, matching Module 02, because both are inside NACE C29.
    R&D and logistics sites are excluded: they are not manufacturing employment
    in the sense C29 measures.

    Columns
    -------
      plant_emp     summed tracked-plant headcount in the region
      region_emp    Eurostat NACE C29 employment for the region
      ratio         plant_emp / region_emp
      residual      region_emp - plant_emp  (the non-assembler tissue)
      n_plants      how many tracked plants contributed
      ratio_flag    'ok' | 'implausible' (>1) — see implausible_ratios()
    """
    cw = fd.plant_nuts_crosswalk()
    key = "nuts2_new" if level == 2 else None
    if key is None:
        raise ValueError("level must be 2; the crosswalk stores NUTS-2 codes. "
                         "For NUTS-1 aggregate the region codes first.")

    ep = fd.employment_panel()
    ep = ep[ep["employees"].notna()]
    # Foundation flags post-closure and frozen (long carried-forward) values.
    # Off by default so the ratio>1 detector can still SEE them; turn on to
    # compute clean ratios.
    if drop_suspect:
        n = int(ep["is_suspect"].sum())
        ep = ep[~ep["is_suspect"]]
        if n:
            warnings.warn(f"dropped {n} suspect plant-years (post-closure / frozen)")
    # Keep plant-level rows everywhere; additionally keep France's commune rows
    # when France is in scope, since `is_plant_level` is False for them by
    # construction (they are commune totals, not plant headcounts).
    if exclude_france:
        ep = ep[ep["is_plant_level"]]
    else:
        ep = ep[ep["is_plant_level"] | (ep["country_iso2"] == "FR")]
        # one row per French plant-year: prefer the commune series (a real time
        # series) over the static reported figure, and never count both
        fr = ep[ep["country_iso2"] == "FR"]
        dup = fr.duplicated(["plant_uid", "year"], keep=False)
        if dup.any():
            keep_idx = (fr[dup].sort_values("is_plant_level")   # False first
                          .groupby(["plant_uid", "year"]).head(1).index)
            ep = ep.drop(index=set(fr[dup].index) - set(keep_idx))
    if scope:
        ep = ep[ep["classification"].isin(list(scope))]

    m = ep.merge(cw[["plant_uid", key, "geo_confidence"]], on="plant_uid", how="left")
    m = m[m[key].notna()]
    if exclude_france:
        m = m[m["country_iso2"] != "FR"]

    plant = (m.groupby([key, "year"], as_index=False)
               .agg(plant_emp=("employees", "sum"),
                    n_plants=("plant_uid", "nunique"),
                    n_imputed=("is_imputed", "sum"),
                    n_suspect=("is_suspect", "sum"),
                    is_plant_level=("is_plant_level", "all"),
                    plants=("plant_name", lambda s: "; ".join(sorted(set(s))))))

    reg = fd.eurostat_panel(nace, "employment", nuts_level=level)
    reg = reg[reg["value"].notna()][["geo_code", "geo_label", "country_iso2",
                                     "year", "value", "is_confidential"]]
    reg = reg.rename(columns={"geo_code": key, "value": "region_emp"})
    reg["region_source"] = "eurostat"

    if not exclude_france:
        # RULE 1 preserved: the French denominator is URSSAF, never Eurostat.
        fr_reg = french_region_totals()
        fr_reg = fr_reg.rename(columns={"nuts2": key})
        fr_reg["geo_label"] = fr_reg[key].map(fd.FRENCH_NUTS2_NAMES).fillna(fr_reg[key])
        fr_reg["is_confidential"] = False
        reg = pd.concat([reg, fr_reg[["geo_code" if key == "geo_code" else key,
                                      "geo_label", "country_iso2", "year",
                                      "region_emp", "is_confidential",
                                      "region_source"]]], ignore_index=True)

    j = plant.merge(reg, on=[key, "year"], how="inner")
    j["ratio"] = j["plant_emp"] / j["region_emp"]
    j["residual"] = j["region_emp"] - j["plant_emp"]
    j["ratio_flag"] = np.where(j["ratio"] > 1, "implausible", "ok")
    j["bloc"] = j["country_iso2"].map(m1.assign_bloc)
    # numerator basis: Germany etc. = plant headcount; France = plant COMMUNE
    # (all auto APE codes), because the reported French plant figure is static.
    # Median commune/plant ratio is 0.96, so the substitution is close — but it
    # is a commune, and every figure must be able to say so.
    j["numerator_basis"] = np.where(j["country_iso2"] == "FR",
                                    "commune_all_APE", "plant_reported")
    j = j.rename(columns={key: "nuts2"})
    return j.sort_values(["nuts2", "year"]).reset_index(drop=True)


# ==============================================================================
# 2. MEASURES
# ==============================================================================

def coverage_ratio_summary(prp: pd.DataFrame,
                           drop_implausible: bool = True) -> pd.DataFrame:
    """Per-region summary of how much of the sector the tracked plants are."""
    d = prp[prp["ratio_flag"] == "ok"] if drop_implausible else prp
    g = (d.groupby(["nuts2", "geo_label", "country_iso2", "bloc"], as_index=False)
          .agg(years=("year", "nunique"),
               n_plants=("n_plants", "max"),
               mean_ratio=("ratio", "mean"),
               first_ratio=("ratio", "first"),
               last_ratio=("ratio", "last"),
               mean_plant_emp=("plant_emp", "mean"),
               mean_region_emp=("region_emp", "mean")))
    g["ratio_change"] = g["last_ratio"] - g["first_ratio"]
    return g.sort_values("mean_ratio", ascending=False).reset_index(drop=True)


def supplier_residual(prp: pd.DataFrame,
                      window: Optional[Tuple[int, int]] = (2012, 2023),
                      drop_implausible: bool = True) -> pd.DataFrame:
    """
    The residual — regional C29 employment NOT in our tracked plants — over time.

    Interpretation: this is bodies, parts, trailers and untracked assemblers.
    If the residual grows while tracked assembly is flat, the supplier tissue is
    thickening around the plants; if it shrinks faster than assembly, the
    ecosystem is hollowing out around surviving plants. Either is a result.
    """
    d = prp[prp["ratio_flag"] == "ok"] if drop_implausible else prp
    if window is not None:
        d = d[(d["year"] >= window[0]) & (d["year"] <= window[1])]
    # balanced set: regions present in every year, so the series is comparable
    yrs = sorted(d["year"].unique())
    counts = d.groupby("nuts2")["year"].nunique()
    keep = counts[counts == len(yrs)].index
    d = d[d["nuts2"].isin(keep)]

    g = (d.groupby(["year", "bloc"], as_index=False)
          .agg(plant_emp=("plant_emp", "sum"),
               region_emp=("region_emp", "sum"),
               residual=("residual", "sum"),
               n_regions=("nuts2", "nunique")))
    g["residual_share"] = g["residual"] / g["region_emp"]
    g.attrs["n_regions_balanced"] = d["nuts2"].nunique()
    g.attrs["years"] = (min(yrs), max(yrs))
    return g


MIN_PLANT_EMP = 100   # below this, the numerator cannot be a real plant


def micro_macro_divergence(prp: pd.DataFrame, y0: int, y1: int,
                           min_plant_emp: int = MIN_PLANT_EMP) -> pd.DataFrame:
    """
    Compare the growth of tracked plants with the growth of their whole region.

    Divergence is the interesting case: if the plants grew while the region
    shrank, the surrounding supplier tissue must have contracted, and vice
    versa. This is the closest we can get, with these data, to the upstream
    transmission question that the internship brief raised.

    WHY A MINIMUM-SIZE FLOOR
    ------------------------
    Growth rates on a near-zero base explode. FRK2 Rhône-Alpes carried a tracked
    figure of 5-7 employees from 2006 to 2019, then 118 by 2023 — a nominal
    +1,900% that dwarfed every real result on the chart.

    The cause is NOT a Eurostat or URSSAF reporting error. The plant is
    "Meyzieu / Villeurbanne (ACI)" — a two-commune name — and the commune the
    crosswalk assigned carries almost no automotive employment, so the figure
    never described the plant at all. It is a failure of the one-plant-per-
    commune assumption in OUR data.

    Regions whose tracked employment falls below `min_plant_emp` at either
    endpoint are therefore excluded and reported, not silently dropped. The
    floor is well separated from real plants: across 43 regions the 5th
    percentile of minimum tracked employment is 121, and only FRK2 falls below.
    """
    d = prp[prp["ratio_flag"] == "ok"]

    small = (d[d["year"].isin([y0, y1])]
             .groupby(["nuts2", "geo_label"])["plant_emp"].min())
    excluded = small[small < min_plant_emp]
    if len(excluded):
        d = d[~d["nuts2"].isin(excluded.index.get_level_values("nuts2"))]
        warnings.warn(
            "excluded from divergence (tracked employment below "
            f"{min_plant_emp}, so growth is a base effect, not a trend): "
            + "; ".join(f"{lbl} ({v:.0f})" for (_, lbl), v in excluded.items()))
    w = (d[d["year"].isin([y0, y1])]
         .pivot_table(index=["nuts2", "geo_label", "country_iso2", "bloc"],
                      columns="year",
                      values=["plant_emp", "region_emp", "residual"]))
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.dropna().reset_index()

    for base in ("plant_emp", "region_emp", "residual"):
        a, b = f"{base}_{y0}", f"{base}_{y1}"
        w[f"{base}_growth"] = (w[b] - w[a]) / w[a]
    w["divergence"] = w["plant_emp_growth"] - w["region_emp_growth"]
    w.attrs.update({"y0": y0, "y1": y1,
                    "excluded": [lbl for (_, lbl) in excluded.index] if len(excluded) else [],
                    "min_plant_emp": min_plant_emp})
    return w.sort_values("divergence", ascending=False).reset_index(drop=True)


# ==============================================================================
# 3. DIAGNOSTICS
# ==============================================================================

def suppression_diagnostic(nace: str = "C29", level: int = 2) -> pd.DataFrame:
    """
    Is Eurostat suppression related to hosting one of our plants?

    Two mechanisms pull in opposite directions and both are visible:
      * host regions are large enough to report at all  -> LESS 'not available'
      * host regions are dominated by one or two plants -> MORE 'confidential'
    Reporting this protects the thesis from being accused of ignoring the
    selection built into its own macro source.
    """
    cw = fd.plant_nuts_crosswalk()
    hosts = set(cw.loc[cw["nuts2_new"].notna() & (cw["country_iso2"] != "FR"),
                       "nuts2_new"])
    e = fd.eurostat_panel(nace, "employment", nuts_level=level).copy()
    e["is_host"] = e["geo_code"].isin(hosts)

    # THE TWO FLAGS ARE NESTED, NOT DISJOINT. Every confidential region-year is
    # also flagged missing -- 334 of 334 in NACE 29 -- and none of them carries
    # a number. So "no figure published" is the parent, and it splits into a
    # figure that was WITHHELD and a figure that was never collected. Treating
    # the two flags as siblings double-counts the suppressed cells and, worse,
    # hides the actual result: in host regions a missing figure is nearly always
    # a suppression, elsewhere it is nearly always a non-collection.
    e["absent_only"] = e["is_missing"] & ~e["is_confidential"]

    g = (e.groupby("is_host", as_index=False)
          .agg(region_years=("value", "size"),
               pct_missing=("is_missing", lambda s: 100 * s.mean()),
               pct_confidential=("is_confidential", lambda s: 100 * s.mean()),
               pct_absent_only=("absent_only", lambda s: 100 * s.mean()),
               n_regions=("geo_code", "nunique")))
    # and by year, for the figure
    by_year = (e.groupby(["year", "is_host"], as_index=False)
                .agg(pct_missing=("is_missing", lambda s: 100 * s.mean()),
                     pct_confidential=("is_confidential", lambda s: 100 * s.mean()),
                     pct_absent_only=("absent_only", lambda s: 100 * s.mean())))
    g.attrs["by_year"] = by_year

    # WHICH host regions are withheld. "A dominant plant makes the figure
    # disclosive" is a claim about dominance, and the number of TRACKED plants
    # does not measure it: Stuttgart has five and is never suppressed, because
    # it also has a large supplier base. What measures it is how much of the
    # region's employment the tracked plants account for. Splitting on that
    # gives a clean gradient; splitting on the plant count does not.
    prp = plant_region_panel(nace=nace, level=level)
    share = prp.groupby("nuts2")["ratio"].median()
    h = e[e["is_host"]].copy()
    h["cov"] = h["geo_code"].map(share)
    h["band"] = pd.cut(h["cov"], [0, .25, .50, np.inf],
                       labels=["under 25%", "25–50%", "over 50%"])
    h["band"] = h["band"].cat.add_categories(["not computable"]).fillna(
        "not computable")
    n_host = len(h)
    by_band = (h.groupby("band", observed=False)
                .agg(region_years=("is_missing", "size"),
                     n_regions=("geo_code", "nunique"),
                     pct_missing=("is_missing", lambda s: 100 * s.mean()),
                     pct_confidential=("is_confidential",
                                       lambda s: 100 * s.mean()),
                     pct_absent_only=("absent_only", lambda s: 100 * s.mean()))
                .reset_index())
    g.attrs["by_band"] = by_band
    return g


def implausible_ratios(prp: pd.DataFrame) -> pd.DataFrame:
    """
    Region-years where tracked plants exceed the regional total — impossible if
    the data were right, so each one names a specific fault to fix.

    This test earns its place: it caught BE22 Limburg, where Ford Genk closed in
    December 2014 but the source panel carries 4,300 employees forward to 2025.
    Candidate causes, in order of likelihood:
      * a carried-forward value for a plant that has closed or shrunk
      * a plant assigned to the wrong NUTS-2 region
      * plant headcount covering activities outside NACE C29
      * a suppressed / underreported regional figure
    """
    bad = prp[prp["ratio"] > 1].copy()
    if bad.empty:
        return bad
    bad["excess"] = bad["plant_emp"] - bad["region_emp"]
    # name the likely cause from the foundation quality flags
    bad["likely_cause"] = np.where(
        bad["n_suspect"] > 0,
        "stale plant value (post-closure / frozen series)",
        "check region assignment or measurement basis")
    return bad[["nuts2", "geo_label", "year", "plants", "n_plants",
                "plant_emp", "region_emp", "ratio", "excess",
                "n_suspect", "likely_cause"]] \
        .sort_values("ratio", ascending=False).reset_index(drop=True)


# ==============================================================================
# 4. FRANCE — the better version, via URSSAF (RULE 1)
# ==============================================================================

def france_macro_micro() -> pd.DataFrame:
    """
    France done properly: URSSAF is commune-level, so a plant's own commune can
    be set against its département.

    This is the one place where the macro-micro link is nearly exact. The
    'plant' figure is the commune total across auto APE codes (the thesis's
    one-plant-per-commune assumption), and the 'region' figure is the whole
    département. The ratio therefore reads as: how much of this département's
    automotive employment sits in the plant's own commune.
    """
    ep = fd.employment_panel()
    fr = ep[(ep["country_iso2"] == "FR") & ep["employees"].notna()].copy()
    if fr.empty:
        raise RuntimeError("no French rows in the employment panel")

    # NOTE ON THE SOURCE FIELDS (they are not what their names suggest):
    #   geo_code    = commune INSEE code, e.g. 57289 (Hambach)
    #   region_code = département NAME, e.g. "Moselle"  -- a name, not a code
    # The département code is therefore derived from the first two digits of the
    # commune code, which is the INSEE convention for metropolitan France. The
    # name is kept as a fallback so a mismatch is visible rather than silent.
    src = fd.load_employment()
    dep = src[(src["country_iso2"] == "FR") &
              src["measurement_basis"].eq("commune_all_APE")][
        ["plant_uid", "year", "region_code", "geo_code"]].drop_duplicates()
    fr = fr.merge(dep, on=["plant_uid", "year"], how="left")

    commune = (pd.to_numeric(fr["geo_code"], errors="coerce")
                 .astype("Int64").astype(str).str.zfill(5))
    fr["departement_code"] = commune.str[:2]
    fr["departement_name_src"] = fr["region_code"].astype(str).str.strip()

    depart = fd.urssaf_panel("C29", "all", "departement")
    depart["departement_code"] = depart["departement_code"].astype(str).str.strip().str.zfill(2)

    j = fr.merge(depart[["departement_code", "departement", "year", "employees"]]
                 .rename(columns={"employees": "dept_emp"}),
                 on=["departement_code", "year"], how="left")

    unmatched = j["dept_emp"].isna().sum()
    if unmatched:
        warnings.warn(
            f"france_macro_micro: {unmatched} plant-years without a département "
            f"total (check Corsica 2A/2B and overseas codes).")
    j = j.rename(columns={"employees": "commune_emp"})
    j["ratio"] = j["commune_emp"] / j["dept_emp"]
    j["residual"] = j["dept_emp"] - j["commune_emp"]
    return j[["plant_uid", "plant_name", "city", "departement_code", "departement",
              "departement_name_src", "year", "commune_emp", "dept_emp",
              "ratio", "residual"]] \
        .dropna(subset=["dept_emp"]).sort_values(["plant_uid", "year"])


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


# Explanatory notes used to be printed under each figure. They now live in
# the LaTeX captions, which lets the chart itself print larger. The text is
# still passed to _foot and kept on the figure as `fig._foot_text`; set this
# to True to draw it again.
SHOW_FOOTNOTES = False


def _foot(fig, text: str, width: int = 110, y: float = -0.02,
          size: float = 8.5) -> None:
    """Figure footnote, hard-wrapped so `bbox_inches='tight'` cannot stretch
    the saved image to the width of a single unwrapped line."""
    fig._foot_text = str(text)
    if not SHOW_FOOTNOTES:
        return
    import textwrap
    lines = []
    for para in str(text).split("\n"):
        lines.extend(textwrap.wrap(para, width=width) or [""])
    fig.text(0.02, y, "\n".join(lines), fontsize=size, color="#5A5A5A",
             ha="left", va="top")


def _save(fig, save):
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def fig_coverage_ratio(prp: pd.DataFrame, summary: pd.DataFrame,
                       save: Optional[str] = None):
    """How much of each region's sector the tracked plants represent."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 6.6))
    fig.suptitle("Tracked plants as a share of regional NACE C29 employment",
                 fontsize=14, fontweight="bold", x=0.02, ha="left")

    ax = axes[0]
    d = prp[prp["ratio_flag"] == "ok"]
    ax.hist(d["ratio"], bins=np.arange(0, 1.05, .05), color="#14746F",
            edgecolor="white")
    med = d["ratio"].median()
    ax.axvline(med, color="#C0563F", ls="--", lw=1.6)
    ax.text(med + .02, ax.get_ylim()[1] * .92, f"median {med:.2f}",
            fontsize=9, color="#C0563F", fontweight="bold")
    ax.set_title("(a) Distribution across region-years", loc="left")
    ax.set_xlabel("tracked plants ÷ regional C29"); ax.set_ylabel("region-years")

    ax = axes[1]
    top = summary.head(16).sort_values("mean_ratio")
    ax.barh(range(len(top)), top["mean_ratio"],
            color=[m1.BLOC_COLOURS.get(b, "#888") for b in top["bloc"]])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"{r.geo_label[:24]} ({int(r.n_plants)})"
                        for r in top.itertuples()], fontsize=7.5)
    ax.set_title("(b) Best-covered regions (plants in brackets)", loc="left")
    ax.set_xlabel("mean ratio")
    ax.legend(handles=[Patch(color=m1.BLOC_COLOURS[b], label=b.replace("_", " "))
                       for b in m1.BLOC_ORDER if b in set(top["bloc"])],
              fontsize=7.5, frameon=False, loc="lower right")

    ax = axes[2]
    g = d.groupby("year").agg(mean_ratio=("ratio", "mean"),
                              median_ratio=("ratio", "median"),
                              n=("nuts2", "nunique"))
    ax.plot(g.index, g["mean_ratio"], lw=2.2, marker="o", ms=3.5,
            color="#14746F", label="mean")
    ax.plot(g.index, g["median_ratio"], lw=2.2, marker="s", ms=3.5,
            color="#2C6FB3", label="median")
    ax.set_title("(c) Ratio over time", loc="left")
    _yrs = sorted(int(v) for v in g.index)
    ax.set_xticks(_yrs[::3]); ax.set_xticklabels([str(v) for v in _yrs[::3]])
    ax.set_ylabel("tracked ÷ regional"); ax.legend(fontsize=8, frameon=False)

    _foot(fig, "Ratios below 1 are expected rather than a sign of error. The regional "
             "statistic covers the whole of NACE C29 \u2014 vehicles, bodies and "
             "parts \u2014 while the plant panel covers assembly and component "
             "sites only. What is left over is the surrounding supplier tissue.", y=-0.02)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    return _save(fig, save)


def fig_supplier_residual(res: pd.DataFrame, save: Optional[str] = None):
    """The non-assembler tissue: does it thicken or hollow out?"""
    import matplotlib.pyplot as plt
    _style()
    blocs = [b for b in m1.BLOC_ORDER if b in set(res["bloc"])]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    fig.suptitle("Automotive employment outside the tracked plants: the supplier tissue",
                 fontsize=14, fontweight="bold", x=0.02, ha="left")

    ax = axes[0]
    for b in blocs:
        s = res[res["bloc"] == b].sort_values("year")
        ax.plot(s["year"], s["plant_emp"] / 1000, lw=2.2, marker="o", ms=3,
                color=m1.BLOC_COLOURS[b], label=f"{b.replace('_',' ')} — tracked plants")
        ax.plot(s["year"], s["residual"] / 1000, lw=1.8, ls="--", marker="s", ms=2.6,
                color=m1.BLOC_COLOURS[b], alpha=.75,
                label=f"{b.replace('_',' ')} — residual")
    ax.set_title("(a) Tracked plants vs residual", loc="left")
    ax.set_ylabel("employees (thousands)"); ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    for b in blocs:
        s = res[res["bloc"] == b].sort_values("year")
        ax.plot(s["year"], 100 * s["residual_share"], lw=2.4, marker="o", ms=3.5,
                color=m1.BLOC_COLOURS[b], label=b.replace("_", " "))
    ax.set_title("(b) Residual share of regional employment", loc="left")
    ax.set_ylabel("% of regional C29 outside tracked plants")
    ax.legend(fontsize=8, frameon=False)

    n = res.attrs.get("n_regions_balanced")
    y = res.attrs.get("years")
    _foot(fig, f"The same {n} regions appear in every year from {y[0]} to {y[1]}, so "
             "the lines cannot move because the sample moved. A rising residual "
             "share means the wider supplier tissue grew faster than the "
             "assembly and component plants tracked here.", y=-0.02)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    return _save(fig, save)


def fig_divergence(div: pd.DataFrame, save: Optional[str] = None, top_n: int = 18):
    """Where plants and their regions moved in opposite directions."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    _style()
    y0, y1 = div.attrs["y0"], div.attrs["y1"]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6),
                             gridspec_kw={"width_ratios": [1, 1.15]})
    fig.suptitle(f"Micro-macro divergence, {y0}–{y1}", fontsize=14,
                 fontweight="bold", x=0.02, ha="left")

    ax = axes[0]
    for b in [b for b in m1.BLOC_ORDER if b in set(div["bloc"])]:
        s = div[div["bloc"] == b]
        ax.scatter(100 * s["region_emp_growth"], 100 * s["plant_emp_growth"],
                   s=34, alpha=.8, color=m1.BLOC_COLOURS[b], label=b.replace("_", " "))
    lim = 1.05 * max(abs(np.r_[div["region_emp_growth"], div["plant_emp_growth"]])) * 100
    ax.plot([-lim, lim], [-lim, lim], ls="--", color="#999", lw=1)
    ax.axhline(0, color="#333", lw=.8); ax.axvline(0, color="#333", lw=.8)
    ax.set_xlabel("regional C29 growth (%)"); ax.set_ylabel("tracked plant growth (%)")
    ax.set_title("(a) Plant growth vs regional growth", loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.text(.02, .97, "above the line:\nplants outgrew their region",
            transform=ax.transAxes, va="top", fontsize=8, color="#6E6E6E")

    ax = axes[1]
    sel = pd.concat([div.head(top_n // 2), div.tail(top_n // 2)]).sort_values("divergence")
    ax.barh(range(len(sel)), 100 * sel["divergence"],
            color=[m1.BLOC_COLOURS.get(b, "#888") for b in sel["bloc"]])
    ax.set_yticks(range(len(sel)))
    ax.set_yticklabels([f"{r.geo_label[:26]}" for r in sel.itertuples()], fontsize=7.5)
    ax.axvline(0, color="#333", lw=1)
    ax.set_title("(b) Divergence (plant growth − regional growth)", loc="left")
    ax.set_xlabel("percentage points")

    note = (f"Balanced regions with tracked employment ≥ {div.attrs.get('min_plant_emp', MIN_PLANT_EMP)} "
            "at both endpoints.")
    if div.attrs.get("excluded"):
        note += (" Excluded for clarity: " + ", ".join(div.attrs["excluded"]) +
                 " — tracked employment there is a handful of people, so the growth rate is a "
                 "base effect from a commune-attribution problem, not a trend.")
    _foot(fig, note, y=-0.02)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    return _save(fig, save)


def fig_suppression(sup: pd.DataFrame, save: Optional[str] = None):
    """Eurostat suppression is not random — and here is the measurement."""
    import matplotlib.pyplot as plt
    _style()
    by_year = sup.attrs["by_year"]
    fig, axes = plt.subplots(1, 3, figsize=(16.6, 5.1),
                             gridspec_kw={"width_ratios": [1, 1, 1.15]})
    fig.suptitle("Unpublished regional employment figures, by reason and region",
                 fontsize=14, fontweight="bold", x=0.02, ha="left")

    C_CONF, C_ABS = "#14746F", "#AFC8C6"
    host = sup[sup["is_host"]].iloc[0]; other = sup[~sup["is_host"]].iloc[0]

    def stack(ax, x, rows, width=.55):
        """One bar per group: withheld at the bottom, never collected on top."""
        ax.set_axisbelow(True)   # gridlines behind the bars, not across them
        conf = np.array([r["pct_confidential"] for r in rows])
        abso = np.array([r["pct_absent_only"] for r in rows])
        ax.bar(x, conf, width, color=C_CONF, label="withheld as confidential")
        ax.bar(x, abso, width, bottom=conf, color=C_ABS,
               edgecolor="white", linewidth=.6, label="no figure collected")
        for i in range(len(x)):
            if conf[i] >= 1.5:
                ax.text(x[i], conf[i] / 2, f"{conf[i]:.1f}%", ha="center",
                        va="center", fontsize=8, color="white", fontweight="bold")
            if abso[i] >= 1.5:
                ax.text(x[i], conf[i] + abso[i] / 2, f"{abso[i]:.1f}%",
                        ha="center", va="center", fontsize=8, color="#33514F")
            ax.text(x[i], conf[i] + abso[i] + .6, f"{conf[i]+abso[i]:.1f}%",
                    ha="center", va="bottom", fontsize=8.6, fontweight="bold",
                    color="#333333")

    # ---- (a) host against the rest ----------------------------------------
    ax = axes[0]
    stack(ax, np.arange(2), [other, host])
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(["regions without\na tracked plant",
                        "regions hosting\na tracked plant"], fontsize=9)
    ax.set_ylabel("% of region-years with no published figure")
    ax.set_ylim(0, 33)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title("(a) Two mechanisms, not one", loc="left")

    # ---- (b) host regions by how much of the region the plants are ---------
    ax = axes[1]
    bands = sup.attrs["by_band"].set_index("band")
    order = [b for b in ["under 25%", "25–50%", "over 50%"] if b in bands.index]
    stack(ax, np.arange(len(order)), [bands.loc[b] for b in order])
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([f"{b}\n({int(bands.loc[b, 'n_regions'])} regions)"
                        for b in order], fontsize=9)
    ax.set_ylim(0, 33)
    ax.set_title("(b) Host regions, by the plants' share of the region",
                 loc="left", fontsize=11)

    # ---- (c) over time -----------------------------------------------------
    ax = axes[2]
    for is_host, lbl, col in ((True, "host regions", "#14746F"),
                              (False, "other regions", "#C6803D")):
        s = by_year[by_year["is_host"] == is_host].sort_values("year")
        ax.plot(s["year"], s["pct_absent_only"], lw=2.2, marker="o", ms=3,
                color=col, label=f"{lbl} — none collected")
        ax.plot(s["year"], s["pct_confidential"], lw=1.8, ls="--", marker="s",
                ms=2.6, color=col, alpha=.85, label=f"{lbl} — withheld")
    ax.set_title("(c) Over time", loc="left"); ax.set_ylabel("% of region-years")
    ax.legend(fontsize=7.5, frameon=False)
    yrs = sorted(int(v) for v in by_year["year"].unique())
    ax.set_xticks(yrs[::3])
    ax.set_xticklabels([str(v) for v in yrs[::3]])

    _foot(fig, "Eurostat's two flags are nested rather than parallel: every "
             "region-year withheld as confidential is also flagged as having no "
             "figure, so the bars split one quantity -- no published figure -- "
             "into a value that was withheld and a value that was never "
             "collected.  Read that way the two groups differ in kind, not "
             "degree. Regions with no tracked plant are missing far more often, "
             "and almost all of it is data never collected. Regions hosting a "
             "plant are missing less often overall, but what is missing is "
             "mostly suppression: a dominant plant makes the figure disclosive.  "
             "Panel (b) locates that suppression. It is essentially absent where "
             "the tracked plants are under a quarter of the region's automotive "
             "employment and common where they are more than half -- mostly "
             "small regions, several of them city-states, where one plant is "
             "most of the sector by construction. The share is computed from the "
             "years a region did report, since it needs the regional figure as "
             "its denominator, so eight host regions cannot be classified and "
             "are left out of panel (b).", y=-0.02)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    return _save(fig, save)


def fig_france_macro_micro(fr: pd.DataFrame, save: Optional[str] = None):
    """France via URSSAF: plant commune inside its département."""
    import matplotlib.pyplot as plt
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))
    fig.suptitle("France (URSSAF) — the plant's commune within its département",
                 fontsize=14, fontweight="bold", x=0.02, ha="left")

    ax = axes[0]
    g = fr.groupby("year").agg(mean_ratio=("ratio", "mean"),
                               median_ratio=("ratio", "median"))
    ax.plot(g.index, g["mean_ratio"], lw=2.4, marker="o", ms=3.5,
            color="#14746F", label="mean")
    ax.plot(g.index, g["median_ratio"], lw=2.2, marker="s", ms=3.5,
            color="#2C6FB3", label="median")
    ax.set_title("(a) Commune share of département employment", loc="left")
    ax.set_ylabel("commune ÷ département"); ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    last = fr.sort_values("year").groupby("plant_uid").tail(1).nlargest(14, "ratio")
    ax.barh(range(len(last)), last["ratio"], color="#C6803D")
    ax.set_yticks(range(len(last)))
    ax.set_yticklabels([f"{r.city[:22]} ({r.departement_code})"
                        for r in last.itertuples()], fontsize=7.5)
    ax.set_title("(b) Most plant-dominated départements (latest year)", loc="left")
    ax.set_xlabel("commune ÷ département")
    _foot(fig, "URSSAF reports at commune level, which makes this the closest the "
             "study comes to matching a plant directly to an official employment "
             "figure. It assumes that one commune contains one tracked plant; "
             "where a commune hosts several employers in the sector, the "
             "comparison attributes all of them to that plant.", y=-0.02)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    return _save(fig, save)


def year_coverage(prp: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Coverage by year — use this to CHOOSE a reference year, never habit.

    2019 looks like the natural pre-COVID reference and is in fact the WORST
    year in the series: only 64% of NUTS-2 regions carry a value, against 93%
    in 2023. Two effects coincide -- 2019-2020 are the weakest years of the
    2008-2020 extract, and the 2021+ extract is far better populated.

        best overall          2023 (93%), 2022 (93%), 2021 (87%)
        best in the old extract  2016-2017 (~77%)
        worst                 2020 (61%), 2019 (64%)
    """
    e = fd.eurostat_panel("C29", "employment", nuts_level=2)
    prp = plant_region_panel() if prp is None else prp
    rows = []
    for y in sorted(e["year"].unique()):
        ey = e[e["year"] == y]
        p = prp[prp["year"] == y]
        rows.append({"year": int(y),
                     "pct_regions_with_value": 100 * ey["value"].notna().mean(),
                     "regions_with_ratio": p["nuts2"].nunique(),
                     "plants_with_ratio": int(p["n_plants"].sum())})
    out = pd.DataFrame(rows)
    out["score"] = out["pct_regions_with_value"] / 100 * out["plants_with_ratio"]
    return out.sort_values("year").reset_index(drop=True)


def map_plants_over_regions(prp: pd.DataFrame, year: int = 2023,
                            save: Optional[str] = None):
    """
    Interactive overlay: regional C29 employment as a choropleth, tracked plants
    as bubbles sized by headcount and coloured by their coverage ratio.

    Default year is 2023, the best-covered year in the series (93% of NUTS-2
    regions carry a value, against 64% in 2019). See `year_coverage()`.
    """
    import folium
    cw = fd.plant_nuts_crosswalk().dropna(subset=["lat", "lon"])
    ep = fd.employment_panel()
    # France's rows are commune totals (is_plant_level False) — keep them, or
    # the map shows an empty France while the choropleth beneath it is shaded.
    ep = ep[(ep["year"] == year) & ep["employees"].notna() &
            (ep["is_plant_level"] | (ep["country_iso2"] == "FR"))]
    ep = ep.sort_values("is_plant_level").drop_duplicates("plant_uid", keep="first")

    geo = fd.load_nuts_geometry(level=2)
    if geo is None:
        raise RuntimeError("NUTS-2 geometry unavailable; see fd.load_nuts_geometry().")
    # France is drawn at departement level from URSSAF, which is the finest
    # level those records support and finer than the NUTS-2 used elsewhere.
    geo3 = fd.load_nuts_geometry(level=3)
    geo0 = fd.load_nuts_geometry(level=0)

    reg = prp[prp["year"] == year].set_index("nuts2")
    ratio = reg["ratio"].to_dict()

    m = folium.Map(location=[50.0, 9.5], zoom_start=4, tiles="cartodbpositron")

    # Shade from the SAME panel the bubbles use, so France (URSSAF) is shaded
    # too rather than left blank. Falls back to the full Eurostat panel for
    # regions that host no tracked plant.
    e = fd.eurostat_panel("C29", "employment", nuts_level=2)
    vals = e[(e["year"] == year) & e["value"].notna()].set_index("geo_code")["value"].to_dict()
    panel_vals = (prp[prp["year"] == year]
                  .drop_duplicates("nuts2").set_index("nuts2")["region_emp"].to_dict())
    vals.update(panel_vals)                      # URSSAF France wins where present
    # French NUTS-2 values are replaced by departement values below, so remove
    # them here: otherwise France is shaded twice, once under the other.
    vals = {k: v for k, v in vals.items() if not str(k).startswith("FR")}

    dep = fd.urssaf_panel("C29", "all", "departement")
    dep = dep[dep["year"] == year]
    dep_vals = dict(zip(dep["departement_code"].astype(str).str.strip()
                        .str.replace(r"\.0$", "", regex=True).str.zfill(2),
                        dep["employees"]))
    dep2nuts = fd.french_departement_to_nuts3(geo3) if geo3 is not None else {}
    fr_vals = {nid: dep_vals[dc] for dc, nid in dep2nuts.items() if dc in dep_vals}
    src = (prp[prp["year"] == year]
           .drop_duplicates("nuts2").set_index("nuts2")["region_source"].to_dict())
    vmax = max(vals.values()) if vals else 1

    def style(feat):
        nid = str(feat["properties"]["NUTS_ID"])
        if nid.startswith("FR"):          # drawn from the departement layer
            return {"fillOpacity": 0, "weight": 0, "color": "#00000000"}
        v = vals.get(nid)
        if v is None:
            return {"fillOpacity": .04, "weight": .3, "color": "#bbbbbb",
                    "fillColor": "#eeeeee"}
        t = min(v / (vmax * .35), 1.0)
        # France is URSSAF, not Eurostat -> different hue, so the reader can
        # never mistake the two sources for one series
        colour = "#7A5C99" if src.get(feat["properties"]["NUTS_ID"]) == "urssaf" else "#2C6FB3"
        return {"fillOpacity": .18 + .55 * t, "weight": .4, "color": "#888888",
                "fillColor": colour}

    def _tip(feat):
        nid = feat["properties"]["NUTS_ID"]
        v = vals.get(nid); s_ = src.get(nid, "eurostat")
        feat["properties"]["_emp"] = f"{v:,.0f}" if v else "n/a"
        feat["properties"]["_src"] = s_ if v else "—"
        return feat
    geo = {**geo, "features": [_tip(f) for f in geo["features"]]}
    folium.GeoJson(geo, style_function=style,
                   name="automotive employment \u2014 NUTS-2 (Eurostat)",
                   tooltip=folium.GeoJsonTooltip(
                       fields=["NUTS_ID", "NAME_LATN", "_emp", "_src"],
                       aliases=["NUTS-2", "region", "sector employment", "source"])
                   ).add_to(m)

    if geo3 is not None and fr_vals:
        fmax = max(fr_vals.values())

        def style_fr(feat):
            nid = str(feat["properties"]["NUTS_ID"])
            v = fr_vals.get(nid)
            if v is None or not np.isfinite(v):
                return {"fillOpacity": .04, "weight": .3, "color": "#bbbbbb",
                        "fillColor": "#eeeeee"}
            t = min(v / (fmax * .55), 1.0)
            return {"fillOpacity": .18 + .55 * t, "weight": .4,
                    "color": "#a99ac4", "fillColor": "#7A5C99"}

        fr3 = {"type": "FeatureCollection", "features": []}
        for f in geo3["features"]:
            nid = str(f["properties"]["NUTS_ID"])
            if not nid.startswith("FR"):
                continue
            p = dict(f["properties"])
            v = fr_vals.get(nid)
            p["_emp"] = f"{v:,.0f}" if v is not None and np.isfinite(v) else "n/a"
            fr3["features"].append({"type": "Feature", "properties": p,
                                    "geometry": f.get("geometry")})
        folium.GeoJson(fr3, style_function=style_fr,
                       name="France \u2014 NUTS-3 departements (URSSAF)",
                       tooltip=folium.GeoJsonTooltip(
                           fields=["NAME_LATN", "NUTS_ID", "_emp"],
                           aliases=["departement", "NUTS-3",
                                    "sector employment (URSSAF)"])
                       ).add_to(m)

    # National borders, drawn heavier than regional ones so the two can be told
    # apart at a glance.
    if geo0 is not None:
        folium.GeoJson(geo0, name="country borders",
                       style_function=lambda f: {"fillOpacity": 0,
                                                 "color": "#1a1a1a",
                                                 "weight": 2.1, "opacity": .9}
                       ).add_to(m)

    # why a region has no ratio: suppressed by Eurostat, or outside its scope
    e_all = fd.eurostat_panel("C29", "employment", nuts_level=2)
    supp = set(e_all.loc[(e_all["year"] == year) & e_all["is_confidential"],
                         "geo_code"])

    for _, r in cw.iterrows():
        row = ep[ep["plant_uid"] == r["plant_uid"]]
        if row.empty:
            continue
        emp = float(row["employees"].iloc[0])
        basis = ("commune (URSSAF)" if r["country_iso2"] == "FR"
                 else "plant headcount")
        # fall back to the legacy code for years before the region was recoded
        nuts = r["nuts2_new"] if pd.notna(r["nuts2_new"]) else r["nuts2_old"]
        rt = ratio.get(nuts)
        if rt is None and pd.notna(r["nuts2_old"]):
            rt, nuts = ratio.get(r["nuts2_old"], rt), (
                r["nuts2_old"] if r["nuts2_old"] in ratio else nuts)

        if rt is not None:
            col = "#C0563F" if rt > 1 else "#14746F" if rt > .5 else "#E9A23B"
            why = f"{rt:.2f}"
        elif nuts in supp:
            col, why = "#8E6BAF", "suppressed (confidential)"
        else:
            col, why = "#9AA0A6", "no regional figure"

        folium.CircleMarker(
            [r["lat"], r["lon"]], radius=float(np.clip(np.sqrt(emp) / 9, 3.5, 22)),
            color=col, fill=True, fill_color=col, fill_opacity=.8, weight=1.2,
            popup=folium.Popup(html=(
                f"<b>{r['plant_name']}</b><br>{r['city']}, {r['country_iso2']}<br>"
                f"group: {r['group']}<br>"
                f"employees {emp:,.0f} <i>({basis})</i><br>"
                f"NUTS-2: {nuts if pd.notna(nuts) else '— outside Eurostat'}<br>"
                f"region ratio: {why}"), max_width=340),
            tooltip=f"{r['plant_name']} — {emp:,.0f}").add_to(m)

    legend = ("<div style='position:fixed;bottom:22px;left:22px;z-index:9999;background:white;"
              "padding:10px 12px;border:1px solid #ccc;border-radius:4px;font:11px sans-serif;"
              "line-height:1.5'>"
              "<b>Plant &divide; region ratio</b><br>"
              "<span style='color:#14746F'>&#9679;</span> &gt; 0.5 &nbsp;"
              "<span style='color:#E9A23B'>&#9679;</span> &le; 0.5 &nbsp;"
              "<span style='color:#C0563F'>&#9679;</span> &gt; 1<br>"
              "<span style='color:#8E6BAF'>&#9679;</span> region suppressed by Eurostat<br>"
              "<span style='color:#9AA0A6'>&#9679;</span> outside Eurostat (UK 2021+, TR, RU)<br>"
              "<b>shading</b> &nbsp;<span style='color:#2C6FB3'>&#9632;</span> "
              "Eurostat, NUTS-2 &nbsp;"
              "<span style='color:#7A5C99'>&#9632;</span> URSSAF, French "
              "d&eacute;partements (NUTS-3)<br>"
              "<i>bubble size = employment; thick lines are country borders, "
              "thin lines regions</i></div>")
    m.get_root().html.add_child(folium.Element(legend))
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        m.save(save)
    return m


# ==============================================================================
# 6. DOES A PLANT FOLLOW ITS REGION?
# ==============================================================================
#
# Everything above compares LEVELS: how much of a region's automotive
# employment sits in the tracked plants. This section compares MOVEMENT. If a
# plant and the region around it rise and fall together, the plant is a fair
# barometer of its local industry and a regional figure can stand in for it.
# If they diverge, the region is telling a different story from the factory at
# its centre -- suppliers growing while assembly shrinks, or the reverse -- and
# that divergence is itself the finding.

MIN_TRACK_YEARS = 8      # fewer overlapping years than this and a correlation
                         # of year-to-year changes is not worth computing


def plant_vs_region(window: Tuple[int, int] = (2012, 2023),
                    nace: str = "C29",
                    min_plant_emp: int = MIN_PLANT_EMP,
                    min_years: int = MIN_TRACK_YEARS) -> pd.DataFrame:
    """
    One row per plant: how closely its employment tracks its region's.

    Returns the total change of each over the window and the correlation of
    their year-on-year changes. Correlating CHANGES rather than levels matters:
    two series that both drift downward will correlate almost perfectly in
    levels whatever they do year to year, which says nothing about whether the
    plant follows the region.

    France is measured from URSSAF and everywhere else from Eurostat, so the
    regional series is not the same instrument in both cases. That affects the
    level of the ratio, not whether the two move together, which is what this
    measures.
    """
    cw = fd.plant_nuts_crosswalk().dropna(subset=["nuts2_new"])
    # Values recorded after a plant has closed are carried forward by the
    # sources and would otherwise contribute a flat line that correlates with
    # nothing. Ford Genk, shut in 2014, is the clearest case.
    ep = fd.employment_panel(drop_post_closure=True, drop_frozen=True)
    ep = ep[ep["employees"].notna() & (ep["employees"] >= min_plant_emp)]
    ep = ep[(ep["year"] >= window[0]) & (ep["year"] <= window[1])]
    # employment_panel already carries plant_name, city, country_iso2 and
    # group; merging them again would suffix every one of them.
    ep = ep.merge(cw[["plant_uid", "nuts2_new"]], on="plant_uid", how="inner")
    ep = ep.rename(columns={"nuts2_new": "nuts2"})

    reg = fd.regional_panel(nace, "employment", level=2, include_france=True)
    reg = reg[(reg["year"] >= window[0]) & (reg["year"] <= window[1])
              & reg["value"].notna()]
    reg = reg.rename(columns={"geo_code": "nuts2", "value": "region_emp"})

    j = ep.merge(reg[["nuts2", "year", "region_emp", "geo_label", "source"]],
                 on=["nuts2", "year"], how="inner")

    rows = []
    for (uid, nuts2), g in j.groupby(["plant_uid", "nuts2"]):
        g = g.sort_values("year")
        if len(g) < min_years:
            continue
        dp = g["employees"].pct_change()
        dr = g["region_emp"].pct_change()
        ok = dp.notna() & dr.notna()
        if ok.sum() < min_years - 1:
            continue
        r = float(dp[ok].corr(dr[ok])) if dp[ok].std() and dr[ok].std() else np.nan
        rows.append(dict(
            plant_uid=uid, nuts2=nuts2,
            plant_name=g["plant_name"].iloc[0], city=g["city"].iloc[0],
            country_iso2=g["country_iso2"].iloc[0], group=g["group"].iloc[0],
            geo_label=g["geo_label"].iloc[0], region_source=g["source"].iloc[0],
            n_years=len(g), y0=int(g["year"].iloc[0]), y1=int(g["year"].iloc[-1]),
            plant_emp_y0=float(g["employees"].iloc[0]),
            plant_emp_y1=float(g["employees"].iloc[-1]),
            region_emp_y0=float(g["region_emp"].iloc[0]),
            region_emp_y1=float(g["region_emp"].iloc[-1]),
            corr_changes=r))
    out = pd.DataFrame(rows)
    if not len(out):
        return out
    out["plant_pct"] = 100 * (out["plant_emp_y1"] / out["plant_emp_y0"] - 1)
    out["region_pct"] = 100 * (out["region_emp_y1"] / out["region_emp_y0"] - 1)
    out["gap_pp"] = out["plant_pct"] - out["region_pct"]
    out.attrs["window"] = window
    return out.sort_values("gap_pp", ascending=False).reset_index(drop=True)


def tracking_series(plant_uids: Sequence[str],
                    window: Tuple[int, int] = (2012, 2023),
                    nace: str = "C29") -> pd.DataFrame:
    """Indexed plant and region series (first year = 100) for named plants."""
    cw = fd.plant_nuts_crosswalk().dropna(subset=["nuts2_new"])
    ep = fd.employment_panel()
    ep = ep[ep["employees"].notna()
            & ep["plant_uid"].isin(plant_uids)
            & (ep["year"] >= window[0]) & (ep["year"] <= window[1])]
    ep = ep.merge(cw[["plant_uid", "nuts2_new"]],
                  on="plant_uid").rename(columns={"nuts2_new": "nuts2"})
    reg = fd.regional_panel(nace, "employment", level=2, include_france=True)
    reg = reg.rename(columns={"geo_code": "nuts2", "value": "region_emp"})
    j = ep.merge(reg[["nuts2", "year", "region_emp", "geo_label"]],
                 on=["nuts2", "year"], how="inner").sort_values(
                     ["plant_uid", "year"])
    out = []
    for uid, g in j.groupby("plant_uid"):
        g = g.sort_values("year").copy()
        g["plant_idx"] = 100 * g["employees"] / g["employees"].iloc[0]
        g["region_idx"] = 100 * g["region_emp"] / g["region_emp"].iloc[0]
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else j


def fig_plant_vs_region(pvr: Optional[pd.DataFrame] = None,
                        window: Tuple[int, int] = (2012, 2023),
                        clip_pct: float = 120.0,
                        n_label: int = 10,
                        save: Optional[str] = None):
    """
    Does a plant rise and fall with the region around it?

    Two readability decisions, both deliberate:

    * The scatter is CLIPPED, not rescaled. One plant grew 412% and, plotted on
      a shared axis, squeezed the other forty-eight into a blob at the origin.
      Points beyond the clip are drawn as triangles on the border and labelled
      with their true value, so nothing is hidden and the bulk stays legible.
    * The correlation panel shows the ten strongest and ten weakest plants
      rather than all forty-nine. Forty-nine labels at a readable size do not
      fit, and the middle of that ranking is where the plants with no clear
      relationship sit — they are counted in the summary instead of listed.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    _style()
    pvr = plant_vs_region(window) if pvr is None else pvr
    y0, y1 = pvr.attrs.get("window", window)

    fig, axes = plt.subplots(1, 2, figsize=(15.4, 6.8),
                             gridspec_kw={"width_ratios": [1, 1.05]})
    fig.suptitle("Plant employment against the employment of the region around "
                 f"it, {y0}\u2013{y1}", fontsize=14.5, fontweight="bold",
                 x=0.02, ha="left")

    # ------------------------------------------------------------------ (a)
    ax = axes[0]
    L = clip_pct
    d = pvr.copy()
    d["x"] = d["region_pct"].clip(-L, L)
    d["y"] = d["plant_pct"].clip(-L, L)
    d["off"] = (d["region_pct"].abs() > L) | (d["plant_pct"].abs() > L)

    ax.fill_between([-L, L], [-L, L], [L, L], color="#14746F", alpha=.05)
    ax.fill_between([-L, L], [-L, -L], [-L, L], color="#B03A2E", alpha=.04)
    ax.plot([-L, L], [-L, L], color="#9AA0A6", lw=1.3, ls="--", zorder=2)
    ax.annotate("plant grew faster\nthan its region", (-L * .93, L * .30),
                fontsize=8.8, color="#14746F", fontweight="bold", va="top")
    ax.annotate("plant grew slower\nthan its region", (L * .93, -L * .62),
                fontsize=8.8, color="#B03A2E", fontweight="bold", ha="right")

    inb, outb = d[~d["off"]], d[d["off"]]
    cols = [m1.BLOC_COLOURS.get(m1.assign_bloc(c), "#888")
            for c in inb["country_iso2"]]
    ax.scatter(inb["x"], inb["y"], s=58, c=cols, alpha=.85, edgecolor="white",
               linewidth=.8, zorder=4)
    # Off-scale plants get a marker on the border and ONE listing in a corner
    # box. Labelling each in place stacked four captions on the same few pixels.
    for r in outb.itertuples():
        col = m1.BLOC_COLOURS.get(m1.assign_bloc(r.country_iso2), "#888")
        ax.scatter([r.x], [r.y], s=110, marker="^", c=[col], edgecolor="#333333",
                   linewidth=1.0, zorder=5)
    if len(outb):
        lines = [f"beyond the axis ({len(outb)}):"]
        for r in outb.sort_values("plant_pct", ascending=False).itertuples():
            lines.append(f"  {r.plant_name[:26]}  "
                         f"plant {r.plant_pct:+.0f}%, region {r.region_pct:+.0f}%")
        ax.annotate("\n".join(lines), xy=(.985, .985), xycoords="axes fraction",
                    ha="right", va="top", fontsize=7.4, color="#333333",
                    bbox=dict(boxstyle="round,pad=0.35", fc="white",
                              ec="#CCCCCC", lw=.8, alpha=.95))

    # label the largest gaps that are still on-scale, alternating side
    lab = inb.reindex(inb["gap_pp"].abs().sort_values(ascending=False).index).head(6)
    for k, r in enumerate(lab.itertuples()):
        ax.annotate(r.plant_name[:24], (r.x, r.y),
                    xytext=(8, 6) if k % 2 == 0 else (-8, -12),
                    textcoords="offset points", fontsize=7.4, color="#444444",
                    ha="left" if k % 2 == 0 else "right")

    ax.axhline(0, color="#333333", lw=1); ax.axvline(0, color="#333333", lw=1)
    ax.set_xlim(-L, L * 1.06); ax.set_ylim(-L, L * 1.52)
    ax.set_xlabel(f"change in regional automotive employment, {y0}\u2013{y1} (%)")
    ax.set_ylabel(f"change in plant employment, {y0}\u2013{y1} (%)")
    blocs = [b for b in m1.BLOC_ORDER
             if b in {m1.assign_bloc(c) for c in pvr["country_iso2"]}
             and b in m1.BLOC_COLOURS]
    ax.legend(handles=[Patch(color=m1.BLOC_COLOURS[b], label=b.replace("_", " "))
                       for b in blocs]
              + [plt.Line2D([], [], marker="^", ls="", color="#666666",
                            markersize=8, label=f"beyond \u00b1{L:.0f}%")],
              fontsize=8.4, frameon=False, loc="lower left")
    # (a) plots every plant with an endpoint change; (b) ranks only those with a
    # computable year-to-year correlation, which is one fewer. Saying "the N
    # plants" in both places implied a single set and the two counts disagreed.
    _n_corr = int(pvr["corr_changes"].notna().sum())
    ax.set_title(f"(a) Endpoint change for each of the {len(pvr)} plants "
                 f"against its own region", loc="left")

    # ------------------------------------------------------------------ (b)
    ax = axes[1]
    t = pvr.dropna(subset=["corr_changes"]).sort_values("corr_changes")
    top = t.tail(n_label); bot = t.head(n_label)
    sel = pd.concat([bot, top])
    y = np.arange(len(sel), dtype=float)
    y[len(bot):] += 1.2                      # a visible break between the two ends
    cols = ["#14746F" if v > .3 else "#B03A2E" if v < -.3 else "#9AA0A6"
            for v in sel["corr_changes"]]
    ax.barh(y, sel["corr_changes"], color=cols, height=.78)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n[:32]} ({c})" for n, c
                        in zip(sel["plant_name"], sel["country_iso2"])],
                       fontsize=8.6)
    for yi, v in zip(y, sel["corr_changes"]):
        ax.annotate(f"{v:+.2f}", (v, yi), xytext=(5 if v >= 0 else -5, 0),
                    textcoords="offset points", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8,
                    color="#444444")
    ax.axvline(0, color="#333333", lw=1.2)
    mid = (y[len(bot) - 1] + y[len(bot)]) / 2
    ax.axhline(mid, color="#CCCCCC", lw=1, ls=":")
    n_mid = len(t) - 2 * n_label
    if n_mid > 0:
        ax.annotate(f"\u2026 {n_mid} further plants, between "
                    f"{bot['corr_changes'].iloc[-1]:+.2f} and "
                    f"{top['corr_changes'].iloc[0]:+.2f}",
                    (0, mid), xytext=(0, 0), textcoords="offset points",
                    ha="center", va="center", fontsize=8.2, color="#777777",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none"))
    ax.set_xlim(-1.05, 1.15)
    ax.set_xlabel("correlation of year-to-year changes with the region")
    ax.set_title(f"(b) The {n_label} closest followers, and the {n_label} least",
                 loc="left")
    n_pos = int((t["corr_changes"] > .3).sum())
    n_neg = int((t["corr_changes"] < -.3).sum())
    _dropped = len(pvr) - len(t)
    _summary = (f"Across the {len(t)} plants with enough overlapping years "
                f"({_dropped} of the {len(pvr)} in panel (a) has too few to "
                f"correlate): {n_pos} move with their region, "
                f"{n_neg} against it, {len(t) - n_pos - n_neg} show no clear "
                f"relationship.\nMedian correlation "
                f"{t['corr_changes'].median():+.2f}.")
    fig._panel_summary = _summary          # carried in the caption instead
    if SHOW_FOOTNOTES:
        ax.annotate(_summary, xy=(.5, -.155), xycoords="axes fraction",
                    ha="center", va="top", fontsize=8.8, color="#333333")

    _foot(fig,
          "Panel (a) sets each plant's change in employment against the change "
          "in its region's automotive employment over the same years. A point "
          "on the dashed line moved exactly with its region; in the shaded area "
          "above it the plant did better than the industry around it, below it "
          f"worse. A handful of plants move by more than \u00b1{L:.0f}% and "
          "are drawn as triangles on the border with their true figures, so "
          "that one plant growing fourfold does not compress all the others "
          "into a single cluster.  Panel (b) correlates the two series year by "
          "year rather than end to end, because two series that both drift "
          "downwards will agree almost perfectly over a decade whatever they "
          "did in between.  Regional figures come from Eurostat, except in "
          "France where they come from URSSAF administrative records; the two "
          "are not the same instrument, which affects how large a gap looks but "
          "not whether plant and region move together.", y=-0.015)
    fig.subplots_adjust(top=.885, bottom=.245, wspace=.44)
    _save(fig, save)
    return fig


def fig_tracking_examples(uids: Optional[Sequence[str]] = None,
                          pvr: Optional[pd.DataFrame] = None,
                          window: Tuple[int, int] = (2012, 2023),
                          n: int = 6, save: Optional[str] = None):
    """
    A handful of plants shown beside their region, indexed to the first year.

    Chosen on evidence rather than by taste: the plants that diverge most from
    their region in each direction, plus the ones that track it most closely.
    """
    import matplotlib.pyplot as plt
    _style()
    pvr = plant_vs_region(window) if pvr is None else pvr
    if uids is None:
        d = pvr.dropna(subset=["corr_changes"])
        picks = pd.concat([d.head(2), d.tail(2),
                           d.reindex(d["corr_changes"].sort_values(
                               ascending=False).index).head(2)])
        uids = list(dict.fromkeys(picks["plant_uid"]))[:n]
    ts = tracking_series(uids, window)
    uids = [u for u in uids if u in set(ts["plant_uid"])]

    ncol = 3
    nrow = int(np.ceil(len(uids) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.9 * ncol, 3.5 * nrow),
                             sharex=True)
    axes = np.atleast_1d(axes).ravel()
    y0, y1 = window
    fig.suptitle(f"Six plants and the regions around them, {y0}\u2013{y1} "
                 f"(both indexed to 100 in {y0})",
                 fontsize=14.5, fontweight="bold", x=0.02, ha="left")

    handles = None
    for ax, uid in zip(axes, uids):
        g = ts[ts["plant_uid"] == uid].sort_values("year")
        row = pvr[pvr["plant_uid"] == uid].iloc[0]
        ax.plot(g["year"], g["plant_idx"], "-o", color="#14746F", lw=2.4, ms=4,
                label="plant")
        ax.plot(g["year"], g["region_idx"], "-o", color="#C6803D", lw=2.0,
                ms=3.5, ls="--", label="region around it")
        ax.axhline(100, color="#BBBBBB", lw=1, ls=":")
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
        ax.set_title(f"{row['plant_name'][:42]}\n{str(row['geo_label'])[:42]}",
                     loc="left", fontsize=9.5)
        # headroom first, then a boxed annotation, so the text never sits on a line
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - .26 * (hi - lo), hi + .06 * (hi - lo))
        ax.annotate(f"plant {row['plant_pct']:+.0f}%    "
                    f"region {row['region_pct']:+.0f}%\n"
                    f"year-to-year correlation {row['corr_changes']:+.2f}",
                    xy=(.02, .02), xycoords="axes fraction", fontsize=8,
                    color="#333333", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.28", fc="white",
                              ec="#DDDDDD", lw=.8, alpha=.92))
        ax.set_ylabel("index, first year = 100", fontsize=8.5)
    for ax in axes[len(uids):]:
        ax.axis("off")
    if handles:
        fig.legend(handles, labels, fontsize=9.4, frameon=False, ncol=2,
                   loc="upper left", bbox_to_anchor=(0.02, 0.935))

    _foot(fig,
          "Each panel shows one plant against the automotive employment of the "
          "region containing it, both set to 100 in the first year so that "
          "shapes can be compared rather than sizes. A plant that hugs the "
          "dashed line is moving with its local industry; one that pulls away "
          "is not. The plants shown are those that diverge most from their "
          "region in each direction together with those that follow it most "
          "closely, chosen from the full set rather than picked by hand.")
    fig.subplots_adjust(top=.845, bottom=.16, hspace=.46, wspace=.30)
    _save(fig, save)
    return fig
