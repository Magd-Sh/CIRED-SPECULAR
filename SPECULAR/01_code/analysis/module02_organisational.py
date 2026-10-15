# -*- coding: utf-8 -*-
"""
================================================================================
 MODULE 02 — ORGANISATIONAL STRUCTURAL CHANGE
 Which firms, which plants, which locations — and who lost ground to whom?
================================================================================

WHAT THIS MODULE ADDS TO MODULE 01
----------------------------------
Module 01 asked *where* employment went, using exhaustive regional statistics.
This module asks *within which firms and plants* it moved, using the micro panel
that this thesis built. It is the level of analysis that Eurostat cannot reach
and that Pavlínek (2019) had to approach through restructuring events and
interviews: we can now do it on plant-level stocks.

Pavlínek's fourth research question is explicitly about **ownership and
nationality** of the firms driving job creation and loss. Section 3 below is the
direct analogue, measured rather than inferred.

THE UNIVERSE — READ THIS FIRST
------------------------------
Throughout this project, "balanced universe" is shorthand for the GERMAN-GROUP
IMPUTED PANEL. It does not mean "statistically representative"; it means "free
of coverage artefacts". The headline figures use it filtered to assembly +
component sites and strictly balanced over 2012-2023 (31 plants).

    it contains ONLY the three German groups
    -> Volkswagen Group (16), Mercedes-Benz (10), BMW Group (5)
    -> 24 plants in Germany, 7 outside (BE, ES, HU x2, PL, PT, SK)
    -> 47% of its cells over 2012-2023 are imputed, not observed

So this module does not describe "European automotive firms" in general. It
describes, precisely and without coverage bias, **German automotive capital**:
how it restructured at home while expanding into the integrated periphery. That
is a narrower claim than Module 01's, but a sharper one -- and it is exactly
Pavlínek's mechanism observed from the inside of the firms driving it.

Renault, Stellantis, Ford and Toyota appear only in the **full-universe context**
figures (Section 7), which carry coverage caveats and must not be read as trends.

UNCERTAINTY IS PROPAGATED
-------------------------
Roughly half the balanced panel is imputed. Every aggregate here carries an
uncertainty band derived from the per-cell `est_sd_log` (Rubin-pooled) via the
delta method. See `aggregate_with_uncertainty()` for the independence caveat --
the bands are a LOWER bound.

Author: Magd Shamashan
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import foundation as fd

__all__ = [
    "GROUP_HOME_COUNTRY", "STELLANTIS_HERITAGE", "DEFAULT_SCOPE",
    "analysis_universe", "aggregate_with_uncertainty",
    "group_trajectories", "home_vs_periphery", "classification_trajectories",
    "within_group_reallocation", "reallocation_two_periods", "data_availability_window", "plant_concentration",
    "ownership_coding",
    "fig_group_trajectories", "fig_home_periphery", "fig_assembly_component",
    "fig_within_group", "fig_within_group_two_periods", "fig_data_availability", "fig_plant_concentration",
    "fig_full_universe_context",
]

DEFAULT_SCOPE: Tuple[str, ...] = ("assembly", "component")

GROUP_COLOURS = {
    "Volkswagen Group": "#2E8B6F", "Mercedes-Benz": "#6A6A6A",
    "BMW Group": "#2C6FB3", "Renault": "#C6803D", "Stellantis": "#7A5C99",
    "Ford": "#30609F", "Toyota": "#B0483C", "Ineos": "#444444",
}

# --- ownership: where the controlling capital is domiciled ---------------------
GROUP_HOME_COUNTRY: Dict[str, str] = {
    "Volkswagen Group": "DE", "Mercedes-Benz": "DE", "BMW Group": "DE",
    "Renault": "FR", "Ford": "US", "Toyota": "JP", "Ineos": "GB",
    # Stellantis is resolved per plant by brand heritage -- see below.
}

# Stellantis (formed 2021 from PSA + FCA) did not exist for most of the window.
# Coding the whole 2006-2025 period as one entity would erase the history that
# matters. Each plant is therefore assigned the nationality of the capital that
# actually ran it: PSA (FR), Fiat (IT) or Opel (DE, GM-owned until 2017).
# Applies only to the full-universe context figures; no Stellantis plant is in
# the balanced universe.
STELLANTIS_HERITAGE: Dict[str, str] = {
    # PSA / Peugeot-Citroën lineage
    "FR_STELLANTIS_SOCHAUX": "FR",
    "FR_STELLANTIS_SAUSHEIM_MULHOUSE": "FR",
    "FR_STELLANTIS_CHARTRES_DE_BRETAGNE_RENNES": "FR",
    "FR_STELLANTIS_CORMELLES_LE_ROYAL_CAEN": "FR",
    "FR_STELLANTIS_METZ": "FR",
    "FR_STELLANTIS_TREMERY": "FR",
    "FR_STELLANTIS_TRITH_SAINT_LEGER_VALENCIENNES": "FR",
    "FR_STELLANTIS_VESOUL": "FR",
    "FR_STELLANTIS_VILLERS_SEMEUSE_CHARLEVILLE_MEZIERES": "FR",
    "FR_STELLANTIS_LIEU_SAINT_AMAND_HORDAIN": "FR",   # Sevelnord, PSA-led JV
    "ES_STELLANTIS_VIGO": "FR",                       # PSA (Citroën) Vigo
    "ES_STELLANTIS_MADRID": "FR",                     # PSA Villaverde
    # Opel lineage (GM until 2017, then PSA/Stellantis)
    "ES_STELLANTIS_ZARAGOZA": "DE",                   # Opel España, Figueruelas
    # Fiat / FCA lineage
    "IT_STELLANTIS_MIRAFIORI": "IT",
    "IT_STELLANTIS_CASSINO": "IT",
    "IT_STELLANTIS_MELFI": "IT",
    "IT_STELLANTIS_POMIGLIANO": "IT",
    "IT_STELLANTIS_ATESSA": "IT",                     # Sevel Sud, Fiat-led JV
}


# ==============================================================================
# 1. UNIVERSE + UNCERTAINTY MACHINERY
# ==============================================================================

def analysis_universe(balanced: bool = True,
                      scope: Sequence[str] = DEFAULT_SCOPE,
                      window: Optional[Tuple[int, int]] = (2012, 2023),
                      strict: bool = True,
                      ep: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    The plant-year employment frame every figure in this module uses.

    balanced : keep only plants in the imputed universe. READ THIS: with
               `strict` over 2012-2023 that set is GERMAN-GROUP ONLY -- VW (16),
               Mercedes (10), BMW (5); 24 plants in Germany plus 7 German-owned
               sites abroad (BE, ES, HU x2, PL, PT, SK). "Balanced" means free of
               coverage artefacts, NOT representative of Europe, and NOT
               observed-only: 47% of the cells are imputed. No
               Renault/Stellantis/Ford/Toyota.
    scope    : plant classifications to keep; default assembly + component
    window   : (y0, y1) analysis window. Defaults to 2012-2023, matching
               Module 01, because that is where support is best.
    strict   : keep only plants with a value in EVERY year of the window.

    WHY `strict` MATTERS -- do not turn it off casually
    ---------------------------------------------------
    "Balanced" in Module 00 means a complete 35 x 20 GRID. It does not mean a
    constant set of contributing plants, because `inactive` cells (outside a
    plant's observed span) carry NaN and drop out. Over 2006-2025 the number of
    contributing plants runs 23 (2006) -> 34 (2019-2023) -> 25 (2025). Summing
    across that produces a spurious rise and a cliff at each end: in an early
    draft this showed up as component employment apparently collapsing 70% in
    2025, which was purely plants leaving the sample.

    With strict=True over 2012-2023 the set is a constant 31 plants, so levels,
    shares and index paths are comparable across years.
    """
    ep = fd.employment_panel() if ep is None else ep
    df = ep.copy()
    if balanced:
        df = df[df["in_imputed_universe"]]
    if scope:
        df = df[df["classification"].isin(list(scope))]
    df = df[df["employees"].notna()]

    if window is not None:
        y0, y1 = window
        df = df[(df["year"] >= y0) & (df["year"] <= y1)]
        if strict:
            years = sorted(df["year"].unique())
            counts = df.groupby("plant_uid")["year"].nunique()
            keep = counts[counts == len(years)].index
            dropped = sorted(set(counts.index) - set(keep))
            df = df[df["plant_uid"].isin(keep)]
            if dropped:
                warnings.warn(
                    f"strict balancing over {y0}-{y1} dropped {len(dropped)} "
                    f"plant(s) with incomplete support: {dropped[:6]}"
                    f"{' ...' if len(dropped) > 6 else ''}")

    df.attrs.update({"balanced": balanced, "scope": tuple(scope),
                     "window": window, "strict": strict,
                     "n_plants": df["plant_uid"].nunique()})
    return df


def aggregate_with_uncertainty(df: pd.DataFrame,
                               by: Sequence[str] = ("year",)) -> pd.DataFrame:
    """
    Sum employment over groups, propagating imputation uncertainty.

    Each imputed cell carries `est_sd_log`, the Rubin-pooled SD on the log
    scale. By the delta method a cell's SD on the natural scale is
    approximately  E_i * s_i,  so for a sum

        SE(Σ E_i) ≈ sqrt( Σ (E_i · s_i)² ).

    CAVEAT -- these bands are a LOWER BOUND. The formula assumes cells are
    independent. They are not: imputations within a plant are serially
    correlated, and all plants share the model's common year effects. A fully
    correct interval would require the M completed panels themselves (aggregate
    within each draw, then apply Rubin's rules across draws). Do that if the
    bands ever carry an argument; use these for indicative width only.
    """
    d = df.copy()
    d["_sd"] = d["est_sd_log"].fillna(0.0)
    d["_var"] = (d["employees"] * d["_sd"]) ** 2
    g = (d.groupby(list(by), as_index=False)
           .agg(employment=("employees", "sum"),
                var=("_var", "sum"),
                n_plants=("plant_uid", "nunique"),
                n_imputed=("is_imputed", "sum"),
                n_cells=("employees", "size")))
    g["se"] = np.sqrt(g["var"])
    g["lo95"] = g["employment"] - 1.96 * g["se"]
    g["hi95"] = g["employment"] + 1.96 * g["se"]
    g["pct_imputed"] = 100 * g["n_imputed"] / g["n_cells"]
    return g.drop(columns=["var"])


# ==============================================================================
# 2. GROUP-LEVEL RECOMPOSITION
# ==============================================================================

def group_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    """Employment by group and year, with shares, index and uncertainty bands."""
    g = aggregate_with_uncertainty(df, by=("year", "group"))
    tot = g.groupby("year")["employment"].transform("sum")
    g["share_pct"] = 100 * g["employment"] / tot
    base = g[g["year"] == g["year"].min()].set_index("group")["employment"]
    g["index_100"] = g.apply(
        lambda r: 100 * r["employment"] / base.get(r["group"], np.nan), axis=1)
    return g


def classification_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    """Assembly vs component trajectories — where electrification bites."""
    g = aggregate_with_uncertainty(df, by=("year", "classification"))
    base = g[g["year"] == g["year"].min()].set_index("classification")["employment"]
    g["index_100"] = g.apply(
        lambda r: 100 * r["employment"] / base.get(r["classification"], np.nan), axis=1)
    return g


# ==============================================================================
# 3. OWNERSHIP AND LOCATION — Pavlínek's axis, measured
# ==============================================================================

def ownership_coding(plants: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Assign each plant the nationality of its controlling capital, and whether
    the plant is at 'home' or 'abroad' relative to that capital.

    home_country    where the controlling group is domiciled
    is_foreign      plant country != home country -> foreign direct investment
    location_type   'home' | 'abroad'

    Stellantis is resolved by brand heritage (PSA/Fiat/Opel), because the group
    only exists from 2021 and coding the whole window as one entity would erase
    the ownership history the analysis is about.
    """
    pl = fd.load_plants() if plants is None else plants.copy()
    home = pl["group"].map(GROUP_HOME_COUNTRY)
    stell = pl["plant_uid"].map(STELLANTIS_HERITAGE)
    pl["home_country"] = home.fillna(stell)
    pl["home_country_source"] = np.where(
        home.notna(), "group_domicile",
        np.where(stell.notna(), "brand_heritage", "unassigned"))
    pl["is_foreign"] = pl["country_iso2"] != pl["home_country"]
    pl["location_type"] = np.where(pl["is_foreign"], "abroad", "home")
    if (pl["home_country_source"] == "unassigned").any():
        miss = pl.loc[pl["home_country_source"] == "unassigned", "plant_uid"].tolist()
        warnings.warn(f"No ownership coding for: {miss}")
    return pl


def home_vs_periphery(df: pd.DataFrame,
                      plants: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    The Pavlínek test on the balanced universe: for a given bloc of capital,
    does employment hold at home while expanding abroad?

    On the balanced universe all three groups are German, so `location_type`
    reduces cleanly to 'Germany (home)' vs 'periphery (abroad)' — 27 vs 7
    plants. That is German automotive capital observed from the inside.
    """
    own = ownership_coding(plants)[["plant_uid", "home_country",
                                    "is_foreign", "location_type"]]
    d = df.merge(own, on="plant_uid", how="left")
    g = aggregate_with_uncertainty(d, by=("year", "location_type"))
    tot = g.groupby("year")["employment"].transform("sum")
    g["share_pct"] = 100 * g["employment"] / tot
    base = g[g["year"] == g["year"].min()].set_index("location_type")["employment"]
    g["index_100"] = g.apply(
        lambda r: 100 * r["employment"] / base.get(r["location_type"], np.nan), axis=1)
    return g


# ==============================================================================
# 4. WITHIN-GROUP REALLOCATION
# ==============================================================================

def within_group_reallocation(df: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    """
    Plant-level shift-share *inside* each group.

        ΔE_p = E_p0·g_G  +  E_p0·(g_p − g_G)
               group trend      reallocation between the group's own plants

    This separates "the whole group shrank" from "the group moved work between
    its plants" — the intra-firm spatial fix. A plant with a large positive
    reallocation term gained work relative to its own parent, which is a
    stronger statement than gaining relative to the industry.
    """
    w = (df[df["year"].isin([y0, y1])]
         .pivot_table(index=["plant_uid", "plant_name", "country_iso2",
                             "group", "classification"],
                      columns="year", values="employees", aggfunc="first")
         .reset_index()
         .rename(columns={y0: "E0", y1: "E1"})
         .dropna(subset=["E0", "E1"]))

    grp = w.groupby("group").agg(G0=("E0", "sum"), G1=("E1", "sum"))
    grp["g_group"] = (grp["G1"] - grp["G0"]) / grp["G0"]
    w = w.merge(grp[["g_group"]], left_on="group", right_index=True)

    w["actual_change"] = w["E1"] - w["E0"]
    w["growth_rate"] = w["actual_change"] / w["E0"]
    w["group_trend_component"] = w["E0"] * w["g_group"]
    w["reallocation"] = w["actual_change"] - w["group_trend_component"]
    w.attrs.update({"y0": y0, "y1": y1, "n_plants": len(w)})
    return w.sort_values("reallocation", ascending=False).reset_index(drop=True)


def reallocation_two_periods(df: pd.DataFrame,
                             y0: int = 2012,
                             ym: int = 2018,
                             y1: int = 2023) -> pd.DataFrame:
    """
    The same within-group shift-share, run twice on halves of the period.

    A single 2012-2023 figure hides whether a plant gained steadily or gained
    early and gave it back. Splitting at `ym` shows which. Only plants observed
    at all three dates are kept, so the two halves are comparable and they add
    to the whole.

    Returns one row per plant with `realloc_1` (y0->ym), `realloc_2` (ym->y1),
    `realloc_total`, and `pattern`, which names the shape of the trajectory.
    """
    a = within_group_reallocation(df, y0, ym)
    b = within_group_reallocation(df, ym, y1)

    keys = ["plant_uid", "plant_name", "country_iso2", "group", "classification"]
    out = (a[keys + ["reallocation", "E0", "E1"]]
           .rename(columns={"reallocation": "realloc_1", "E0": "E_y0", "E1": "E_ym"})
           .merge(b[keys + ["reallocation", "E1"]]
                    .rename(columns={"reallocation": "realloc_2", "E1": "E_y1"}),
                  on=keys, how="inner"))

    out["realloc_total"] = out["realloc_1"] + out["realloc_2"]

    def label(r):
        s1, s2 = r["realloc_1"] > 0, r["realloc_2"] > 0
        if s1 and s2:
            return "gains in both halves"
        if not s1 and not s2:
            return "loses in both halves"
        return "gains then loses" if s1 else "loses then gains"

    out["pattern"] = out.apply(label, axis=1)
    out.attrs.update({"y0": y0, "ym": ym, "y1": y1, "n_plants": len(out)})
    return out.sort_values("realloc_total", ascending=False).reset_index(drop=True)


# ==============================================================================
# 5. PLANT CHURN AND CONCENTRATION
# ==============================================================================

def data_availability_window(scope: Sequence[str] = DEFAULT_SCOPE) -> pd.DataFrame:
    """
    Each plant's DATA-AVAILABILITY window in the imputed panel.

    THIS IS NOT A PLANT OPERATING RECORD. An earlier version of this function
    read the panel's `inactive` flags as plant entry/exit. That was wrong, and
    the data says so plainly: for every plant, `inactive` years sit exactly
    OUTSIDE its observed span (VW Kassel inactive 2006-2018 / observed
    2019-2025; Untertürkheim inactive 2025 / observed 2014-2024). Kassel has
    operated since 1958 and Untertürkheim still operates. `inactive` marks
    where the imputation declined to extrapolate beyond the evidence — a
    boundary of KNOWLEDGE, not of existence.

    What this function gives you is the window over which each plant's series
    is supported by data. That is genuinely useful — it shows where the panel
    is evidence and where it is silence — but genuine plant openings and
    closures must come from the registry (`classification == 'historic'`) or
    external verification, never from here.
    """
    imp = fd.load_employment_imputed()
    # the imputed CSV carries its own group/country/brand columns; drop them so
    # the registry stays authoritative and the merge does not create _x/_y pairs
    imp = imp.drop(columns=[c for c in ("group", "country", "brand")
                            if c in imp.columns])
    pl = fd.load_plants()[["plant_uid", "plant_name", "city", "country_iso2",
                           "group", "classification"]]
    imp = imp.merge(pl, on="plant_uid", how="left")
    if scope:
        imp = imp[imp["classification"].isin(list(scope))]

    act = imp[~imp["is_inactive"]]
    yr_min, yr_max = int(imp["year"].min()), int(imp["year"].max())
    t = (act.groupby(["plant_uid", "plant_name", "city", "country_iso2",
                      "group", "classification"], as_index=False)
            .agg(first_active=("year", "min"), last_active=("year", "max"),
                 n_active_years=("year", "nunique")))
    t = t.rename(columns={"first_active": "first_supported_year",
                          "last_active": "last_supported_year",
                          "n_active_years": "n_supported_years"})
    t["spans_panel_start"] = t["first_supported_year"] <= yr_min
    t["spans_panel_end"] = t["last_supported_year"] >= yr_max
    t["coverage_pct"] = 100 * t["n_supported_years"] / (yr_max - yr_min + 1)
    # the ONLY defensible existence claim available from this data
    t["is_historic_site"] = t["classification"].eq("historic")
    t.attrs.update({"year_min": yr_min, "year_max": yr_max})
    return t.sort_values(["first_supported_year", "plant_uid"]).reset_index(drop=True)


def plant_concentration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Concentration of employment across PLANTS (not regions).

    Rising HHI = work consolidating into fewer, larger plants; falling =
    spreading across the network. Complements Module 01's regional measure.
    """
    rows = []
    for year, g in df.groupby("year"):
        v = g["employees"].sort_values(ascending=False).values
        tot = v.sum()
        if tot <= 0:
            continue
        sh = v / tot
        rows.append({
            "year": year, "n_plants": len(v),
            "hhi": float((sh ** 2).sum() * 10_000),
            "top5_pct": float(100 * sh[:5].sum()),
            "largest_plant_pct": float(100 * sh[0]),
            "mean_size": float(v.mean()), "median_size": float(np.median(v)),
            "total_employment": float(tot),
        })
    return pd.DataFrame(rows)


# ==============================================================================
# 6. FIGURES
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


_UNC_NOTE = ("Shaded bands show the uncertainty carried over from filling gaps "
             "in the employment series (\u00b11.96 standard errors). Because "
             "the errors are treated as independent, the bands are a lower "
             "bound: the true uncertainty is at least this wide.")


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


def fig_group_trajectories(gt: pd.DataFrame, save: Optional[str] = None):
    """Levels with uncertainty bands, shares, and indexed paths by group."""
    import matplotlib.pyplot as plt
    _style()
    groups = list(gt.groupby("group")["employment"].mean()
                  .sort_values(ascending=False).index)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    fig.suptitle("Employment by group — German-group imputed panel "
                 "(VW · Mercedes · BMW only; assembly + component)",
                 fontsize=14, fontweight="bold", x=0.02, ha="left")

    ax = axes[0]
    for g in groups:
        s = gt[gt["group"] == g].sort_values("year")
        c = GROUP_COLOURS.get(g, "#888888")
        ax.plot(s["year"], s["employment"] / 1000, lw=2.2, marker="o", ms=3, color=c, label=g)
        ax.fill_between(s["year"], s["lo95"] / 1000, s["hi95"] / 1000, color=c, alpha=.15, lw=0)
    ax.set_title("(a) Employment level", loc="left")
    ax.set_ylabel("employees (thousands)"); ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    piv = gt.pivot(index="year", columns="group", values="share_pct")[groups]
    ax.stackplot(piv.index, *[piv[g] for g in groups],
                 colors=[GROUP_COLOURS.get(g, "#888888") for g in groups],
                 labels=groups, alpha=.92)
    ax.set_title("(b) Share of the panel", loc="left")
    ax.set_ylabel("% of total"); ax.set_ylim(0, 100)
    ax.legend(fontsize=8, frameon=False, loc="lower left")

    ax = axes[2]
    for g in groups:
        s = gt[gt["group"] == g].sort_values("year")
        ax.plot(s["year"], s["index_100"], lw=2.2, marker="o", ms=3,
                color=GROUP_COLOURS.get(g, "#888888"), label=g)
    ax.axhline(100, color="#999999", ls="--", lw=1)
    ax.set_title(f"(c) Indexed ({int(gt['year'].min())} = 100)", loc="left")
    ax.set_ylabel("index")

    _foot(fig, _UNC_NOTE, y=-0.01)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    return _save(fig, save)


def fig_home_periphery(hp: pd.DataFrame, save: Optional[str] = None):
    """The Pavlínek test from inside the firms: home vs abroad."""
    import matplotlib.pyplot as plt
    _style()
    cols = {"home": "#14746F", "abroad": "#2C6FB3"}
    labels = {"home": "Germany", "abroad": "Abroad"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.4))
    fig.suptitle("Employment in German plants and in plants abroad",
                 fontsize=14, fontweight="bold", x=0.02, ha="left")

    ax = axes[0]
    for k in ("home", "abroad"):
        s = hp[hp["location_type"] == k].sort_values("year")
        ax.plot(s["year"], s["employment"] / 1000, lw=2.4, marker="o", ms=3,
                color=cols[k], label=labels[k])
        ax.fill_between(s["year"], s["lo95"] / 1000, s["hi95"] / 1000,
                        color=cols[k], alpha=.15, lw=0)
    ax.set_title("(a) Employment level", loc="left")
    ax.set_ylabel("employees (thousands)"); ax.legend(fontsize=9, frameon=False)

    ax = axes[1]
    s = hp[hp["location_type"] == "abroad"].sort_values("year")
    ax.plot(s["year"], s["share_pct"], lw=2.4, marker="o", ms=3.5, color="#2C6FB3")
    ax.fill_between(s["year"], s["share_pct"], alpha=.15, color="#2C6FB3")
    ax.set_title("(b) Share of employment located abroad", loc="left")
    ax.set_ylabel("% of panel employment")

    ax = axes[2]
    for k in ("home", "abroad"):
        s = hp[hp["location_type"] == k].sort_values("year")
        ax.plot(s["year"], s["index_100"], lw=2.4, marker="o", ms=3,
                color=cols[k], label=labels[k])
    ax.axhline(100, color="#999999", ls="--", lw=1)
    ax.set_title(f"(c) Indexed ({int(hp['year'].min())} = 100)", loc="left")
    ax.set_ylabel("index"); ax.legend(fontsize=9, frameon=False)

    # Counts come from `hp`, never typed by hand: the previous version hard-coded
    # "27 plants in Germany" and drifted out of step with the text when the
    # universe changed. `hp` carries n_plants per year and location type.
    _n = hp.groupby("location_type")["n_plants"].max()
    _home, _abroad = int(_n.get("home", 0)), int(_n.get("abroad", 0))
    _pct_imp = 100 * hp["n_imputed"].sum() / max(hp["n_cells"].sum(), 1)
    _foot(fig, f"This chart covers {_home} plants in Germany and {_abroad} abroad "
             "(Belgium, Spain, Hungary \u00d72, Poland, Portugal, Slovakia), all "
             "owned by Volkswagen, Mercedes-Benz or BMW. It is a German-group "
             "view, not a European one. These are the plants with a complete "
             f"series over the window; {_pct_imp:.0f}% of the cells behind the "
             "lines are filled by imputation rather than observed, which is what "
             "the shaded bands measure. " + _UNC_NOTE, y=-0.01)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    return _save(fig, save)


def fig_assembly_component(ct: pd.DataFrame, save: Optional[str] = None):
    """Assembly vs component — where the powertrain transition bites."""
    import matplotlib.pyplot as plt
    _style()
    cols = {"assembly": "#14746F", "component": "#C6803D"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle("Assembly vs component sites", fontsize=14, fontweight="bold",
                 x=0.02, ha="left")
    ax = axes[0]
    for k in ct["classification"].unique():
        s = ct[ct["classification"] == k].sort_values("year")
        c = cols.get(k, "#888888")
        ax.plot(s["year"], s["employment"] / 1000, lw=2.4, marker="o", ms=3, color=c, label=k)
        ax.fill_between(s["year"], s["lo95"] / 1000, s["hi95"] / 1000, color=c, alpha=.15, lw=0)
    ax.set_title("(a) Employment level", loc="left")
    ax.set_ylabel("employees (thousands)"); ax.legend(fontsize=9, frameon=False)

    ax = axes[1]
    for k in ct["classification"].unique():
        s = ct[ct["classification"] == k].sort_values("year")
        ax.plot(s["year"], s["index_100"], lw=2.4, marker="o", ms=3,
                color=cols.get(k, "#888888"), label=k)
    ax.axhline(100, color="#999999", ls="--", lw=1)
    ax.set_title(f"(b) Indexed ({int(ct['year'].min())} = 100)", loc="left")
    ax.set_ylabel("index"); ax.legend(fontsize=9, frameon=False)
    _foot(fig, _UNC_NOTE, y=-0.02)
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    return _save(fig, save)


def fig_within_group(wg: pd.DataFrame, save: Optional[str] = None, top_n: int = 22):
    """Reallocation between plants inside each group — the intra-firm spatial fix."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    _style()
    y0, y1 = wg.attrs["y0"], wg.attrs["y1"]
    sel = pd.concat([wg.head(top_n // 2), wg.tail(top_n // 2)]).sort_values("reallocation")
    fig, ax = plt.subplots(figsize=(11.5, 8))
    cols = [GROUP_COLOURS.get(g, "#888888") for g in sel["group"]]
    ax.barh(range(len(sel)), sel["reallocation"] / 1000, color=cols)
    ax.set_yticks(range(len(sel)))
    ax.set_yticklabels([f"{r.plant_name[:34]}  ({r.country_iso2})"
                        for r in sel.itertuples()], fontsize=8)
    ax.axvline(0, color="#333333", lw=1)
    ax.set_title(f"Reallocation between plants within each group, {y0}–{y1}",
                 loc="left", fontsize=13, fontweight="bold")
    ax.set_xlabel("thousands of jobs relative to the plant's own group trend")
    ax.legend(handles=[Patch(color=GROUP_COLOURS[g], label=g)
                       for g in sel["group"].unique() if g in GROUP_COLOURS],
              fontsize=8, frameon=False, loc="lower right")
    _foot(fig, "A positive value means the plant gained work relative to its own parent "
             "group. That is a stronger statement than gaining relative to the "
             "industry as a whole, because it holds the corporate strategy fixed "
             "and compares sites competing for the same investment.", y=-0.02)
    fig.tight_layout()
    return _save(fig, save)


def fig_within_group_two_periods(tp: pd.DataFrame, save: Optional[str] = None,
                                 top_n: int = 20):
    """The same reallocation, split into two halves so the timing is visible."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    _style()
    y0, ym, y1 = tp.attrs["y0"], tp.attrs["ym"], tp.attrs["y1"]
    sel = (pd.concat([tp.head(top_n // 2), tp.tail(top_n // 2)])
             .sort_values("realloc_total").reset_index(drop=True))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6),
                             gridspec_kw={"width_ratios": [1.55, 1]})

    # --- (a) the two halves side by side, per plant -------------------------
    ax = axes[0]
    y = np.arange(len(sel))
    h = 0.38
    c1, c2 = "#2C6FB3", "#C6803D"
    ax.barh(y + h / 2, sel["realloc_1"] / 1000, height=h, color=c1,
            label=f"{y0}–{ym}")
    ax.barh(y - h / 2, sel["realloc_2"] / 1000, height=h, color=c2,
            label=f"{ym}–{y1}")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.plant_name[:32]}  ({r.country_iso2})"
                        for r in sel.itertuples()], fontsize=8)
    ax.axvline(0, color="#333333", lw=1)
    ax.set_title(f"(a) Change relative to the plant's own group, {y0}–{ym} and {ym}–{y1}", loc="left",
                 fontsize=11.5, fontweight="bold")
    ax.set_xlabel("thousands of jobs gained or lost relative to the group's growth")
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")

    # --- (b) how many plants follow each pattern ----------------------------
    ax = axes[1]
    order = ["gains in both halves", "gains then loses",
             "loses then gains", "loses in both halves"]
    counts = tp["pattern"].value_counts().reindex(order).fillna(0)
    colours = ["#14746F", "#C6803D", "#2C6FB3", "#B0413E"]
    ax.barh(range(len(order)), counts.values, color=colours)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([o.replace(" in both halves", "\nin both halves")
                         .replace(" then ", "\nthen ") for o in order], fontsize=9)
    ax.invert_yaxis()
    for i, v in enumerate(counts.values):
        ax.text(v + .3, i, f"{int(v)}", va="center", fontsize=9.5,
                color="#333333")
    ax.set_xlim(0, max(counts.values) * 1.22)
    ax.set_title(f"(b) Plants by direction of change ({int(counts.sum())} plants)",
                 loc="left", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("number of plants")

    fig.suptitle(f"Reallocation between plants within each group, split at {ym}",
                 x=0.02, ha="left", fontsize=13.5, fontweight="bold")
    _foot(fig, "A positive value means the plant gained work relative to its own parent "
             "group over that half. Splitting the period separates a plant that gained "
             "steadily from one that gained early and gave it back. Only plants observed "
             f"in {y0}, {ym} and {y1} are shown, so the two halves are comparable and "
             "they add up approximately to the whole-period figure.", y=-0.02)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return _save(fig, save)


def fig_data_availability(tl: pd.DataFrame, save: Optional[str] = None):
    """Where each plant's series is evidence, and where it is silence."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    _style()
    t = tl.sort_values(["group", "first_supported_year", "plant_uid"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, max(6, 0.28 * len(t))))
    for i, r in t.iterrows():
        c = GROUP_COLOURS.get(r["group"], "#888888")
        ax.barh(i, r["last_supported_year"] - r["first_supported_year"] + 1,
                left=r["first_supported_year"], color=c, alpha=.85, height=.62)
        if r["is_historic_site"]:
            ax.plot(r["last_supported_year"] + .6, i, marker="x", color="#C0563F", ms=7)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels([f"{r.plant_name[:32]} ({r.country_iso2})"
                        for r in t.itertuples()], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(tl.attrs["year_min"] - 1, tl.attrs["year_max"] + 1)
    ax.set_title("Data-availability window per plant (NOT an operating record)",
                 loc="left", fontsize=13, fontweight="bold")
    handles = [Patch(color=GROUP_COLOURS[g], label=g)
               for g in t["group"].unique() if g in GROUP_COLOURS]
    handles += [plt.Line2D([], [], marker="x", ls="", color="#C0563F",
                           label="historic site (per registry)")]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="lower right")
    _foot(fig, "Each bar shows the years for which employment figures could be found, "
             "not the years the plant operated. Volkswagen Kassel has been "
             "running since 1958 and Untert\u00fcrkheim is still open, yet both "
             "appear truncated here. Read the bars as data availability, never "
             "as opening and closing dates.", y=-0.01)
    fig.tight_layout()
    return _save(fig, save)


def fig_plant_concentration(pc: pd.DataFrame, save: Optional[str] = None):
    """Is work consolidating into fewer, bigger plants?"""
    import matplotlib.pyplot as plt
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.2))
    fig.suptitle("Concentration of employment across plants (German-group imputed panel)",
                 fontsize=14, fontweight="bold", x=0.02, ha="left")
    ax = axes[0]
    ax.plot(pc["year"], pc["hhi"], marker="o", ms=4, lw=2.2, color="#14746F")
    ax.set_title("(a) Herfindahl across plants (×10,000)", loc="left"); ax.set_ylabel("HHI")
    ax = axes[1]
    ax.plot(pc["year"], pc["top5_pct"], marker="o", ms=4, lw=2.2, color="#14746F",
            label="top 5 plants")
    ax.plot(pc["year"], pc["largest_plant_pct"], marker="s", ms=4, lw=2.2,
            color="#C6803D", label="largest plant")
    ax.set_title("(b) Share held by the largest plants", loc="left")
    ax.set_ylabel("% of panel employment"); ax.legend(fontsize=8, frameon=False)
    ax = axes[2]
    ax.plot(pc["year"], pc["mean_size"], marker="o", ms=4, lw=2.2, color="#14746F",
            label="mean")
    ax.plot(pc["year"], pc["median_size"], marker="s", ms=4, lw=2.2, color="#2C6FB3",
            label="median")
    ax.set_title("(c) Plant size", loc="left")
    ax.set_ylabel("employees per plant"); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return _save(fig, save)


def fig_full_universe_context(scope: Sequence[str] = DEFAULT_SCOPE,
                              save: Optional[str] = None):
    """
    CONTEXT ONLY — all 89 plants, observed data, coverage shown alongside.
    Renault / Stellantis / Ford / Toyota appear only here. Levels are
    coverage-driven and must NOT be read as trends.
    """
    import matplotlib.pyplot as plt
    _style()
    ep = fd.employment_panel()
    d = ep[ep["classification"].isin(list(scope)) & ep["employees"].notna()]
    # France is kept: its rows are commune totals (is_plant_level False), which
    # is a different quantity but belongs in a CONTEXT figure that is explicitly
    # about coverage rather than levels. The caption says so.
    # The balanced set Section 4.2 actually uses, so that the comparison the
    # caption claims is drawn on the figure instead of left to the reader.
    bal = analysis_universe(balanced=True, scope=scope,
                            window=(2012, 2023), strict=True, ep=ep)
    bal_uids = set(bal["plant_uid"].unique())
    bal_series = (bal.groupby("year")["employees"].sum() / 1000)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))
    fig.suptitle("Full universe — context only (coverage-driven, not a trend)",
                 fontsize=14, fontweight="bold", x=0.02, ha="left")
    ax = axes[0]
    piv = d.pivot_table(index="year", columns="group", values="employees", aggfunc="sum")
    for g in piv.columns:
        ax.plot(piv.index, piv[g] / 1000, lw=2, marker="o", ms=3, alpha=.85,
                color=GROUP_COLOURS.get(g, "#888888"), label=g)
    ax.plot(bal_series.index, bal_series.values, lw=3.2, ls="--", color="black",
            zorder=5, label=f"balanced set used in §4.2 ({len(bal_uids)} plants)")
    ax.set_title("(a) Employment by group, all collected plants", loc="left")
    ax.set_ylabel("employees (thousands)")
    _h, _l = ax.get_legend_handles_labels()
    fig.legend(_h, _l, fontsize=8, frameon=False, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.02))
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax = axes[1]
    cov = d.pivot_table(index="year", columns="group", values="plant_uid", aggfunc="nunique")
    for g in cov.columns:
        ax.plot(cov.index, cov[g], lw=2, marker="o", ms=3, alpha=.85,
                color=GROUP_COLOURS.get(g, "#888888"), label=g)
    ax.axhline(len(bal_uids), ls="--", lw=2.2, color="black", zorder=5)
    ax.annotate(f"balanced set: {len(bal_uids)} plants, every year",
                xy=(cov.index.min(), len(bal_uids)), xytext=(3, 4),
                textcoords="offset points", fontsize=8, fontweight="bold")
    ax.set_title("(b) Plants with a figure, observed or imputed", loc="left")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("plants with data")
    _foot(fig, "The dashed black line is the balanced set of "
             f"{len(bal_uids)} plants that Section 4.2 rests on: Volkswagen, "
             "Mercedes-Benz and BMW sites reporting in every year 2012-2023. The "
             "coloured lines are the whole collected universe, including Renault, "
             "Stellantis, Ford, Toyota and Ineos, which appear here and nowhere "
             "else in the study. Panel (a) should be read only alongside panel "
             "(b): the coloured lines largely reflect how many plants report "
             "figures in each year, not how employment moved. France is included, "
             "but measured as commune totals from URSSAF rather than plant "
             "headcount, so French and non-French figures are not exactly the "
             "same quantity. Any claim about a trend should rest on the balanced "
             "set.", y=-0.02)
    fig.tight_layout(rect=[0, 0.1, 1, 0.93])
    return _save(fig, save)
