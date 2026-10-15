# -*- coding: utf-8 -*-
"""
================================================================================
 IEA FACILITY MAP  ->  VALUE-CHAIN CONTEXT FOR THE PLANT PANEL
 Supporting layer for Module 04
================================================================================

WHAT THIS ADDS
--------------
Module 04 measures a residual: the automotive employment recorded in a region
that does NOT sit in one of the plants tracked here. That residual is defined
by subtraction, which makes it correct but abstract — it is whatever is left
over once assembly is accounted for.

The IEA facility map turns the residual into something visible. It geolocates
17,156 automotive facilities and, crucially, says what each one IS:

    Parts supplier                  15,812
    Tier 1 component supplier          554
    Steel plant                        275
    Assembly - car                     248
    In-house component                 194
    Battery cell manufacturing          39
    Battery component manufacturing     34

So the residual is not a statistical artefact. It is roughly sixty parts
suppliers for every assembly plant, and they can be counted region by region.

TWO USES, AND THE LIMITS OF EACH
--------------------------------
1. COVERAGE AUDIT. The 248 assembly points can be matched against the plant
   panel by proximity, which answers a question no internal check can: are
   there car assembly plants in Europe that the panel simply does not contain?

   The limit is that the IEA file carries NO NAMES — only coordinates, a type
   and a cluster label. Matching is therefore purely spatial, and a match means
   "there is a tracked plant within X km of this point", not "these are the
   same factory". The threshold is a judgement, so `coverage_audit` reports how
   the answer changes across a range of thresholds rather than fixing one.

2. THE RESIDUAL, GIVEN A SHAPE. Supplier counts per region can be set beside
   the employment residual for the same region. If the residual really is the
   supplier tissue, regions with more supplier sites should carry more of it.
   That is a check on the interpretation, not a measurement of it.

WHAT THIS IS NOT
----------------
The IEA map has no employment figures and no dates. It is a snapshot of where
facilities are, not of how many people work in them or when they opened. It
cannot be used for any trend, and nothing here does.

Author: Magd Shamashan
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import foundation as fd

__all__ = [
    "IEA_DIR", "IEA_TYPES", "SUPPLIER_TYPES",
    "load_iea", "iea_in_europe", "coverage_audit", "unmatched_assembly",
    "supplier_density", "residual_vs_suppliers",
    "fig_coverage_audit", "fig_residual_vs_suppliers", "map_value_chain",
]

# Resolved from the project root when available; the relative fallbacks below
# are kept so the module still works if run from the analysis folder directly.
try:
    from specular_paths import P as _P
    IEA_DIR = _P.inputs / "iea"
except Exception:
    IEA_DIR = Path("../datasets/IEA json")
IEA_FILE = "IEA_2025 (1).json"

IEA_TYPES = ("Assembly - car", "Tier 1 component supplier", "In-house component",
             "Parts supplier", "Battery cell manufacturing",
             "Battery component manufacturing", "Steel plant")

# Everything that is NOT final assembly: this is what the Module 04 residual
# should be made of.
SUPPLIER_TYPES = ("Parts supplier", "Tier 1 component supplier",
                  "In-house component", "Battery cell manufacturing",
                  "Battery component manufacturing", "Steel plant")

# Bounding box for Europe west of the Urals. The IEA file is global; without
# this, Chinese and American clusters dominate every count.
EUROPE_BOX = dict(lat_min=34.0, lat_max=71.0, lon_min=-11.0, lon_max=32.0)


def _iea_path(directory: Optional[str] = None) -> Path:
    for cand in ([Path(directory)] if directory else []) + [
            IEA_DIR, Path("datasets/IEA json"), Path("../datasets/IEA json")]:
        p = cand / IEA_FILE
        if p.exists():
            return p
    raise FileNotFoundError(
        f"{IEA_FILE} not found. Pass directory= explicitly.")


def load_iea(directory: Optional[str] = None) -> pd.DataFrame:
    """The IEA facility map as a dataframe with numeric coordinates."""
    recs = json.loads(_iea_path(directory).read_text())
    d = pd.DataFrame(recs)
    for c in ("lat", "lon"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["lat", "lon"])
    d["is_assembly"] = d["type"] == "Assembly - car"
    d["is_supplier"] = d["type"].isin(SUPPLIER_TYPES)
    return d.reset_index(drop=True)


def iea_in_europe(d: Optional[pd.DataFrame] = None,
                  restrict_to_nuts: bool = True,
                  level: int = 0) -> pd.DataFrame:
    """
    IEA facilities inside Europe.

    A latitude/longitude box is not enough on its own: a box wide enough to
    reach Finland also reaches Oran and Kaliningrad, and those arrive as
    "European assembly plants the panel is missing" when they are simply
    outside the reporting area. `restrict_to_nuts` keeps only points that fall
    inside a NUTS country polygon, which is the same territory the employment
    statistics cover. The box is kept as a cheap first pass and as a fallback
    for when the geometry cannot be loaded.
    """
    d = load_iea() if d is None else d
    b = EUROPE_BOX
    out = d[(d.lat.between(b["lat_min"], b["lat_max"]))
            & (d.lon.between(b["lon_min"], b["lon_max"]))].reset_index(drop=True)
    if not restrict_to_nuts:
        return out
    geo = fd.load_nuts_geometry(level=level)
    if geo is None:
        warnings.warn("NUTS geometry unavailable; falling back to the "
                      "bounding box, which also contains North Africa and "
                      "Kaliningrad.")
        return out
    keep = _assign_regions(out, geo, code_len=2 + level)
    out = out[pd.notna(keep)].copy()
    out["cntr"] = [k for k in keep if pd.notna(k)]
    return out.reset_index(drop=True)


def _assign_regions(pts: pd.DataFrame, geo: dict, code_len: int) -> np.ndarray:
    """Point-in-polygon assignment of every point to a NUTS code."""
    feats = [f for f in geo["features"]
             if len(str(f["properties"]["NUTS_ID"])) == code_len]
    lat = pts["lat"].to_numpy(); lon = pts["lon"].to_numpy()
    assigned = np.full(len(pts), None, dtype=object)
    for f in feats:
        nid = str(f["properties"]["NUTS_ID"])
        rings = _rings(f.get("geometry"))
        if not rings:
            continue
        xs = np.concatenate([np.asarray(r)[:, 0] for r in rings])
        ys = np.concatenate([np.asarray(r)[:, 1] for r in rings])
        box = ((lon >= xs.min()) & (lon <= xs.max())
               & (lat >= ys.min()) & (lat <= ys.max()) & (assigned == None))
        idx = np.flatnonzero(box)
        if not len(idx):
            continue
        inside = np.zeros(len(idx), dtype=bool)
        for ring in rings:
            inside ^= _point_in_ring(lon[idx], lat[idx], np.asarray(ring))
        assigned[idx[inside]] = nid
    return assigned


# ==============================================================================
# 1. COVERAGE AUDIT
# ==============================================================================

def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance, vectorised over the second pair."""
    R = 6371.0
    p1, p2 = math.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def coverage_audit(radii_km: Sequence[float] = (10, 25, 50, 75, 100),
                   iea: Optional[pd.DataFrame] = None,
                   plants: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    How many IEA assembly plants have a tracked plant nearby, and vice versa?

    Reported across a RANGE of distance thresholds on purpose. The IEA file has
    no names, so "the same factory" can only ever mean "close enough", and the
    right threshold is a judgement about how precisely both sources place a
    site. Showing the whole curve lets a reader see how much the answer depends
    on that judgement instead of taking one number on trust.
    """
    iea = iea_in_europe(iea)
    pl = fd.plant_nuts_crosswalk() if plants is None else plants
    pl = pl.dropna(subset=["lat", "lon"])
    ia = iea[iea["is_assembly"]].reset_index(drop=True)

    # distance from every IEA assembly point to its nearest tracked plant
    nn_iea = np.array([_haversine_km(r.lat, r.lon, pl["lat"].values,
                                     pl["lon"].values).min()
                       for r in ia.itertuples()])
    nn_pl = np.array([_haversine_km(r.lat, r.lon, ia["lat"].values,
                                    ia["lon"].values).min()
                      for r in pl.itertuples()])

    rows = []
    for r in radii_km:
        rows.append(dict(
            radius_km=r,
            iea_assembly=len(ia),
            iea_with_tracked_plant=int((nn_iea <= r).sum()),
            iea_missing=int((nn_iea > r).sum()),
            pct_iea_covered=100 * (nn_iea <= r).mean(),
            tracked_plants=len(pl),
            tracked_with_iea_point=int((nn_pl <= r).sum()),
            pct_tracked_confirmed=100 * (nn_pl <= r).mean()))
    out = pd.DataFrame(rows)
    out.attrs["nn_iea_km"] = nn_iea
    out.attrs["nn_plant_km"] = nn_pl
    out.attrs["iea_assembly"] = ia
    out.attrs["plants"] = pl
    return out


def unmatched_assembly(radius_km: float = 50,
                       iea: Optional[pd.DataFrame] = None,
                       plants: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    IEA assembly points with no tracked plant within `radius_km`.

    These are the candidate gaps in the panel. The cluster label is the only
    identifying information the IEA file carries, so it is reported as-is; a
    reader can look up what sits at those coordinates.
    """
    iea = iea_in_europe(iea)
    pl = fd.plant_nuts_crosswalk() if plants is None else plants
    pl = pl.dropna(subset=["lat", "lon"])
    ia = iea[iea["is_assembly"]].copy()
    ia["nearest_tracked_km"] = [
        _haversine_km(r.lat, r.lon, pl["lat"].values, pl["lon"].values).min()
        for r in ia.itertuples()]
    miss = ia[ia["nearest_tracked_km"] > radius_km].copy()
    miss = miss.sort_values("nearest_tracked_km", ascending=False)
    miss.attrs["radius_km"] = radius_km
    cols = [c for c in ("cntr", "lat", "lon", "Cluster_su",
                        "nearest_tracked_km") if c in miss.columns]
    return miss[cols].reset_index(drop=True)


# ==============================================================================
# 2. THE RESIDUAL, GIVEN A SHAPE
# ==============================================================================

def supplier_density(level: int = 2, iea: Optional[pd.DataFrame] = None,
                     geo: Optional[dict] = None) -> pd.DataFrame:
    """
    Count IEA facilities per NUTS region, by type.

    Points are assigned to regions by point-in-polygon against the GISCO
    geometry. Implemented with a ray-casting test rather than a spatial library,
    so it runs anywhere; it is exact for these polygons.
    """
    iea = iea_in_europe(iea)
    geo = fd.load_nuts_geometry(level=level) if geo is None else geo
    if geo is None:
        raise RuntimeError("NUTS geometry unavailable — cannot assign points "
                           "to regions.")

    iea = iea.copy()
    iea["nuts"] = _assign_regions(iea, geo, code_len=2 + level)
    d = iea.dropna(subset=["nuts"])
    g = (d.pivot_table(index="nuts", columns="type", values="lat",
                       aggfunc="size").fillna(0).astype(int))
    g["assembly"] = g.get("Assembly - car", 0)
    g["suppliers"] = sum(g.get(t, 0) for t in SUPPLIER_TYPES)
    g["total"] = g[[c for c in g.columns
                    if c not in ("assembly", "suppliers")]].sum(axis=1)
    out = g.reset_index().rename(columns={"nuts": f"nuts{level}"})
    out.attrs["n_assigned"] = len(d)
    out.attrs["n_total"] = len(iea)
    return out


def _rings(geom) -> List[list]:
    if geom is None:
        return []
    t, c = geom.get("type"), geom.get("coordinates") or []
    if t == "Polygon":
        return list(c)
    if t == "MultiPolygon":
        return [r for poly in c for r in poly]
    return []


def _point_in_ring(px: np.ndarray, py: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Vectorised ray-casting point-in-polygon."""
    x, y = ring[:, 0], ring[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    inside = np.zeros(len(px), dtype=bool)
    for i in range(len(x)):
        cond = ((y[i] > py) != (y2[i] > py))
        if not cond.any():
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (x2[i] - x[i]) * (py - y[i]) / (y2[i] - y[i]) + x[i]
        inside ^= cond & (px < xint)
    return inside


def residual_vs_suppliers(prp: pd.DataFrame, year: int = 2023,
                          dens: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Regional employment residual beside the number of IEA supplier sites.

    If the residual is the supplier tissue, the two should move together across
    regions. This is a consistency check on the interpretation, and nothing
    more: the IEA map has no employment figures, so a supplier count cannot
    predict a headcount, only agree or disagree with its geography.
    """
    dens = supplier_density(level=2) if dens is None else dens
    d = prp[(prp["year"] == year) & (prp["ratio_flag"] == "ok")].copy()
    m = d.merge(dens.rename(columns={"nuts2": "nuts2"}), on="nuts2", how="left")
    for c in ("assembly", "suppliers", "total"):
        if c in m:
            m[c] = m[c].fillna(0).astype(int)
    m = m[m["residual"].notna()]
    if len(m) > 3:
        m.attrs["pearson_r"] = float(m["suppliers"].corr(m["residual"]))
        m.attrs["spearman_r"] = float(
            m["suppliers"].rank().corr(m["residual"].rank()))
    m.attrs["year"] = year
    return m


# ==============================================================================
# 3. FIGURES
# ==============================================================================

def _style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200,
        "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11.5, "axes.titleweight": "bold", "font.size": 10})


# Explanatory notes used to be printed under each figure. They now live in
# the LaTeX captions, which lets the chart itself print larger. The text is
# still passed to _foot and kept on the figure as `fig._foot_text`; set this
# to True to draw it again.
SHOW_FOOTNOTES = False


def _foot(fig, text: str, width: int = 110, y: float = -0.02, size: float = 8.5):
    fig._foot_text = str(text)
    if not SHOW_FOOTNOTES:
        return
    import textwrap
    lines = []
    for para in str(text).split("\n"):
        lines.extend(textwrap.wrap(para, width=width) or [""])
    fig.text(0.02, y, "\n".join(lines), fontsize=size, color="#5A5A5A",
             ha="left", va="top")


def fig_coverage_audit(cov: Optional[pd.DataFrame] = None,
                       save: Optional[str] = None):
    """How completely does the plant panel cover European car assembly?"""
    import matplotlib.pyplot as plt
    _style()
    cov = coverage_audit() if cov is None else cov
    nn_iea = cov.attrs["nn_iea_km"]

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.6))
    fig.suptitle("Coverage of European car assembly, checked against the IEA "
                 "facility map", fontsize=14.5, fontweight="bold", x=0.02,
                 ha="left")

    ax = axes[0]
    ax.plot(cov["radius_km"], cov["pct_iea_covered"], "-o", color="#14746F",
            lw=2.6, ms=6, label="IEA assembly sites with a tracked plant nearby")
    ax.plot(cov["radius_km"], cov["pct_tracked_confirmed"], "-o",
            color="#C6803D", lw=2.2, ms=5, ls="--",
            label="tracked plants with an IEA point nearby")
    for _, r in cov.iterrows():
        ax.annotate(f"{r['pct_iea_covered']:.0f}%", (r["radius_km"],
                    r["pct_iea_covered"]), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8.8,
                    fontweight="bold", color="#14746F")
    ax.set_xlabel("how close two points must be to count as the same site (km)")
    ax.set_ylabel("% matched")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8.8, frameon=False, loc="lower right")
    ax.set_title("(a) Share of sites matched, by distance threshold", loc="left")

    ax = axes[1]
    ax.hist(np.clip(nn_iea, 0, 300), bins=30, color="#4E6E8E",
            edgecolor="white")
    ax.axvline(50, color="#B03A2E", lw=1.6, ls="--")
    ax.annotate("50 km", (50, ax.get_ylim()[1] * .92), xytext=(6, 0),
                textcoords="offset points", fontsize=9.2, color="#B03A2E",
                fontweight="bold")
    ax.set_xlabel("distance from an IEA assembly site to the nearest tracked "
                  "plant (km)")
    ax.set_ylabel("IEA assembly sites")
    ax.set_title("(b) How far away the nearest tracked plant is", loc="left")

    r50 = cov[cov["radius_km"] == 50]
    txt = ""
    if len(r50):
        r = r50.iloc[0]
        txt = (f"At 50 km, {int(r['iea_with_tracked_plant'])} of "
               f"{int(r['iea_assembly'])} IEA car-assembly sites in Europe have "
               f"a tracked plant nearby and {int(r['iea_missing'])} do not.  ")
    _foot(fig, txt +
          "The IEA facility map records coordinates, a facility type and a "
          "cluster label, but no plant names. Matching is therefore purely by "
          "distance: a match means a tracked plant lies within the stated "
          "radius, not that the two records are confirmed to be the same "
          "factory. Because that threshold is a judgement, the whole curve is "
          "shown rather than a single figure. Sites without a match are "
          "candidates for gaps in the panel and are listed separately.")
    fig.subplots_adjust(top=.875, bottom=.20, wspace=.26)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def fig_residual_vs_suppliers(rvs: Optional[pd.DataFrame] = None,
                              save: Optional[str] = None):
    """Is the employment residual where the supplier sites are?"""
    import matplotlib.pyplot as plt
    _style()
    if rvs is None:
        raise ValueError("pass residual_vs_suppliers(prp)")
    year = rvs.attrs.get("year", "")

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 5.8))
    fig.suptitle("The employment residual and the supplier sites that should "
                 f"explain it, {year}", fontsize=14.5, fontweight="bold",
                 x=0.02, ha="left")

    ax = axes[0]
    ax.scatter(rvs["suppliers"], rvs["residual"] / 1000, s=46, alpha=.72,
               color="#14746F", edgecolor="white", zorder=3)
    big = rvs.nlargest(6, "residual")
    for r in big.itertuples():
        ax.annotate(str(r.geo_label)[:22], (r.suppliers, r.residual / 1000),
                    xytext=(6, 4), textcoords="offset points", fontsize=7.6,
                    color="#444444")
    if "pearson_r" in rvs.attrs and rvs["suppliers"].std() > 0:
        z = np.polyfit(rvs["suppliers"], rvs["residual"] / 1000, 1)
        xs = np.linspace(0, rvs["suppliers"].max(), 20)
        ax.plot(xs, np.polyval(z, xs), color="#B03A2E", lw=1.5, ls="--",
                label=f"least squares (r = {rvs.attrs['pearson_r']:.2f}, "
                      f"rank r = {rvs.attrs['spearman_r']:.2f})")
        ax.legend(fontsize=8.8, frameon=False, loc="upper left")
    ax.set_xlabel("IEA supplier and component sites in the region")
    ax.set_ylabel("employment outside the tracked plants (thousands)")
    ax.set_title("(a) One point per region", loc="left")

    ax = axes[1]
    t = rvs.nlargest(14, "residual").sort_values("residual")
    y = np.arange(len(t))
    # Stacked, not overlaid: the two parts sum to the region's total, and
    # drawing them from a common zero made the smaller one look larger.
    ax.barh(y, t["plant_emp"] / 1000, color="#14746F",
            label="employment in tracked plants")
    ax.barh(y, t["residual"] / 1000, left=t["plant_emp"] / 1000,
            color="#7FB2AE", label="employment outside them")
    for yi, (p, r) in enumerate(zip(t["plant_emp"] / 1000, t["residual"] / 1000)):
        ax.annotate(f"{100 * r / (p + r):.0f}% outside", (p + r, yi),
                    xytext=(5, 0), textcoords="offset points", va="center",
                    fontsize=7.4, color="#555555")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{str(l)[:26]} ({int(s)} sites)"
                        for l, s in zip(t["geo_label"], t["suppliers"])],
                       fontsize=7.8)
    ax.set_xlabel("employees (thousands)")
    ax.set_xlim(0, ((t["plant_emp"] + t["residual"]) / 1000).max() * 1.22)
    ax.legend(fontsize=8.4, frameon=False, loc="lower right")
    ax.set_title("(b) The fourteen regions with the largest residual", loc="left")

    _foot(fig,
          "The residual is the automotive employment a region records that does "
          "not sit in one of the plants tracked here. The horizontal axis "
          "counts the parts suppliers, tier-1 suppliers, in-house component "
          "plants, battery and steel sites the IEA locates in the same region. "
          "A positive relationship supports reading the residual as the "
          "supplier tissue around assembly, which is what it is meant to be. "
          "It cannot do more than that: the IEA map records where facilities "
          "are, never how many people work in them, so this is a check on the "
          "geography of the interpretation and not a measurement of it.")
    fig.subplots_adjust(top=.885, bottom=.20, wspace=.34)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


def map_value_chain(radius_km: float = 50, year: int = 2023,
                    show_suppliers: bool = True, max_suppliers: int = 4000,
                    save: Optional[str] = None):
    """
    The whole value chain on one map: tracked plants, IEA assembly sites that
    the panel does and does not reach, and the supplier tissue behind them.
    """
    import folium
    iea = iea_in_europe()
    pl = fd.plant_nuts_crosswalk().dropna(subset=["lat", "lon"])
    miss = unmatched_assembly(radius_km, iea, pl)
    missing_pts = set(zip(miss["lat"].round(5), miss["lon"].round(5)))
    ia = iea[iea["is_assembly"]]

    m = folium.Map(location=[50.0, 10.0], zoom_start=4, tiles="cartodbpositron")

    if show_suppliers:
        sup = iea[iea["is_supplier"]]
        if len(sup) > max_suppliers:
            sup = sup.sample(max_suppliers, random_state=0)
        fg = folium.FeatureGroup(name=f"IEA supplier & component sites "
                                      f"({len(sup):,} shown)", show=True)
        for r in sup.itertuples():
            folium.CircleMarker([r.lat, r.lon], radius=1.6, color="#9AA0A6",
                                weight=0, fill=True, fill_color="#9AA0A6",
                                fill_opacity=.45).add_to(fg)
        fg.add_to(m)

    fg = folium.FeatureGroup(name="IEA car assembly — no tracked plant nearby",
                             show=True)
    for r in ia.itertuples():
        key = (round(r.lat, 5), round(r.lon, 5))
        if key not in missing_pts:
            continue
        folium.CircleMarker(
            [r.lat, r.lon], radius=8, color="#B03A2E", weight=2.4, fill=True,
            fill_color="#B03A2E", fill_opacity=.25,
            tooltip=f"IEA assembly, no tracked plant within {radius_km:.0f} km",
            popup=folium.Popup(html=(
                f"<b>IEA car assembly site</b><br>cluster: {r.Cluster_su or '—'}"
                f"<br>{r.lat:.4f}, {r.lon:.4f}<br>"
                f"<i>not represented in the plant panel</i>"), max_width=300)
        ).add_to(fg)
    fg.add_to(m)

    fg = folium.FeatureGroup(name="IEA car assembly — matched", show=True)
    for r in ia.itertuples():
        key = (round(r.lat, 5), round(r.lon, 5))
        if key in missing_pts:
            continue
        folium.CircleMarker([r.lat, r.lon], radius=4.5, color="#14746F",
                            weight=1.4, fill=True, fill_color="#ffffff",
                            fill_opacity=.9,
                            tooltip="IEA assembly, matched to a tracked plant"
                            ).add_to(fg)
    fg.add_to(m)

    ep = fd.employment_panel()
    ep = ep[(ep["year"] == year) & ep["employees"].notna()]
    emp = dict(zip(ep["plant_uid"], ep["employees"]))
    fg = folium.FeatureGroup(name="plants tracked in this study", show=True)
    for r in pl.itertuples():
        e = emp.get(r.plant_uid)
        folium.CircleMarker(
            [r.lat, r.lon],
            radius=float(np.clip(np.sqrt(e) / 12, 4, 14)) if e else 5,
            color="#1B3A6B", weight=1.6, fill=True, fill_color="#2C6FB3",
            fill_opacity=.75,
            tooltip=(f"{r.plant_name}" + (f" — {e:,.0f} employees" if e else "")),
            popup=folium.Popup(html=(
                f"<b>{r.plant_name}</b><br>{r.city}, {r.country_iso2}<br>"
                f"group: {r.group}"
                + (f"<br>employees {year}: {e:,.0f}" if e else "")), max_width=300)
        ).add_to(fg)
    fg.add_to(m)

    # National borders, heavier than anything else on the map, so a country
    # boundary can be told from the supplier cloud behind it.
    geo0 = fd.load_nuts_geometry(level=0)
    if geo0 is not None:
        folium.GeoJson(geo0, name="country borders",
                       style_function=lambda f: {"fillOpacity": 0,
                                                 "color": "#1a1a1a",
                                                 "weight": 2.1, "opacity": .9}
                       ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    n_missing = len(miss)
    legend = (
        "<div style='position:fixed;bottom:22px;left:22px;z-index:9999;"
        "background:rgba(255,255,255,.96);padding:11px 13px;border:1px solid "
        "#b9b9b9;border-radius:5px;font:11.5px/1.55 -apple-system,sans-serif;"
        "max-width:320px;box-shadow:0 1px 5px rgba(0,0,0,.18)'>"
        "<b>The automotive value chain in Europe</b><br>"
        "<span style='color:#2C6FB3'>&#9679;</span> plant tracked in this "
        "study, sized by employment<br>"
        "<span style='color:#14746F'>&#9675;</span> IEA car assembly site with "
        "a tracked plant nearby<br>"
        "<span style='color:#B03A2E'>&#9679;</span> IEA car assembly site with "
        f"<b>none</b> within {radius_km:.0f} km &mdash; {n_missing} of these<br>"
        "<span style='color:#9AA0A6'>&#8226;</span> supplier, component, "
        "battery and steel sites<br>"
        "<span style='color:#1a1a1a'>&#9473;</span> country border<br>"
        "<hr style='margin:6px 0;border:0;border-top:1px solid #ddd'>"
        "<i>The grey points are the value chain behind assembly: the IEA counts "
        "roughly sixty of them for every car plant. They are what the "
        "employment residual is made of. The map records where facilities are, "
        "never how many people work in them.</i></div>")
    m.get_root().html.add_child(folium.Element(legend))

    print(f"Value-chain map · {len(pl)} tracked plants · {len(ia)} IEA assembly "
          f"sites in Europe, {n_missing} with no tracked plant within "
          f"{radius_km:.0f} km")
    if show_suppliers:
        print(f"                · {len(iea[iea['is_supplier']]):,} supplier and "
              f"component sites in the file")
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        m.save(save)
    return m
