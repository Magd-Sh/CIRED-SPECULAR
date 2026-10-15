"""
Static maps for Chapter 4, built with matplotlib only.

No geopandas, no folium, no selenium: a choropleth is filled polygons coloured
by a value, and the NUTS GeoJSON in analysis/outputs/ already holds the
polygons. Screenshotting the interactive maps would drag in basemap tiles, zoom
controls and attribution text, none of which belong in a printed thesis.

Usage from a notebook in analysis/:

    import static_maps as sm
    sm.map_employment_change(save="../thesis/figures/map01_change.png")
    sm.map_value_chain(save="../thesis/figures/map02_valuechain.png")
    sm.map_battery_france(save="../thesis/figures/map03_battery.png")
"""
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.cm import ScalarMappable

import foundation as fd

# Resolved from the project root, not from the caller's working directory: the
# original read "outputs/" relative to wherever python happened to be started.
try:
    from specular_paths import P as _P
    GEO_DIR = _P.geometry
except Exception:
    GEO_DIR = Path(__file__).resolve().parents[2] / "02_data_inputs" / "geometry"
EUROPE = dict(lon=(-11, 32), lat=(34, 71))

# Europe is wide and far north; without a projection the map looks stretched.
# A cosine-of-mid-latitude aspect is the cheap, honest fix for a figure of this
# size and needs no pyproj.
def _aspect(lat_mid: float) -> float:
    return 1.0 / np.cos(np.radians(lat_mid))


def _tracked_in_scope() -> pd.DataFrame:
    """
    Tracked plants inside the scope of the study.

    A bounding box does NOT do this: St Petersburg sits at 30.3E, inside any
    frame wide enough to include Romania and Bulgaria, so the two Russian sites
    survived the box filter and were drawn while the caption claimed they were
    not. Excluding by country is the honest test.
    """
    cw = fd.plant_nuts_crosswalk().dropna(subset=["lat", "lon"])
    return cw[~cw["country_iso2"].isin(OUT_OF_SCOPE)]


OUT_OF_SCOPE = ("RU",)


def _load_geo(level: int) -> Dict[str, list]:
    """NUTS_ID -> list of (N,2) lon/lat rings."""
    path = GEO_DIR / f"nuts{level}_2021.geojson"
    gj = json.load(open(path, encoding="utf-8"))
    out: Dict[str, list] = {}
    for ft in gj["features"]:
        nid = ft["properties"]["NUTS_ID"]
        geom = ft["geometry"]
        polys = []
        if geom["type"] == "Polygon":
            polys = [geom["coordinates"]]
        elif geom["type"] == "MultiPolygon":
            polys = geom["coordinates"]
        rings = []
        for poly in polys:
            if poly:
                rings.append(np.asarray(poly[0], dtype=float))   # outer ring
        if rings:
            out[nid] = rings
    return out


def _draw(ax, rings_by_id, values=None, cmap=None, norm=None,
          face="#EDEDED", edge="#FFFFFF", lw=.35, zorder=1):
    """Fill polygons; colour by `values` where a code has one."""
    patches, colours = [], []
    for nid, rings in rings_by_id.items():
        for ring in rings:
            patches.append(MplPolygon(ring, closed=True))
            if values is not None and nid in values.index and pd.notna(values[nid]):
                colours.append(cmap(norm(values[nid])))
            else:
                colours.append(face)
    pc = PatchCollection(patches, facecolors=colours, edgecolors=edge,
                         linewidths=lw, zorder=zorder)
    ax.add_collection(pc)
    return pc


def _frame(ax, title, subtitle=None):
    ax.set_xlim(*EUROPE["lon"]); ax.set_ylim(*EUROPE["lat"])
    ax.set_aspect(_aspect(np.mean(EUROPE["lat"])))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    # The subtitle sits above the axes and grows upward, so the title has to be
    # pushed clear of it by one line-height per subtitle line.
    n_lines = 1 + (subtitle or "").count("\n")
    ax.set_title(title, loc="left", fontsize=13.2, fontweight="bold",
                 pad=18 + 13 * n_lines)
    if subtitle:
        ax.text(0, 1.012, subtitle, transform=ax.transAxes, fontsize=9.0,
                color="#5A5A5A", va="bottom", linespacing=1.35)


# ==============================================================================
#  MAP 1 - where employment moved, 2012-2023
# ==============================================================================
def shift_share_strict(y0: int = 2012, y1: int = 2023):
    """
    The 68-region strict-balanced shift-share the thesis reports.

    NOT `balanced_subset`, which balances on the two ENDPOINTS and returns 76
    regions and -149k/+133k. The thesis uses strict balance -- present in every
    year of the window -- which gives 68 regions and -142k/+128k. Using the
    wrong one here would put a map in the chapter that disagrees with the
    figure two pages earlier.
    """
    import module01_spatial as m1
    p = m1.european_panel(level=1, france="urssaf", drop_other_bloc=True)
    years = list(range(y0, y1 + 1))
    piv = p.pivot_table(index="geo_code", columns="year", values="value",
                        aggfunc="first")
    keep = piv[years].dropna().index
    strict = p[p["geo_code"].isin(keep) & p["year"].isin([y0, y1])].copy()
    return m1.shift_share_1sector(strict, y0, y1).set_index("geo_code")


def map_employment_change(y0: int = 2012, y1: int = 2023,
                          save: Optional[str] = None, label_n: int = 5):
    """Figure 4.2 as a map: the regional component of the shift-share."""
    import module01_spatial as m1
    ss = shift_share_strict(y0, y1)
    v = ss["differential_shift"] / 1000.0

    rings = _load_geo(1)
    lim = float(np.nanpercentile(np.abs(v), 96))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    cmap = plt.get_cmap("RdBu_r")

    fig, ax = plt.subplots(figsize=(9.0, 8.2))
    _draw(ax, rings, values=v, cmap=cmap, norm=norm)
    _frame(ax, f"Regional shift in automotive employment, {y0}\u2013{y1}",
           "thousands of jobs gained or lost against the European growth rate")

    # A map of 68 regions cannot carry 68 labels. Take the largest movers and
    # drop any whose label would sit on top of one already placed; the colour
    # scale carries the rest.
    top = pd.concat([v.nlargest(label_n), v.nsmallest(label_n)])
    top = top.reindex(top.abs().sort_values(ascending=False).index)
    placed: list = []
    for code, val in top.items():
        if code not in rings:
            continue
        pts = max(rings[code], key=len)
        x, y = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        if any((x - px) ** 2 + ((y - py) * 1.6) ** 2 < 3.6 ** 2 for px, py in placed):
            continue
        placed.append((x, y))
        ax.annotate(f"{m1.region_label(code)}\n{val:+,.0f}k", (x, y),
                    fontsize=6.4, ha="center", va="center", zorder=6,
                    color="#111111",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                              edgecolor="#B0B0B0", linewidth=.5, alpha=.9))

    sm_ = ScalarMappable(norm=norm, cmap=cmap); sm_.set_array([])
    cb = fig.colorbar(sm_, ax=ax, fraction=.026, pad=.015, extend="both")
    cb.set_label("thousands of jobs, relative to the European trend", fontsize=8.6)
    cb.ax.tick_params(labelsize=8)
    fig.subplots_adjust(top=.93, bottom=.03, left=.02, right=.96)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


# ==============================================================================
#  MAP 2 - the value chain the plant panel cannot see
# ==============================================================================
def map_value_chain(save: Optional[str] = None):
    """
    Supplier and component sites against assembly plants.

    The argument of Section 4.4 in one picture: the grey cloud is the part of
    the industry that regional statistics count and a panel built on assembly
    plants cannot reach.
    """
    import iea_value_chain as iv
    iea = iv.iea_in_europe()
    sup = iea[iea["is_supplier"]]
    asm = iea[iea["is_assembly"]]
    tracked = _tracked_in_scope()

    rings0 = _load_geo(0)
    fig, ax = plt.subplots(figsize=(9.0, 8.2))
    _draw(ax, rings0, face="#F4F4F4", edge="#D0D0D0", lw=.5)

    ax.scatter(sup["lon"], sup["lat"], s=2.0, c="#9A9A9A", alpha=.35,
               linewidths=0, zorder=2,
               label=f"supplier and component sites ({len(sup):,})")
    ax.scatter(asm["lon"], asm["lat"], s=26, facecolor="none",
               edgecolor="#1F4E79", linewidths=1.0, zorder=3,
               label=f"car assembly sites, IEA ({len(asm)})")
    ax.scatter(tracked["lon"], tracked["lat"], s=30, c="#C0392B", alpha=.9,
               linewidths=0, zorder=4,
               label=f"plants tracked in this study ({len(tracked)})")

    _frame(ax, "The automotive value chain",
           "every site the IEA locates inside the European statistical area")
    leg = ax.legend(loc="upper left", fontsize=8.8, frameon=True,
                    framealpha=.92, borderpad=.7, labelspacing=.7,
                    scatterpoints=1, markerscale=1.6)
    leg.get_frame().set_edgecolor("#CCCCCC")
    fig.subplots_adjust(top=.93, bottom=.03, left=.02, right=.98)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


# ==============================================================================
#  MAP 3 - battery manufacturing in France
# ==============================================================================
NEW_CELL_PLANTS = {"BILLY-BERCLAU": "62", "BRUGES": "33",
                   "GRENOBLE": "38", "BOURBOURG": "59"}


def map_battery_france(y0: int = 2019, y1: int = 2024,
                       save: Optional[str] = None):
    """
    Battery employment by departement, and where it did not exist in y0.

    Communes have no coordinates in the URSSAF extract, so the four new cell
    plants are marked by shading their departement rather than by a point.
    """
    dep = fd.urssaf_panel(nace="C27", scope="battery", level="departement")
    dep = dep[dep["year"].isin([y0, y1])]
    piv = dep.pivot_table(index="departement_code", columns="year",
                          values="employees", aggfunc="sum")
    d2n = fd.french_departement_to_nuts3()
    piv["nuts3"] = [d2n.get(str(i).zfill(2)) for i in piv.index]
    piv = piv.dropna(subset=["nuts3"]).set_index("nuts3")

    rings = {k: v for k, v in _load_geo(3).items() if k.startswith("FR")}
    v = piv[y1]
    norm = Normalize(vmin=0, vmax=float(np.nanpercentile(v, 97)))
    cmap = plt.get_cmap("YlGnBu")

    fig, ax = plt.subplots(figsize=(7.6, 8.0))
    _draw(ax, rings, values=v, cmap=cmap, norm=norm, face="#F2F2F2",
          edge="#FFFFFF", lw=.3)

    new = [d2n.get(c) for c in NEW_CELL_PLANTS.values()]
    for nid, name in zip(new, NEW_CELL_PLANTS):
        if nid not in rings:
            continue
        for ring in rings[nid]:
            ax.add_patch(MplPolygon(ring, closed=True, fill=False,
                                    edgecolor="#B03030", linewidth=1.6, zorder=5))
        pts = max(rings[nid], key=len)
        dy = {"BILLY-BERCLAU": -0.34, "BOURBOURG": 0.34}.get(name, 0.0)
        ax.annotate(name.title(), (pts[:, 0].mean(), pts[:, 1].mean() + dy),
                    fontsize=7.4, ha="center", va="center", zorder=6,
                    fontweight="bold", color="#7A1E1E",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="#B03030", linewidth=.6, alpha=.92))

    ax.set_xlim(-5.5, 9.8); ax.set_ylim(41.2, 51.4)
    ax.set_aspect(_aspect(46.5)); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Battery manufacturing in France, {y1}", loc="left",
                 fontsize=13.2, fontweight="bold", pad=30)
    ax.text(0, 1.012, "APE 2720Z, salaried employment by departement; outlined "
            f"departements recorded no battery employment in {y0}",
            transform=ax.transAxes, fontsize=9.0, color="#5A5A5A", va="bottom")
    sm_ = ScalarMappable(norm=norm, cmap=cmap); sm_.set_array([])
    cb = fig.colorbar(sm_, ax=ax, fraction=.03, pad=.015, extend="max")
    cb.set_label("employees", fontsize=8.6); cb.ax.tick_params(labelsize=8)
    fig.subplots_adjust(top=.93, bottom=.03, left=.02, right=.95)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


# ==============================================================================
#  MAP 2b - the value chain, aggregated to regions
# ==============================================================================
def map_value_chain_regional(level: int = 2, save: Optional[str] = None,
                             label_n: int = 8):
    """
    The same argument as `map_value_chain`, at the level of division Figure 4.13
    uses: supplier sites counted per NUTS region rather than drawn one by one.

    Twelve thousand individual grey dots show that the supplier tissue is dense
    but not where it is dense -- the points overplot into a single smear across
    the middle of the continent. Counting them per region turns the same data
    into a readable surface, and puts the map on the same unit as the residual
    analysis it supports.
    """
    import iea_value_chain as iv
    dens = iv.supplier_density(level=level).set_index(f"nuts{level}")
    tracked = _tracked_in_scope()

    rings = _load_geo(level)
    v = dens["suppliers"]
    norm = Normalize(vmin=0, vmax=float(np.nanpercentile(v[v > 0], 92)))
    cmap = plt.get_cmap("BuPu")

    fig, ax = plt.subplots(figsize=(9.2, 8.4))
    _draw(ax, rings, values=v, cmap=cmap, norm=norm, face="#F4F4F4",
          edge="#FFFFFF", lw=.3)
    ax.scatter(tracked["lon"], tracked["lat"], s=26, c="#C0392B", alpha=.92,
               linewidths=.5, edgecolor="white", zorder=4,
               label=f"plants tracked in this study ({len(tracked)})")

    top = v.nlargest(label_n)
    placed: list = []
    for code, n in top.items():
        if code not in rings:
            continue
        pts = max(rings[code], key=len)
        x, y = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        if any((x - px) ** 2 + ((y - py) * 1.6) ** 2 < 3.4 ** 2 for px, py in placed):
            continue
        placed.append((x, y))
        ax.annotate(f"{code}\n{int(n)}", (x, y), fontsize=6.6, ha="center",
                    va="center", zorder=6, color="#111111",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                              edgecolor="#B0B0B0", linewidth=.5, alpha=.9))

    # Two short lines instead of one long one: the subtitle sets the figure's
    # minimum width, and at one line it was forcing the map narrower than the
    # page allows.
    _frame(ax, f"The automotive value chain \u2014 by NUTS-{level}",
           "parts, tier-one, in-house component, battery and steel\n"
           "sites counted per region, tracked plants drawn on top")
    ax.legend(loc="upper left", fontsize=8.8, frameon=True, framealpha=.92,
              markerscale=1.4, borderpad=.6)
    sm_ = ScalarMappable(norm=norm, cmap=cmap); sm_.set_array([])
    cb = fig.colorbar(sm_, ax=ax, fraction=.026, pad=.015, extend="max")
    cb.set_label("supplier and component sites in the region", fontsize=8.6)
    cb.ax.tick_params(labelsize=8)
    fig.subplots_adjust(top=.93, bottom=.03, left=.02, right=.96)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig


# ==============================================================================
#  MAP 4 - the plant registry: what carries a figure and what does not
# ==============================================================================
GROUP_COLOURS = {
    "Volkswagen Group": "#1F4E79",
    "Mercedes-Benz":    "#0E7C7B",
    "BMW Group":        "#7A4E9E",
    "Stellantis":       "#C0392B",
    "Renault":          "#E08A1E",
    "Ford":             "#4E6E58",
    "Toyota":           "#8C6D46",
    "Ineos":            "#6B6B6B",
}


def map_plant_panel(save: Optional[str] = None, annotate_absent: bool = False):
    """
    Every plant in the master list, coloured by group, with the plants that
    carry no employment figure drawn hollow.

    This is a map of the SOURCE MATERIAL, not of a result: the point is that the
    registry is 89 plants and only 76 of them have a number attached, and that
    the thirteen without one are not scattered at random. They are Stellantis in
    Italy and Spain, Renault in Spain and France, and the two Russian sites,
    which are outside the study's scope in any case.

    Coordinates come from the plant crosswalk, which has them for all 89.
    """
    cw = fd.plant_nuts_crosswalk().dropna(subset=["lat", "lon"]).copy()
    have = set(fd.load_employment()["plant_uid"].unique())
    cw["has_employment"] = cw["plant_uid"].isin(have)

    rings0 = _load_geo(0)
    fig, ax = plt.subplots(figsize=(9.4, 7.2))
    _draw(ax, rings0, face="#F4F4F4", edge="#D0D0D0", lw=.5)

    # groups ordered by how many plants they contribute, so the legend reads as
    # a statement about the panel's composition rather than alphabetically
    order = (cw[cw["has_employment"]]["group"].value_counts().index.tolist()
             + [g for g in cw["group"].unique()
                if g not in set(cw[cw["has_employment"]]["group"])])
    for g in order:
        s = cw[(cw["group"] == g) & cw["has_employment"]]
        if s.empty:
            continue
        ax.scatter(s["lon"], s["lat"], s=44,
                   c=GROUP_COLOURS.get(g, "#6B6B6B"), alpha=.92,
                   linewidths=.6, edgecolor="white", zorder=4,
                   label=f"{g} ({len(s)})")

    absent = cw[~cw["has_employment"]]
    ax.scatter(absent["lon"], absent["lat"], s=46, facecolor="none",
               edgecolor="#4A4A4A", linewidths=1.1, zorder=5,
               label=f"in the list, no employment figure ({len(absent)})")

    if annotate_absent:
        for _, r in absent.iterrows():
            ax.annotate(str(r["city"]), (r["lon"], r["lat"]), fontsize=6.2,
                        xytext=(4, 3), textcoords="offset points",
                        color="#333333", zorder=6)

    # The NUTS geometry stops at the EU and EFTA border, so the two Russian
    # sites would otherwise sit on blank paper and read as a plotting fault.
    ru = cw[cw["country_iso2"].isin(OUT_OF_SCOPE)]
    if not ru.empty:
        ax.annotate(f"Russia ({len(ru)}), outside\nthe scope of the study",
                    (float(ru["lon"].mean()), float(ru["lat"].mean())),
                    fontsize=7.2, ha="right", va="center", color="#4A4A4A",
                    xytext=(-10, 0), textcoords="offset points", zorder=6)

    n_ok = int(cw["has_employment"].sum())
    _frame(ax, "The plants in the panel",
           f"{len(cw)} plants in the master list; {n_ok} carry at least one "
           f"employment figure,\n{len(absent)} carry none")
    # The continental frame used by the other maps reaches to 71N and leaves the
    # top third of this one empty: no plant sits above 60N. Crop to the plants.
    ax.set_xlim(-11.5, 33.0)
    ax.set_ylim(35.0, 62.0)
    ax.set_aspect(_aspect(48.5))
    leg = ax.legend(loc="upper left", fontsize=8.4, frameon=True,
                    framealpha=.93, borderpad=.6, labelspacing=.45,
                    scatterpoints=1, markerscale=1.15)
    leg.get_frame().set_edgecolor("#CCCCCC")
    fig.subplots_adjust(top=.93, bottom=.03, left=.02, right=.98)
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200, bbox_inches="tight")
    return fig
