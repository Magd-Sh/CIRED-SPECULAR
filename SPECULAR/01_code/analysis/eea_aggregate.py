# -*- coding: utf-8 -*-
"""
================================================================================
 EEA CO2 MONITORING -> COMPACT MODEL-LEVEL MASS LOOKUP
 Preparation step for Module 03b (vehicle size)
================================================================================

THE PROBLEM
-----------
The EEA monitoring dataset (Regulation (EU) 2019/631) is per-REGISTRATION:
roughly 10-15 million rows per year, well over 100 million rows for 2010-2023,
several GB per year as CSV. It will not fit in memory on a normal machine, and
it does not need to.

THE POINT
---------
We do not want registrations. We want, per (make, commercial name, year):

    mass in running order  (the size variable the thesis needs)
    fuel mix               (BEV share, to cross-check the nameplate classifier)
    registrations          (a weight, so heavy-selling models count for more)

That is a table of tens of thousands of rows -- a few MB. So we aggregate AT THE
DOOR, streaming the file in chunks and never holding more than one chunk plus
the running aggregate. Memory is bounded by chunk size, not by file size, so
this runs on a laptop as happily as on a server.

COLUMN NAMES VARY BY YEAR
-------------------------
EEA changed headers across vintages. The resolver below accepts the known
aliases and reports what it matched, rather than assuming one schema:

    make            Mk, Make
    commercial name Cn, CN, Commercial name
    mass            m (kg), Mt, M (kg), Mass (kg)
    fuel            Ft, Fuel type
    year            year, Year, Reporting year
    country         MS, Ms, Member state
    electric range  z (Wh/km), Ewltp (g/km) ... (optional, ignored if absent)

USAGE
-----
    # one file or a whole directory of yearly files
    python eea_aggregate.py --input ~/Downloads/eea/ --out data/eea_model_mass.csv

    # restrict to the makes we actually need (much smaller output)
    python eea_aggregate.py --input ~/Downloads/eea/ --out data/eea_model_mass.csv \\
        --makes "VOLKSWAGEN,AUDI,PORSCHE,BMW,MERCEDES,SEAT,SKODA,RENAULT,DACIA,\\
PEUGEOT,CITROEN,OPEL,FIAT,ALFA ROMEO,LANCIA,JEEP,DS,FORD,TOYOTA,LEXUS"

Then Module 03b joins the result onto tbl_model_year on (brand, model).

Author: Magd Shamashan
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ aliases --
ALIASES: Dict[str, List[str]] = {
    # --- essential ------------------------------------------------------------
    "make":  ["Mk", "MK", "Make", "make", "MAKE"],
    "cname": ["Cn", "CN", "Commercial name", "commercial_name", "CommercialName"],
    "mass":  ["m (kg)", "M (kg)", "mass", "Mass (kg)", "m_kg", "M_kg"],
    "year":  ["year", "Year", "Reporting year", "reporting_year"],
    # `r` = TOTAL NEW REGISTRATIONS for that version. Rows are vehicle VERSIONS,
    # not individual cars, so every statistic must be weighted by it. Without
    # this a one-off special counts as much as a 200,000-unit Golf.
    "regs":  ["r", "R", "Total new registrations", "registrations"],
    # --- strongly recommended -------------------------------------------------
    "fuel":  ["Ft", "FT", "Fuel type", "fuel_type", "FuelType"],
    "fmode": ["Fm", "FM", "Fuel mode"],
    "cat":   ["Ct", "CT", "Category of vehicle type approved"],
    "country": ["Country", "MS", "Ms", "Member state"],
    # --- useful ---------------------------------------------------------------
    "co2":   ["Ewltp (g/km)", "Enedc (g/km)", "e (g/km)", "co2"],
    "testmass": ["Mt", "MT", "WLTP test mass"],
    "engine_cm3": ["ec (cm3)", "ec (cm³)", "Engine capacity"],
    "power_kw": ["ep (KW)", "ep (kW)", "Engine power"],
    "elec_range": ["Electric range (km)", "Electric range", "z (Wh/km)"],
    "pool":  ["Mp", "MP", "Manufacturer pooling"],
    "status": ["Status", "status"],
}


def resolve_columns(header: Iterable[str]) -> Dict[str, str]:
    """Map our canonical names onto whatever this vintage calls them."""
    cols = {str(c).strip(): str(c) for c in header}
    lower = {k.lower(): v for k, v in cols.items()}
    found: Dict[str, str] = {}
    for canon, opts in ALIASES.items():
        for o in opts:
            if o in cols:
                found[canon] = cols[o]; break
            if o.lower() in lower:
                found[canon] = lower[o.lower()]; break
    return found


def _clean_text(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.strip().str.upper()
             .str.replace(r"\s+", " ", regex=True)
             .replace({"NAN": np.nan, "": np.nan}))


# Grouping keys for the aggregate. `by_fuel()` switches the powertrain class in
# and out, so the same streaming code produces either the compact model table or
# the model x powertrain table without a second pass over 2.4 GB.
GROUP_KEYS: List[str] = ["make", "cname", "year"]


def by_fuel(on: bool = True) -> None:
    """Include (or drop) `fuel_class` in the aggregation key."""
    global GROUP_KEYS
    GROUP_KEYS = (["make", "cname", "year", "fuel_class"] if on
                  else ["make", "cname", "year"])


def aggregate_file(path: str, chunksize: int = 500_000,
                   makes: Optional[List[str]] = None,
                   category: Optional[List[str]] = ("M1", "M1G"),
                   verbose: bool = True) -> pd.DataFrame:
    """
    Stream one EEA file and return a (make, cname, year) aggregate.

    Memory use is bounded by `chunksize`, not by file size. Only the columns we
    need are parsed, which is most of the speed-up.
    """
    head = pd.read_csv(path, nrows=5, sep=None, engine="python",
                       encoding_errors="replace")
    cmap = resolve_columns(head.columns)
    missing = [k for k in ("make", "cname", "mass") if k not in cmap]
    if missing:
        raise ValueError(
            f"{Path(path).name}: could not find column(s) {missing}. "
            f"Header was: {list(head.columns)[:20]}")
    if verbose:
        print(f"  columns matched: { {k: v for k, v in cmap.items()} }")

    usecols = [v for v in cmap.values()]
    parts: List[pd.DataFrame] = []
    n_rows = 0
    t0 = time.time()

    reader = pd.read_csv(path, usecols=usecols, chunksize=chunksize,
                         low_memory=True, encoding_errors="replace",
                         on_bad_lines="skip")
    for i, chunk in enumerate(reader, 1):
        chunk = chunk.rename(columns={v: k for k, v in cmap.items()})
        n_rows += len(chunk)

        chunk["make"] = _clean_text(chunk["make"])
        chunk["cname"] = _clean_text(chunk["cname"])
        if makes:
            chunk = chunk[chunk["make"].isin(makes)]
            if chunk.empty:
                continue
        chunk["mass"] = pd.to_numeric(chunk["mass"], errors="coerce")
        # plausible passenger-car / LCV kerb range; drops sentinels and typos
        chunk = chunk[chunk["mass"].between(500, 4_000)]
        # registrations = the weight. Absent (older vintages) -> one row, one car.
        if "regs" in chunk.columns:
            chunk["regs"] = pd.to_numeric(chunk["regs"], errors="coerce").fillna(0)
            chunk = chunk[chunk["regs"] > 0]
        else:
            chunk["regs"] = 1
        if category and "cat" in chunk.columns:
            chunk = chunk[_clean_text(chunk["cat"]).isin(category)]
        if chunk.empty:
            continue
        for c in ("co2", "power_kw", "engine_cm3"):
            if c in chunk.columns:
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
        if "year" not in chunk.columns:
            chunk["year"] = _year_from_name(path)
        chunk["year"] = pd.to_numeric(chunk["year"], errors="coerce").astype("Int64")
        if "fuel" in chunk.columns:
            chunk["fuel"] = _clean_text(chunk["fuel"])
            chunk["is_bev"] = chunk["fuel"].str.contains(
                "ELECTRIC", na=False) & ~chunk["fuel"].str.contains(
                "PETROL|DIESEL|HYBRID|PLUG", na=False)
        else:
            chunk["is_bev"] = np.nan

        # weighted sums: every quantity is registration-weighted
        # Powertrain class. The thesis needs to separate "cars got bigger"
        # from "batteries are heavy": without this the two are confounded in a
        # single mean, and the 2020 inflection could be read either way.
        if "fuel" in chunk.columns:
            f = chunk["fuel"].fillna("")
            chunk["fuel_class"] = np.select(
                [chunk["is_bev"] == True,
                 f.str.contains("PLUG", na=False),
                 f.str.contains("ELECTRIC", na=False) & f.str.contains("PETROL|DIESEL", na=False)],
                ["BEV", "PHEV", "HEV"], default="ICE")
        else:
            chunk["fuel_class"] = "UNKNOWN"

        chunk["mass_w"] = chunk["mass"] * chunk["regs"]
        chunk["bev_regs"] = chunk["regs"].where(chunk["is_bev"] == True, 0)
        # CO2 NEEDS ITS OWN DENOMINATOR.
        # Ewltp is empty before 2018 (WLTP replaced NEDC in 2017-2019). Dividing
        # a NaN-summed numerator by TOTAL registrations silently produced
        # co2_mean = 0.0 g/km for 2010-2017 -- a pandas groupby sum skips NaN
        # and returns 0, so "no measurement" became "zero emissions", which are
        # opposites. Carrying the registrations of rows that actually have a
        # reading keeps the mean NaN when nothing was measured, and correct when
        # only part of the year was.
        if "co2" in chunk.columns:
            has_co2 = chunk["co2"].notna()
            chunk["co2_w"] = (chunk["co2"] * chunk["regs"]).where(has_co2)
            chunk["co2_regs"] = chunk["regs"].where(has_co2, 0.0)
        else:
            chunk["co2_w"] = np.nan
            chunk["co2_regs"] = 0.0
        g = (chunk.groupby(GROUP_KEYS, dropna=True)
                  .agg(n_versions=("mass", "size"),
                       regs=("regs", "sum"),
                       mass_w=("mass_w", "sum"),
                       co2_w=("co2_w", "sum"),
                       co2_regs=("co2_regs", "sum"),
                       mass_min=("mass", "min"),
                       mass_max=("mass", "max"),
                       bev_regs=("bev_regs", "sum"))
                  .reset_index())
        parts.append(g)

        if verbose and i % 20 == 0:
            print(f"    {n_rows:,} rows  ({time.time()-t0:.0f}s)", flush=True)

        # keep the running aggregate small
        if len(parts) >= 40:
            parts = [_combine(parts)]

    if not parts:
        return pd.DataFrame()
    out = _combine(parts)
    if verbose:
        print(f"  -> {n_rows:,} rows read, {len(out):,} (make, model, year) groups "
              f"in {time.time()-t0:.0f}s")
    return out


def _combine(parts: List[pd.DataFrame]) -> pd.DataFrame:
    return (pd.concat(parts, ignore_index=True)
              .groupby(GROUP_KEYS, as_index=False)
              .agg(n_versions=("n_versions", "sum"), regs=("regs", "sum"),
                   mass_w=("mass_w", "sum"), co2_w=("co2_w", "sum"),
                   co2_regs=("co2_regs", "sum"),
                   mass_min=("mass_min", "min"), mass_max=("mass_max", "max"),
                   bev_regs=("bev_regs", "sum")))


def _year_from_name(path: str) -> Optional[int]:
    import re
    m = re.search(r"(20[0-2]\d)", Path(path).name)
    return int(m.group(1)) if m else None


def build_lookup(input_path: str, out_path: str,
                 chunksize: int = 500_000,
                 makes: Optional[List[str]] = None,
                 category: Optional[List[str]] = ("M1",)) -> pd.DataFrame:
    """Aggregate one file or a directory of files into the compact lookup."""
    p = Path(input_path)
    files = sorted(glob.glob(str(p / "*.csv"))) if p.is_dir() else [str(p)]
    if not files:
        raise FileNotFoundError(f"no .csv found at {input_path}")
    print(f"{len(files)} file(s) to process")

    parts = []
    for f in files:
        size_gb = os.path.getsize(f) / 1e9
        print(f"\n{Path(f).name}  ({size_gb:.2f} GB)")
        try:
            parts.append(aggregate_file(f, chunksize=chunksize, makes=makes,
                                        category=category))
        except Exception as exc:
            print(f"  SKIPPED: {exc}")

    parts = [p_ for p_ in parts if len(p_)]
    if not parts:
        raise RuntimeError("nothing aggregated — check the column mapping above")
    agg = _combine(parts)

    # registration-weighted means -- the quantity the thesis needs
    agg["mass_mean_kg"] = agg["mass_w"] / agg["regs"]
    # NaN, not 0, where nothing was measured
    agg["co2_mean_g_km"] = np.where(agg["co2_regs"] > 0,
                                    agg["co2_w"] / agg["co2_regs"].replace(0, np.nan),
                                    np.nan)
    agg["co2_coverage"] = agg["co2_regs"] / agg["regs"]
    agg["bev_share"] = agg["bev_regs"] / agg["regs"]
    agg = agg.rename(columns={"make": "eea_make", "cname": "eea_model",
                              "regs": "registrations"})
    agg = agg[["eea_make", "eea_model", "year", "registrations", "n_versions",
               "mass_mean_kg", "mass_min", "mass_max", "co2_mean_g_km",
               "co2_coverage", "bev_share"]]
    agg = agg.sort_values(["eea_make", "eea_model", "year"])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}  ({len(agg):,} rows, "
          f"{os.path.getsize(out_path)/1e6:.1f} MB)")
    print(f"  makes: {agg['eea_make'].nunique():,} | "
          f"models: {agg['eea_model'].nunique():,} | "
          f"years: {int(agg['year'].min())}-{int(agg['year'].max())} | "
          f"registrations covered: {agg['registrations'].sum():,}")
    return agg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="EEA .csv file or directory of them")
    ap.add_argument("--out", default="data/eea_model_mass.csv")
    ap.add_argument("--chunksize", type=int, default=500_000,
                    help="rows per chunk; lower it if memory is tight")
    ap.add_argument("--makes", default="",
                    help="comma-separated makes to keep (much smaller output)")
    ap.add_argument("--category", default="M1,M1G",
                    help="vehicle categories to keep. M1=passenger cars, "
                         "M1G=off-road passenger cars (G-Class etc. — keep them), "
                         "N1=light commercial vans. '' keeps all")
    a = ap.parse_args(argv)
    makes = [m.strip().upper() for m in a.makes.split(",") if m.strip()] or None
    cats = [c.strip().upper() for c in a.category.split(",") if c.strip()] or None
    build_lookup(a.input, a.out, chunksize=a.chunksize, makes=makes, category=cats)


if __name__ == "__main__":
    main()
