"""
Build the EEA inputs of the analysis from the full European registration files.

    python eea_build_eu27.py aggregate 2010 2011 ...   # one pass per yearly file
    python eea_build_eu27.py hist      2010 2011 ...   # mass histograms, same files
    python eea_build_eu27.py combine                   # write the four inputs

Input: one CSV per year in RAW_DIR (EEA "Monitoring of CO2 emissions from
passenger cars - Regulation (EU) 2019/631"; final data 2010-2024, provisional
2025), every country and every make. The yearly files are several GB, so each
is streamed in chunks and reduced to small intermediate files in WORK_DIR; a
year that has been aggregated is not read again.

Cleaning is the same as `eea_aggregate.py`: passenger cars only (M1, M1G),
mass 500-4,000 kg, every quantity weighted by registrations, powertrain class
ICE / HEV / BEV from the fuel type. Two additions:

* COUNTRY. The aggregates keep the member state, so the analysis can be run on
  the EU27 (the default) or on any subset. The UK (to 2020), Iceland (from
  2018) and Norway (from 2019) are in the files but outside the EU27; keeping
  them would make the fleet change because countries enter and leave it.
* EARLY HYBRIDS. In 2010-2012 the EEA files list full hybrids (Prius, Auris
  Hybrid, Lexus RX450h and CT200h, Cayenne S Hybrid, Jazz Hybrid ...) with fuel
  type "Electric". Left as they are, they enter the battery-electric class and
  set its first-year mass. From 2013 most full hybrids are filed under their
  fuel (petrol) and so count as combustion cars, while the hybrid class
  (petrol/electric, diesel/electric) is mostly plug-in hybrids. The early full
  hybrids are therefore moved to the combustion class, so the classes mean the
  same thing in every year (changed 7 Oct 2026; before, they went to hybrid).

Outputs, in OUT_DIR (02_data_inputs/eea/):
  eea_model_mass.csv          make x model x year, EU27      (as before)
  eea_model_mass_by_fuel.csv  make x model x year x class    (as before)
  eea_mass_hist.csv           year x class x 10-kg bin, registrations
  eea_country_class.csv       country x seven-groups flag x year x class
"""
import glob, os, re, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eea_aggregate import resolve_columns, _clean_text

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RAW_DIR = os.environ.get("EEA_FULL_DIR", os.path.join(ROOT, "03_data_raw", "eea_registrations"))
WORK_DIR = os.environ.get("EEA_WORK_DIR", os.path.join(ROOT, "03_data_raw", "eea_full_work"))
OUT_DIR = os.path.join(ROOT, "02_data_inputs", "eea")

EU27 = ("AT BE BG CY CZ DE DK EE ES FI FR GR HR HU IE IT LT LU LV MT NL PL PT "
        "RO SE SI SK").split()
KEYS = ["country", "make", "cname", "year", "fuel_class"]
HYB = (r"HYBRID|PRIUS|RX ?450|CT ?200|GS ?450|LS ?600|IS ?300|AURIS|YARIS|JAZZ|"
       r"INSIGHT|CR-?Z|CIVIC|CAYENNE|PANAMERA|ACTIVE ?HYBRID|S ?400|HSD")
BINS = np.arange(500, 4010, 10)
# brands of the seven groups in the plant panel
GROUP7 = ("VOLKSWAGEN", "VW", "AUDI", "SKODA", "KODA", "SEAT", "CUPRA", "PORSCHE",
          "BENTLEY", "LAMBORGHINI", "BUGATTI", "BMW", "MINI", "ROLLSROYCE",
          "MERCEDES", "SMART", "MAYBACH", "DAIMLER", "RENAULT", "DACIA", "ALPINE",
          "PEUGEOT", "AUTOMOBILESPEUGEOT", "CITRO", "AUTOMOBILESCITRO", "DS",
          "OPEL", "VAUXHALL", "FIAT", "ABARTH", "ALFA", "LANCIA", "JEEP",
          "CHRYSLER", "DODGE", "MASERATI", "FORD", "TOYOTA", "LEXUS")


def _file(y):
    """The file for year y. On INARI each year has the full EEA file
    (constructeurs_final_YYYY.csv) and a light extract of it with the nine columns
    used here (light_constructeurs_final_YYYY.csv); the light one is read. A year
    with a single file (2025: constructeurs_provisional_2025.csv) uses that file."""
    f = sorted(glob.glob(os.path.join(RAW_DIR, f"*_{y}.csv")))
    light = [x for x in f if os.path.basename(x).startswith("light_")]
    if len(light) == 1:
        return light[0]
    if len(f) == 1:
        return f[0]
    raise FileNotFoundError(f"expected one light file, or a single file, for {y} in {RAW_DIR}; found {f}")


def _classes(fuel, cname, make, year):
    fu = _clean_text(fuel).fillna("")
    bev = fu.str.contains("ELECTRIC") & ~fu.str.contains("PETROL|DIESEL|HYBRID|PLUG")
    hev = (fu.str.contains("ELECTRIC") & fu.str.contains("PETROL|DIESEL")) | fu.str.contains("PLUG")
    if year <= 2012:
        early = bev & (_clean_text(cname).fillna("").str.contains(HYB, regex=True)
                       | _clean_text(make).fillna("").str.contains("LEXUS"))
        bev = bev & ~early          # full hybrids: combustion, as from 2013
    return np.where(bev, "BEV", np.where(hev, "HEV", "ICE")), bev


def _chunks(y):
    f = _file(y)
    cmap = resolve_columns(pd.read_csv(f, nrows=5).columns)
    cols = [cmap[k] for k in ("country", "make", "cname", "cat", "regs", "mass", "fuel")]
    if "co2" in cmap:
        cols.append(cmap["co2"])
    for ch in pd.read_csv(f, usecols=cols, chunksize=1_000_000, low_memory=False,
                          encoding_errors="replace", on_bad_lines="skip"):
        ch = ch.rename(columns={v: k for k, v in cmap.items()})
        ch["country"] = _clean_text(ch["country"]).replace({"UK": "GB"})
        ch = ch[_clean_text(ch["cat"]).isin(["M1", "M1G"])]
        ch["mass"] = pd.to_numeric(ch["mass"], errors="coerce")
        ch = ch[ch["mass"].between(500, 4000)]
        ch["regs"] = pd.to_numeric(ch["regs"], errors="coerce").fillna(0)
        ch = ch[ch["regs"] > 0]
        if ch.empty:
            continue
        ch["make"] = _clean_text(ch["make"]); ch["cname"] = _clean_text(ch["cname"])
        ch["fuel_class"], bev = _classes(ch["fuel"], ch["cname"], ch["make"], y)
        ch["bev_regs"] = ch["regs"].where(bev, 0)
        yield ch


def aggregate(years):
    os.makedirs(WORK_DIR, exist_ok=True)
    agg = dict(n_versions=("mass", "size"), regs=("regs", "sum"), mass_w=("mass_w", "sum"),
               co2_w=("co2_w", "sum"), co2_regs=("co2_regs", "sum"),
               mass_min=("mass", "min"), mass_max=("mass", "max"), bev_regs=("bev_regs", "sum"))
    re_agg = dict(n_versions=("n_versions", "sum"), regs=("regs", "sum"), mass_w=("mass_w", "sum"),
                  co2_w=("co2_w", "sum"), co2_regs=("co2_regs", "sum"),
                  mass_min=("mass_min", "min"), mass_max=("mass_max", "max"),
                  bev_regs=("bev_regs", "sum"))
    for y in years:
        t0, parts = time.time(), []
        for ch in _chunks(y):
            ch["year"] = y
            ch["co2"] = pd.to_numeric(ch.get("co2"), errors="coerce")
            ch["mass_w"] = ch["mass"] * ch["regs"]
            has = ch["co2"].notna()
            ch["co2_w"] = (ch["co2"] * ch["regs"]).where(has, 0.0)
            ch["co2_regs"] = ch["regs"].where(has, 0.0)
            parts.append(ch.groupby(KEYS, dropna=False).agg(**agg).reset_index())
            if len(parts) >= 20:
                parts = [pd.concat(parts).groupby(KEYS, dropna=False, as_index=False).agg(**re_agg)]
        out = pd.concat(parts).groupby(KEYS, dropna=False, as_index=False).agg(**re_agg)
        out.to_csv(os.path.join(WORK_DIR, f"agg_{y}.csv.gz"), index=False)
        print(y, f"{out['regs'].sum()/1e6:.2f}m registrations", f"{time.time()-t0:.0f}s", flush=True)


def hist(years):
    os.makedirs(WORK_DIR, exist_ok=True)
    for y in years:
        H = {c: np.zeros(len(BINS) - 1) for c in ("ICE", "HEV", "BEV")}
        for ch in _chunks(y):
            ch = ch[ch["country"].isin(EU27)]
            for c in H:
                s = ch["fuel_class"] == c
                H[c] += np.histogram(ch["mass"][s], BINS, weights=ch["regs"][s])[0]
        rows = [dict(year=y, fuel_class=c, bin_lo=int(b), registrations=float(v))
                for c, h in H.items() for b, v in zip(BINS[:-1], h) if v > 0]
        pd.DataFrame(rows).to_csv(os.path.join(WORK_DIR, f"hist_{y}.csv.gz"), index=False)
        print(y, "histogram written", flush=True)


def _lookup(d, keys):
    g = d.groupby(keys, as_index=False).agg(
        registrations=("regs", "sum"), n_versions=("n_versions", "sum"),
        mass_w=("mass_w", "sum"), co2_w=("co2_w", "sum"), co2_regs=("co2_regs", "sum"),
        mass_min=("mass_min", "min"), mass_max=("mass_max", "max"), bev_regs=("bev_regs", "sum"))
    g["mass_mean_kg"] = g["mass_w"] / g["registrations"]
    g["co2_mean_g_km"] = np.where(g["co2_regs"] > 0,
                                  g["co2_w"] / g["co2_regs"].replace(0, np.nan), np.nan)
    g["co2_coverage"] = g["co2_regs"] / g["registrations"]
    g["bev_share"] = g["bev_regs"] / g["registrations"]
    return g.rename(columns={"make": "eea_make", "cname": "eea_model"})


def combine():
    a = pd.concat([pd.read_csv(f, low_memory=False)
                   for f in sorted(glob.glob(os.path.join(WORK_DIR, "agg_*.csv.gz")))],
                  ignore_index=True)
    eu = a[a["country"].isin(EU27)].dropna(subset=["make", "cname"])
    m = _lookup(eu, ["make", "cname", "year"])
    m[["eea_make", "eea_model", "year", "registrations", "n_versions", "mass_mean_kg",
       "mass_min", "mass_max", "co2_mean_g_km", "co2_coverage", "bev_share"]] \
        .sort_values(["eea_make", "eea_model", "year"]) \
        .to_csv(os.path.join(OUT_DIR, "eea_model_mass.csv"), index=False)
    f = _lookup(eu, ["make", "cname", "year", "fuel_class"])
    f[["eea_make", "eea_model", "year", "fuel_class", "registrations", "n_versions",
       "mass_mean_kg", "mass_min", "mass_max", "co2_mean_g_km", "co2_coverage"]] \
        .sort_values(["eea_make", "eea_model", "year", "fuel_class"]) \
        .to_csv(os.path.join(OUT_DIR, "eea_model_mass_by_fuel.csv"), index=False)
    a = a.dropna(subset=["make", "cname"])      # same rows as the two lookups
    norm = a["make"].astype(str).str.upper().str.replace(r"[^A-Z]", "", regex=True)
    a["group7"] = norm.str.startswith(GROUP7)
    (a.groupby(["country", "group7", "year", "fuel_class"], as_index=False)
       .agg(registrations=("regs", "sum"), mass_w=("mass_w", "sum"))
       .to_csv(os.path.join(OUT_DIR, "eea_country_class.csv"), index=False))
    h = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(WORK_DIR, "hist_*.csv.gz")))])
    h.to_csv(os.path.join(OUT_DIR, "eea_mass_hist.csv"), index=False)
    print("wrote the four inputs to", OUT_DIR,
          f"| EU27 registrations {m['registrations'].sum()/1e6:.1f}m")


if __name__ == "__main__":
    stage, args = sys.argv[1], [int(x) for x in sys.argv[2:]]
    {"aggregate": aggregate, "hist": hist}.get(stage, lambda _: combine())(args)
