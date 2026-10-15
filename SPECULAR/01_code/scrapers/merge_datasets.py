#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge the CURATED dataset (hand-verified, models_output_curated/) with the
SCRAPED dataset (automated scraper, models_output/) into one master dataset
with a `method` column (curated | scraped | classification).

- Every (plant, model) row is tagged with how it was produced.
- Where a plant exists in BOTH, curated is authoritative; the scraped rows are
  kept in the long file (method=scraped, superseded=yes) so you can compare.
- All 43 plants appear, including the component/powertrain/historic sites, which
  get a single explicit "no passenger-car models" row (method=classification).

Outputs (models_output_master/):
  master_models_long.csv            union, with method + superseded + confidence
  master_models_binary_matrix.csv   authoritative 0/1 matrix, all 43 plants, + method col
  master_employment.csv             curated + scraped employment, method-tagged

Run:  python3 merge_datasets.py
Set SCRAPED_LONG / SCRAPED_EMP env vars to point at your scraper output if it
isn't at the default models_output/ location.
"""
import os
import csv
from pathlib import Path

BASE = Path(__file__).resolve().parent
CUR = BASE / "models_output_curated"
OUT = BASE / "models_output_master"
START, END = 2006, 2026
YEARS = list(range(START, END + 1))

# Scraper outputs (override via env). Default to the user's scraper folder.
SCRAPED_LONG = Path(os.environ.get("SCRAPED_LONG", str(BASE.parent / "models_output" / "models_long_evidence.csv")))
SCRAPED_EMP = Path(os.environ.get("SCRAPED_EMP", str(BASE.parent / "models_output" / "employment_from_wikipedia.csv")))


def read_csv(p):
    if not Path(p).exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cur_long = read_csv(CUR / "models_long_evidence_curated.csv")
    cls = read_csv(CUR / "plant_classification.csv")
    scr_long = read_csv(SCRAPED_LONG)
    cur_emp = read_csv(CUR / "employment_from_wikipedia_curated.csv")
    scr_emp = read_csv(SCRAPED_EMP)

    curated_plants = {r["plant_id"] for r in cur_long}
    noncar = {r["plant_id"]: r for r in cls if r["classification"] in ("component", "historic")}

    # ---- master long (union) ----
    long_rows = []
    for r in cur_long:
        long_rows.append({**{k: r.get(k, "") for k in
            ["plant_id", "plant_name", "brand", "country", "model", "year_start", "year_end", "confidence", "source", "evidence"]},
            "method": "curated", "superseded": "no"})
    for r in scr_long:
        pid = r.get("plant_id", "")
        long_rows.append({"plant_id": pid, "plant_name": r.get("plant_name", ""),
            "brand": r.get("brand", ""), "country": r.get("country", ""),
            "model": r.get("model", ""), "year_start": r.get("year_start", ""),
            "year_end": r.get("year_end", ""), "confidence": "",
            "source": "automated scraper (Wikipedia+Gemini)",
            "evidence": r.get("evidence", ""),
            "method": "scraped", "superseded": "yes" if (pid in curated_plants or pid in noncar) else "no"})
    # non-car plants as explicit informational rows
    for pid, c in noncar.items():
        long_rows.append({"plant_id": pid, "plant_name": "", "brand": "", "country": "",
            "model": f"— no passenger-car models ({c['classification']})", "year_start": "",
            "year_end": "", "confidence": "High", "source": "classification", "evidence": c["note"],
            "method": "classification", "superseded": "no"})
    _w(OUT / "master_models_long.csv",
       ["plant_id", "plant_name", "brand", "country", "model", "year_start", "year_end",
        "confidence", "method", "superseded", "source", "evidence"], long_rows)

    # ---- master binary matrix (authoritative: curated; scraped only for plants curated lacks) ----
    bin_rows = []
    scraped_extra_plants = {r["plant_id"] for r in scr_long} - curated_plants - set(noncar)
    # curated assembly rows
    for r in cur_long:
        bin_rows.append(_binrow(r["plant_id"], r["plant_name"], r["brand"], r["country"],
                                r["model"], to_int(r["year_start"]), to_int(r["year_end"]) or END, "curated"))
    # scraped-only plants (coverage curated doesn't have)
    for r in scr_long:
        if r["plant_id"] in scraped_extra_plants:
            bin_rows.append(_binrow(r["plant_id"], r.get("plant_name", ""), r.get("brand", ""),
                r.get("country", ""), r["model"], to_int(r.get("year_start")),
                to_int(r.get("year_end")) or END, "scraped"))
    # non-car plants: one zero row each
    for pid, c in noncar.items():
        row = {"plant_id": pid, "plant_name": "", "brand": "", "country": "", "method": "classification",
               "car model ": f"— no passenger-car models ({c['note']})"}
        for y in YEARS:
            row[y] = 0
        bin_rows.append(row)
    cols = ["plant_id", "plant_name", "brand", "country", "method", "car model "] + YEARS
    _w(OUT / "master_models_binary_matrix.csv", cols, bin_rows)

    # ---- master employment ----
    emp_rows = []
    for r in cur_emp:
        emp_rows.append({**r, "method": "curated"})
    for r in scr_emp:
        emp_rows.append({"plant_id": r.get("plant_id", ""), "plant_name": r.get("plant_name", ""),
            "brand": r.get("brand", ""), "country": r.get("country", ""),
            "employees": r.get("employees", ""), "reference_year": r.get("reference_year", ""),
            "pattern": r.get("pattern", ""), "snippet": r.get("snippet", ""), "method": "scraped"})
    _w(OUT / "master_employment.csv",
       ["plant_id", "plant_name", "brand", "country", "employees", "reference_year", "pattern", "snippet", "method"], emp_rows)

    print(f"master_models_long: {len(long_rows)} rows  "
          f"(curated={sum(1 for r in long_rows if r['method']=='curated')}, "
          f"scraped={sum(1 for r in long_rows if r['method']=='scraped')}, "
          f"classification={sum(1 for r in long_rows if r['method']=='classification')})")
    print(f"master_models_binary_matrix: {len(bin_rows)} rows  "
          f"(non-car plants added: {len(noncar)})")
    print(f"master_employment: {len(emp_rows)} rows")
    print(f"scraped file present: {SCRAPED_LONG.exists()}  ({SCRAPED_LONG})")
    print(f"Written to {OUT}")


def _binrow(pid, name, brand, country, model, ys, ye, method):
    row = {"plant_id": pid, "plant_name": name, "brand": brand, "country": country,
           "method": method, "car model ": model}
    ys = ys or START
    for y in YEARS:
        row[y] = 1 if (ys <= y <= ye) else 0
    return row


def _w(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)


if __name__ == "__main__":
    main()
