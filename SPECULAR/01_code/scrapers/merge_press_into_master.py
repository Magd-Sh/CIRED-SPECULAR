#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge verified press / established-plant gap-fills into your master employment
dataset. Works with the grid file (employment_merged_by_archive_year_flagged.xlsx)
AND the older long CSV.

- Grid master (one row per plant-year): BLANK rows are FILLED IN PLACE.
- Long master: new (plant, year) rows are APPENDED.
- Either way an existing employee value is NEVER overwritten.
- Press-sourced fills are marked low_confidence = "Y".

USAGE
  MASTER_CSV="$SPECULAR_ROOT/03_data_raw/scraper_outputs/employment_merged_by_archive_year_flagged.xlsx" \
  PRESS_CSV="$SPECULAR_ROOT/03_data_raw/scraper_outputs/press_gap_fills_clean.csv" \
  python3 merge_press_into_master.py
Writes <master>_with_press.<ext> next to the master.
Set DROP_SUSPECT=1 to exclude rows flagged suspect=cross_plant_repeat.
"""
import os, sys
from pathlib import Path
import pandas as pd

MASTER_CSV = os.environ.get("MASTER_CSV", "")
PRESS_CSV = os.environ.get("PRESS_CSV",
    str(Path(__file__).resolve().parent / "press_gap_output" / "press_gap_fills_clean.csv"))
DROP_SUSPECT = os.environ.get("DROP_SUSPECT", "0") != "0"

COPY_COLS = ["employees", "value_source", "selection_basis", "extraction_method", "pattern",
             "source_url", "snippet", "source_tier", "reference_year"]


def read_table(path):
    p = str(path)
    if p.lower().endswith((".xlsx", ".xlsm", ".xls")):
        xl = pd.ExcelFile(path)
        sheet = "By archive year" if "By archive year" in xl.sheet_names else xl.sheet_names[0]
        return pd.read_excel(xl, sheet_name=sheet)
    return pd.read_csv(path)


def write_table(df, path):
    if str(path).lower().endswith((".xlsx", ".xlsm")):
        df.to_excel(path, index=False, sheet_name="By archive year")
    else:
        df.to_csv(path, index=False)


if not MASTER_CSV or not os.path.exists(MASTER_CSV):
    sys.exit("[!] set MASTER_CSV to your master dataset (.xlsx or .csv)")
if not os.path.exists(PRESS_CSV):
    sys.exit(f"[!] press file not found: {PRESS_CSV}")

master = read_table(MASTER_CSV)
press = read_table(PRESS_CSV)
ycol = "year" if "year" in master.columns else "reference_year"
if DROP_SUSPECT and "suspect" in press.columns:
    press = press[press["suspect"].fillna("") == ""]

# index existing (plant_id, year) -> first row position
idx = {}
for i in master.index:
    y = pd.to_numeric(master.at[i, ycol], errors="coerce")
    if pd.notna(y):
        idx.setdefault((str(master.at[i, "plant_id"]), int(y)), i)

empnum = pd.to_numeric(master["employees"], errors="coerce")
updated, appended, skipped = 0, 0, 0
new_rows = []
for _, pr in press.iterrows():
    try:
        y = int(pr["year"])
    except (TypeError, ValueError):
        continue
    key = (str(pr["plant_id"]), y)
    if key in idx:
        i = idx[key]
        if pd.isna(empnum.at[i]):                       # blank cell -> fill in place
            for c in COPY_COLS:
                if c in pr and c in master.columns and pd.notna(pr[c]):
                    master.at[i, c] = pr[c]
            if "low_confidence" in master.columns:
                master.at[i, "low_confidence"] = "Y"
            updated += 1
        else:
            skipped += 1                                # already has a value -> never overwrite
    else:                                               # no such cell -> append
        row = {c: pr[c] for c in master.columns if c in pr}
        row["plant_id"] = pr["plant_id"]; row[ycol] = y
        if "low_confidence" in master.columns:
            row["low_confidence"] = "Y"
        new_rows.append(row)
        appended += 1

if new_rows:
    master = pd.concat([master, pd.DataFrame(new_rows)], ignore_index=True, sort=False)

ext = Path(MASTER_CSV).suffix
out_path = MASTER_CSV[: -len(ext)] + "_with_press" + ext
write_table(master, out_path)
print(f"master rows: {len(master) - len(new_rows)} -> {len(master)}")
print(f"filled-in-place: {updated} | appended: {appended} | skipped (already had value): {skipped}")
print(f"written: {out_path}")
