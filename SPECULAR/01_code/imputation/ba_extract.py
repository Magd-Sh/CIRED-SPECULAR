# -*- coding: utf-8 -*-
"""
Extract car-industry employment (WZ 2008 division 29 and groups 291/292/293) from the BA
quarterly tables "Beschäftigte nach Wirtschaftszweigen (WZ 2008) - Länder"
(files wz-heft-<state>-0-<YYYYMM>-xls(x).xlsx;jsessionid=...).

Level: federal STATE x quarter (no district breakdown in these tables).
Measure: employees subject to social insurance at place of work (sheet "SVB - Tabelle I"
or the first SVB table in older files), column "Insgesamt".
Suppressed cells ("*", confidentiality / dominance cases) -> NaN.

Usage: python ba_extract.py <BA_DIR> <OUT_CSV>
"""
import os, re, sys, glob
import numpy as np
import pandas as pd

BA_DIR, OUT = sys.argv[1], sys.argv[2]
STATE = {"01": "Schleswig-Holstein", "02": "Hamburg", "03": "Niedersachsen", "04": "Bremen",
         "05": "Nordrhein-Westfalen", "06": "Hessen", "07": "Rheinland-Pfalz",
         "08": "Baden-Wuerttemberg", "09": "Bayern", "10": "Saarland", "11": "Berlin",
         "12": "Brandenburg", "13": "Mecklenburg-Vorpommern", "14": "Sachsen",
         "15": "Sachsen-Anhalt", "16": "Thueringen"}
CODES = ["29", "291", "292", "293"]


def pick_sheet(xl):
    """First sheet holding SVB employees by WZ (name varies across years)."""
    for s in xl.sheet_names:
        if re.search(r"SVB.*(Tabelle ?I\b|Tabelle ?1\b)", s):
            return s
    for s in xl.sheet_names:                    # older layouts: scan for the WZ-29 row
        df = xl.parse(s, header=None, nrows=400)
        if df[0].astype(str).str.match(r"\s*29 ").any():
            return s
    return None


rows = []
for path in sorted(glob.glob(os.path.join(BA_DIR, "wz-heft-*"))):
    base = os.path.basename(path).split(";")[0]
    m = re.match(r"wz-heft-(\d{2})-0-(\d{4})(\d{2})", base)
    if not m:
        continue
    st, yr, mo = m.group(1), int(m.group(2)), int(m.group(3))
    engine = "xlrd" if base.endswith(".xls") and not base.endswith(".xlsx") else "openpyxl"
    try:
        xl = pd.ExcelFile(path, engine=engine)
    except Exception:                            # mislabelled extension -> try the other engine
        xl = pd.ExcelFile(path, engine="openpyxl" if engine == "xlrd" else "xlrd")
    sh = pick_sheet(xl)
    df = xl.parse(sh, header=None)
    lab = df[0].astype(str).str.strip()
    for code in CODES:
        hit = df[lab.str.match(rf"^{code}\s")]
        val = hit.iloc[0, 1] if len(hit) else np.nan
        num = pd.to_numeric(val, errors="coerce")
        rows.append({"state_code": st, "state": STATE.get(st, st), "year": yr, "month": mo,
                     "wz": code, "employees": num,
                     "suppressed": isinstance(val, str) and val.strip() == "*",
                     "sheet": sh, "file": base})

out = pd.DataFrame(rows).sort_values(["state_code", "wz", "year", "month"])
out.to_csv(OUT, index=False)
print(f"[OK] {OUT}: {len(out)} rows, {out.file.nunique()} files, "
      f"{out.employees.isna().sum()} missing ({out.suppressed.sum()} suppressed '*')")
