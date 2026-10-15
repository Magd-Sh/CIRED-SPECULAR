# -*- coding: utf-8 -*-
"""
Match EEA registration data (eea_model_year.csv) to the plant models
(models_binary_matrix) at MODEL-FAMILY level, and build family x year vehicle attributes.

Both sides are mapped to a family id:
  - plant model names  -> family via ordered regex rules (FAMILY_OURS; first match wins)
  - EEA (make, cn)     -> family via canonical make + regex on the commercial name (FAMILIES)

Output:
  eea_family_year.csv : family, year, regs, mass_eea (registration-weighted mean mass in
                        running order, kg), ev_share_eea, phev_share_eea, n_cn
  eea_match_report.csv: one row per plant model -> family, matched?, years covered

Usage: python eea_match.py eea_model_year.csv MODELS_v5.xlsx
"""
import re, sys
import numpy as np
import pandas as pd

EEA_PATH = sys.argv[1] if len(sys.argv) > 1 else "eea_model_year.csv"
MOD_PATH = sys.argv[2] if len(sys.argv) > 2 else "auto_models_production_consolidated_v5.xlsx"

# ---------------------------------------------------------------- canonical make
def canon_make(m):
    m = str(m)
    for key, val in [("VOLKSWAGEN", "VW"), ("AUDI", "AUDI"), ("PORSCHE", "PORSCHE"),
                     ("MERCEDES", "MB"), ("DAIMLER", "MB"), ("CUPRA", "CUPRA"), ("SEAT", "SEAT"),
                     ("KODA", "SKODA"), ("BENTLEY", "BENTLEY"), ("LAMBORGHINI", "LAMBO"),
                     ("FORD", "FORD")]:
        if key in m:
            return val
    if m in ("VW", "V.W", "V W"):
        return "VW"
    if m.startswith("BMW") or m in ("BWM", "BWW"):
        return "BMW"
    return None

A = r"^(AUDI ?)?"          # Audi cn sometimes carries the make ("AUDIA4", "AUDI A4")
MBP = r"^(MERCEDES[- ]?BENZ ?|MERCEDES[- ]?AMG ?|AMG ?)?"
def bmw_series(n):
    return rf"^(SERIE {n}|{n} ?ER ?REIHE|{n} ?ER\b|{n} SERIES|M{n}\b|M?{n}\d\d[A-Z]*\b)"

# family id -> (set of canonical makes, regex on EEA commercial name)
FAMILIES = {
    # --- Volkswagen
    "VW_GOLF_SPORTSVAN": ({"VW"}, r"^GOLF ?SPORTSVAN"),
    "VW_EGOLF":          ({"VW"}, r"^E-? ?GOLF"),
    "VW_GOLF":           ({"VW"}, r"^GOLF(?! ?SPORTSVAN| ?PLUS)"),
    "VW_POLO":           ({"VW"}, r"^POLO"),
    "VW_PASSAT":         ({"VW"}, r"^PASSAT(?! ?CC)"),
    "VW_TIGUAN":         ({"VW"}, r"^TIGUAN"),
    "VW_TOUAREG":        ({"VW"}, r"^TOUAREG"),
    "VW_TOURAN":         ({"VW"}, r"^TOURAN"),
    "VW_SHARAN":         ({"VW"}, r"^SHARAN"),
    "VW_UP":             ({"VW"}, r"^(E-? ?)?UP\b"),
    "VW_ID3":            ({"VW"}, r"^ID\.? ?3"),
    "VW_ID4":            ({"VW"}, r"^ID\.? ?4"),
    "VW_ID5":            ({"VW"}, r"^ID\.? ?5"),
    "VW_SCIROCCO":       ({"VW"}, r"^SCIROCCO"),
    "VW_EOS":            ({"VW"}, r"^EOS"),
    "VW_PHAETON":        ({"VW"}, r"^PHAETON"),
    "VW_TAIGO":          ({"VW"}, r"^TAIGO"),
    "VW_TAYRON":         ({"VW"}, r"^TAYRON"),
    "VW_AMAROK":         ({"VW"}, r"^AMAROK"),
    "VW_CADDY":          ({"VW"}, r"^CADDY"),
    "VW_TRANSPORTER":    ({"VW"}, r"^(MULTIVAN|CARAVELLE|CALIFORNIA|TRANSPORTER|GRAND CALIFORNIA)"),
    "VW_CRAFTER":        ({"VW"}, r"^E-? ?CRAFTER|^CRAFTER"),
    # --- Audi
    "AUDI_ETRON_GT":     ({"AUDI"}, A + r"(RS ?|S ?)?E-?TRON ?GT"),
    "AUDI_Q8_ETRON":     ({"AUDI"}, A + r"(S?Q8 ?)?E-?TRON(?! ?GT)"),
    "AUDI_Q4":           ({"AUDI"}, A + r"Q4"),
    "AUDI_A1":           ({"AUDI"}, A + r"(A1|S1)\b"),
    "AUDI_A3":           ({"AUDI"}, A + r"(A3|S3|RS ?3)\b"),
    "AUDI_A4":           ({"AUDI"}, A + r"(A4|S4|RS ?4)\b"),
    "AUDI_A5":           ({"AUDI"}, A + r"(A5|S5|RS ?5)\b"),
    "AUDI_A6":           ({"AUDI"}, A + r"(A6|S6|RS ?6)\b"),
    "AUDI_A7":           ({"AUDI"}, A + r"(A7|S7|RS ?7)\b"),
    "AUDI_A8":           ({"AUDI"}, A + r"(A8|S8)\b"),
    "AUDI_Q3":           ({"AUDI"}, A + r"(RS ?)?Q3"),
    "AUDI_Q7":           ({"AUDI"}, A + r"S?Q7"),
    "AUDI_Q8":           ({"AUDI"}, A + r"(RS ?|S)?Q8(?! ?E-?TRON)"),
    "AUDI_R8":           ({"AUDI"}, A + r"R8"),
    "AUDI_TT":           ({"AUDI"}, A + r"TT"),
    # --- BMW
    "BMW_IX1":           ({"BMW"}, r"^IX1"),
    "BMW_IX":            ({"BMW"}, r"^IX(?!\d)"),
    "BMW_I3":            ({"BMW"}, r"^I3"),
    "BMW_I4":            ({"BMW"}, r"^I4"),
    "BMW_X1":            ({"BMW"}, r"^X1"),
    "BMW_X3":            ({"BMW"}, r"^X3"),
    "BMW_Z4":            ({"BMW"}, r"^Z4"),
    "BMW_1":             ({"BMW"}, bmw_series(1)),
    "BMW_3":             ({"BMW"}, bmw_series(3)),
    "BMW_4":             ({"BMW"}, bmw_series(4)),
    "BMW_5":             ({"BMW"}, bmw_series(5)),
    "BMW_6":             ({"BMW"}, bmw_series(6)),
    "BMW_7":             ({"BMW"}, bmw_series(7)),
    "BMW_8":             ({"BMW"}, bmw_series(8)),
    # --- Mercedes-Benz (order matters: CLA/CLS/CLE before CL, GLx before G)
    "MB_CLA":            ({"MB"}, MBP + r"CLA(?!SSE)"),
    "MB_CLS":            ({"MB"}, MBP + r"CLS"),
    "MB_CLE":            ({"MB"}, MBP + r"CLE"),
    "MB_CL":             ({"MB"}, MBP + r"(CL ?\d{3}|CL-?KLASSE|CL-CLASS)"),
    "MB_GLA":            ({"MB"}, MBP + r"GLA"),
    "MB_GLB":            ({"MB"}, MBP + r"GLB"),
    "MB_GLC":            ({"MB"}, MBP + r"GLC"),
    "MB_GLK":            ({"MB"}, MBP + r"GLK"),
    "MB_SLK":            ({"MB"}, MBP + r"(SLK|SLC)"),
    "MB_SL":             ({"MB"}, r"^(MERCEDES[- ]?BENZ ?|MERCEDES[- ]?AMG ?|AMG ?|MAYBACH ?)?(SL ?\d{2,3}|SL-?KLASSE|SL\b)"),
    "MB_AMG_GT":         ({"MB"}, r"^(MERCEDES[- ]?AMG ?|AMG ?)GT"),
    "MB_EQA":            ({"MB"}, r"^EQA"),
    "MB_EQC":            ({"MB"}, r"^EQC"),
    "MB_EQE":            ({"MB"}, r"^EQE"),
    "MB_EQS":            ({"MB"}, r"^EQS"),
    "MB_A":              ({"MB"}, MBP + r"(A ?\d{2,3}\b|A-?KLASSE|A KLASSE|CLASSE A|A-CLASS)"),
    "MB_B":              ({"MB"}, MBP + r"(B ?\d{3}\b|B-?KLASSE|CLASSE B|B-CLASS)"),
    "MB_C":              ({"MB"}, MBP + r"(C ?\d{2,3}\b|C-?KLASSE|CLASSE C|C-CLASS)"),
    "MB_E":              ({"MB"}, MBP + r"(E ?\d{2,3}\b|E-?KLASSE|CLASSE E|E-CLASS)"),
    "MB_S":              ({"MB"}, r"^(MERCEDES[- ]?BENZ ?|MERCEDES[- ]?AMG ?|AMG ?|MAYBACH ?)?(S ?\d{2,3}\b|S-?KLASSE|CLASSE S|S-CLASS)"),
    "MB_V":              ({"MB"}, r"^(V ?\d{3}\b|V-?KLASSE|CLASSE V|VIANO|VITO|MARCO POLO|EQV)"),
    "MB_SPRINTER":       ({"MB"}, r"^E-? ?SPRINTER|^SPRINTER"),
    "MB_CITAN":          ({"MB"}, r"^CITAN"),
    # --- Porsche
    "POR_911":           ({"PORSCHE"}, r"^911"),
    "POR_BOXSTER":       ({"PORSCHE"}, r"^(718 ?)?BOXSTER"),
    "POR_CAYMAN":        ({"PORSCHE"}, r"^(718 ?)?CAYMAN"),
    "POR_CAYENNE":       ({"PORSCHE"}, r"^CAYENNE"),
    "POR_MACAN":         ({"PORSCHE"}, r"^MACAN"),
    "POR_PANAMERA":      ({"PORSCHE"}, r"^PANAMERA"),
    "POR_918":           ({"PORSCHE"}, r"^918"),
    # --- other VW-group brands / others
    "SEAT_ALHAMBRA":     ({"SEAT"}, r"^ALHAMBRA"),
    "SEAT_MII":          ({"SEAT"}, r"^(E-? ?)?MII"),
    "SEAT_TARRACO":      ({"SEAT"}, r"^TARRACO"),
    "CUPRA_BORN":        ({"CUPRA", "SEAT"}, r"^(CUPRA ?)?BORN"),
    "CUPRA_TERRAMAR":    ({"CUPRA", "SEAT"}, r"^(CUPRA ?)?TERRAMAR"),
    "SKODA_CITIGO":      ({"SKODA"}, r"^CITIGO"),
    "SKODA_KAROQ":       ({"SKODA"}, r"^KAROQ"),
    "SKODA_OCTAVIA":     ({"SKODA"}, r"^OCTAVIA"),
    "SKODA_SUPERB":      ({"SKODA"}, r"^SUPERB"),
    "BENTLEY_BENTAYGA":  ({"BENTLEY"}, r"^BENTAYGA"),
    "LAMBO_URUS":        ({"LAMBO"}, r"^URUS"),
    "FORD_GALAXY":       ({"FORD"}, r"^GALAXY"),
}

# plant-model name -> family (ordered, first match wins; case-insensitive)
FAMILY_OURS = [
    (r"golf sportsvan", "VW_GOLF_SPORTSVAN"), (r"e-golf", "VW_EGOLF"), (r"\bgolf\b", "VW_GOLF"),
    (r"\bpolo\b", "VW_POLO"), (r"passat", "VW_PASSAT"), (r"tiguan", "VW_TIGUAN"),
    (r"touareg", "VW_TOUAREG"), (r"touran", "VW_TOURAN"), (r"sharan", "VW_SHARAN"),
    (r"volkswagen up\b", "VW_UP"), (r"id\.?3", "VW_ID3"), (r"id\.?4", "VW_ID4"), (r"id\.?5", "VW_ID5"),
    (r"scirocco", "VW_SCIROCCO"), (r"\beos\b", "VW_EOS"), (r"phaeton", "VW_PHAETON"),
    (r"taigo", "VW_TAIGO"), (r"tayron", "VW_TAYRON"), (r"amarok", "VW_AMAROK"),
    (r"caddy", "VW_CADDY"), (r"transporter|california", "VW_TRANSPORTER"), (r"crafter", "VW_CRAFTER"),
    # Audi (e-tron GT and Q8 e-tron before plain Q8)
    (r"e-tron gt", "AUDI_ETRON_GT"),
    (r"q8 e-tron|audi e-tron|sq8 e-tron|suv électrique", "AUDI_Q8_ETRON"),
    (r"\bq4\b", "AUDI_Q4"), (r"\ba1\b|\bs1\b", "AUDI_A1"), (r"\ba3\b|\bs3\b|rs ?3", "AUDI_A3"),
    (r"\ba4\b|\bs4\b|rs ?4", "AUDI_A4"), (r"\ba5\b|\bs5\b|rs ?5", "AUDI_A5"),
    (r"\ba6\b|\bs6\b|rs ?6", "AUDI_A6"), (r"\ba7\b|\bs7\b|rs ?7", "AUDI_A7"), (r"\ba8\b", "AUDI_A8"),
    (r"\bq3\b|rs q3", "AUDI_Q3"), (r"\bq7\b", "AUDI_Q7"), (r"\bq8\b", "AUDI_Q8"),
    (r"\br8\b", "AUDI_R8"), (r"\btt\b", "AUDI_TT"),
    # BMW
    (r"\bix1\b", "BMW_IX1"), (r"\bix\b|inext|ix series", "BMW_IX"), (r"\bi3", "BMW_I3"), (r"\bi4\b", "BMW_I4"),
    (r"\bx1\b", "BMW_X1"), (r"\bx3\b", "BMW_X3"), (r"\bz4\b", "BMW_Z4"),
    (r"bmw 1er|bmw 1 series", "BMW_1"), (r"bmw 3er|bmw 3 series|bmw 320i|bmw m3", "BMW_3"),
    (r"bmw 4er|bmw 4 series|bmw m4", "BMW_4"), (r"bmw 5er|bmw 5 series|bmw m5", "BMW_5"),
    (r"bmw 6er|bmw 6 series", "BMW_6"), (r"bmw 7er|bmw 7 series", "BMW_7"),
    (r"bmw 8er|bmw 8 series|bmw m8", "BMW_8"),
    # Mercedes
    (r"\bcla\b|cla-|classe cla|cla coup|cla shooting|mercedes cla", "MB_CLA"),
    (r"\bcls\b", "MB_CLS"), (r"\bcle\b", "MB_CLE"),
    (r"\bcl\b|cl-class|cl-klasse|clase cl|classe cl|c215|c216|\bcl \d", "MB_CL"),
    (r"\bgla\b|gla-klasse|classe gla", "MB_GLA"), (r"\bglb\b", "MB_GLB"),
    (r"\bglc\b", "MB_GLC"), (r"\bglk\b", "MB_GLK"), (r"\bslk\b", "MB_SLK"),
    (r"\bsl\b|sl roadster", "MB_SL"), (r"amg gt|^gt$", "MB_AMG_GT"),
    (r"\beqa\b", "MB_EQA"), (r"\beqc\b", "MB_EQC"), (r"\beqe\b", "MB_EQE"), (r"\beqs\b", "MB_EQS"),
    (r"a-class|a-klasse|clase a|classe a", "MB_A"), (r"b-class|b-klasse|clase b|classe b", "MB_B"),
    (r"c-class|c-klasse|classe c|c-coup", "MB_C"), (r"e-class|e-klasse", "MB_E"),
    (r"s-class|s-klasse", "MB_S"), (r"viano|vito", "MB_V"), (r"sprinter", "MB_SPRINTER"),
    (r"citan", "MB_CITAN"),
    # Porsche
    (r"carrera gt|\b918\b", "POR_918"), (r"\b911\b|\b997\b", "POR_911"),
    (r"boxster|\b987\b", "POR_BOXSTER"), (r"cayman", "POR_CAYMAN"), (r"cayenne", "POR_CAYENNE"),
    (r"macan", "POR_MACAN"), (r"panamera", "POR_PANAMERA"),
    # others
    (r"alhambra", "SEAT_ALHAMBRA"), (r"\bmii\b", "SEAT_MII"), (r"tarraco", "SEAT_TARRACO"),
    (r"cupra born", "CUPRA_BORN"), (r"terramar", "CUPRA_TERRAMAR"),
    (r"citigo", "SKODA_CITIGO"), (r"karoq", "SKODA_KAROQ"), (r"octavia", "SKODA_OCTAVIA"),
    (r"superb", "SKODA_SUPERB"), (r"bentayga", "BENTLEY_BENTAYGA"), (r"urus", "LAMBO_URUS"),
    (r"galaxy", "FORD_GALAXY"),
]

def our_family(name):
    n = re.sub(r"\[\[|\]\]|<ref.*", "", str(name)).lower()
    for pat, fam in FAMILY_OURS:
        if re.search(pat, n):
            return fam
    return None


if __name__ == "__main__":
    eea = pd.read_csv(EEA_PATH)
    eea["cm"] = eea["make"].map(canon_make)
    eea = eea[eea["cm"].notna() & eea["mass_mean"].notna()].copy()
    eea["cn"] = eea["cn"].astype(str)

    # assign each EEA row to the FIRST family whose make set + regex match
    eea["family"] = None
    for fam, (makes, rx) in FAMILIES.items():
        m = eea["family"].isna() & eea["cm"].isin(makes) & eea["cn"].str.contains(rx, regex=True, na=False)
        eea.loc[m, "family"] = fam
    fe = eea[eea["family"].notna()].copy()
    fe["rm"] = fe["regs"] * fe["mass_mean"]
    fe["rev"] = fe["regs"] * fe["ev_share"]
    fe["rphev"] = fe["regs"] * fe["phev_share"]
    fy = fe.groupby(["family", "year"]).agg(regs=("regs", "sum"), rm=("rm", "sum"),
                                            rev=("rev", "sum"), rphev=("rphev", "sum"),
                                            n_cn=("cn", "nunique")).reset_index()
    fy["mass_eea"] = fy["rm"] / fy["regs"]
    fy["ev_share_eea"] = fy["rev"] / fy["regs"]
    fy["phev_share_eea"] = fy["rphev"] / fy["regs"]
    fy[["family", "year", "regs", "mass_eea", "ev_share_eea", "phev_share_eea", "n_cn"]] \
        .to_csv("eea_family_year.csv", index=False)

    # match report for plant models
    bm = pd.read_excel(MOD_PATH, sheet_name="models_binary_matrix")
    yc = [str(y) for y in range(2006, 2026)]
    bm = bm[bm[yc].sum(axis=1) > 0][["plant_id", "model"]].drop_duplicates()
    bm["family"] = bm["model"].map(our_family)
    fam_years = fy.groupby("family")["year"].agg(["min", "max", "count"])
    bm = bm.join(fam_years, on="family")
    bm["eea_matched"] = bm["count"].notna()
    bm.to_csv("eea_match_report.csv", index=False)

    print(f"[OK] eea_family_year.csv: {fy.family.nunique()} families, {len(fy)} family-years")
    print(f"plant-model rows matched to an EEA family: {bm.eea_matched.sum()}/{len(bm)}")
    print("unmatched plant models:", sorted(bm.loc[~bm.eea_matched, "model"].unique()))
