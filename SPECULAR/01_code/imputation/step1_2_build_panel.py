# -*- coding: utf-8 -*-
"""
STEPS 1 & 2 - Build the plant x year panel and the covariates.

Output: panel_assembly.csv  (26 assembly plants x 20 years, 2006-2025)
Key columns:
  log_emp        : ln(employment)                     [to impute]
  log_prod       : ln(vehicle production)             [to impute, very sparse]
  kerb_weight    : average kerb weight (kg) - proxy for vehicle SIZE [covariate / sometimes imputed]
  seg_score      : average segment (ordinal 1=Economy ... 6=Luxury)
  ev_share       : share of electric models
  n_models       : number of active models
  + reliability flags (kind, source) for overimputation
  + emp_reliability : confidence weight of the observed employment value

Usage:
  python step1_2_build_panel.py EMP.xlsx MODELS_v5.xlsx  [out_dir]
"""
import sys, os
import numpy as np
import pandas as pd

EMP_PATH  = sys.argv[1] if len(sys.argv) > 1 else "employment_timeseries_full.xlsx"
MOD_PATH  = sys.argv[2] if len(sys.argv) > 2 else "auto_models_production_consolidated_v5.xlsx"
OUT_DIR   = sys.argv[3] if len(sys.argv) > 3 else "."
SCOPE     = sys.argv[4] if len(sys.argv) > 4 else "assembly"   # 'assembly' or 'all'
SIZE_SRC  = sys.argv[5] if len(sys.argv) > 5 else "eea"        # 'eea' (default) or 'ademe' (legacy v1)
EEA_FY    = sys.argv[6] if len(sys.argv) > 6 else "eea_family_year.csv"
EEA_MY    = sys.argv[7] if len(sys.argv) > 7 else "eea_model_year.csv"   # for yearly registration totals
YEARS     = list(range(2006, 2026))

os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------------------------------------------- STEP 1
# 1a. Scope: assembly plants only (research decision) OR all plants
xl = pd.ExcelFile(MOD_PATH)
pm = xl.parse("plant_master")
sel = pm if SCOPE == "all" else pm[pm.classification == "assembly"]
assembly = sel[["plant_id", "country", "group", "brand", "classification"]].copy()
plants = sorted(assembly.plant_id.unique())
PANEL_FILE = ("panel_all" if SCOPE == "all" else "panel_assembly") + \
             ("_ademe" if SIZE_SRC == "ademe" else "") + ".csv"
print(f"[1] scope='{SCOPE}': {len(plants)} plants retained "
      f"({dict(assembly.classification.value_counts())})")

# 1b. Full plant x year grid
grid = pd.MultiIndex.from_product([plants, YEARS], names=["plant_id", "year"]).to_frame(index=False)
grid = grid.merge(assembly, on="plant_id", how="left")

# 1c. Employment (keyed on reference_year: the year the figure actually refers to)
emp = pd.read_excel(EMP_PATH)
emp = emp[emp.plant_id.isin(plants)].copy()
# one value per plant-year; prefer reference_year if available, else year
emp["yr"] = emp["reference_year"].fillna(emp["year"]).astype(int)
emp = emp[emp.yr.between(2006, 2025)]
# keep the best value per plant-year: priority 'fresh' > 'carried', then source
kind_rank = {"fresh": 0, "carried": 1}
emp["krank"] = emp["kind"].map(kind_rank).fillna(2)
emp = emp.sort_values(["plant_id", "yr", "krank"]).dropna(subset=["employees"])
empg = emp.groupby(["plant_id", "yr"]).agg(
    employees=("employees", "first"),
    kind=("kind", "first"),
    value_source=("value_source", "first"),
    carried_years=("carried_years", "first"),
    snippet=("snippet", "first"),
).reset_index().rename(columns={"yr": "year"})

# 1c-bis. Documented data corrections (data_corrections.csv): clear errors found by the audit
#   (non-headcount figures, whole-site totals, one-year date shifts, duplicate plant). The source
#   spreadsheet is NOT modified; every correction is listed with its reason and evidence.
CORR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_corrections.csv")
if os.path.exists(CORR):
    corr = pd.read_csv(CORR, dtype={"year": str})
    n0 = len(empg)
    for c in corr.itertuples():
        sel = empg.plant_id.eq(c.plant_id) & (True if c.year == "ALL" else empg.year.eq(int(c.year)))
        if c.action == "drop":
            empg = empg[~sel]
        elif c.action == "set":
            empg.loc[sel, "employees"] = float(c.new_value)
            empg.loc[sel, "value_source"] = empg.loc[sel, "value_source"].astype(str) + "+corrected"
        elif c.action == "move":
            empg.loc[sel, "year"] = int(c.new_year)
    assert not empg.duplicated(["plant_id", "year"]).any(), "a correction created a duplicate plant-year"
    print(f"[1] data corrections applied: {len(corr)} rules ({n0} -> {len(empg)} plant-year values)")

# staleness: the same (undated) text re-used for a later year = archived copy of an unchanged page
empg = empg.sort_values(["plant_id", "year"])
first_year = empg.groupby(["plant_id", "snippet"])["year"].transform("min")
empg["stale_years"] = np.where(empg["snippet"].notna(), empg["year"] - first_year, 0)

panel = grid.merge(empg, on=["plant_id", "year"], how="left")

# 1d. Reliability weight of the observed employment value (used for over-imputation)
def reliability(row):
    if pd.isna(row.employees):
        return np.nan
    vs = str(row.value_source or "")
    base = 1.00
    if row.kind == "carried":
        cy = row.carried_years if pd.notna(row.carried_years) else 1
        base -= 0.10 * float(cy)                           # degrade with age of the carry
    if pd.notna(row.stale_years) and row.stale_years > 0:
        base -= 0.10 * float(row.stale_years)              # same text as an earlier year (stale)
    if vs.startswith("wiki"):
        base -= 0.15                                       # wiki = rounded ('ca. 20,000')
    return float(np.clip(base, 0.30, 1.00))
panel["emp_reliability"] = panel.apply(reliability, axis=1)
panel = panel.drop(columns=["snippet"])

print(f"[1] observed employment: {panel.employees.notna().sum()}/{len(panel)} cells "
      f"({panel.employees.notna().mean():.0%})")

# ----------------------------------------------------------------- STEP 2
# 2a. Model attributes (ADEME): mass, segment, energy; propagated per model
ml = xl.parse("models_long")
SEG_ORD = {"Economy": 1, "Lower-medium": 2, "Upper-medium": 3,
           "Executive": 4, "Luxury": 6}  # ordinal for size/tier

def seg_to_score(s):
    if pd.isna(s):
        return np.nan
    vals = [SEG_ORD.get(p.strip()) for p in str(s).split(";")]
    vals = [v for v in vals if v is not None]
    return np.mean(vals) if vals else np.nan

def kerb_mid(r):
    lo, hi = r.get("kerb_weight_min_kg"), r.get("kerb_weight_max_kg")
    if pd.notna(lo) and pd.notna(hi):
        return (lo + hi) / 2
    return lo if pd.notna(lo) else hi

ml = ml.copy()
ml["seg_score"] = ml["market_segment"].map(seg_to_score)
ml["kerb"] = ml.apply(kerb_mid, axis=1)
ml["is_ev"] = ml["energy_fuel"].astype(str).str.contains("Electric", case=False, na=False) & \
              ~ml["energy_fuel"].astype(str).str.contains("hybrid", case=False, na=False)

# 2b. NOMINAL FALLBACK for major models not matched to ADEME.
#     Approximate manufacturer figures (typical kerb weight, kg) and ordinal segment.
#     >>> TO VERIFY / refine: these are nominal values, tagged kerb_source='fallback'.
#     They rescue the "size" covariate for 10 assembly plants otherwise at 0%.
#     (case-insensitive substring -> (seg_score, kerb_kg))
FALLBACK = [
    ("tiguan",   (3.0, 1650)), ("nivus", (2.0, 1250)), ("taigo", (2.0, 1250)),
    ("t-sport",  (2.0, 1250)),
    ("q7",       (6.0, 2200)), ("q8", (6.0, 2300)), ("bentayga", (6.0, 2400)),
    ("urus",     (6.0, 2350)), ("cayenne", (6.0, 2100)),
    ("phaeton",  (6.0, 2250)),
    ("passat",   (3.0, 1500)), ("polo", (1.5, 1150)), ("mii", (1.0, 950)),
    ("golf",     (2.0, 1300)),
    ("id.3",     (2.0, 1650)), ("id.4", (3.0, 2100)), ("id.5", (3.0, 2150)),
    ("e-golf",   (2.0, 1600)), ("born", (2.0, 1700)), ("q4 e-tron", (3.5, 2100)),
    ("a-class",  (2.0, 1400)), ("clase a", (2.0, 1400)), ("b-class", (2.0, 1450)),
    ("clase b",  (2.0, 1450)), ("cla", (2.5, 1500)), ("glb", (3.0, 1700)),
    ("c-class",  (3.0, 1600)),
    ("sprinter", (3.0, 2200)),            # heavy commercial van
    ("a5",       (4.0, 1600)), ("s5", (4.0, 1650)),
]
def fallback_lookup(model):
    m = str(model).lower()
    for pat, (seg, kerb) in FALLBACK:
        if pat in m:
            return seg, kerb
    return None, None

# median kerb weight per segment (from real ADEME values) as a 2nd-tier fallback
seg_kerb = ml.dropna(subset=["seg_score", "kerb"]).groupby("seg_score")["kerb"].median()

def fill_kerb_row(r):
    """Return (kerb, seg_score, source)."""
    if pd.notna(r["kerb"]):
        return r["kerb"], r["seg_score"], "ademe"
    fseg, fkerb = fallback_lookup(r["model"])
    if fkerb is not None:
        return float(fkerb), float(fseg), "fallback"
    if pd.notna(r["seg_score"]) and len(seg_kerb):
        idx = (seg_kerb.index - r["seg_score"]).abs().idxmin()
        return float(seg_kerb.get(idx)), r["seg_score"], "seg_median"
    return np.nan, r["seg_score"], "none"

_f = ml.apply(fill_kerb_row, axis=1, result_type="expand")
ml["kerb_filled"] = _f[0]
ml["seg_filled"]  = _f[1]
ml["kerb_source"] = _f[2]
print(f"[2] model mass source: {ml['kerb_source'].value_counts().to_dict()}")

# 2c. Binary model x year presence matrix -> covariates aggregated per plant-year
bm = xl.parse("models_binary_matrix")
ycols = [str(y) for y in YEARS]
attr = ml[["plant_id", "model", "seg_filled", "kerb_filled", "is_ev"]].drop_duplicates(["plant_id", "model"])
attr = attr.rename(columns={"seg_filled": "seg_score"})
bm2 = bm[["plant_id", "model"] + ycols].merge(attr, on=["plant_id", "model"], how="left")

# 2c-bis. OPTIONAL: vehicle mass & EV share from the EEA registration data (SIZE_SRC='eea').
#   Year-specific, registration-weighted mass per model family (eea_family_year.csv, built by
#   eea_aggregate.py + eea_match.py). EEA mass = "mass in running order" (kerb + ~75 kg driver);
#   non-EEA models (ADEME/fallback kerb) are shifted by the median EEA-minus-old offset so all
#   values are on one scale. Years before a family's first EEA year are backfilled with the
#   nearest EEA year. seg_score is unchanged (EEA has no segment variable).
USE_EEA = (SIZE_SRC == "eea")
if USE_EEA:
    from eea_match import our_family
    fy = pd.read_csv(EEA_FY)
    eea_mass = {(r.family, r.year): r.mass_eea for r in fy.itertuples()}
    eea_ev = {(r.family, r.year): r.ev_share_eea for r in fy.itertuples()}
    fam_years = fy.groupby("family")["year"].apply(lambda s: np.array(sorted(s))).to_dict()
    bm2["family"] = bm2["model"].map(our_family)

    def eea_lookup(fam, y):
        """(mass, ev, source) for a family-year: exact year, else nearest EEA year, else None."""
        if fam in fam_years:
            if (fam, y) in eea_mass:
                return eea_mass[(fam, y)], eea_ev[(fam, y)], "eea"
            yrs = fam_years[fam]
            ny = int(yrs[np.argmin(np.abs(yrs - y))])
            return eea_mass[(fam, ny)], eea_ev[(fam, ny)], "eea_nearest_year"
        return None, None, None

    # calibration offset: EEA running-order mass minus old ADEME/fallback kerb, on exact matches
    long = bm2.melt(id_vars=["plant_id", "model", "family", "kerb_filled"], value_vars=ycols,
                    var_name="year", value_name="active")
    long = long[(long.active == 1) & long.kerb_filled.notna() & long.family.notna()]
    long["year"] = long["year"].astype(int)
    long["eea"] = [eea_mass.get((f, y), np.nan) for f, y in zip(long.family, long.year)]
    diffs = (long["eea"] - long["kerb_filled"]).dropna().values
    OFFSET = float(np.median(diffs)) if len(diffs) else 0.0
    print(f"[2] EEA calibration offset (EEA mass - old kerb, median over {len(diffs)} model-years): {OFFSET:+.0f} kg")

    # --- registration-volume OUTPUT PROXY (per plant-year)
    #   share of the family's EU registrations in that year's EEA total (normalises for the
    #   varying EEA sample size), split equally across the dataset plants building that family
    #   that year, summed over the plant's models. Exact EEA years only (no backfill: a flat
    #   backfilled proxy would carry no information). Defined only when EVERY active model that
    #   has an EEA family is covered that year (avoids jumps when a family enters coverage).
    eea_regs = {(r.family, r.year): r.regs for r in fy.itertuples()}
    tot_regs = pd.read_csv(EEA_MY).groupby("year")["regs"].sum().to_dict()
    long_all = bm2.melt(id_vars=["plant_id", "family"], value_vars=ycols,
                                       var_name="year", value_name="active")
    long_all = long_all[(long_all.active == 1) & long_all.family.notna()]
    long_all["year"] = long_all["year"].astype(int)
    n_plants_fam = long_all.groupby(["family", "year"])["plant_id"].nunique().to_dict()

    def regs_proxy(act, y):
        fams = [f for f in act["family"] if f is not None and pd.notna(f)]
        if not fams or y not in tot_regs:
            return np.nan
        if any((f, y) not in eea_regs for f in fams):
            return np.nan
        return sum(eea_regs[(f, y)] / tot_regs[y] / n_plants_fam.get((f, y), 1) for f in fams)

rows, src_count = [], {}
for p in plants:
    sub = bm2[bm2.plant_id == p]
    for y in YEARS:
        act = sub[sub[str(y)] == 1]
        n = len(act)
        if USE_EEA and n:
            masses, evs, n_eea = [], [], 0
            for r in act.itertuples():
                m_, e_, s_ = eea_lookup(r.family, y)
                if m_ is None:                      # no EEA family -> calibrated old value
                    m_ = r.kerb_filled + OFFSET if pd.notna(r.kerb_filled) else np.nan
                    e_ = float(r.is_ev) if pd.notna(r.is_ev) else np.nan
                    s_ = "old_calibrated" if pd.notna(m_) else "none"
                else:
                    n_eea += 1
                src_count[s_] = src_count.get(s_, 0) + 1
                masses.append(m_); evs.append(e_)
            kerb_val = np.nanmean(masses) if np.isfinite(masses).any() else np.nan
            ev_val = np.nanmean(evs) if np.isfinite(evs).any() else np.nan
            eea_frac = n_eea / n
            rp = regs_proxy(act, y)
        else:
            kerb_val = act["kerb_filled"].mean() if n else np.nan
            ev_val = act["is_ev"].mean() if n else np.nan
            eea_frac = np.nan
            rp = np.nan
        rows.append({
            "plant_id": p, "year": y,
            "n_models": n,
            "kerb_weight": kerb_val,
            "seg_score":   act["seg_score"].mean() if n else np.nan,
            "ev_share":    ev_val,
            "size_eea_frac": eea_frac,
            "regs_proxy": rp,
        })
cov = pd.DataFrame(rows)
if USE_EEA:
    print(f"[2] model-year mass source (EEA mode): {src_count}")

# 2d. Plant-year production
prod = xl.parse("production_plant_year")
prod = prod[prod.plant_id.isin(plants)]
prodg = prod.groupby(["plant_id", "year"])["production_units"].first().reset_index()
cov = cov.merge(prodg, on=["plant_id", "year"], how="left")

panel = panel.merge(cov, on=["plant_id", "year"], how="left")

# 2e. Transforms
panel["log_emp"]  = np.log(panel["employees"])
panel["log_prod"] = np.log(panel["production_units"])
panel["log_nmodels"] = np.log1p(panel["n_models"])
panel["log_regs"] = np.log(panel["regs_proxy"]) if "regs_proxy" in panel else np.nan

# 2f. "Legitimate blank": a plant with no active model that year = not yet open / converted
#     -> we DO NOT impute it (cf. Hebdo 9). We flag it.
panel["legit_empty"] = (panel["n_models"].fillna(0) == 0)

# ----------------------------------------------------------------- coverage
cover = panel.groupby("plant_id").agg(
    emp_cov=("log_emp", lambda s: s.notna().mean()),
    prod_cov=("log_prod", lambda s: s.notna().mean()),
    size_cov=("kerb_weight", lambda s: s.notna().mean()),
).round(2)
print("\n[2] coverage per plant (employment / prod / size):")
print(cover.sort_values("emp_cov").to_string())

print("\n[2] global covariate coverage:")
for c in ["log_emp", "log_prod", "kerb_weight", "seg_score", "ev_share", "n_models"]:
    print(f"    {c:14s} {panel[c].notna().mean():.0%}")

out = os.path.join(OUT_DIR, PANEL_FILE)
panel.to_csv(out, index=False)
print(f"\n[OK] panel written: {out}  ({panel.shape[0]} rows x {panel.shape[1]} columns)")
