# -*- coding: utf-8 -*-
"""
STEP 5 - Run the multiple imputation and write the deliverables.

Input: panel_assembly.csv
Outputs (outputs/ folder):
  imputations/imp_01.csv ... imp_M.csv   -> M completed panels (Rubin's rules downstream)
  employment_completed_summary.csv       -> 1 row per plant-year:
        mean, standard deviation, 95% CI, status (observed / imputed_internal /
        imputed_edge / inactive / imputed_low_coverage), and imputed flag.
"""
import sys, os
import numpy as np
import pandas as pd
from features import active_span
from impute_within import impute, point_estimate

PANEL = sys.argv[1] if len(sys.argv) > 1 else "panel_assembly.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "outputs"
M = int(sys.argv[3]) if len(sys.argv) > 3 else 50
MIN_OBS = int(sys.argv[4]) if len(sys.argv) > 4 else 0   # drop plants with fewer observed emp years (0 = keep all)
COV_THRESHOLD = 0.25                      # per-plant employment coverage threshold (research decision)

os.makedirs(os.path.join(OUT, "imputations"), exist_ok=True)

raw = pd.read_csv(PANEL)

# exclude data-poor plants entirely (too few anchors -> destabilize the imputation)
if MIN_OBS > 0:
    obs_ct = raw.groupby("plant_id")["log_emp"].apply(lambda s: s.notna().sum())
    keep = set(obs_ct[obs_ct >= MIN_OBS].index)
    dropped = sorted(pl.split("_", 1)[1] for pl in set(raw.plant_id) - keep)
    raw = raw[raw.plant_id.isin(keep)].copy()
    print(f"[5] excluded {len(dropped)} data-poor plants (< {MIN_OBS} observed emp years): {dropped}")
    print(f"[5] {len(keep)} plants retained")
panel = active_span(raw)
act = panel[panel.active].sort_values(["plant_id", "year"]).reset_index(drop=True)
inactive = panel[~panel.active].copy()

# per-plant employment coverage WITHIN the active span
cov = act.groupby("plant_id")["log_emp"].apply(lambda s: s.notna().mean())
low_cov_plants = set(cov[cov < COV_THRESHOLD].index)
print(f"[5] plants below the {COV_THRESHOLD:.0%} employment-coverage threshold: "
      f"{sorted(pl.split('_',1)[1] for pl in low_cov_plants)}")

# --- multiple imputation
comp = impute(act, M=M, seed=0)
print(f"[5] {M} completed panels generated")

# --- write each completed panel (rows sorted plant, year - same order as the engine output)
key_cols = ["plant_id", "country", "group", "brand", "year"]
assert (comp[0][["plant_id", "year"]].values == act[["plant_id", "year"]].values).all()
rel = act["emp_reliability"].values if "emp_reliability" in act else np.full(len(act), np.nan)
for i, c in enumerate(comp, 1):
    dfo = c[key_cols].copy()
    dfo["log_emp_imputed"] = c["log_emp_imp"]
    dfo["employees_imputed"] = np.exp(c["log_emp_imp"])
    dfo["was_observed"] = act["log_emp"].notna().values
    dfo["employees_recorded"] = np.exp(act["log_emp"].values)     # recorded value (NaN if missing)
    dfo["emp_reliability"] = rel
    dfo.to_csv(os.path.join(OUT, "imputations", f"imp_{i:02d}.csv"), index=False)

# --- central estimate: NOISELESS point estimate (draws scatter around it);
#     band: 2.5th-97.5th percentiles of the M draws (calibrated to ~95% coverage, step4b).
#     Observed values with reliability < 1 are over-imputed, so they also get a (small) band.
pt = point_estimate(act)
center_log = pt["log_emp_point"].values
stack_log = np.stack([c["log_emp_imp"].values for c in comp])   # (M, n)
mean_log = center_log
sd_log = stack_log.std(0)
lo_log, hi_log = np.percentile(stack_log, 2.5, axis=0), np.percentile(stack_log, 97.5, axis=0)

def status(row_i):
    if act.loc[row_i, "plant_id"] in low_cov_plants and not np.isfinite(act.loc[row_i, "log_emp"]):
        return "imputed_low_coverage"
    if np.isfinite(act.loc[row_i, "log_emp"]):
        return "observed"
    # internal vs edge: are there observations on both sides within the plant?
    sub = act[act.plant_id == act.loc[row_i, "plant_id"]]
    yr = act.loc[row_i, "year"]
    oy = sub.loc[sub["log_emp"].notna(), "year"]
    if len(oy) and oy.min() < yr < oy.max():
        return "imputed_internal"
    return "imputed_edge"

summary = act[key_cols].copy()
summary["status"] = [status(i) for i in range(len(act))]
summary["employees_obs"] = np.exp(act["log_emp"].values)
summary["employees_est"] = np.exp(mean_log)                    # central estimate (= recorded if observed)
summary["est_sd_log"] = sd_log
summary["employees_lo95"] = np.exp(lo_log)
summary["employees_hi95"] = np.exp(hi_log)
summary["emp_reliability"] = rel
summary["imputed"] = summary["status"] != "observed"
summary["noisy_observed"] = (summary["status"] == "observed") & (np.nan_to_num(rel, nan=1.0) < 1.0)

# attach inactive cells (legitimate blank), not imputed
if len(inactive):
    inact = inactive[key_cols].copy()
    inact["status"] = "inactive"
    inact["employees_obs"] = np.exp(inactive["log_emp"].values)
    for c in ["employees_est", "employees_lo95", "employees_hi95", "est_sd_log"]:
        inact[c] = np.nan
    inact["imputed"] = False
    summary = pd.concat([summary, inact], ignore_index=True)

summary = summary.sort_values(["plant_id", "year"]).reset_index(drop=True)
summary.to_csv(os.path.join(OUT, "employment_completed_summary.csv"), index=False)

print("\n[5] status breakdown:")
print(summary["status"].value_counts().to_string())
print(f"\n[OK] deliverables written to {OUT}/")
print(f"    - imputations/imp_01..{M:02d}.csv")
print(f"    - employment_completed_summary.csv  ({len(summary)} rows)")
