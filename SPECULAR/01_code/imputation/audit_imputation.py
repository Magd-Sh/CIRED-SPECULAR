# -*- coding: utf-8 -*-
"""
Audit of the final imputed employment dataset.

Usage: python audit_imputation.py EMP.xlsx panel_assembly.csv outputs [min_obs]
Prints PASS/FAIL/WARN lines; exits with code 1 if any FAIL.
"""
import sys, glob, os
import numpy as np
import pandas as pd
from features import active_span
from impute_within import point_estimate, OI_SCALE

EMP, PANEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
MIN_OBS = int(sys.argv[4]) if len(sys.argv) > 4 else 0
fails, warns = [], []

def check(ok, msg, warn=False):
    tag = "PASS" if ok else ("WARN" if warn else "FAIL")
    print(f"[{tag}] {msg}")
    if not ok:
        (warns if warn else fails).append(msg)

panel = pd.read_csv(PANEL)
summ = pd.read_csv(os.path.join(OUT, "employment_completed_summary.csv"))
files = sorted(glob.glob(os.path.join(OUT, "imputations", "imp_*.csv")))
P = [pd.read_csv(f) for f in files]
plants = sorted(summ.plant_id.unique())
print(f"== {OUT}: {len(plants)} plants, {len(summ)} summary rows, {len(P)} panels")

# ---------------------------------------------------------------- 1. structure
check(len(P) == 50, f"50 imputation files present ({len(P)})")
check(not summ.duplicated(["plant_id", "year"]).any(), "summary: one row per plant-year")
check(len(summ) == len(plants) * 20, f"summary covers every plant x 2006-2025 ({len(summ)} = {len(plants)}x20)")
keys0 = P[0][["plant_id", "year"]]
check(all((p[["plant_id", "year"]].values == keys0.values).all() for p in P), "all 50 panels have identical rows in identical order")
check(not keys0.duplicated().any(), "panels: one row per plant-year")
check(all(p.employees_imputed.notna().all() for p in P), "no missing value left in any panel")
check(all((p.employees_imputed > 0).all() and np.isfinite(p.employees_imputed).all() for p in P), "all imputed values positive and finite")
check(all(np.allclose(np.exp(p.log_emp_imputed), p.employees_imputed) for p in P), "employees_imputed = exp(log_emp_imputed)")
act_keys = set(map(tuple, summ.loc[summ.status != "inactive", ["plant_id", "year"]].values))
check(set(map(tuple, keys0.values)) == act_keys, "panel rows = summary rows that are not 'inactive'")

# ---------------------------------------------------------------- 2. source fidelity
src = pd.read_excel(EMP)
src = src[src.plant_id.isin(plants)].dropna(subset=["employees"]).copy()
src["yr"] = src["reference_year"].fillna(src["year"]).astype(int)
src = src[src.yr.between(2006, 2025)]
# apply the documented corrections (independently of step1_2) so the comparison is like-for-like
corr = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_corrections.csv"), dtype={"year": str})
for c in corr.itertuples():
    sel = src.plant_id.eq(c.plant_id) & (True if c.year == "ALL" else src.yr.eq(int(c.year)))
    if c.action == "drop":
        src = src[~sel]
    elif c.action == "set":
        src.loc[sel, "employees"] = float(c.new_value)
    elif c.action == "move":
        src.loc[sel, "yr"] = int(c.new_year)
print(f"       {len(corr)} documented corrections applied to the source before comparison")
rec = summ.dropna(subset=["employees_obs"]).set_index(["plant_id", "year"])["employees_obs"]
srcvals = src.groupby(["plant_id", "yr"])["employees"].apply(set)
missing_from_src = [k for k, v in rec.items() if k not in srcvals.index or not any(np.isclose(v, s) for s in srcvals[k])]
check(len(missing_from_src) == 0, f"every recorded value exists in the source file for that plant & reference year ({len(rec)} values)")
not_used = [k for k in srcvals.index if k not in rec.index]
check(len(not_used) == 0, f"every source plant-year (reference-year keyed) is present as an observation ({len(srcvals)} source cells)", warn=True)
conflict = srcvals[srcvals.apply(len) > 1]
check(len(conflict) == 0, f"source has a single value per plant & reference year ({len(conflict)} plant-years with conflicting values)", warn=True)
if len(conflict):
    c = conflict.apply(lambda s: (min(s), max(s), max(s) / min(s) - 1))
    big = c[c.apply(lambda t: t[2] > 0.10)]
    print(f"       conflicts > 10% apart: {len(big)}")
    for k, (lo, hi, r) in big.items():
        print(f"         {k[0]} {k[1]}: {lo:.0f} vs {hi:.0f} (+{r:.0%}); kept {rec.get(k, np.nan):.0f}")

# ---------------------------------------------------------------- 3. observed values
obs_mask = P[0].was_observed.values
summ_act = summ[summ.status != "inactive"].set_index(["plant_id", "year"]).loc[list(map(tuple, keys0.values))]
check((summ_act.status.eq("observed").values == obs_mask).all(), "was_observed (panels) = status 'observed' (summary)")
check(np.allclose(P[0].employees_recorded.values[obs_mask], summ_act.employees_obs.values[obs_mask]), "employees_recorded = summary employees_obs on observed cells")
rel = P[0].emp_reliability.values
reliable = obs_mask & ~(np.nan_to_num(rel, nan=1.0) < 1)
noisy = obs_mask & (np.nan_to_num(rel, nan=1.0) < 1)
stack = np.stack([p.log_emp_imputed.values for p in P])
recl = np.log(P[0].employees_recorded.values)
check(np.allclose(stack[:, reliable], recl[reliable]), f"reliable observed values identical in all 50 panels ({reliable.sum()} cells)")
if noisy.sum():
    z = (stack[:, noisy] - recl[noisy]) / (OI_SCALE * (1 - rel[noisy]))
    check(abs(z.mean()) < 0.15 and 0.8 < z.std() < 1.2,
          f"over-imputed observed cells ({noisy.sum()}) vary around the recorded value with the intended sd (z mean {z.mean():+.2f}, sd {z.std():.2f})")
check(np.allclose(summ_act.employees_est.values[obs_mask], summ_act.employees_obs.values[obs_mask]), "central estimate = recorded value on all observed cells")
check((rel[obs_mask] >= 0.3).all() and (rel[obs_mask] <= 1).all(), "reliability within [0.3, 1] on observed cells")

# ---------------------------------------------------------------- 4. summary vs panels
act_panel = active_span(panel[panel.plant_id.isin(plants)])
act_panel = act_panel[act_panel.active].sort_values(["plant_id", "year"]).reset_index(drop=True)
pt = point_estimate(act_panel)
check(np.allclose(np.exp(pt.log_emp_point.values), summ_act.employees_est.values), "summary employees_est = engine central estimate")
lo, hi = np.exp(np.percentile(stack, 2.5, 0)), np.exp(np.percentile(stack, 97.5, 0))
check(np.allclose(lo, summ_act.employees_lo95.values) and np.allclose(hi, summ_act.employees_hi95.values),
      "summary 95% band = 2.5/97.5 percentiles of the 50 panels")
check(np.allclose(stack.std(0), summ_act.est_sd_log.values), "summary est_sd_log = sd of the 50 panels")
inside = (summ_act.employees_est.values >= lo * 0.999) & (summ_act.employees_est.values <= hi * 1.001)
check(inside.mean() > 0.97, f"central estimate inside its own 95% band ({inside.mean():.1%} of cells)")

# ---------------------------------------------------------------- 5. status logic
ok_status = True
for pid, g in summ.groupby("plant_id"):
    oy = g.loc[g.status == "observed", "year"]
    for r in g.itertuples():
        if r.status == "imputed_internal" and not (oy.min() < r.year < oy.max()):
            ok_status = False
        if r.status == "imputed_edge" and (oy.min() < r.year < oy.max()):
            ok_status = False
check(ok_status, "internal/edge status consistent with the position of observations")
check(summ.imputed.eq(summ.status.str.startswith("imputed")).all(), "'imputed' flag consistent with status")
inact = summ[summ.status == "inactive"]
sig = panel.set_index(["plant_id", "year"])
bad_inact = [k for k in map(tuple, inact[["plant_id", "year"]].values)
             if pd.notna(sig.loc[k, "log_emp"]) or pd.notna(sig.loc[k, "log_prod"]) or (sig.loc[k, "n_models"] or 0) > 0]
check(len(bad_inact) == 0, f"'inactive' cells have no employment, production or model signal ({len(inact)} cells)")

# ---------------------------------------------------------------- 6. plausibility
pl_rng = summ[summ.status == "observed"].groupby("plant_id").employees_obs.agg(["min", "max"])
est = summ_act.reset_index().merge(pl_rng, on="plant_id")
imp_est = est[est.status.str.startswith("imputed")]
out_range = imp_est[(imp_est.employees_est < imp_est["min"] * 0.999) | (imp_est.employees_est > imp_est["max"] * 1.001)]
check(len(out_range) == 0, f"central estimates of imputed cells stay within the plant's observed min-max ({len(imp_est)} cells)")
ratio = np.exp(stack.max(0) - stack.min(0))
check(np.quantile(ratio[~obs_mask], 0.99) < 6, f"draw spread reasonable (99th pct max/min across draws = {np.quantile(ratio[~obs_mask], 0.99):.1f}x)", warn=True)
# suspicious recorded values: year-on-year jump > 30% that reverses within 2 years
sus = []
for pid, g in summ[summ.status == "observed"].sort_values("year").groupby("plant_id"):
    y, v = g.year.values, np.log(g.employees_obs.values)
    for i in range(1, len(v) - 1):
        if y[i + 1] - y[i - 1] <= 3:
            up, down = v[i] - v[i - 1], v[i + 1] - v[i]
            if abs(up) > np.log(1.3) and abs(down) > np.log(1.3) and np.sign(up) != np.sign(down):
                sus.append((pid, int(y[i]), round(np.exp(v[i - 1])), round(np.exp(v[i])), round(np.exp(v[i + 1]))))
check(len(sus) == 0, f"no recorded spike/dip > 30% that reverses within 2-3 years ({len(sus)} found)", warn=True)
for s in sus:
    print(f"         {s[0]} {s[1]}: {s[2]} -> {s[3]} -> {s[4]}")

# ---------------------------------------------------------------- 7. regression usability
X = panel[["plant_id", "year", "kerb_weight"]]
m = P[0].merge(X, on=["plant_id", "year"], how="left")
check(len(m) == len(P[0]), "panels join 1:1 with the covariate panel on plant_id, year")
print(f"       kerb_weight available on {m.kerb_weight.notna().mean():.0%} of panel rows")
if MIN_OBS:
    oc = panel.groupby("plant_id").log_emp.apply(lambda s: s.notna().sum())
    check(set(plants) == set(oc[oc >= MIN_OBS].index), f"plants retained = plants with >= {MIN_OBS} observed years")

print(f"\nSUMMARY {OUT}: {len(fails)} FAIL, {len(warns)} WARN")
sys.exit(1 if fails else 0)
