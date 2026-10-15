# -*- coding: utf-8 -*-
"""
STEP 4b - Compare engine variants and calibrate the uncertainty (masking, several seeds).

Hides observed employment cells, imputes them, and scores:
  - RMSE (log) of the CENTRAL estimate (point_estimate)            -> accuracy
  - share of hidden true values inside the 95% band of the draws   -> calibration (target 95%)
Designs: A = 15% of observed cells at random (mostly internal gaps),
         C = erase the pre-2013 block of plants with >= 3 early observations (edges).

Usage: python step4b_calibrate.py panel_assembly.csv [n_seeds] [M]
"""
import sys, itertools
import numpy as np
import pandas as pd
from features import active_span
from impute_within import impute, point_estimate, COVARS

PANEL = sys.argv[1] if len(sys.argv) > 1 else "panel_assembly.csv"
NSEED = int(sys.argv[2]) if len(sys.argv) > 2 else 5
M = int(sys.argv[3]) if len(sys.argv) > 3 else 40
MODE = sys.argv[4] if len(sys.argv) > 4 else "variants"          # 'variants' or 'calibrate'

act = active_span(pd.read_csv(PANEL))
act = act[act.active].sort_values(["plant_id", "year"]).reset_index(drop=True)
truth_idx = act.index[act.log_emp.notna()].to_numpy()


def masks(seed):
    rng = np.random.default_rng(seed)
    A = rng.choice(truth_idx, int(0.15 * len(truth_idx)), replace=False)
    early = act[(act.year <= 2012) & act.log_emp.notna()]
    late = set(act[(act.year > 2012) & act.log_emp.notna()].plant_id)   # must keep >= 1 anchor
    cnt = early.groupby("plant_id").size()
    elig = [p for p in cnt[cnt >= 3].index if p in late]
    sel = rng.choice(elig, min(8, len(elig)), replace=False)
    C = early[early.plant_id.isin(sel)].index.to_numpy()
    return {"A_internal": A, "C_edge": C}


def score(mask, **kw):
    dfm = act.copy(); dfm.loc[mask, "log_emp"] = np.nan
    pt = point_estimate(dfm, covars=kw.get("covars", COVARS), bridge=kw.get("bridge", False))
    key = pt.set_index(["plant_id", "year"])["log_emp_point"]
    tr = act.loc[mask].set_index(["plant_id", "year"])["log_emp"]
    rmse = float(np.sqrt(np.mean((key.loc[tr.index].values - tr.values) ** 2)))
    draws = impute(dfm, M=M, seed=7, **kw)
    D = np.stack([d.set_index(["plant_id", "year"])["log_emp_imp"].loc[tr.index].values for d in draws])
    lo, hi = np.percentile(D, 2.5, axis=0), np.percentile(D, 97.5, axis=0)
    cov = float(np.mean((tr.values >= lo) & (tr.values <= hi)))
    width = float(np.mean(hi - lo))
    return rmse, cov, width


def run(configs):
    res = []
    for name, kw in configs.items():
        for seed in range(NSEED):
            for dname, mk in masks(seed).items():
                r, c, w = score(mk, **kw)
                res.append((name, dname, seed, r, c, w))
    R = pd.DataFrame(res, columns=["variant", "design", "seed", "rmse", "cov95", "width"])
    print(R.groupby(["design", "variant"])[["rmse", "cov95", "width"]].mean().round(3).to_string())
    return R


if MODE == "variants":
    run({
        "baseline (linear, no regs)": dict(covars=COVARS, bridge=False),
        "+ regs proxy":               dict(covars=COVARS + ["log_regs"], bridge=False),
        "+ bridge":                   dict(covars=COVARS, bridge=True),
        "+ bridge + regs":            dict(covars=COVARS + ["log_regs"], bridge=True),
    })
else:
    cfg = {}
    for ki, ke in itertools.product([0.75, 1.0, 1.5, 2.0], [1.0, 1.5, 2.0, 3.0]):
        cfg[f"k_int={ki}, k_edge={ke}"] = dict(covars=COVARS, bridge=False, k_int=ki, k_edge=ke)
    run(cfg)
