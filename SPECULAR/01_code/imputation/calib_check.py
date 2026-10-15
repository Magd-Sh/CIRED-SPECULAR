# -*- coding: utf-8 -*-
"""
Coverage of the 95% bands by cell type (internal / short edge 1-2 yrs / long edge >= 3 yrs),
pooling three masking designs (random 15%, early block, late block) over several seeds.

Usage: python calib_check.py PANEL n_seeds M "k_int,k_edge,k_nug;k_int,k_edge,k_nug;..."
"""
import sys
import numpy as np
import pandas as pd
import impute_within as iw
from features import active_span

PANEL, NSEED, M, GRID = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
act = active_span(pd.read_csv(PANEL))
act = act[act.active].sort_values(["plant_id", "year"]).reset_index(drop=True)
truth = act.index[act.log_emp.notna()].to_numpy()


def masks(seed):
    rng = np.random.default_rng(seed)
    out = {"A": rng.choice(truth, int(0.15 * len(truth)), replace=False)}
    e = act[(act.year <= 2012) & act.log_emp.notna()]
    late = set(act[(act.year > 2012) & act.log_emp.notna()].plant_id)
    c = e.groupby("plant_id").size(); el = [p for p in c[c >= 3].index if p in late]
    out["C"] = e[e.plant_id.isin(rng.choice(el, min(8, len(el)), replace=False))].index.to_numpy()
    l2 = act[(act.year >= 2020) & act.log_emp.notna()]
    early = set(act[(act.year < 2020) & act.log_emp.notna()].plant_id)
    c2 = l2.groupby("plant_id").size(); el2 = [p for p in c2[c2 >= 2].index if p in early]
    out["L"] = l2[l2.plant_id.isin(rng.choice(el2, min(8, len(el2)), replace=False))].index.to_numpy()
    return out


rows = []
for cfg in GRID.split(";"):
    ki, ke, kn = map(float, cfg.split(","))
    for seed in range(NSEED):
        for d, mk in masks(seed).items():
            dfm = act.copy(); dfm.loc[mk, "log_emp"] = np.nan
            tr = act.loc[mk].set_index(["plant_id", "year"])["log_emp"]
            rem = dfm[dfm.log_emp.notna()].groupby("plant_id")["year"].apply(np.array)
            dr = iw.impute(dfm, M=M, seed=7, k_int=ki, k_edge=ke, k_nug=kn)
            D = np.stack([x.set_index(["plant_id", "year"])["log_emp_imp"].loc[tr.index].values for x in dr])
            lo, hi = np.percentile(D, 2.5, 0), np.percentile(D, 97.5, 0)
            for (p, y), t, l, h in zip(tr.index, tr.values, lo, hi):
                yrs = rem.get(p, np.array([]))
                if len(yrs) == 0:
                    continue
                if yrs.min() < y < yrs.max():
                    typ = "internal"
                else:
                    typ = "edge 1-2y" if np.abs(yrs - y).min() <= 2 else "edge 3y+"
                rows.append((f"k_int={ki} k_edge={ke} k_nug={kn}", typ, float(l <= t <= h), h - l))
R = pd.DataFrame(rows, columns=["config", "cell", "covered", "width"])
print(R.groupby(["config", "cell"]).agg(coverage=("covered", "mean"), width=("width", "mean"),
                                        n=("covered", "size")).round(3).to_string())
