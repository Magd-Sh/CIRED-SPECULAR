# -*- coding: utf-8 -*-
"""
STEP 4 - Masking validation: justifies the METHOD choice.

We hide observed employment cells, impute them, and compare to the truth (log RMSE).
Three families compared:
  - within-plant interpolation (naive baseline)
  - EMB / Amelia MVN (the "heavy" joint method recommended in the literature)
  - within-plant MI (our retained method)
Two designs: A) 15% at random (internal gaps) . C) erased start blocks (edges).

Expected (and verified) conclusion: the joint MVN UNDERPERFORMS interpolation on this
panel (little structure across plants); within-plant MI matches the best prediction
everywhere AND provides the multiple-imputation uncertainty (Rubin) that interpolation lacks.
"""
import sys
import numpy as np
import pandas as pd
from features import active_span, build_matrix
from emb import emb
from impute_within import impute, point_estimate

PANEL = sys.argv[1] if len(sys.argv) > 1 else "panel_assembly.csv"
panel = active_span(pd.read_csv(PANEL))
act = panel[panel.active].copy().reset_index(drop=True)
truth_idx = act.index[act["log_emp"].notna()].tolist()
print(f"{len(act)} active rows . {len(truth_idx)} observed employment cells\n")


def rmse(pred, true):
    e = np.asarray(pred) - np.asarray(true)
    return float(np.sqrt(np.nanmean(e ** 2)))


def m_interp(dfm, keys):
    p = dfm.groupby("plant_id").log_emp.transform(lambda s: s.interpolate(limit_direction="both")).values
    look = {(pl, yr): val for pl, yr, val in zip(dfm.plant_id, dfm.year, p)}
    return [look.get(k, np.nan) for k in keys]


def m_emb(dfm, keys, M=5):
    X, cols, ie, df = build_matrix(dfm, use_lags=True, plant_trend=False)
    comp = emb(X, M=M, seed=1)
    mean = np.stack([c[:, ie] for c in comp]).mean(0)
    look = {(df.loc[i, "plant_id"], df.loc[i, "year"]): mean[i] for i in range(len(df))}
    return [look.get(k, np.nan) for k in keys]


def m_within(dfm, keys, M=10):
    pt = point_estimate(dfm)                      # central estimate of the retained engine
    look = {(p, y): v for p, y, v in zip(pt.plant_id, pt.year, pt.log_emp_point)}
    return [look.get(k, np.nan) for k in keys]


def run(name, mask):
    keys = [(act.loc[i, "plant_id"], act.loc[i, "year"]) for i in mask]
    tv = act.loc[mask, "log_emp"].values
    dfm = act.copy(); dfm.loc[mask, "log_emp"] = np.nan
    print(f"== design {name}  ({len(mask)} hidden cells)")
    for label, fn in [("within-plant interpolation", m_interp),
                      ("Amelia joint MVN (EMB)", m_emb),
                      ("within-plant MI (retained)", m_within)]:
        print(f"   {label:28s} RMSE(log)={rmse(fn(dfm, keys), tv):.3f}")
    print()


rng = np.random.default_rng(0)
run("A - 15% at random (internal gaps)",
    list(rng.choice(truth_idx, int(0.15 * len(truth_idx)), replace=False)))

early = act[(act.year <= 2012) & act.log_emp.notna()]
cnt = early.groupby("plant_id").size()
sel = cnt[cnt >= 3].index.tolist()
sel = list(np.random.default_rng(1).choice(sel, min(10, len(sel)), replace=False))
run("C - erased start blocks (edges)", early[early.plant_id.isin(sel)].index.tolist())
