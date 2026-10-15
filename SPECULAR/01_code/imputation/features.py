# -*- coding: utf-8 -*-
"""
Builds the numeric matrix for imputation (shared by validation & run).

- Restricts each plant to its ACTIVE SPAN [first, last year with a signal], so we don't
  impute employment where the plant did not exist ("legitimate blank", cf. Hebdo 9).
- Analysis variables (may be missing -> imputed):
    log_emp, log_prod, kerb_weight, seg_score, ev_share, log_nmodels [, emp_lag1, emp_lead1]
- Design variables (always observed, condition only):
    plant dummies (drop first) + time (centered) linear/quadratic/cubic
"""
import numpy as np
import pandas as pd

ANALYSIS = ["log_emp", "log_prod", "kerb_weight", "seg_score", "ev_share", "log_nmodels"]


def active_span(panel):
    """Flag the rows within each plant's active span."""
    panel = panel.copy()
    signal = panel["log_emp"].notna() | panel["log_prod"].notna() | (panel["n_models"].fillna(0) > 0)
    panel["_signal"] = signal
    keep = np.zeros(len(panel), dtype=bool)
    for p, sub in panel.groupby("plant_id"):
        yrs = sub.loc[sub["_signal"], "year"]
        if len(yrs) == 0:
            continue
        lo, hi = yrs.min(), yrs.max()
        keep |= (panel["plant_id"] == p) & panel["year"].between(lo, hi)
    panel["active"] = keep
    return panel


def build_matrix(panel_active, use_lags=True, plant_trend=True):
    """
    panel_active : subset already filtered (active==True), sorted by plant then year.
    Returns X (n x p), column names, and the index of the log_emp column.
    """
    df = panel_active.sort_values(["plant_id", "year"]).reset_index(drop=True).copy()

    cols = list(ANALYSIS)
    data = {c: df[c].astype(float).values for c in ANALYSIS}

    if use_lags:
        lag = np.full(len(df), np.nan)
        lead = np.full(len(df), np.nan)
        for p, idx in df.groupby("plant_id").groups.items():
            idx = list(idx)
            e = df.loc[idx, "log_emp"].values
            lag[idx[1:]] = e[:-1]
            lead[idx[:-1]] = e[1:]
        data["emp_lag1"] = lag
        data["emp_lead1"] = lead
        cols += ["emp_lag1", "emp_lead1"]

    # design: plant dummies (drop the 1st) + centered time
    plants = sorted(df["plant_id"].unique())
    for p in plants[1:]:
        data[f"pl_{p}"] = (df["plant_id"] == p).astype(float).values
        cols.append(f"pl_{p}")
    yc = (df["year"].values - 2015.5) / 10.0
    data["t1"] = yc
    data["t2"] = yc ** 2
    data["t3"] = yc ** 3
    cols += ["t1", "t2", "t3"]

    # plant-specific linear trend: captures each plant's own trajectory,
    # which is what interpolation does implicitly. Key to competing with persistence.
    if plant_trend:
        for p in plants:
            col = f"tr_{p}"
            data[col] = np.where(df["plant_id"].values == p, yc, 0.0)
            cols.append(col)

    X = np.column_stack([data[c] for c in cols])
    return X, cols, cols.index("log_emp"), df
