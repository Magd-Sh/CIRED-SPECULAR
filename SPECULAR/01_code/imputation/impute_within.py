# -*- coding: utf-8 -*-
"""
Within-plant multiple imputation engine (v3) - empirically adapted to THIS panel.

Why within-plant (documented by the validation, cf. step4):
  The dominant signal is WITHIN-PLANT persistence (employment moves slowly). Joint methods
  (Amelia MVN, matrix completion) were tested and do worse than within-plant interpolation:
  there is little factor structure across plants and the covariates are sparse. So the
  imputation is built AROUND each plant's own trajectory; the common year effect and the
  covariates only shape the path where the plant has no anchor (edges) or, optionally,
  inside gaps (covariate-shaped bridge).

Model for log-employment e_t of a plant (per draw m = 1..M):
  - Increments:  e_t - e_{t-1} = delta_t + slopes . d(covariates)_t + u_t,
    delta_t = common year effect, slopes = covariate effects (ridge on observed consecutive
    pairs, bootstrapped per draw), u_t ~ N(0, s^2) innovation with robust scale s.
  - INTERNAL gap between observed anchors L < j < R:
      deterministic path = linear interpolation (bridge=False) or the covariate/year-shaped
      path pinned to both anchors (bridge=True);
      noise = Brownian bridge with step sd k_int * s  ->  zero at the anchors, largest in the
      middle, growing with gap length.
  - EDGE gap (before first / after last observation):
      deterministic path = walk from the nearest anchor adding the predicted increments;
      noise = random walk with step sd k_edge * s  ->  grows with distance from the anchor.
  - OVER-IMPUTATION (overimpute=True, Blackwell et al. 2015): an observed value with
    reliability r < 1 (rounded wiki figure, carried value) is treated as measured with error:
    in each draw it is replaced by  e_obs + N(0, (oi_scale * (1 - r))^2),  and the gaps next to
    it are filled from the perturbed anchors, so measurement noise propagates.
  - point_estimate(): same deterministic path, recorded anchors, no noise, no bootstrap
    (the plotted central line).

API:
  impute(panel_active, M, seed, covars=..., bridge=..., overimpute=..., k_int, k_edge, oi_scale)
      -> list of M completed dataframes (column log_emp_imp)
  point_estimate(panel_active, covars=..., bridge=...) -> dataframe (column log_emp_point)
  innovation_scale(panel_active, covars=...) -> robust s
"""
import numpy as np
import pandas as pd

COVARS = ["log_prod", "kerb_weight", "seg_score", "ev_share", "log_nmodels"]
YEARS = list(range(2006, 2026))

# Calibrated defaults (masking, 5-10 seeds; see step4b_calibrate.py and README):
# - EDGE_DRIFT = 'none': the edge central path is a flat carry from the nearest observation.
#   Over 10 seeds it beats the walk with year effects + covariates (RMSE early edges 0.178 vs
#   0.192, late edges 0.144 vs 0.167); internal gaps are unaffected (0.151).
# - Uncertainty scales, in units of the robust innovation scale s (calib_check.py, 10 seeds,
#   after the data corrections): K_INT = 3.5 (Brownian bridge in internal gaps), K_EDGE = 1.75
#   (random-walk step at edges) and K_NUG = 3.5 (one-off level noise from the first edge year,
#   for jumps right next to an observation). 95% bands then contain 94.5% of hidden true values in
#   internal gaps, 96.2% at edges 1-2 years from an observation and 93.2% further out. Without the
#   nugget, short edges were covered only ~78-90%; with the raw scale (k = 1) everything ~80%.
# - OI_SCALE = 0.3: measurement sd of an observed value = 0.3 x (1 - reliability), in logs.
K_INT, K_EDGE, OI_SCALE = 3.5, 1.75, 0.3
K_NUG = 3.5              # one-off level noise at the first edge year (x s)
EDGE_DRIFT = "none"      # edge central path: 'full' | 'year' | 'none'


def _precov(df, covars):
    """Pre-impute covariates (within-plant interp, then global mean) so they can serve as predictors."""
    d = df.copy()
    for c in covars:
        if c not in d:
            d[c] = np.nan
        d[c] = d.groupby("plant_id")[c].transform(lambda s: s.interpolate(limit_direction="both"))
        d[c] = d[c].fillna(d[c].mean() if d[c].notna().any() else 0.0)
    return d


def _pairs(d, covars):
    rows = []
    for p, sub in d.sort_values("year").groupby("plant_id"):
        sub = sub.reset_index(drop=True)
        e = sub["log_emp"].values
        C = sub[covars].values
        for i in range(1, len(sub)):
            if np.isfinite(e[i]) and np.isfinite(e[i - 1]):
                rows.append([sub.loc[i, "year"], e[i] - e[i - 1]] + list(C[i] - C[i - 1]))
    return pd.DataFrame(rows, columns=["year", "dy"] + [f"d_{c}" for c in covars])


def _year_effects(d, covars, rng=None, lam=5.0, bootstrap=True):
    """Ridge on observed consecutive pairs: dlog_emp ~ year dummies + dcovariates.
    Returns (delta per year, covariate slopes, robust residual scale)."""
    A = _pairs(d, covars)
    if len(A) < 10:
        return {y: 0.0 for y in YEARS}, np.zeros(len(covars)), 0.05
    B = A.sample(len(A), replace=True, random_state=int(rng.integers(1e9))) if bootstrap else A
    yrs = sorted(A["year"].unique())
    X = lambda T: np.column_stack([(T["year"].values == y).astype(float) for y in yrs] +
                                  [T[f"d_{c}"].values for c in covars])
    Xb = X(B)
    beta = np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ B["dy"].values)
    delta = {y: beta[k] for k, y in enumerate(yrs)}
    mean_d = float(np.mean(list(delta.values())))
    for y in YEARS:
        delta.setdefault(y, mean_d)
    slopes = beta[len(yrs):]
    res = A["dy"].values - X(A) @ beta
    s = 1.4826 * np.median(np.abs(res - np.median(res)))        # robust (MAD) innovation scale
    return delta, slopes, max(float(s), 0.01)


def innovation_scale(panel_active, covars=COVARS):
    d = _precov(panel_active, covars)
    return _year_effects(d, covars, bootstrap=False)[2]


def _fill_plant(e, yr, cov, delta, slopes, bridge, rng=None, s_int=0.0, s_edge=0.0,
                edge_drift=None, s_nug=0.0):
    """Fill the NaNs of one plant's log-employment series e (anchors already set).
    edge_drift: 'full' (year effects + covariates), 'year' (year effects only),
                'none' (flat carry from the anchor = pure persistence)."""
    edge_drift = edge_drift or EDGE_DRIFT
    out = e.copy()
    obs = np.where(np.isfinite(e))[0]
    if len(obs) == 0:
        return out
    n = len(e)
    inc = np.zeros(n)                                            # predicted increment into t
    for t in range(1, n):
        inc[t] = delta.get(yr[t], 0.0) + (cov[t] - cov[t - 1]) @ slopes
    inc_edge = {"full": inc,
                "year": np.array([0.0] + [delta.get(yr[t], 0.0) for t in range(1, n)]),
                "none": np.zeros(n)}[edge_drift]
    # internal gaps
    for a, b in zip(obs[:-1], obs[1:]):
        if b - a < 2:
            continue
        L, R = a, b
        steps = np.arange(L + 1, R + 1)
        cum = np.cumsum(inc[L + 1:R + 1]) if bridge else np.zeros(R - L)
        frac = (steps - L) / (R - L)
        path = e[L] + cum + frac * (e[R] - e[L] - cum[-1])       # pinned to both anchors
        if rng is not None and s_int > 0:
            W = np.cumsum(rng.normal(0, s_int, R - L))
            path = path + (W - frac * W[-1])                     # Brownian bridge
        out[L + 1:R] = path[:-1]
    # edges
    lo, hi = obs.min(), obs.max()
    for anchor, rng_idx, step in [(lo, range(lo - 1, -1, -1), -1), (hi, range(hi + 1, n), 1)]:
        val, noise = e[anchor], 0.0
        if rng is not None and s_nug > 0 and len(rng_idx):
            noise = rng.normal(0, s_nug)        # one-off level uncertainty (jump right after/before the anchor)
        for j in rng_idx:
            k = j if step == 1 else j + 1                        # increment linking j and its neighbour
            val += step * inc_edge[k]
            if rng is not None and s_edge > 0:
                noise += rng.normal(0, s_edge)
            out[j] = val + noise
    return out


def _run(panel_active, covars, bridge, rng=None, bootstrap=False, overimpute=False,
         k_int=None, k_edge=None, oi_scale=None, k_nug=None):
    k_int = K_INT if k_int is None else k_int
    k_edge = K_EDGE if k_edge is None else k_edge
    oi_scale = OI_SCALE if oi_scale is None else oi_scale
    k_nug = K_NUG if k_nug is None else k_nug
    base = _precov(panel_active, covars).sort_values(["plant_id", "year"]).reset_index(drop=True)
    delta, slopes, s = _year_effects(base, covars, rng=rng, bootstrap=bootstrap)
    imp = base["log_emp"].values.astype(float).copy()
    rel = base["emp_reliability"].values if "emp_reliability" in base else np.full(len(base), np.nan)
    for p, sub in base.groupby("plant_id"):
        idx = sub.index.values
        e = imp[idx].copy()
        if overimpute and rng is not None:
            r = np.where(np.isfinite(rel[idx]), rel[idx], 1.0)
            sd = oi_scale * (1.0 - r)
            obs = np.isfinite(e)
            e[obs] = e[obs] + rng.normal(0, 1, obs.sum()) * sd[obs]
        imp[idx] = _fill_plant(e, sub["year"].values, sub[covars].values, delta, slopes, bridge,
                               rng=rng, s_int=k_int * s if rng is not None else 0.0,
                               s_edge=k_edge * s if rng is not None else 0.0,
                               s_nug=k_nug * s if rng is not None else 0.0)
    return base, imp


def impute(panel_active, M=10, seed=0, covars=COVARS, bridge=False, overimpute=True,
           k_int=None, k_edge=None, oi_scale=None, k_nug=None):
    """M completed dataframes (column log_emp_imp)."""
    rng = np.random.default_rng(seed)
    completed = []
    for m in range(M):
        base, imp = _run(panel_active, covars, bridge, rng=rng, bootstrap=True,
                         overimpute=overimpute, k_int=k_int, k_edge=k_edge, oi_scale=oi_scale,
                         k_nug=k_nug)
        out = base.copy()
        out["log_emp_imp"] = imp
        completed.append(out)
    return completed


def point_estimate(panel_active, covars=COVARS, bridge=False):
    """Noiseless central estimate (recorded anchors, no noise, no bootstrap)."""
    base, imp = _run(panel_active, covars, bridge, rng=None, bootstrap=False)
    out = base.copy()
    out["log_emp_point"] = imp
    return out
