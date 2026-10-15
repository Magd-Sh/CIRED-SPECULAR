# -*- coding: utf-8 -*-
"""
Rubin's rules - pooling regression estimates across the M imputed panels.

To use at the final stage (regressions employment ~ vehicle size + controls):
run the SAME regression on each of the M panels, then call rubin_pool() on the lists of
coefficients and variances. The combined standard error incorporates the imputation
uncertainty (BETWEEN-panel variance) - that is the whole point of multiple imputation.

Example (with statsmodels/sklearn or a homemade OLS):
    betas, vars_ = [], []
    for m in range(1, M+1):
        d = pd.read_csv(f"outputs/imputations/imp_{m:02d}.csv")
        b, v = my_ols(d)          # b: (k,), v: (k,) coefficient variances
        betas.append(b); vars_.append(v)
    res = rubin_pool(np.array(betas), np.array(vars_))
"""
import numpy as np


def rubin_pool(betas, variances):
    """
    betas      : (M, k) coefficients per panel
    variances  : (M, k) variances (SE^2) of the coefficients per panel
    Returns dict: estimate, se, within_var, between_var, total_var, df, t, ci_lo, ci_hi.
    """
    betas = np.asarray(betas, float)
    variances = np.asarray(variances, float)
    M = betas.shape[0]
    qbar = betas.mean(0)                          # combined estimate
    ubar = variances.mean(0)                      # within variance
    B = betas.var(0, ddof=1)                      # between variance
    T = ubar + (1 + 1 / M) * B                    # total variance
    se = np.sqrt(T)
    # degrees of freedom (Rubin 1987)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (1 + 1 / M) * B / ubar
        df = (M - 1) * (1 + 1 / r) ** 2
    df = np.where(np.isfinite(df), df, 1e6)
    t = qbar / se
    return {
        "estimate": qbar, "se": se,
        "within_var": ubar, "between_var": B, "total_var": T,
        "df": df, "t": t,
        "ci_lo": qbar - 1.96 * se, "ci_hi": qbar + 1.96 * se,
        "frac_missing_info": (r + 2 / (df + 3)) / (r + 1),
    }
