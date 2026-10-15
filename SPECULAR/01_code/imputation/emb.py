# -*- coding: utf-8 -*-
"""
EMB engine (Expectation-Maximization with Bootstrapping) - Honaker & King 2010 (Amelia).
Multiple imputation under a multivariate normal, in pure numpy (no heavy dependency).

Idea:
  Variables (analysis + design) are assumed jointly multivariate normal N(mu, Sigma).
  We estimate (mu, Sigma) by EM on incomplete data, then impute each missing cell with a
  draw from its conditional distribution given the observed cells in the same row. To
  propagate estimation uncertainty, we repeat over M bootstrap resamples -> M completed
  datasets (Rubin's rules downstream).

Key functions:
  em_mvn(X)                -> mu, Sigma          (EM on a matrix with NaN)
  impute_draw(X, mu, Sig)  -> completed X (1 conditional draw per missing cell)
  emb(X, M)                -> list of M completed matrices
"""
import numpy as np


def _standardize(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X - mu) / sd, mu, sd


def em_mvn(X, ridge=0.05, max_iter=200, tol=1e-4, verbose=False):
    """
    EM for N(mu, Sigma) with missing values (NaN).
    X : (n, p) already standardized. ridge : regularization added to Sigma's diagonal
        (= Amelia's empirical prior; guarantees Sigma_oo is invertible).
    """
    n, p = X.shape
    obs_mask = ~np.isnan(X)
    # init: mean 0, Sigma = identity
    mu = np.zeros(p)
    Sigma = np.eye(p)

    # pre-group rows by missingness pattern (speeds up: one inversion per pattern)
    patterns = {}
    for i in range(n):
        key = obs_mask[i].tobytes()
        patterns.setdefault(key, []).append(i)

    prev = None
    for it in range(max_iter):
        S1 = np.zeros(p)
        S2 = np.zeros((p, p))
        for key, rows in patterns.items():
            o = np.frombuffer(key, dtype=bool)
            m = ~o
            idx_o = np.where(o)[0]
            idx_m = np.where(m)[0]
            Xr = X[rows][:, idx_o]                       # (k, |o|) observed values
            if idx_m.size == 0:
                Xhat = X[rows]
                S1 += Xhat.sum(0)
                S2 += Xhat.T @ Xhat
                continue
            So = Sigma[np.ix_(idx_o, idx_o)]
            Smo = Sigma[np.ix_(idx_m, idx_o)]
            Smm = Sigma[np.ix_(idx_m, idx_m)]
            inv_So = np.linalg.pinv(So)
            beta = Smo @ inv_So                          # (|m|, |o|)
            cond_cov = Smm - beta @ Smo.T                # E-step correction
            # conditional mean for each row in the group
            resid = Xr - mu[idx_o]                        # (k, |o|)
            cond_mean = mu[idx_m] + resid @ beta.T        # (k, |m|)
            k = len(rows)
            Xhat = np.empty((k, p))
            Xhat[:, idx_o] = Xr
            Xhat[:, idx_m] = cond_mean
            S1 += Xhat.sum(0)
            S2 += Xhat.T @ Xhat
            # add conditional covariance (uncertainty over the missing cells)
            S2[np.ix_(idx_m, idx_m)] += k * cond_cov

        mu = S1 / n
        Sigma = S2 / n - np.outer(mu, mu)
        Sigma[np.diag_indices(p)] += ridge               # regularization
        Sigma = (Sigma + Sigma.T) / 2                    # symmetrize (numerical stability)

        if prev is not None:
            delta = np.max(np.abs(Sigma - prev))
            if verbose and it % 20 == 0:
                print(f"    EM it {it}: delta={delta:.5f}")
            if delta < tol:
                break
        prev = Sigma.copy()
    return mu, Sigma


def impute_draw(X, mu, Sigma, rng):
    """One conditional draw per missing cell (conditional normal law)."""
    n, p = X.shape
    obs_mask = ~np.isnan(X)
    Xc = X.copy()
    for i in range(n):
        o = obs_mask[i]
        m = ~o
        if not m.any():
            continue
        idx_o = np.where(o)[0]
        idx_m = np.where(m)[0]
        So = Sigma[np.ix_(idx_o, idx_o)]
        Smo = Sigma[np.ix_(idx_m, idx_o)]
        Smm = Sigma[np.ix_(idx_m, idx_m)]
        beta = Smo @ np.linalg.pinv(So)
        cond_mean = mu[idx_m] + beta @ (X[i, idx_o] - mu[idx_o])
        cond_cov = Smm - beta @ Smo.T
        cond_cov = (cond_cov + cond_cov.T) / 2 + 1e-8 * np.eye(len(idx_m))
        try:
            L = np.linalg.cholesky(cond_cov)
            draw = cond_mean + L @ rng.standard_normal(len(idx_m))
        except np.linalg.LinAlgError:
            draw = cond_mean                              # non-PD cov: use conditional mean
        Xc[i, idx_m] = draw
    return Xc


def emb(X, M=10, ridge=0.05, seed=0, verbose=False):
    """
    Multiple imputation by EMB.
    Returns a list of M completed matrices (in ORIGINAL, de-standardized space).
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    Xs, mu0, sd0 = _standardize(X)
    completed = []
    for mrep in range(M):
        # bootstrap the rows -> estimate (mu, Sigma) on the resampled sample
        boot = rng.integers(0, n, size=n)
        mu_b, Sig_b = em_mvn(Xs[boot], ridge=ridge, verbose=verbose)
        # impute the missing cells of the ORIGINAL data with a draw
        Xc = impute_draw(Xs, mu_b, Sig_b, rng)
        completed.append(Xc * sd0 + mu0)                 # de-standardize
        if verbose:
            print(f"  [EMB] completed dataset {mrep+1}/{M}")
    return completed
