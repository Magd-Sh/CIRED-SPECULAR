# -*- coding: utf-8 -*-
"""
Tests de validation par masquage (slide 5 du Point Hebdo 11).

Principe : on cache des cellules OBSERVÉES de la BDD emploi, on les prédit
avec chaque méthode, puis on compare aux vraies valeurs (échelle log).

Trois designs de masquage :
  A) 15 % de cellules observées au hasard            -> mime les trous internes
  B) 40 % des cellules observées de 2006-2015        -> mime notre missingness réelle
  C) effacer TOUT le début (2006-2013) de 12 usines   -> mime les 209 trous de bord
     (aucun point d'ancrage à gauche : test d'extrapolation pure)

Méthodes comparées :
  - moyenne de l'usine
  - interpolation linéaire intra-usine (+ report à plat sur les bords)
  - effets fixes usine + année
  - MC-NNM façon Athey et al. 2021 (soft-impute + effets fixes non régularisés)
  - (design C) report arrière ancré sur le sentier des effets année

Usage : python tests_masquage_imputation.py employment_timeseries_full.xlsx
"""
import sys
import numpy as np
import pandas as pd

PATH = sys.argv[1] if len(sys.argv) > 1 else "employment_timeseries_full.xlsx"

# ------------------------------------------------------------------ données
df = pd.read_excel(PATH)
df = df[df.year >= 2006]                          # on écarte Győr 2001-2005
Y = df.pivot(index="plant_id", columns="year", values="employees")
M = np.log(Y.values)                              # log emploi
obs = ~np.isnan(M)                                # masque des cellules observées
N, T = M.shape
print(f"matrice {N}x{T}, {obs.sum()} cellules observées")


# ------------------------------------------------------------------ méthodes
def plant_mean(Xtr):
    """Moyenne de chaque usine sur ses cellules d'entraînement."""
    pm = np.nanmean(Xtr, 1)
    return np.tile(pm[:, None], (1, T))


def interp(Xtr):
    """Interpolation linéaire intra-usine ; bords = report à plat."""
    P = np.full_like(Xtr, np.nan)
    for i in range(N):
        s = pd.Series(Xtr[i])
        P[i] = s.interpolate(limit_direction="both").values
        if np.isnan(P[i]).all():                  # usine sans aucune obs d'entraînement
            P[i] = np.nanmean(Xtr)
    return P


def two_way_fe(Xtr):
    """mu + alpha_i (effet usine) + delta_t (effet année)."""
    mu = np.nanmean(Xtr)
    a = np.nanmean(Xtr, 1) - mu
    a[np.isnan(a)] = 0
    b = np.nanmean(Xtr - a[:, None], 0) - mu
    b[np.isnan(b)] = 0
    return mu + a[:, None] + b[None, :], a, b, mu


def softimpute(X, train, lam, iters=300):
    """
    MC-NNM simplifié (Athey et al. 2021) :
    itère { SVD -> rétrécir les valeurs singulières de lam -> recompléter },
    avec effets fixes usine/année ré-estimés à chaque passe et NON pénalisés.
    """
    L = np.zeros((N, T))
    g = np.zeros(N)                                # effets usine
    d = np.zeros(T)                                # effets année
    for _ in range(iters):
        # SVD de la matrice courante (obs vraies là où observé, L sinon)
        U, s, Vt = np.linalg.svd(
            np.where(train, X - g[:, None] - d[None, :], L), full_matrices=False
        )
        s = np.maximum(s - lam, 0)                 # pénalité nuclear norm
        L = U @ np.diag(s) @ Vt
        # ré-estimation des effets fixes sur les résidus observés
        r = np.where(train, X - L, np.nan)
        g = np.nan_to_num(np.nanmean(r, 1))
        r2 = np.where(train, X - L - g[:, None], np.nan)
        d = np.nan_to_num(np.nanmean(r2, 0))
    return L + g[:, None] + d[None, :], s


def mcnnm_best(Xtr, train, test, M_true, lams=(0.1, 0.3, 0.5, 1.0, 2.0)):
    """Essaie plusieurs lambda, garde le meilleur sur le test.
    NB : choisir lambda sur le test AVANTAGE la méthode (fait exprès :
    si elle perd quand même, l'argument est plus fort)."""
    best = None
    for lam in lams:
        P, _ = softimpute(np.where(train, M_true, 0), train, lam)
        err = np.sqrt(np.nanmean((P[test] - M_true[test]) ** 2))
        if best is None or err < best[1]:
            best = (lam, err, P)
    return best


def score(P, test):
    """RMSE log + erreur relative moyenne approx. sur les cellules cachées."""
    e = P[test] - M[test]
    return np.sqrt(np.nanmean(e ** 2)), np.nanmean(np.abs(np.exp(e) - 1))


def report(name, preds, test):
    print(f"\n== design {name}  ({test.sum()} cellules cachées)")
    for k, P in preds.items():
        rmse, rel = score(P, test)
        print(f"  {k:32s} RMSE(log)={rmse:.3f}   err. relative ~{rel:.1%}")


# ------------------------------------------------------- designs A et B
rng = np.random.default_rng(0)                    # seed = reproductible
oi = np.argwhere(obs)                             # indices des cellules observées

mask_A = oi[rng.choice(len(oi), size=int(0.15 * len(oi)), replace=False)]
early = oi[oi[:, 1] < 10]                         # colonnes 0-9 = 2006-2015
mask_B = early[rng.choice(len(early), size=int(0.4 * len(early)), replace=False)]

for name, mask in [("A — 15 % au hasard", mask_A), ("B — début de période", mask_B)]:
    test = np.zeros_like(obs)
    test[tuple(mask.T)] = True
    train = obs & ~test
    Xtr = np.where(train, M, np.nan)
    fe, _, _, _ = two_way_fe(Xtr)
    lam, _, mc = mcnnm_best(Xtr, train, test, M)
    report(name, {
        "moyenne usine": plant_mean(Xtr),
        "interpolation intra-usine": interp(Xtr),
        "effets fixes usine+année": fe,
        f"MC-NNM (lam={lam})": mc,
    }, test)

# ------------------------------------------------------- design C (bords)
rng = np.random.default_rng(1)
early_cov = obs[:, :8].sum(1)                     # obs dans 2006-2013
cands = np.where(early_cov >= 3)[0]               # usines assez couvertes tôt
sel = rng.choice(cands, size=min(12, len(cands)), replace=False)

test = np.zeros_like(obs)
for i in sel:
    test[i, :8] = obs[i, :8]                      # on efface TOUT leur début
train = obs & ~test
Xtr = np.where(train, M, np.nan)

fe, a, b, mu = two_way_fe(Xtr)
# report arrière ancré : première obs + sentier des effets année
anchored = interp(Xtr).copy()
for i in sel:
    if train[i].any():
        fo = np.argmax(train[i])                  # première année observée
        anchored[i, :fo] = M[i, fo] + (b[:fo] - b[fo])
lam, _, mc = mcnnm_best(Xtr, train, test, M)
report("C — blocs de bord effacés", {
    "interp + report à plat": interp(Xtr),
    "effets fixes usine+année": fe,
    "ancré + sentier effets année": anchored,
    f"MC-NNM (lam={lam})": mc,
}, test)

# ------------------------------------- diagnostic : structure factorielle
L, s = softimpute(np.where(obs, M, 0), obs, 0.3)
print("\nvaleurs singulières après retrait des effets fixes :", np.round(s[:6], 2))
print("(chute brutale = pas de structure factorielle riche entre usines)")
