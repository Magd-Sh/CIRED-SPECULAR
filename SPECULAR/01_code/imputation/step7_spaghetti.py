# -*- coding: utf-8 -*-
"""
STEP 7 - Spaghetti plots: all M imputation draws as thin lines, one figure per group.

For each plant, one panel showing:
  - the M=10 completed panels, each as a thin light line (the imputation draws)
  - observed employment points (black dots) on top
Only active-span years are drawn (imputation files already contain active rows only).

Outputs:
  outputs/plots/spaghetti_<group>.png   (one per manufacturer group)
"""
import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys
OUTBASE = sys.argv[1] if len(sys.argv) > 1 else "outputs"
IMPDIR = os.path.join(OUTBASE, "imputations")
OUTDIR = os.path.join(OUTBASE, "plots")
os.makedirs(OUTDIR, exist_ok=True)

# load the M completed panels, tag each with its draw index
files = sorted(glob.glob(os.path.join(IMPDIR, "imp_*.csv")))
draws = []
for k, f in enumerate(files, 1):
    d = pd.read_csv(f)
    d["draw"] = k
    draws.append(d)
alld = pd.concat(draws, ignore_index=True)
M = len(files)
print(f"[7] loaded {M} imputation draws")

GROUP_FILE = {"Volkswagen Group": "VW", "Mercedes-Benz": "MERCEDES", "BMW Group": "BMW"}


def plot_group(group):
    sub = alld[alld.group == group]
    plants = sorted(sub.plant_id.unique())
    ncol = 4
    nrow = int(np.ceil(len(plants) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.0, nrow * 2.3), squeeze=False)
    axes = axes.flatten()
    for ax in axes[len(plants):]:
        ax.set_visible(False)

    for ax, plant in zip(axes, plants):
        p = sub[sub.plant_id == plant]
        # each draw as a thin translucent line
        alpha = max(0.10, min(0.35, 3.5 / M))       # thinner when many draws
        for k in range(1, M + 1):
            dk = p[p.draw == k].sort_values("year")
            ax.plot(dk["year"], dk["employees_imputed"],
                    color="#1f77b4", lw=0.6, alpha=alpha, zorder=2)
        # observed points at their RECORDED value (low-reliability ones vary slightly across draws)
        d1 = p[p.draw == 1].sort_values("year")
        obs = d1[d1["was_observed"]]
        yobs = obs["employees_recorded"] if "employees_recorded" in obs else obs["employees_imputed"]
        ax.scatter(obs["year"], yobs, s=20, color="black", zorder=5)

        ax.set_title(plant.split("_", 1)[1].replace("_", " "), fontsize=8)
        ax.set_xlim(2005.5, 2025.5)
        ax.set_xticks([2006, 2010, 2015, 2020, 2025])
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

    legend = [
        Line2D([0], [0], color="black", marker="o", ls="", ms=5, label="observed"),
        Line2D([0], [0], color="#1f77b4", lw=1.2, alpha=0.5,
               label=f"{M} imputation draws (thin lines)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Employment - {M} imputation draws per plant - {group} (2006-2025)",
                 fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0.04, 1, 0.99])
    out = os.path.join(OUTDIR, f"spaghetti_{GROUP_FILE[group]}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[7] written: {out}  ({len(plants)} plants)")


for g in GROUP_FILE:
    plot_group(g)
