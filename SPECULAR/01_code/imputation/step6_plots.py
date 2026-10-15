# -*- coding: utf-8 -*-
"""
STEP 6 - Time-series plots: before (simple interpolation) vs after (multiple imputation).

For each assembly plant, one panel showing:
  - observed employment points (black dots)
  - simple within-plant interpolation (dashed grey line)   [the "before"]
  - multiple-imputation estimate (blue line) + 95% band     [the "after"]
  - imputed points coloured by status (internal / edge / low-coverage)

Outputs:
  outputs/plots/employment_grid.png       (all 26 plants, small multiples)
  outputs/plots/by_plant/<plant>.png      (one file per plant)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from features import active_span

import sys
PANEL = sys.argv[1] if len(sys.argv) > 1 else "panel_assembly.csv"
OUTBASE = sys.argv[2] if len(sys.argv) > 2 else "outputs"
SUMMARY = os.path.join(OUTBASE, "employment_completed_summary.csv")
OUTDIR = os.path.join(OUTBASE, "plots")
os.makedirs(os.path.join(OUTDIR, "by_plant"), exist_ok=True)

panel = active_span(pd.read_csv(PANEL))
summ = pd.read_csv(SUMMARY)

STATUS_COLOR = {
    "imputed_internal": "#1f77b4",
    "imputed_edge": "#d62728",
    "imputed_low_coverage": "#ff7f0e",
}


def interp_series(sub):
    """Simple within-plant interpolation of employment (the 'before')."""
    s = sub.set_index("year")["employees"].interpolate(limit_direction="both")
    return s


def plot_plant(ax, plant):
    p = panel[panel.plant_id == plant].sort_values("year")
    s = summ[summ.plant_id == plant].sort_values("year")
    years = p["year"].values

    # active span only (leave inactive years blank)
    act = p[p.active]
    if len(act) == 0:
        ax.set_visible(False)
        return
    y0, y1 = act["year"].min(), act["year"].max()

    # BEFORE: simple interpolation over the active span
    isr = interp_series(act)
    ax.plot(isr.index, isr.values, ls="--", color="#555555", lw=1.5, zorder=2)

    # AFTER: imputation estimate + 95% band
    sa = s[s["status"] != "inactive"].sort_values("year")
    ax.fill_between(sa["year"], sa["employees_lo95"], sa["employees_hi95"],
                    color="#1f77b4", alpha=0.15, zorder=1)
    ax.plot(sa["year"], sa["employees_est"], color="#1f77b4", lw=1.6, zorder=3)

    # imputed points coloured by status
    imp = sa[sa["imputed"]]
    for st, col in STATUS_COLOR.items():
        pts = imp[imp["status"] == st]
        ax.scatter(pts["year"], pts["employees_est"], s=18, color=col, zorder=4)

    # observed points on top
    obs = sa[~sa["imputed"]]
    ax.scatter(obs["year"], obs["employees_obs"], s=22, color="black", zorder=5)

    ax.set_title(plant.split("_", 1)[1].replace("_", " "), fontsize=8)
    ax.set_xlim(2005.5, 2025.5)
    ax.set_xticks([2006, 2010, 2015, 2020, 2025])
    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.2)


# only plants that were actually imputed (present in the summary)
plants = sorted(set(summ.plant_id.unique()) & set(panel[panel.active].plant_id.unique()))

# ---- grid of small multiples
ncol = 5
nrow = int(np.ceil(len(plants) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.7, nrow * 2.0))
axes = axes.flatten()
for ax in axes[len(plants):]:
    ax.set_visible(False)
for ax, plant in zip(axes, plants):
    plot_plant(ax, plant)

legend = [
    Line2D([0], [0], color="black", marker="o", ls="", ms=5, label="observed"),
    Line2D([0], [0], color="#555555", ls="--", lw=1.5, label="before: simple interpolation"),
    Line2D([0], [0], color="#1f77b4", lw=1.6, label="after: imputation (central estimate)"),
    Line2D([0], [0], color="#1f77b4", alpha=0.25, lw=8, label="95% imputation band"),
    Line2D([0], [0], color="#1f77b4", marker="o", ls="", ms=5, label="imputed (internal)"),
    Line2D([0], [0], color="#d62728", marker="o", ls="", ms=5, label="imputed (edge)"),
    Line2D([0], [0], color="#ff7f0e", marker="o", ls="", ms=5, label="imputed (low coverage)"),
]
fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=8, frameon=False,
           bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Plant employment: simple interpolation vs multiple imputation (2006-2025)",
             fontsize=12, y=1.0)
fig.tight_layout(rect=[0, 0.03, 1, 0.99])
fig.savefig(os.path.join(OUTDIR, "employment_grid.png"), dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"[6] grid written: {OUTDIR}/employment_grid.png")

# ---- one file per plant
for plant in plants:
    fig, ax = plt.subplots(figsize=(6, 3.4))
    plot_plant(ax, plant)
    ax.set_ylabel("employees")
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "by_plant", f"{plant}.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
print(f"[6] {len(plants)} per-plant plots written: {OUTDIR}/by_plant/")
