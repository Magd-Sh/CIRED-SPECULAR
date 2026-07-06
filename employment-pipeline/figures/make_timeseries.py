#!/usr/bin/env python3
"""Employment time series per group, from the pipeline's stability time-series grid.
Run AFTER pipeline.py:  python figures/make_timeseries.py
Outputs figures/timeseries_<group>.png  (one line per plant; carried-forward
observations, i.e. a value confirmed unchanged by a later snapshot, are ringed)."""
from pathlib import Path
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
def num(s): return pd.to_numeric(s, errors="coerce")

G = pd.read_csv(ROOT/"outputs"/"employment_timeseries_full.csv")
G["emp"] = num(G["employees"]); G["year"] = num(G["obs_year"]); G["carr"] = num(G["carried_years"]).fillna(0) > 0
G = G[G["emp"].notna() & G["year"].between(2006, 2025)]

GROUPS = {"Volkswagen Group":"volkswagen", "Mercedes-Benz":"mercedes", "BMW Group":"bmw"}
meta = G.drop_duplicates("plant_id").set_index("plant_id")
size = G.groupby("plant_id")["emp"].median()
YEARS = list(range(2006, 2026))

for grp, tag in GROUPS.items():
    d = G[G["group"] == grp]
    plants = sorted(d["plant_id"].unique(), key=lambda p: size.get(p, 0), reverse=True)
    cmap = plt.get_cmap("tab20"); fig, ax = plt.subplots(figsize=(12.5, 7))
    for k, p in enumerate(plants):
        s = d[d["plant_id"] == p].sort_values("year"); c = cmap(k % 20)
        ax.plot(s["year"], s["emp"], lw=1.8, color=c, label=meta.loc[p, "plant_name"], zorder=2)
        fr, ca = s[~s["carr"]], s[s["carr"]]
        ax.scatter(fr["year"], fr["emp"], s=24, color=c, zorder=3)
        ax.scatter(ca["year"], ca["emp"], s=30, color=c, edgecolor="#E8820C", linewidth=1.5, zorder=4)
    ax.set_xticks(YEARS); ax.set_xticklabels(YEARS, rotation=90, fontsize=8.5)
    ax.set_xlim(2005.5, 2025.5); ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylabel("Employees"); ax.set_xlabel("Year")
    ax.set_title(f"{grp} — employment over time", fontsize=12, pad=10)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#EEEEEE", lw=0.7); ax.set_axisbelow(True)
    h, _ = ax.get_legend_handles_labels()
    h.append(Line2D([0],[0], marker="o", color="w", markerfacecolor="#888",
                    markeredgecolor="#E8820C", markeredgewidth=1.5, markersize=8, label="carried (stability)"))
    ax.legend(handles=h, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    plt.tight_layout(); plt.savefig(ROOT/"figures"/f"timeseries_{tag}.png", dpi=150, bbox_inches="tight", facecolor="white"); plt.close()
    print("wrote", f"figures/timeseries_{tag}.png")
