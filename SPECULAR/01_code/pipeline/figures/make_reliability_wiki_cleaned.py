#!/usr/bin/env python3
"""Reliability scatter, but with Wikipedia's false positives removed, to test whether
its *clean* values behave like the archived sources.
Run AFTER pipeline.py:  python figures/make_reliability_wiki_cleaned.py
Outputs figures/reliability_scatter_wiki_cleaned.png
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/"src"))
import pipeline as P

C = P.load_all(); C["reason"] = P.contamination_mask(C)
W = P.load_source(P.REG["wikipedia"]["label"], P.REG["wikipedia"]["file"], "wikipedia", 99)
W["place_year"] = W["ref"]; W["reason"] = P.wiki_contamination_mask(W)
pool = pd.concat([C, W], ignore_index=True)
pool["is_contam"] = pool["reason"].ne("")
pool["is_out"] = P.outlier_mask(pool) & ~pool["is_contam"]
pool["is_block"] = P.block_mask(pool) & ~pool["is_contam"] & ~pool["is_out"]
pool["clean"] = ~(pool["is_contam"] | pool["is_out"] | pool["is_block"])
pool["yr"] = pd.to_numeric(pool["place_year"], errors="coerce")
V = P.main(); cons = {(r.plant_id, int(r.year)): int(r.employees) for r in V.itertuples()}
nsrc = pool[pool["clean"]].groupby(["plant_id","yr"])["source"].transform("nunique")
pool["multi"] = False; pool.loc[nsrc.index, "multi"] = nsrc.ge(2)

fam = {**{s["label"]:"archived" for s in P.REG["archived"]},
       **{s["label"]:"press" for s in P.REG["press"]}, "wiki":"wikipedia"}
def metrics(g, drop_fp=False):
    n = len(g); use = g[g["clean"]] if drop_fp else g
    fp = 0.0 if drop_fp else 100*(g["is_contam"]|g["is_block"]|g["is_out"]).sum()/n
    cl = g[g["clean"]]
    dg = cl[cl["multi"]]; dg = dg[[(p,int(y)) in cons for p,y in zip(dg["plant_id"],dg["yr"])]]
    dis = np.mean([cons.get((r.plant_id,int(r.yr)))!=int(r.emp) for r in dg.itertuples()])*100 if len(dg) else np.nan
    return n, round(fp,1), round(dis,1) if dis==dis else np.nan

rows = []
for src, g in pool.groupby("source"):
    n, fp, dis = metrics(g, drop_fp=(src=="wiki"))   # <-- Wikipedia counted WITHOUT its false positives
    rows.append({"source":("wikipedia (clean)" if src=="wiki" else src), "family":fam.get(src,"?"),
                 "n":n, "fp":fp, "dis":dis})
S = pd.DataFrame(rows)
# also keep the raw wikipedia point for the "before" marker
gw = pool[pool["source"]=="wiki"]; n_raw, fp_raw, dis_raw = metrics(gw, drop_fp=False)

FAM = {"archived":"#1f77b4","wikipedia":"#2ca02c","press":"#E08A1E"}
fig, ax = plt.subplots(figsize=(11, 7.5))
for f, c in FAM.items():
    s = S[S["family"] == f]
    ax.scatter(s["fp"], s["dis"], s=s["n"]*2.2, color=c, alpha=.65, edgecolor="white", linewidth=1, label=f)
for r in S.itertuples():
    if not np.isnan(r.dis): ax.annotate(r.source, (r.fp, r.dis), fontsize=8, xytext=(5,4), textcoords="offset points", color="#333")
# raw wikipedia (before) as a faded hollow marker + arrow to the cleaned point
ax.scatter([fp_raw], [dis_raw], s=n_raw*2.2, facecolor="none", edgecolor="#2ca02c", linewidth=1.6, alpha=.7)
ax.annotate("wikipedia (raw)", (fp_raw, dis_raw), fontsize=8, xytext=(5,-12), textcoords="offset points", color="#2ca02c")
wc = S[S["source"]=="wikipedia (clean)"].iloc[0]
ax.add_patch(FancyArrowPatch((fp_raw, dis_raw), (wc.fp, wc.dis), arrowstyle="-|>", mutation_scale=15,
             color="#2ca02c", linewidth=1.4, linestyle=(0,(4,2)), connectionstyle="arc3,rad=0.0"))

ax.set_xlabel("False-positive rate (contamination + outliers, %)")
ax.set_ylabel("Disagreement vs consensus (multi-source cells, %)")
ax.set_title("Reliability with Wikipedia's false positives removed\n"
             "(arrow: raw → clean Wikipedia · size = number of values)", fontsize=12, pad=10)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
ax.grid(color="#EEE", lw=0.7); ax.set_axisbelow(True); ax.legend(title="Family", fontsize=9, frameon=False)
ax.text(0.99, 0.02, "Bottom-left = more reliable", transform=ax.transAxes, ha="right", fontsize=9, color="#888", style="italic")
plt.tight_layout(); plt.savefig(ROOT/"figures"/"reliability_scatter_wiki_cleaned.png", dpi=150, bbox_inches="tight", facecolor="white"); plt.close()
print("wrote figures/reliability_scatter_wiki_cleaned.png")
