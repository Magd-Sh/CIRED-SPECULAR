#!/usr/bin/env python3
"""Source-reliability scorecard + matrix + scatter, computed uniformly from the
pipeline's own candidate pool. Run AFTER pipeline.py:
    python figures/make_reliability.py

For each raw source it measures, against the final consolidated value:
  - false positives = share of its values dropped as contamination or outliers
  - disagreement    = on cells seen by >=2 sources, how often it differs from consensus
  - adoption        = how often its value becomes the final value
Outputs figures/reliability_scorecard.csv, reliability_matrix.png, reliability_scatter.png.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
import pipeline as P

# ---- build the full candidate pool (non-wiki + the single wiki) with flags ----
C = P.load_all(); C["reason"] = P.contamination_mask(C)
W = P.load_source(P.REG["wikipedia"]["label"], P.REG["wikipedia"]["file"], "wikipedia", 99)
W["place_year"] = W["ref"]; W["reason"] = P.wiki_contamination_mask(W)
pool = pd.concat([C, W], ignore_index=True)
pool["is_contam"] = pool["reason"].ne("")
pool["is_out"]    = P.outlier_mask(pool) & ~pool["is_contam"]
pool["is_block"]  = P.block_mask(pool) & ~pool["is_contam"] & ~pool["is_out"]
pool["clean"]     = ~(pool["is_contam"] | pool["is_out"] | pool["is_block"])
pool["yr"]        = pd.to_numeric(pool["place_year"], errors="coerce")

# ---- final consolidated value per (plant, year) ----
V = P.main()
cons = {(r.plant_id, int(r.year)): int(r.employees) for r in V.itertuples()}
nsrc = pool[pool["clean"]].groupby(["plant_id","yr"])["source"].transform("nunique")
pool["multi"] = False; pool.loc[pool["clean"].index, "multi"] = False
pool.loc[nsrc.index, "multi"] = nsrc.ge(2)

fam = {**{s["label"]:"archived" for s in P.REG["archived"]},
       **{s["label"]:"press" for s in P.REG["press"]}, "wiki":"wikipedia"}
rows = []
for src, g in pool.groupby("source"):
    n = len(g)
    contam = int((g["is_contam"] | g["is_block"]).sum()); out = int(g["is_out"].sum())
    cl = g[g["clean"]]
    inc = cl[[ (p,int(y)) in cons for p,y in zip(cl["plant_id"], cl["yr"]) ]] if len(cl) else cl
    adopt = np.mean([cons.get((r.plant_id,int(r.yr)))==int(r.emp) for r in inc.itertuples()])*100 if len(inc) else np.nan
    dg = cl[cl["multi"]]; dg = dg[[ (p,int(y)) in cons for p,y in zip(dg["plant_id"], dg["yr"]) ]]
    disag = np.mean([cons.get((r.plant_id,int(r.yr)))!=int(r.emp) for r in dg.itertuples()])*100 if len(dg) else np.nan
    rows.append({"source":("wikipedia" if src=="wiki" else src), "family":fam.get(src,"?"), "n_values":n,
                 "contamination_pct":round(100*contam/n,1), "outlier_pct":round(100*out/n,1),
                 "false_positive_pct":round(100*(contam+out)/n,1),
                 "disagreement_pct":round(disag,1) if disag==disag else np.nan,
                 "adoption_pct":round(adopt,1) if adopt==adopt else np.nan})
S = pd.DataFrame(rows).sort_values("false_positive_pct").reset_index(drop=True)
S.to_csv(ROOT/"figures"/"reliability_scorecard.csv", index=False)

FAM = {"archived":"#1f77b4","wikipedia":"#2ca02c","press":"#E08A1E"}
# ---- matrix ----
emax = max(S[c].max() for c in ["contamination_pct","outlier_pct","false_positive_pct"])
cols = [("n_values","Volume\n(# values)","vol"),("contamination_pct","Contamination %\n(wrong entity)","err"),
        ("outlier_pct","Outliers %\n(off plant scale)","err"),("false_positive_pct","= False positives %\n(total)","err"),
        ("disagreement_pct","Disagreement %\n(vs consensus)","noise"),("adoption_pct","Adoption %\n(value kept)","adopt")]
cm = {"err":plt.get_cmap("Reds"),"vol":plt.get_cmap("Blues"),"noise":plt.get_cmap("Oranges"),"adopt":plt.get_cmap("Greens")}
fig, ax = plt.subplots(figsize=(12.5, 0.55*len(S)+3))
for j,(col,hdr,kind) in enumerate(cols):
    v = S[col].astype(float).values
    lo,hi = np.nanmin(v), np.nanmax(v)
    for i,x in enumerate(v):
        if np.isnan(x): ax.add_patch(plt.Rectangle((j,i),1,1,color="#F2F2F2")); ax.text(j+.5,i+.5,"—",ha="center",va="center",color="#999"); continue
        norm = x/emax if kind=="err" else ((x-lo)/(hi-lo) if hi>lo else .4)
        ax.add_patch(plt.Rectangle((j,i),1,1,color=cm[kind](0.12+0.75*norm)))
        ax.text(j+.5,i+.5, f"{int(x)}" if col=="n_values" else f"{x:.1f}", ha="center", va="center",
                fontsize=9, color="white" if norm>.6 else "#222", fontweight="bold" if col=="false_positive_pct" else "normal")
for x in (1,4): ax.plot([x,x],[0,len(S)],color="#333",lw=2)
ax.text(2.5,-0.7,"ERRORS  (false positives = contamination + outliers)",ha="center",fontsize=10.5,color="#B00",fontweight="bold")
ax.text(5,-0.7,"QUALITY / USAGE",ha="center",fontsize=10.5,color="#1F6E1F",fontweight="bold")
ax.set_xlim(0,len(cols)); ax.set_ylim(0,len(S))
ax.set_xticks([j+.5 for j in range(len(cols))]); ax.set_xticklabels([c[1] for c in cols],fontsize=9)
ax.set_yticks([i+.5 for i in range(len(S))]); ax.set_yticklabels(S["source"],fontsize=10)
for t,f in zip(ax.get_yticklabels(),S["family"]): t.set_color(FAM.get(f,"#333"))
ax.xaxis.tick_top(); ax.invert_yaxis()
for sp in ax.spines.values(): sp.set_visible(False)
ax.tick_params(length=0)
ax.set_title("Source reliability (raw)\nred = more errors · green = more often kept · label colored by family",fontsize=12,pad=40)
plt.tight_layout(); plt.savefig(ROOT/"figures"/"reliability_matrix.png",dpi=150,bbox_inches="tight",facecolor="white"); plt.close()

# ---- scatter ----
fig, ax = plt.subplots(figsize=(11,7.5))
for f,c in FAM.items():
    s = S[S["family"]==f]; ax.scatter(s["false_positive_pct"],s["disagreement_pct"],s=s["n_values"]*2.2,color=c,alpha=.65,edgecolor="white",linewidth=1,label=f)
for r in S.itertuples():
    if not np.isnan(r.disagreement_pct): ax.annotate(r.source,(r.false_positive_pct,r.disagreement_pct),fontsize=8,xytext=(4,4),textcoords="offset points",color="#333")
ax.set_xlabel("False-positive rate (contamination + outliers, %)"); ax.set_ylabel("Disagreement vs consensus (multi-source cells, %)")
ax.set_title("Reliability: false positives vs noise\n(size = number of values)",fontsize=12,pad=10)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
ax.grid(color="#EEE",lw=0.7); ax.set_axisbelow(True); ax.legend(title="Family",fontsize=9,frameon=False)
ax.text(0.99,0.02,"Bottom-left = more reliable",transform=ax.transAxes,ha="right",fontsize=9,color="#888",style="italic")
plt.tight_layout(); plt.savefig(ROOT/"figures"/"reliability_scatter.png",dpi=150,bbox_inches="tight",facecolor="white"); plt.close()
print("wrote figures/reliability_scorecard.csv, reliability_matrix.png, reliability_scatter.png")
