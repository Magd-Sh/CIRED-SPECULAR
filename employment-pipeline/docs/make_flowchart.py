#!/usr/bin/env python3
"""Generate docs/flowchart.png - a detailed diagram of the merge/filter pipeline.
Deterministic; re-run to regenerate.  python docs/make_flowchart.py

Counts shown are the ones the pipeline actually produces on the frozen data/raw/.
"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "flowchart.png"
C = {"src":"#1f77b4", "auto":"#C0392B", "human":"#E08A1E", "out":"#2E7D32",
     "cfg":"#5D6D7E", "chip":"#8E44AD", "note":"#7F8C8D"}
fig, ax = plt.subplots(figsize=(14, 20)); ax.set_xlim(0, 14); ax.set_ylim(0, 50); ax.axis("off")

def box(x, y, w, h, text, color, fs=10, bold=False, align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                 linewidth=1.7, edgecolor=color, facecolor=color+"18"))
    ha = {"center":"center","left":"left"}[align]
    tx = x+w/2 if align=="center" else x+0.25
    ax.text(tx, y+h/2, text, ha=ha, va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#1a1a1a")

def chip(x, y, text, w=2.15, h=0.62):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.1",
                 linewidth=1.2, edgecolor=C["chip"], facecolor=C["chip"]+"14"))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=7.6, color="#1a1a1a")

def tag(x, y, text):                       # config-file tag on the right rail
    ax.add_patch(FancyBboxPatch((x, y), 3.0, 0.6, boxstyle="round,pad=0.04,rounding_size=0.08",
                 linewidth=1.0, edgecolor=C["cfg"], facecolor=C["cfg"]+"12"))
    ax.text(x+0.15, y+0.3, text, ha="left", va="center", fontsize=7.4, color=C["cfg"], family="monospace")

def arrow(x, y0, y1, lab=None):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=15, linewidth=1.6, color="#555"))
    if lab: ax.text(x-0.25, (y0+y1)/2, lab, ha="right", va="center", fontsize=8.5, color=C["auto"], fontweight="bold")

CX = 5.4
ax.text(7, 49.1, "Employment-merge pipeline", ha="center", fontsize=18, fontweight="bold")
ax.text(7, 48.3, "many noisy plant scrapes  →  one clean value per plant × year (2006–2025)  ·  every value keeps its source, citation and URL",
        ha="center", fontsize=9.5, color="#555")

# ---------- SOURCES ----------
box(0.3, 44.3, 4.0, 3.0,
    "ARCHIVED SCRAPES  (10 files)\nbase · filtered · gemini\nv3_2 · de_v3_1 · de_v2\nmodels/prod ×3 · xlsx_de\n\nkeyed to the REFERENCE year", C["src"], 8.6)
box(4.7, 44.3, 4.0, 3.0,
    "PRESS / GAP-FILL  (4 files)\npress_tavily · established_press\nbackfill · wide\n\n\nkeyed to the YEAR reported", C["src"], 8.6)
box(9.1, 44.3, 4.4, 3.0,
    "WIKIPEDIA  (1 file)\nemployment_from_wikipediaa.csv\nthe single FINAL pass —\nreplaces 6 older ones\n\nkeyed to the REFERENCE year", C["src"], 8.6)
for x0 in (2.3, 6.7): arrow(x0, 44.3, 43.2)
# Wikipedia bypasses the archived/press consolidation and is placed at step 6
ax.add_patch(FancyArrowPatch((11.3, 44.3), (9.35, 21.5), arrowstyle="-|>", mutation_scale=14,
             linewidth=1.5, color="#2ca02c", linestyle=(0,(4,3)),
             connectionstyle="arc3,rad=-0.12"))
ax.text(12.4, 33, "held back →\nplaced at step 6", ha="center", va="center",
        fontsize=7.8, color="#2ca02c")

# ---------- 1 LOAD ----------
box(2.0, 41.4, 7.0, 1.6,
    "1 · LOAD → one long candidate table\n2,028 archived+press candidates  ·  101 Wikipedia rows\ncolumns: plant · snapshot yr · reference yr · value · source · citation · URL", C["auto"], 8.8)
arrow(CX, 41.4, 40.6)

# ---------- 2 CONTAMINATION ----------
box(0.8, 34.3, 9.2, 6.1, "", C["auto"])
ax.text(5.4, 39.9, "2 · CONTAMINATION FILTERS   (rule-based text checks — no LLM)", ha="center", fontsize=9.6, fontweight="bold", color=C["auto"])
chips = ["cross-plant duplicate\n(Wikipedia only)","wrong-entity keyword\n(Kamenz, Opel, Hannover…)","city-mismatch\n(Munich group figure)",
         "production-units\n(N Einheiten / vehicles)","job-announcement\n(“will create N jobs”)","historical\n(19th-century figure)]
for i,txt in enumerate(chips):
    cx = 1.15 + (i % 4)*2.25; cy = 37.2 - (i//4)*1.55
    chip(cx, cy, txt)
ax.text(5.4, 34.75, "every dropped row logged → outputs/contamination_removed.csv", ha="center", fontsize=7.8, color="#555", style="italic")
tag(10.4, 37.0, "contamination_rules.json")
arrow(CX, 34.3, 33.4, "−54  /  −45 wiki")

# ---------- 3 OUTLIER ----------
box(2.0, 31.4, 7.0, 2.0,
    "3 · OUTLIER RULE   (per plant, ≥3 values)\nflag if  > 3.5× or < 1/3.5× the plant median   AND   > 10,000 apart\n(absolute floor keeps real swings at small plants)", C["auto"], 8.7)
tag(10.4, 32.1, "parameters.json")
arrow(CX, 31.4, 30.5, "−33")

# ---------- 4 BLOCKLIST ----------
box(2.0, 29.0, 7.0, 1.5,
    "4 · MANUAL BLOCKLIST   (human judgment)\ncurated wrong values no rule catches — combined totals, sub-units, boilerplate", C["human"], 8.7)
tag(10.4, 29.4, "manual_blocks.csv  (14)")
arrow(CX, 29.0, 28.1, "−13")

# ---------- 5 CONSOLIDATE ----------
box(1.4, 23.2, 8.2, 4.6, "", C["auto"])
ax.text(5.4, 27.3, "5 · CONSOLIDATE → one value per (plant, snapshot year)", ha="center", fontsize=9.6, fontweight="bold", color=C["auto"])
box(1.9, 25.4, 2.3, 1.2, "≥2 archived\nsources AGREE", C["auto"], 8.0)
box(4.55, 25.4, 2.3, 1.2, "else archived\nby PRIORITY", C["auto"], 8.0)
box(7.2, 25.4, 2.3, 1.2, "else a\nPRESS value", C["auto"], 8.0)
ax.add_patch(FancyArrowPatch((4.2,26.0),(4.55,26.0), arrowstyle="-|>", mutation_scale=11, color="#888"))
ax.add_patch(FancyArrowPatch((6.85,26.0),(7.2,26.0), arrowstyle="-|>", mutation_scale=11, color="#888"))
box(2.6, 23.6, 6.4, 1.1, "then curated + hand-typed press figures fill still-empty cells\n(non-archived never overrides an archived value)", C["auto"], 7.9)
tag(10.4, 25.8, "parameters.json")
tag(10.4, 25.0, "curated_additions.csv (4)")
tag(10.4, 24.2, "manual_additions.csv (16)")
arrow(CX, 23.2, 22.3)

# ---------- 6 WIKI FILL ----------
box(2.0, 20.7, 7.0, 1.6,
    "6 · WIKIPEDIA FILL   (final file only)\nkept 56 of 101 rows · placed on the reference year\nfills only cells still empty  →  +32 values", C["src"], 8.7)
tag(10.4, 21.2, "employment_from_wikipediaa.csv")
arrow(CX, 20.7, 19.8, "+32")

# ---------- 7 MANUAL EDITS ----------
box(2.0, 18.3, 7.0, 1.5,
    "7 · MANUAL EDITS   (human judgment)\nremove BMW Berlin, Irlbach-Straßkirchen, Debrecen · drop Sindelfingen 2011", C["human"], 8.7)
tag(10.4, 18.7, "manual_edits.csv  (4)")
arrow(CX, 18.3, 17.4)

# ---------- FINAL ----------
box(2.6, 15.4, 5.6, 1.8, "FINAL DATASET\n374 values · 40 plants · 2006–2025\noutputs/employment_final.csv", C["out"], 10, bold=True)
arrow(CX, 15.4, 14.5)

# ---------- 8 STABILITY ----------
box(1.6, 12.4, 8.0, 2.1,
    "8 · STABILITY DERIVATION\nfresh value at each reference year, PLUS carry a value forward\nONLY if a later snapshot reports the identical number\n(→ evidence employment was unchanged)", C["auto"], 8.7)
arrow(CX, 12.4, 11.5)

box(2.2, 9.4, 6.8, 2.0,
    "employment_observations_stability.csv\n414 observations  (85 carried-forward)\n+ full 2006–2025 time-series grid", C["out"], 9, bold=True)

# ---------- reproducibility note ----------
box(0.5, 6.0, 13.0, 2.4,
    "REPRODUCIBILITY   —   everything from step 1 down is deterministic code driven entirely by the files in config/.\n"
    "Given the frozen data/raw/, `python src/pipeline.py` reproduces every output, and each dropped/added value is logged with its reason.\n"
    "NOT reproducible (and documented as such): the upstream extraction of the raw numbers from web pages by the Mistral LLM, which is\n"
    "non-deterministic — so those raw CSVs are frozen and treated as the starting point.", C["note"], 8.4, align="left")

# ---------- legend ----------
ax.text(0.5, 4.6, "Legend", fontsize=10, fontweight="bold")
leg = [("data source",C["src"]),("automatic (code)",C["auto"]),("human judgment (versioned config)",C["human"]),
       ("output",C["out"]),("filter rule",C["chip"]),("config file",C["cfg"])]
for i,(lab,col) in enumerate(leg):
    x = 0.5 + (i % 3)*4.5; y = 3.6 - (i//3)*0.8
    ax.add_patch(FancyBboxPatch((x, y), 0.35, 0.35, boxstyle="round,pad=0.02", edgecolor=col, facecolor=col+"18", linewidth=1.3))
    ax.text(x+0.55, y+0.17, lab, fontsize=8.2, va="center")

plt.tight_layout(); plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white"); plt.close()
print("wrote", OUT)
