#!/usr/bin/env python3
"""Generate docs/panel_flowchart.png - how the plant panel was built, end to end.

The companion to make_flowchart.py. That one opens up a single stage, the merge;
this one is the whole chain at one level up, from the four collection passes to
the completed employment grid the results rest on, and it is the diagram meant
for the main text.

Deterministic; re-run to regenerate.  python docs/make_panel_flowchart.py

Every count is one the shipped code produces on the frozen inputs.
"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "panel_flowchart.png"
C = {"src": "#1f77b4", "auto": "#C0392B", "human": "#E08A1E", "out": "#2E7D32",
     "cfg": "#5D6D7E", "chip": "#8E44AD", "note": "#7F8C8D"}

fig, ax = plt.subplots(figsize=(14, 14.6))
ax.set_xlim(0, 14); ax.set_ylim(7.5, 44); ax.axis("off")


def box(x, y, w, h, text, color, fs=10, bold=False, align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.08,rounding_size=0.15",
                 linewidth=1.7, edgecolor=color, facecolor=color + "18"))
    ha = {"center": "center", "left": "left"}[align]
    tx = x + w / 2 if align == "center" else x + 0.25
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#1a1a1a")


def chip(x, y, text, w=2.6, h=0.78, color=None):
    color = color or C["chip"]
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.04,rounding_size=0.1",
                 linewidth=1.2, edgecolor=color, facecolor=color + "14"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=7.6, color="#1a1a1a")


def tag(x, y, text, w=3.2):
    ax.add_patch(FancyBboxPatch((x, y), w, 0.6,
                 boxstyle="round,pad=0.04,rounding_size=0.08",
                 linewidth=1.0, edgecolor=C["cfg"], facecolor=C["cfg"] + "12"))
    ax.text(x + 0.15, y + 0.3, text, ha="left", va="center", fontsize=7.4,
            color=C["cfg"], family="monospace")


def arrow(x, y0, y1, lab=None, side="right"):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>",
                 mutation_scale=15, linewidth=1.6, color="#555"))
    if lab:
        dx, ha = (0.25, "left") if side == "right" else (-0.25, "right")
        ax.text(x + dx, (y0 + y1) / 2, lab, ha=ha, va="center", fontsize=8.5,
                color=C["auto"], fontweight="bold")


CX = 5.6
ax.text(7, 43.1, "Building the plant panel", ha="center", fontsize=18,
        fontweight="bold")
ax.text(7, 42.3, "four collection passes  →  one value per plant × year  →  a "
        "completed grid that carries its own uncertainty",
        ha="center", fontsize=9.5, color="#555")

# ---------- COLLECTION -------------------------------------------------------
box(0.4, 37.2, 13.2, 4.2, "", C["src"])
ax.text(7.0, 40.9, "1 · COLLECTION   (four passes, each added because the "
        "previous one ran out of material)",
        ha="center", fontsize=9.6, fontweight="bold", color=C["src"])
passes = [
    "PASS 1 · archived plant pages\nInternet Archive snapshots,\none per year, read by\nper-brand patterns",
    "PASS 2 · + a language model\nbehind the patterns\njudges plant vs group figure,\nreads pages no pattern fits",
    "PASS 3 · Wikipedia\nand its revision history\nadds coverage; often\nattached to the wrong year",
    "PASS 4 · the local press\nsearch API in the local\nlanguage, model reads the page,\ncode verifies every row",
]
for i, t in enumerate(passes):
    box(0.75 + i * 3.25, 38.05, 3.0, 2.55, t, C["src"], 7.9)
ax.text(7.0, 37.62, "each scraper run three times and the runs unioned — one "
        "pass returns a sample of what is retrievable, not all of it   ·   "
        "coverage 40% → 56%",
        ha="center", fontsize=7.8, color="#555", style="italic")
arrow(CX, 37.2, 35.9)

# ---------- MERGE ------------------------------------------------------------
box(1.2, 31.4, 8.8, 4.4, "", C["auto"])
ax.text(5.6, 35.3, "2 · MERGE   (deterministic; detailed in the appendix)",
        ha="center", fontsize=9.6, fontweight="bold", color=C["auto"])
steps = ["contamination\nfilters", "outlier\nrule", "manual\nblocklist",
         "consolidation\norder", "Wikipedia fills\nempty cells", "manual\nedits"]
for i, t in enumerate(steps):
    cx = 1.55 + (i % 3) * 2.85
    cy = 33.5 - (i // 3) * 1.15
    chip(cx, cy, t, w=2.6, h=0.95,
         color=C["human"] if "manual" in t else C["auto"])
ax.text(5.6, 31.85, "keyed to the year the figure DESCRIBES, not the year the "
        "page was captured   ·   15 source files, 2,129 candidate values",
        ha="center", fontsize=7.8, color="#555", style="italic")
tag(10.4, 33.6, "config/  (38 curated rows)")
tag(10.4, 32.7, "src/pipeline.py")
arrow(CX, 31.4, 30.4)

box(2.4, 28.4, 6.4, 1.9,
    "374 values · 40 plants · 2006–2025\n"
    "414 observations once carried values are counted separately, 85 of them carried\n"
    "(carried only where a later snapshot reports the identical number)",
    C["out"], 8.6, bold=False)
arrow(CX, 28.4, 27.4)

# ---------- REGISTRY ---------------------------------------------------------
box(1.2, 24.2, 8.8, 3.2,
    "3 · JOIN TO THE PLANT REGISTRY\n"
    "89 plants: name, city, group, brand, assembly or component, coordinates, NUTS region\n"
    "76 of them carry at least one employment figure — 988 plant–year values, "
    "12 countries, 8 groups\n\n"
    "two measurement bases, never mixed silently: 568 plant headcounts and\n"
    "420 French commune totals read from URSSAF, each row flagged with which it is",
    C["src"], 8.6)
tag(10.4, 25.6, "tbl_plants · tbl_employment")
arrow(CX, 24.2, 23.2, "35 plants of the\nthree German groups", side="right")

# ---------- IMPUTATION -------------------------------------------------------
box(1.2, 18.4, 8.8, 4.8, "", C["auto"])
ax.text(5.6, 22.7, "4 · FILLING THE GAPS   (multiple imputation)",
        ha="center", fontsize=9.6, fontweight="bold", color=C["auto"])
ax.text(5.6, 22.15, "about half the grid was empty, and not at random: 85–90% "
        "coverage for 2022–2025 against 25–40% before 2015",
        ha="center", fontsize=7.9, color="#555", style="italic")

box(1.6, 20.1, 3.95, 1.75,
    "METHOD\nwithin-plant multiple imputation,\nchosen by masking 15% of the\nobserved values: matrix completion\nand joint MVN both did worse",
    C["auto"], 7.9)
box(5.7, 20.1, 3.95, 1.75,
    "SCOPE AND DRAWS\n40 → 35 plants, dropping five with\nfewer than five observed years\nM = 50 draws, so every filled\nvalue carries an interval",
    C["auto"], 7.9)
box(1.6, 18.75, 8.05, 1.05,
    "the line reported in every figure is the point estimate; the band around it is the spread of the 50 draws",
    C["auto"], 8.0)
tag(10.4, 20.9, "01_code/imputation/")
tag(10.4, 20.0, "imp_01..50.csv")
arrow(CX, 18.4, 17.4)

box(2.0, 14.9, 7.2, 2.4,
    "THE COMPLETED GRID\n35 plants × 20 years = 700 cells\n"
    "317 observed · 203 filled inside a series · 105 filled at an edge\n"
    "75 outside the plant's observed span, left empty and never read as zero\n"
    "49% of the live cells are inference rather than a source",
    C["out"], 9, bold=False)

# ---------- reproducibility note --------------------------------------------
box(0.5, 11.4, 13.0, 2.6,
    "WHAT REPRODUCES, AND WHAT DOES NOT   —   stages 2 to 4 are deterministic: given the frozen raw files, the merge returns the same\n"
    "374 values every time, and the imputation returns the same central estimate to the last digit (the interval is drawn, so it moves slightly).\n"
    "Stage 1 does not reproduce, by nature. The Internet Archive returns a different sample of snapshots on each run and the language models are\n"
    "hosted services whose behaviour changes without notice, so the raw extractions are frozen and treated as the starting point. Every value in\n"
    "the panel keeps its source, its exact quotation and its URL, and every dropped or hand-added value is logged with its reason.",
    C["note"], 8.2, align="left")

# ---------- legend -----------------------------------------------------------
ax.text(0.5, 10.2, "Legend", fontsize=10, fontweight="bold")
leg = [("data source", C["src"]), ("automatic (code)", C["auto"]),
       ("human judgment (versioned config)", C["human"]),
       ("output", C["out"]), ("file", C["cfg"])]
for i, (lab, col) in enumerate(leg):
    x = 0.5 + (i % 3) * 4.5
    y = 9.2 - (i // 3) * 0.8
    ax.add_patch(FancyBboxPatch((x, y), 0.35, 0.35, boxstyle="round,pad=0.02",
                 edgecolor=col, facecolor=col + "18", linewidth=1.3))
    ax.text(x + 0.55, y + 0.17, lab, fontsize=8.2, va="center")

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print("wrote", OUT)
