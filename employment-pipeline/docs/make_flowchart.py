#!/usr/bin/env python3
"""Generate docs/flowchart.png - the merge/filter pipeline as a diagram.
Deterministic; re-run to regenerate.  python docs/make_flowchart.py"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "flowchart.png"
C = {"src":"#1f77b4", "auto":"#C0392B", "human":"#E08A1E", "out":"#2E7D32", "note":"#7F8C8D"}
fig, ax = plt.subplots(figsize=(12, 15)); ax.set_xlim(0, 12); ax.set_ylim(0, 40); ax.axis("off")

def box(x, y, w, h, text, color, fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                 linewidth=1.6, edgecolor=color, facecolor=color+"22"))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#222")

def arrow(x, y0, y1):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=16,
                 linewidth=1.6, color="#555"))

CX = 6
# ---- title
ax.text(CX, 39.2, "Employment-merge pipeline", ha="center", fontsize=16, fontweight="bold")
ax.text(CX, 38.4, "raw plant scrapes  ->  one clean value per plant x year (2006-2025)",
        ha="center", fontsize=10, color="#555")

# ---- upstream note
box(0.3, 36.3, 4.6, 1.4, "Upstream (NOT in repo, non-deterministic)\nLLM extraction of numbers from web pages\n(Mistral / Haiku)", C["note"], 8.5)

# ---- sources
box(0.3, 33.6, 3.5, 1.9, "Archived scrapes\nbase, filtered, gemini,\nv3_2, de_v3_1, de_v2,\nmodels/prod x3, xlsx_de", C["src"], 8.5)
box(4.25, 33.6, 3.5, 1.9, "Press / gap-fill\npress_tavily,\nestablished_press,\nbackfill, wide", C["src"], 8.5)
box(8.2, 33.6, 3.5, 1.9, "Wikipedia\n(single FINAL file)\nreplaces 6 older passes", C["src"], 8.5)
for x0 in (2.05, 6.0, 9.95): arrow(x0, 33.6, 32.7)

box(3.3, 31.2, 5.4, 1.4, "1. LOAD  ->  candidate table\n(plant, snapshot yr, reference yr, value, source, citation, URL)", C["auto"], 9)
arrow(CX, 31.2, 30.3)

box(2.0, 27.2, 8.0, 2.9, "2. CONTAMINATION FILTERS  (rule-based, no LLM)\n"
    "cross-plant duplicate (wiki) . wrong-entity keyword . city-mismatch (Munich)\n"
    "production-units . job-announcement . historical . artifact\n"
    "wiki-specific: Porsche-AG totals, sub-units, Wrzesnia pre-2016\n"
    "-> every drop logged in contamination_removed.csv", C["auto"], 8.7)
arrow(CX, 27.2, 26.3)

box(3.0, 24.6, 6.0, 1.4, "3. OUTLIER RULE (per plant)\n>3.5x or <1/3.5x median  AND  >10,000 apart", C["auto"], 9)
arrow(CX, 24.6, 23.7)

box(3.0, 22.0, 6.0, 1.4, "4. MANUAL BLOCKLIST  (human judgment)\ncurated wrong values no rule catches - manual_blocks.csv", C["human"], 9)
arrow(CX, 22.0, 21.1)

box(2.4, 18.3, 7.2, 2.5, "5. CONSOLIDATE  ->  one value per (plant, snapshot year)\n"
    ">=2 archived agree  ->  else archived priority  ->  else press\n"
    "then curated_additions + manual_additions fill gaps\n"
    "(non-archived never overrides archived)", C["auto"], 8.7)
arrow(CX, 18.3, 17.4)

box(2.6, 15.6, 6.8, 1.4, "6. WIKIPEDIA FILL  (final file only)\nplaced on reference_year, fills still-empty cells", C["src"], 9)
arrow(CX, 15.6, 14.7)

box(3.0, 13.0, 6.0, 1.4, "7. MANUAL EDITS  (human judgment)\nremove BMW Berlin, Irlbach, Debrecen; drop Sindelfingen 2011", C["human"], 9)
arrow(CX, 13.0, 12.1)

box(3.2, 10.6, 5.6, 1.2, "FINAL DATASET\none value per plant x year", C["out"], 10, bold=True)
arrow(CX, 10.6, 9.7)

box(2.2, 7.3, 7.6, 2.1, "8. STABILITY DERIVATION\ncarry a value forward ONLY if a later snapshot\nreports the identical number  (fresh + carried observations)", C["auto"], 9)
arrow(CX, 7.3, 6.4)

box(2.6, 4.8, 6.8, 1.2, "employment_observations_stability.csv\n+ full 2006-2025 time-series grid", C["out"], 9, bold=True)

# ---- legend
ax.text(0.3, 2.9, "Legend", fontsize=10, fontweight="bold")
for i,(lab,col) in enumerate([("data source",C["src"]),("automatic (code)",C["auto"]),
                              ("human judgment (versioned config)",C["human"]),("output",C["out"])]):
    ax.add_patch(FancyBboxPatch((0.3+i*3.0, 1.9), 0.35, 0.35, boxstyle="round,pad=0.02",
                 edgecolor=col, facecolor=col+"22", linewidth=1.4))
    ax.text(0.75+i*3.0, 2.07, lab, fontsize=7.8, va="center")

plt.tight_layout(); plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white"); plt.close()
print("wrote", OUT)
