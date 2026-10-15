"""
Rebuild the Section 4.1 figures on the STRICT-balanced panel.

Why this file exists
--------------------
`m1.balanced_subset` keeps regions observed at the two ENDPOINTS: 75 regions,
European growth +5.8%, core -148.9k, east +132.8k. The thesis reports the STRICT
balance -- observed in EVERY year of the window -- which gives 68 regions, +6.7%,
-142.4k and +127.7k. Figure 4.2 was built from the first and described with the
second, so its in-plot note contradicted the text that cites it.

The figures self-describe from `ss.attrs`, so nothing in the plotting code was
wrong; it was fed the wrong panel. Run this and the notes agree with Chapter 4.

    python regen_module01_strict.py
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); os.chdir(HERE)

import matplotlib; matplotlib.use("Agg")
import foundation as fd, module01_spatial as m1

fd.CONFIG.data_dir = "data"
OUT = "../thesis/figures/"
Y0, Y1 = 2012, 2023


def strict_panel(level=1, france="urssaf", y0=Y0, y1=Y1):
    """Regions with a value in EVERY year of the window, endpoints returned."""
    p = m1.european_panel(level=level, france=france, drop_other_bloc=True)
    years = list(range(y0, y1 + 1))
    piv = p.pivot_table(index="geo_code", columns="year", values="value",
                        aggfunc="first")
    keep = piv[years].dropna().index
    out = p[p["geo_code"].isin(keep) & p["year"].isin([y0, y1])].copy()
    out.attrs.update(p.attrs); out.attrs["balance"] = "strict"
    p = p.copy(); p.attrs["balance"] = "strict"
    return p, out


full, strict = strict_panel()
ss = m1.shift_share_1sector(strict, Y0, Y1)
print(f"strict: {ss.attrs['n_regions']} regions | growth {100*ss.attrs['eu_growth']:+.2f}%")
print(f"  core {ss[ss.bloc=='core'].differential_shift.sum()/1000:+.1f}k"
      f" | east {ss[ss.bloc=='eastern_periphery'].differential_shift.sum()/1000:+.1f}k")

# `fig_shift_share_explained` is the reader-facing two-panel version the thesis
# caption describes ("Panel (a) works the calculation through for four regions").
# `fig_shift_share` is the older continuity figure and does NOT match that
# caption -- an earlier pass regenerated the wrong one.
m1.fig_shift_share_explained(ss, panel=full, save=OUT + "fig01_shift_share.png")
print("wrote fig01_shift_share.png")
