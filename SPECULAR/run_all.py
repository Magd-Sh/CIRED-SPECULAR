#!/usr/bin/env python3
"""
Regenerate every figure in the thesis that can be regenerated, into 04_outputs/.

    mamba activate DATA_ECON        # or your own env; see README
    python run_all.py

Reads only 02_data_inputs/. No network, no API keys: the collection step that
needed them is frozen in 03_data_raw/ and is not re-run here.

WHAT THIS DOES AND DOES NOT COVER
---------------------------------
This rebuilds 43 figures. The thesis uses 48 images: 38 of these 43 appear in
it, the other 5 are the maps and the plant-level decomposition added late and
present only in the current draft.

The 9 images the thesis uses that this script does not rebuild are listed at the
end of every run, so nobody assumes the set is complete. None of them is lost:

  imp_grid, imp_spaghetti_vw    come from the imputation pipeline, which is a
                                stage of its own -- 01_code/imputation/, three
                                commands, see the README there
  proc_*  (7 figures)           come from the source-testing notebooks in
                                01_code/exploration/, which are shipped with
                                their outputs

All nine are also shipped as files with the thesis.

Each figure is attempted independently: one failure does not stop the rest, and
the exit code is non-zero if any failed.
"""
from __future__ import annotations
import sys, time, warnings, traceback
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "01_code"))
sys.path.insert(0, str(HERE / "01_code" / "analysis"))

import matplotlib
matplotlib.use("Agg")

from specular_paths import P
import foundation as fd

fd.CONFIG.data_dir = str(P.inputs)
fd.CONFIG.out_dir  = str(P.outputs)
P.outputs.mkdir(parents=True, exist_ok=True)

import module01_spatial as m1
import module02_organisational as m2
import module03_product as m3      # models AND vehicle mass (03b merged in)
import module04_macro_micro as m4
import iea_value_chain as iv
import static_maps as sm

OUT = str(P.outputs) + "/"
Y0, Y1 = 2012, 2023

NOT_REGENERABLE = [
    ("imp_grid.png",           "01_code/imputation/step6_plots.py"),
    ("imp_spaghetti_vw.png",   "01_code/imputation/step7_spaghetti.py"),
    ("proc_pavlinek_fig6.png", "01_code/exploration/erm_pavlinek_reproduction.ipynb"),
    ("proc_pavlinek_fig7.png", "01_code/exploration/erm_pavlinek_reproduction.ipynb"),
    ("proc_pavlinek_fig8.png", "01_code/exploration/erm_pavlinek_reproduction.ipynb"),
    ("proc_iea_assembly.png",  "01_code/exploration/iea_facility_counts.ipynb"),
    ("proc_iea_allcats.png",   "01_code/exploration/iea_facility_counts.ipynb"),
    ("proc_eurostat_units.png","01_code/exploration/eurostat_units_vs_employment.ipynb"),
    ("proc_commune_check.png", "01_code/exploration/urssaf_commune_check.ipynb"),
]


def strict_regional_panel(level: int = 1, nace: str = "C29"):
    """
    Regions with a value in EVERY year of the window.

    NOT `m1.balanced_subset`, which balances on the two endpoints and returns 75
    regions, +5.8% growth, core -149k. The thesis reports strict balance: 68
    regions, +6.7%, -142k. Getting this wrong silently changes the headline.
    """
    p = m1.european_panel(level=level, nace=nace, france="urssaf",
                          drop_other_bloc=True)
    years = list(range(Y0, Y1 + 1))
    piv = p.pivot_table(index="geo_code", columns="year", values="value",
                        aggfunc="first")
    keep = piv[years].dropna().index
    out = p[p["geo_code"].isin(keep) & p["year"].isin([Y0, Y1])].copy()
    out.attrs.update(p.attrs); out.attrs["balance"] = "strict"
    p = p.copy(); p.attrs["balance"] = "strict"
    return p, out


def build_steps():
    """(name, thunk) for every regenerable figure, grouped by chapter."""
    S = []

    # ---- Chapter 4.1 / Appendix C : the spatial picture ---------------------
    full, strict = strict_regional_panel()
    ss = m1.shift_share_1sector(strict, Y0, Y1)
    print(f"regional panel: {ss.attrs['n_regions']} regions, "
          f"growth {100*ss.attrs['eu_growth']:+.2f}%")

    conc      = m1.concentration(full, Y0, Y1, balanced=True)
    conc_bloc = m1.concentration_by_bloc(full, Y0, Y1)

    S += [
      ("fig01_bloc_trajectories", lambda: m1.fig_bloc_trajectories(
          m1.bloc_trajectories(full, balanced=True, y0=Y0, y1=Y1),
          save=OUT+"fig01_bloc_trajectories.png")),
      ("fig01_shift_share", lambda: m1.fig_shift_share_explained(
          ss, panel=full, save=OUT+"fig01_shift_share.png")),
      ("fig01_hhi", lambda: m1.fig_hhi(conc, conc_bloc, panel=full,
          save=OUT+"fig01_hhi.png")),
      ("fig01_lorenz", lambda: m1.fig_lorenz(full, years=(Y0, Y1),
          save=OUT+"fig01_lorenz.png")),
      ("fig01_france", lambda: m1.fig_france_urssaf(
          save=OUT+"fig01_france.png")),
    ]

    # the two-sector version needs the NACE 27 panel alongside
    def _ss2():
        full27, strict27 = strict_regional_panel(nace="C27")
        ss2 = m1.shift_share_2sector(strict, strict27, Y0, Y1)
        return m1.fig_shift_share_2sector(ss2, panel=full,
                                          save=OUT+"fig01_ss2.png")
    S.append(("fig01_ss2", _ss2))

    # ---- maps --------------------------------------------------------------
    S += [
      ("map01_employment_change", lambda: sm.map_employment_change(
          save=OUT+"map01_employment_change.png", label_n=7)),
      ("map02_value_chain", lambda: sm.map_value_chain(
          save=OUT+"map02_value_chain.png")),
      ("map02b_value_chain_regional", lambda: sm.map_value_chain_regional(
          save=OUT+"map02b_value_chain_regional.png")),
      ("map03_battery_france", lambda: sm.map_battery_france(
          save=OUT+"map03_battery_france.png")),
      ("map04_plant_panel", lambda: sm.map_plant_panel(
          save=OUT+"map04_plant_panel.png")),
    ]

    # ---- the two pipeline diagrams -----------------------------------------
    # Drawn, not computed: they have no inputs, they run as scripts in their own
    # folder, and their counts are hard-coded from what the pipeline produces on
    # the frozen data. Kept here so the archive rebuilds every thesis image from
    # one command rather than most of them.
    def _diagram(script, produced):
        import subprocess, shutil
        d = HERE / "01_code" / "pipeline" / "docs"
        subprocess.run([sys.executable, str(d / script)], cwd=d, check=True,
                       capture_output=True)
        shutil.copy(d / produced, OUT + produced)

    S += [
      ("merge_flowchart", lambda: _diagram("make_flowchart.py", "flowchart.png")),
      ("panel_flowchart", lambda: _diagram("make_panel_flowchart.py",
                                           "panel_flowchart.png")),
    ]

    # ---- Chapter 4.2 : inside the firm -------------------------------------
    uni = m2.analysis_universe(balanced=True, window=(Y0, Y1), strict=True)
    S += [
      ("fig02_home_periphery", lambda: m2.fig_home_periphery(
          m2.home_vs_periphery(uni), save=OUT+"fig02_home_periphery.png")),
      ("fig02_assembly_component", lambda: m2.fig_assembly_component(
          m2.classification_trajectories(uni),
          save=OUT+"fig02_assembly_component.png")),
      ("fig02_within_group", lambda: m2.fig_within_group(
          m2.within_group_reallocation(uni, Y0, Y1),
          save=OUT+"fig02_within_group.png")),
      ("fig02_within_group_2p", lambda: m2.fig_within_group_two_periods(
          m2.reallocation_two_periods(uni, Y0, 2018, Y1),
          save=OUT+"fig02_within_group_2p.png")),
      ("fig02_plant_concentration", lambda: m2.fig_plant_concentration(
          m2.plant_concentration(uni), save=OUT+"fig02_plant_concentration.png")),
      ("fig02_data_availability", lambda: m2.fig_data_availability(
          m2.data_availability_window(), save=OUT+"fig02_data_availability.png")),
      ("fig02_full_universe_context", lambda: m2.fig_full_universe_context(
          save=OUT+"fig02_full_universe_context.png")),
    ]

    # ---- Chapter 4.3 : the product -----------------------------------------
    mp = m3.model_panel()
    S += [
      ("fig03_model_mix", lambda: m3.fig_model_mix(
          m3.plant_model_mix(mp), save=OUT+"fig03_model_mix.png")),
      ("fig03_churn", lambda: m3.fig_churn(
          m3.model_churn(mp), save=OUT+"fig03_churn.png")),
      ("fig03_electrification", lambda: m3.fig_electrification(
          mp, save=OUT+"fig03_electrification.png")),
      ("fig03_multi_sourcing", lambda: m3.fig_multi_sourcing(
          m3.multi_sourcing(mp), save=OUT+"fig03_multi_sourcing.png")),
      ("fig03_attribute_readiness", lambda: m3.fig_attribute_readiness(
          m3.attribute_readiness(), save=OUT+"fig03_attribute_readiness.png")),
      ("fig03_mass_by_bloc", lambda: m3.fig_mass_by_bloc(
          save=OUT+"fig03_mass_by_bloc.png")),
      ("fig03_plant_mass", lambda: m3.fig_plant_mass(
          save=OUT+"fig03_plant_mass.png")),
      ("fig03_mass_trend", lambda: m3.fig_mass_trend(
          save=OUT+"fig03_mass_trend.png")),
      ("fig03_plant_traj", lambda: m3.fig_plant_trajectories(
          save=OUT+"fig03_plant_traj.png")),
      ("fig03_mass_percentiles", lambda: m3.fig_mass_percentiles(
          save=OUT+"fig03_mass_percentiles.png")),
      ("fig03_powertrain", lambda: m3.fig_powertrain(
          save=OUT+"fig03_powertrain.png")),
      ("fig03_mass_by_nameplate", lambda: m3.fig_mass_by_nameplate(
          save=OUT+"fig03_mass_by_nameplate.png")),
      ("fig03_powertrain_tracked", lambda: m3.fig_powertrain_tracked(
          save=OUT+"fig03_powertrain_tracked.png")),
      ("fig03_powertrain_tracked_vs_fleet", lambda: m3.fig_powertrain_tracked_vs_fleet(
          save=OUT+"fig03_powertrain_tracked_vs_fleet.png")),
      ("fig03_three_decompositions", lambda: m3.fig_three_decompositions(
          save=OUT+"fig03_three_decompositions.png")),
      ("fig03_co2", lambda: m3.fig_plant_co2(save=OUT+"fig03_co2.png")),
      ("fig03_evmass", lambda: m3.fig_electrification_vs_mass(
          save=OUT+"fig03_evmass.png")),
    ]

    # ---- Chapter 4.4 : plants against regions -------------------------------
    prp = m4.plant_region_panel()
    S += [
      ("fig04_coverage_ratio", lambda: m4.fig_coverage_ratio(
          prp, m4.coverage_ratio_summary(prp),
          save=OUT+"fig04_coverage_ratio.png")),
      ("fig04_supplier_residual", lambda: m4.fig_supplier_residual(
          m4.supplier_residual(prp), save=OUT+"fig04_supplier_residual.png")),
      ("fig04_suppression", lambda: m4.fig_suppression(
          m4.suppression_diagnostic(), save=OUT+"fig04_suppression.png")),
      ("fig04_france", lambda: m4.fig_france_macro_micro(
          m4.france_macro_micro(), save=OUT+"fig04_france.png")),
      ("fig04_plant_vs_region", lambda: m4.fig_plant_vs_region(
          save=OUT+"fig04_plant_vs_region.png")),
      ("fig04_tracking_examples", lambda: m4.fig_tracking_examples(
          save=OUT+"fig04_tracking_examples.png")),
      ("fig04_iea_residual", lambda: iv.fig_residual_vs_suppliers(
          iv.residual_vs_suppliers(prp), save=OUT+"fig04_iea_residual.png")),
      ("fig04_iea_coverage", lambda: iv.fig_coverage_audit(
          save=OUT+"fig04_iea_coverage.png")),
    ]

    # ---- diagnostics that need the whole built dataset ----------------------
    # `build_all` loads every layer and takes about a minute, so it is called
    # once here and shared rather than rebuilt per figure.
    _built = {}
    def built():
        if not _built:
            _built["d"] = fd.build_all(verbose=False)
        return _built["d"]

    S += [
      ("fig00_imputation", lambda: fd.plot_imputation_diagnostics(
          built(), save=OUT+"fig00_imputation.png")),
      ("fig00_nuts_seam", lambda: fd.plot_nuts_seam_diagnostic(
          built(), save=OUT+"fig00_nuts_seam.png")),
    ]
    return S


def main() -> int:
    if P.check():
        print("\nInputs are missing — see above. Nothing was regenerated.")
        return 1

    steps = build_steps()
    failed, t0 = [], time.time()
    for name, fn in steps:
        try:
            fn(); print(f"  ok    {name}")
        except Exception as e:
            failed.append(name)
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")

    n_ok = len(steps) - len(failed)
    print(f"\n{n_ok}/{len(steps)} figures written to {P.outputs} "
          f"in {time.time()-t0:.0f}s")
    if failed:
        print("failed: " + ", ".join(failed))

    print(f"\n{len(NOT_REGENERABLE)} further images in the thesis come from "
          "another stage, not from here. Shipped as files; rebuilt by:")
    for f, why in NOT_REGENERABLE:
        print(f"  {f:<26} {why}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
