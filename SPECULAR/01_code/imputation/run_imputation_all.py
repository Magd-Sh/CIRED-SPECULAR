# -*- coding: utf-8 -*-
"""
Orchestrator - runs the whole employment imputation chain, in order.

    python run_all.py EMP.xlsx MODELS_v5.xlsx [EEA_DIR] [M]

Steps:
  0  (only if eea_family_year.csv is absent and EEA_DIR is given)
     aggregate the EEA registration files and match them to model families
  1-2  build the panels with EEA vehicle size        -> panel_assembly.csv, panel_all.csv
  4    method-choice validation (interpolation vs Amelia MVN vs within-plant MI)
  4b   variant comparison + uncertainty calibration (several masking seeds)
  5    multiple imputation + deliverables            -> outputs/ (26 assembly), outputs_all/ (35)
  6-7  figures (before/after grid, per plant, 50-draw spaghetti by group)
"""
import os, sys, subprocess

EMP = sys.argv[1] if len(sys.argv) > 1 else "../uploads/employment_timeseries_full.xlsx"
MOD = sys.argv[2] if len(sys.argv) > 2 else "../uploads/auto_models_production_consolidated_v5.xlsx"
EEA_DIR = sys.argv[3] if len(sys.argv) > 3 else ""
M = sys.argv[4] if len(sys.argv) > 4 else "50"


def sh(cmd):
    print("\n" + "=" * 70 + f"\n$ {' '.join(cmd)}\n" + "=" * 70, flush=True)
    subprocess.run([sys.executable] + cmd, check=True)


if not os.path.exists("eea_family_year.csv"):
    if not EEA_DIR:
        sys.exit("eea_family_year.csv missing: pass EEA_DIR (folder with light_constructeurs_*.csv)")
    years = [str(y) for y in range(2010, 2026)]
    sh(["eea_aggregate.py", "part", EEA_DIR, "eea_parts"] + years)
    sh(["eea_aggregate.py", "combine", "eea_parts", "eea_model_year.csv"])
    sh(["eea_match.py", "eea_model_year.csv", MOD])

sh(["step1_2_build_panel.py", EMP, MOD, ".", "assembly"])
sh(["step1_2_build_panel.py", EMP, MOD, ".", "all"])
sh(["step4_validate.py", "panel_assembly.csv"])
sh(["step4b_calibrate.py", "panel_assembly.csv", "4", "25", "variants"])
sh(["step5_run_imputation.py", "panel_assembly.csv", "outputs", M])
sh(["step5_run_imputation.py", "panel_all.csv", "outputs_all", M, "5"])
for base, panel in [("outputs", "panel_assembly.csv"), ("outputs_all", "panel_all.csv")]:
    sh(["step6_plots.py", panel, base])
    sh(["step7_spaghetti.py", base])
# final audit (exits with an error if any check fails)
sh(["audit_imputation.py", EMP, "panel_assembly.csv", "outputs"])
sh(["audit_imputation.py", EMP, "panel_all.csv", "outputs_all", "5"])
print("\n[OK] Chain complete. Deliverables in outputs/ and outputs_all/.")
