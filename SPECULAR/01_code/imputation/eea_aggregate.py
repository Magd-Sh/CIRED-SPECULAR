# -*- coding: utf-8 -*-
"""
Aggregate the EEA CO2-monitoring registration files (light_constructeurs_<year>.csv)
to one row per (make, commercial name, year), streaming in chunks (files up to ~350 MB).

Two modes:
  python eea_aggregate.py part    <EEA_DIR> <PARTS_DIR> <year> [<year> ...]
      -> writes PARTS_DIR/eea_<year>.csv  (one file per year; run in batches)
  python eea_aggregate.py combine <PARTS_DIR> <OUT_CSV>
      -> merges the parts into OUT_CSV with columns:
         make, cn, year, regs (sum of r), mass_mean (registration-weighted mean
         mass in running order, kg), ev_share, phev_share, n_rows
"""
import sys, os, glob
import numpy as np
import pandas as pd


def norm(s):
    return (s.astype(str).str.upper().str.strip()
             .str.replace(r"[\.\,]+$", "", regex=True)
             .str.replace(r"\s+", " ", regex=True))


def aggregate_year(path):
    chunks = []
    for ch in pd.read_csv(path, chunksize=2_000_000, low_memory=False,
                          usecols=["Mk", "Cn", "r", "m (kg)", "Ft", "year"]):
        ch = ch.rename(columns={"m (kg)": "m"})
        ch["m"] = pd.to_numeric(ch["m"], errors="coerce")
        ch["r"] = pd.to_numeric(ch["r"], errors="coerce").fillna(1)
        ch["make"] = norm(ch["Mk"]); ch["cn"] = norm(ch["Cn"])
        ft = ch["Ft"].astype(str).str.lower()
        ev = (ft == "electric")
        phev = ft.str.contains("electric") & ~ev
        ch["rm"] = np.where(ch["m"].notna(), ch["r"] * ch["m"], 0.0)
        ch["rmw"] = np.where(ch["m"].notna(), ch["r"], 0.0)
        ch["rev"] = ch["r"] * ev
        ch["rphev"] = ch["r"] * phev
        chunks.append(ch.groupby(["make", "cn", "year"]).agg(
            regs=("r", "sum"), rm=("rm", "sum"), rmw=("rmw", "sum"),
            rev=("rev", "sum"), rphev=("rphev", "sum"), n_rows=("r", "size")))
    return pd.concat(chunks).groupby(level=[0, 1, 2]).sum().reset_index()


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "part":
        eea_dir, parts_dir, years = sys.argv[2], sys.argv[3], sys.argv[4:]
        os.makedirs(parts_dir, exist_ok=True)
        for y in years:
            df = aggregate_year(os.path.join(eea_dir, f"light_constructeurs_{y}.csv"))
            df.to_csv(os.path.join(parts_dir, f"eea_{y}.csv"), index=False)
            print(f"{y}: {len(df)} model-year rows", flush=True)
    elif mode == "combine":
        parts_dir, out = sys.argv[2], sys.argv[3]
        out_df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(parts_dir, "eea_*.csv")))])
        out_df = out_df.groupby(["make", "cn", "year"], as_index=False).sum()
        out_df["mass_mean"] = out_df["rm"] / out_df["rmw"].replace(0, np.nan)
        out_df["ev_share"] = out_df["rev"] / out_df["regs"]
        out_df["phev_share"] = out_df["rphev"] / out_df["regs"]
        out_df[["make", "cn", "year", "regs", "mass_mean", "ev_share", "phev_share", "n_rows"]].to_csv(out, index=False)
        print(f"[OK] {out}: {len(out_df)} rows", flush=True)
