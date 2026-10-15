#!/usr/bin/env python3
"""
Stage 2: rebuild the plant panel from the frozen raw scrapes.

    mamba activate specular
    python run_merge.py

This does NOT scrape anything. It reads the sixteen source files the scrapers
produced months ago, applies the contamination filters, the per-plant outlier
rule, the curated blocklist and the consolidation order, and writes one value
per plant and year.

Why it is worth running
-----------------------
Collection cannot be reproduced: the Internet Archive returns a different sample
of snapshots on each run and the language models are hosted services. The merge
that runs on those frozen files CAN be, exactly. So this is the step that lets
someone check the chain from raw scrape to final panel without re-collecting
anything, and it is the claim Appendix A of the thesis makes.

Expected output, on the shipped inputs:

    final values      : 374   (non-wiki 342 + wiki_final 32)
    stability obs     : 414   (85 carried)
    plants            : 40

Those two counts measure different things -- the consolidated panel and the
stability derivation -- so they are not meant to reconcile against each other.

After this, `run_all.py` regenerates the figures.
"""
from __future__ import annotations
import subprocess, sys, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "01_code"))
from specular_paths import P

PIPELINE = P.pipeline
ENTRY = PIPELINE / "src" / "pipeline.py"


def main() -> int:
    if not ENTRY.exists():
        print(f"pipeline not found at {ENTRY}")
        return 1

    raw = PIPELINE / "data" / "raw"
    n_raw = len(list(raw.glob("*.csv"))) + len(list(raw.glob("*.xlsx")))
    if n_raw == 0:
        print(f"no raw source files in {raw} — nothing to merge")
        return 1
    print(f"merging {n_raw} raw source files from {raw}\n")

    # The pipeline resolves config/ and data/ relative to its own folder, so it
    # has to be run from there rather than from the project root.
    r = subprocess.run([sys.executable, "src/pipeline.py"],
                       cwd=str(PIPELINE), text=True)
    if r.returncode != 0:
        print("\nmerge failed")
        return r.returncode

    out = PIPELINE / "outputs"
    made = sorted(p.name for p in out.glob("*") if p.is_file())
    print(f"\nwrote {len(made)} file(s) to {out}")
    for m in made[:8]:
        print(f"  {m}")
    print("\nNote: the analysis reads the frozen copy in 02_data_inputs/plants/,")
    print("not these outputs. Compare them if you want to verify the chain;")
    print("copy them across only if you mean to change the panel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
