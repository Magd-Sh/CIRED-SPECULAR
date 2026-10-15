"""
One place where every path in this project is defined.

WHY THIS FILE EXISTS
--------------------
The scripts were written on a laptop and carried absolute paths such as
/Users/majdshamashan/Downloads/... . Nineteen of them. On any other machine each
one fails on its first line, which is the single thing that stopped the analysis
being re-runnable by someone else.

Everything now resolves from PROJECT_ROOT, which is found by walking up from
this file. Nothing needs editing after a move; if you want the data somewhere
else, set SPECULAR_ROOT in the environment and it wins.

    from specular_paths import P
    df = pd.read_excel(P.plants)          # not a hard-coded string
"""
from __future__ import annotations
import os
from pathlib import Path

_here = Path(__file__).resolve()
PROJECT_ROOT = Path(os.environ.get("SPECULAR_ROOT", _here.parents[1]))


class _Paths:
    root = PROJECT_ROOT
    code        = PROJECT_ROOT / "01_code"
    analysis    = code / "analysis"
    scrapers    = code / "scrapers"
    pipeline    = code / "pipeline"

    inputs      = PROJECT_ROOT / "02_data_inputs"
    plants_dir  = inputs / "plants"
    eurostat    = inputs / "eurostat"
    urssaf      = inputs / "urssaf"
    eea         = inputs / "eea"
    geometry    = inputs / "geometry"

    raw         = PROJECT_ROOT / "03_data_raw"
    eea_raw     = raw / "eea_registrations"
    scraper_out = raw / "scraper_outputs"
    caches      = raw / "scraper_caches"

    outputs     = PROJECT_ROOT / "04_outputs"

    # individual files the code asks for by name
    plants          = plants_dir / "auto_plants_models_harmonised.xlsx"
    employment_sum  = plants_dir / "employment_completed_summary.csv"
    eea_mass        = eea / "eea_model_mass.csv"
    eea_mass_fuel   = eea / "eea_model_mass_by_fuel.csv"

    def check(self) -> int:
        """Report which expected inputs are missing. Returns the count."""
        missing = 0
        for name in ("plants", "employment_sum", "eea_mass", "eea_mass_fuel"):
            p = getattr(self, name)
            if not p.exists():
                print(f"  MISSING  {name}: {p}")
                missing += 1
        for d in (self.eurostat, self.urssaf, self.geometry):
            if not any(d.glob("*")):
                print(f"  EMPTY    {d}")
                missing += 1
        return missing


P = _Paths()

if __name__ == "__main__":
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    n = P.check()
    print("all expected inputs present" if n == 0 else f"{n} problem(s)")
