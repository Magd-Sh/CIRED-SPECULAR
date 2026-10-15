# CIRED-SPECULAR

Material for the master's thesis *Mapping employment change in the European
automotive industry: regions, plants, and the vehicles they built from 2006 to 2025*
(Magd Shamashan, Université Paris Dauphine–PSL and CIRED, SPECULAR project, 2026).

- `docs/` — the interactive maps referenced in the thesis, served by GitHub Pages
  at <https://magd-sh.github.io/CIRED-SPECULAR/docs/>.
- `SPECULAR/` — the full archive: code, data inputs, the plant panel and its
  imputed version, and every figure. Start with `SPECULAR/README.md`; running
  `python run_all.py` from `SPECULAR/` regenerates the figures.
- `employment-pipeline/` — an earlier version of the employment merge, kept for the
  record. The current one is `SPECULAR/01_code/pipeline/`.

Two inputs are too large for GitHub and are not here: the raw EEA registration
files (about 25 GB, one CSV per year; public on the EEA data portal) and the
scraper caches (about 5 GB). Both are on the CIRED server INARI, in
`/data/shared/SPECULAR/03_data_raw/`. The analysis runs without them: it reads the
aggregates in `SPECULAR/02_data_inputs/`.

Third-party data in `SPECULAR/02_data_inputs/` come from Eurostat, URSSAF, the
European Environment Agency and the International Energy Agency, under their own
terms of reuse; the sources are listed in `SPECULAR/README.md`.
