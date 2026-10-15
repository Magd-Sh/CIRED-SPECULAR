Raw EEA CO2-monitoring files (passenger cars, Regulation (EU) 2019/631).

On INARI this folder holds, for each year 2010-2024, the full EEA file
(constructeurs_final_YYYY.csv) and a light extract of it with the nine columns
the analysis uses (light_constructeurs_final_YYYY.csv), and the provisional 2025
file (constructeurs_provisional_2025.csv). They are not part of the pushed archive.

01_code/analysis/eea_build_eu27.py reads this folder by default: the light file
for each year, and the single 2025 file. It writes the four aggregated inputs in
02_data_inputs/eea/ (EU27 only), which are all the analysis reads.

The 2025 figures in the thesis were built from an earlier download of the
provisional 2025 data. The provisional file here was downloaded later, so 2025
aggregates rebuilt from it can differ slightly from those in the thesis.
Updated 15 October 2026.
