"""
Regenerate the four maps that changed:
  * France drawn at departement level from URSSAF, not NUTS-2
  * heavier country borders than regional borders, on all four
  * "(check data)" removed from the ratio legend

Run it from anywhere:

    python ~/Desktop/cired_stage/analysis/regenerate_two_maps.py

Needs folium and plotly installed (they are not available in the assistant's
sandbox, which is why this has to run on your machine).
"""
import os, sys, shutil, warnings
warnings.filterwarnings("ignore")

# Run from anywhere: make sure this file's own folder (the analysis folder) is
# importable, and work relative to it rather than to the shell's directory.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

import foundation as fd
import module04_macro_micro as m4
import iea_value_chain as iv

fd.CONFIG.data_dir = "./data"
fd.CONFIG.out_dir = "./outputs"

# --- map 1: plants over regional employment ---------------------------------
prp = m4.plant_region_panel()
m4.map_plants_over_regions(prp, year=2023,
                           save="outputs/map04_plants_over_regions.html")
print("rebuilt outputs/map04_plants_over_regions.html")

# --- map 2: regional employment choropleth ----------------------------------
panel = fd.regional_panel("C29", "employment", level=2, include_france=True)
fd.map_nuts_choropleth(
    panel, 2019, level=2,
    title="Automotive employment by region, 2019 "
          "(NUTS-2; France at departement level, URSSAF)",
    save="outputs/map00_nuts_c29_2019.html")
print("rebuilt outputs/map00_nuts_c29_2019.html")

# --- map 3: the automotive value chain --------------------------------------
iv.map_value_chain(radius_km=50, year=2023,
                   save="outputs/map04_value_chain.html")
print("rebuilt outputs/map04_value_chain.html")

# --- map 4: all tracked plants ----------------------------------------------
fd.map_plants_folium(fd.plant_nuts_crosswalk(),
                     employment=fd.employment_panel(), year=2023,
                     save="outputs/map00_plants.html")
print("rebuilt outputs/map00_plants.html")

# --- copy into the published site -------------------------------------------
site = "../CIRED-SPECULAR/docs/maps"
if os.path.isdir(site):
    shutil.copy("outputs/map04_plants_over_regions.html",
                f"{site}/plants-over-regions.html")
    shutil.copy("outputs/map00_nuts_c29_2019.html",
                f"{site}/nuts-employment-2019.html")
    shutil.copy("outputs/map04_value_chain.html",
                f"{site}/automotive-value-chain.html")
    shutil.copy("outputs/map00_plants.html",
                f"{site}/tracked-plants.html")
    print("copied all four into", site)
    print("\nNow publish them:")
    print("  cd ~/Desktop/cired_stage/CIRED-SPECULAR")
    print("  git add docs && git commit -m 'Maps: France at departement level, "
          "heavier country borders' && git push")
else:
    print("site folder not found at", site, "- copy the two files manually")
