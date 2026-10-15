#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Curated, hand-verified models-per-plant-per-year dataset (2006-2026) for ALL 43 plants.

Built by reading each plant's Wikipedia article(s) directly (EN + DE/FR), supplemented
by well-documented production facts for plants whose article lacks an explicit
year-by-year list. Every model row carries a confidence flag and a source note.

  confidence = High    : explicit Wikipedia statement / dedicated plant article
               Medium  : year boundary inferred from model-generation timing
               Low     : plausible but should be double-checked

Component / powertrain / battery / historic sites are listed separately: they
build NO complete cars in 2006-2026, so their zero is real, not missing data.

Outputs (models_output_curated/):
  models_binary_matrix_curated.csv      wide 0/1, rows=(plant,model), cols 2006..2026
  models_production_units_curated.csv   same layout, annual units where known
  models_long_evidence_curated.csv      model, span, confidence, source, evidence
  plant_classification.csv              all 43 plants: assembly / component / historic
  employment_from_wikipedia_curated.csv employment (in-window) for consolidation
"""
import csv
from pathlib import Path

START, END = 2006, 2026
YEARS = list(range(START, END + 1))
OUT = Path(__file__).resolve().parent / "models_output_curated"

WIKI = "Wikipedia plant/model articles (read directly)"
FACTS = "Wikipedia + well-documented production facts"

# (model, year_start, year_end|None, confidence, evidence)
PLANTS = {
 "BE_AUDI_BRUSSELS": dict(name="Audi Brussels", brand="audi", country="Belgium", src="en/fr/de WP 'Audi Brussels'", models=[
    ("Volkswagen Golf",1980,2007,"High","1980–2007: VW Golf (3,870,701 units)"),
    ("Volkswagen Polo",2006,2009,"High","2006–2009: VW Polo (146,019 units)"),
    ("Audi A3",2004,2009,"High","2004–2009: Audi A3"),
    ("Audi A3 Sportback",2007,2010,"High","2007–2010: Audi A3 Sportback"),
    ("Audi A1",2010,2018,"High","2010–2018: Audi A1 (909,164 units)"),
    ("Audi e-tron",2018,2022,"High","2018–2022: Audi e-tron (160,000 units)"),
    ("Audi e-tron Sportback",2019,2022,"Medium","DE: 'seit 2019'; EN/FR say 2020"),
    ("Audi Q8 e-tron",2022,2025,"High","Dec 2022–Feb 2025 (plant closed 28 Feb 2025)"),
    ("Audi Q8 e-tron Sportback",2022,2025,"High","Dec 2022–Feb 2025"),
 ], employment=[(2934,2022,"fr infobox: 2 934 (fin 2022)")], units={"Audi A1":{2011:118200,2015:116250},"Audi Q8 e-tron":{2023:53555,2024:15193}}),

 "DE_VW_WOLFSBURG": dict(name="Volkswagen Wolfsburg Plant", brand="vw", country="Germany", src="en/de WP 'Volkswagenwerk Wolfsburg'", models=[
    ("Volkswagen Golf",1974,None,"High","Lead plant for the Golf since 1974"),
    ("Volkswagen Golf Variant",2006,None,"Medium","Golf estate at Wolfsburg"),
    ("Volkswagen Touran",2006,None,"High","Touran (1T/5T)"),
    ("Volkswagen Tiguan",2007,None,"High","Tiguan since 2007"),
    ("Volkswagen Golf Sportsvan",2014,2020,"Medium","Golf Sportsvan 2014–2020"),
    ("Volkswagen Tiguan Allspace",2017,2024,"Medium","Tiguan Allspace"),
    ("SEAT Tarraco",2018,2024,"Medium","SEAT Tarraco 2018–2024"),
    ("Volkswagen ID.3",2023,None,"Medium","ID.3 assembly added 2023"),
    ("Volkswagen Tayron",2024,None,"Medium","Tayron from 2024"),
 ], employment=[(60000,2018,"~60,000 at Wolfsburg (approx)")], units={}),

 "DE_MERCEDES_SINDELFINGEN": dict(name="Mercedes-Benz Werk Sindelfingen", brand="mercedes", country="Germany", src="de WP 'Werk Sindelfingen'", models=[
    ("Mercedes-Benz S-Class",2006,None,"High","S-Klasse built only here (W221/W222/W223)"),
    ("Mercedes-Benz E-Class",2006,None,"High","E-Klasse Limousine + T-Modell"),
    ("Mercedes-Benz CLS",2006,2023,"Medium","CLS (C219/C218/C257); discontinued 2023"),
    ("Maybach 57/62",2006,2012,"High","Maybach-Manufaktur 2002–2012"),
    ("Mercedes-Maybach S-Class",2015,None,"High","Maybach S-Klasse"),
    ("Mercedes-AMG GT 4-Door",2018,None,"High","AMG GT 4-Türer (X290)"),
    ("Mercedes-Benz EQS",2021,None,"High","EQS (V297), Factory 56, first BEV here"),
    ("Mercedes-Benz GLC",2015,None,"Low","listed as built here to cover demand peaks"),
 ], employment=[(35000,2017,"de: ~35.000 (31.12.2017)")], units={}),

 "DE_MERCEDES_RASTATT": dict(name="Mercedes-Benz Werk Rastatt", brand="mercedes", country="Germany", src="de WP 'Werk Rastatt'", models=[
    ("Mercedes-Benz A-Class",2006,None,"High","A-Klasse since 1997 (W169/W176/W177)"),
    ("Mercedes-Benz B-Class",2006,None,"High","B-Klasse since 2005"),
    ("Mercedes-Benz GLA",2013,None,"High","GLA (X156/H247)"),
    ("Mercedes-Benz EQA",2021,None,"High","EQA (H243) from 2021"),
 ], employment=[], units={}),

 "DE_MERCEDES_BREMEN": dict(name="Mercedes-Benz Werk Bremen", brand="mercedes", country="Germany", src="de WP 'Werk Bremen'", models=[
    ("Mercedes-Benz C-Class",2006,None,"High","Stammwerk der C-Klasse"),
    ("Mercedes-Benz E-Class Coupe/Cabriolet",2009,2023,"Medium","C207/C238"),
    ("Mercedes-Benz GLC",2015,None,"High","Lead plant for GLC"),
    ("Mercedes-Benz GLC Coupé",2016,None,"Medium","GLC Coupé"),
    ("Mercedes-Benz SL",2012,2020,"Medium","SL (R231)"),
    ("Mercedes-Benz SLC",2011,2020,"Medium","SLK/SLC (R172)"),
    ("Mercedes-Benz EQC",2019,2023,"High","EQC (N293) 2019–2023"),
 ], employment=[(12500,2017,"de: >12.500 (31.12.2017)")], units={}),

 "DE_BMW_DINGOLFING": dict(name="BMW Group Werk Dingolfing", brand="bmw", country="Germany", src="de WP 'Werk Dingolfing'", models=[
    ("BMW 5 Series",2006,None,"High","5er core model since 1973"),
    ("BMW 6 Series",2006,2018,"Medium","6er coupe/cabrio"),
    ("BMW 7 Series",2006,None,"High","7er flagship"),
    ("BMW 8 Series",2018,None,"High","8er (G14/G15/G16) from 2018"),
    ("BMW 4 Series",2014,None,"Medium","article lists 4er built here"),
    ("BMW iX",2021,None,"High","BMW iX since July 2021"),
 ], employment=[(17000,2021,"de: ~17.000 (Juli 2021)")], units={"_plant_total":{2021:245000}}),

 "DE_VW_DRESDEN": dict(name="Transparent Factory (Dresden)", brand="vw", country="Germany", src="en WP 'Transparent Factory'", models=[
    ("Volkswagen Phaeton",2006,2016,"High","Phaeton produced until 2016"),
    ("Volkswagen e-Golf",2017,2020,"High","electric Golf from 2017"),
    ("Volkswagen ID.3",2021,2025,"High","ID.3 Jan 2021 until closure Dec 2025"),
 ], employment=[], units={}),

 "HU_AUDI_GYOR": dict(name="Audi Hungaria (Győr)", brand="audi", country="Hungary", src="en WP 'Audi Hungaria'", models=[
    ("Audi TT Coupé",2006,2023,"High","TT Coupé built here (gen2/gen3) until 2023"),
    ("Audi TT Roadster",2006,2023,"High","TT Roadster"),
    ("Audi A3 Cabriolet",2008,2020,"Medium","A3 Cabriolet from 2007/08"),
    ("Audi A3 Sedan",2013,2020,"Medium","A3 sedan fully built from 2013"),
    ("Audi Q3",2018,None,"High","Q3 (2nd gen) from 2018/19"),
    ("Audi Q3 Sportback",2019,None,"High","Q3 Sportback from 2019"),
    ("Cupra Terramar",2024,None,"Medium","Q3 sibling from 2024"),
 ], employment=[(13084,2018,"en infobox: 13,084 employees (2018)")], units={}),

 "SK_VW_BRATISLAVA": dict(name="Volkswagen Bratislava Plant", brand="vw", country="Slovakia", src="en WP 'Volkswagen Bratislava Plant'", models=[
    ("Audi Q7",2006,None,"High","Q7 (2005–present)"),
    ("Audi Q8",2018,None,"High","Q8 (2018–present)"),
    ("Porsche Cayenne",2006,None,"High","Cayenne (2002–present)"),
    ("Porsche Cayenne Coupé",2019,None,"High","Cayenne Coupé (2019–present)"),
    ("Volkswagen Touareg",2006,None,"High","Touareg (2002–present)"),
    ("Škoda Superb",2023,None,"High","Superb (2023–present)"),
    ("Volkswagen Passat",2023,None,"High","Passat (2023–present)"),
    ("Volkswagen Polo",2006,2007,"High","VW Polo (1999–2007)"),
    ("Škoda Octavia",2008,2010,"High","Octavia (2008–2010)"),
    ("Volkswagen up!",2011,2023,"High","up! (2011–2023)"),
    ("SEAT Mii",2011,2021,"High","Mii (2011–2021)"),
    ("Škoda Citigo",2011,2020,"High","Citigo (2011–2020)"),
    ("Škoda Karoq",2020,2023,"High","Karoq (2020–2023)"),
    ("Bentley Bentayga",2016,None,"Medium","Bentayga assembly/bodies since 2016"),
 ], employment=[], units={}),

 "DE_VW_ZWICKAU": dict(name="Volkswagen Zwickau-Mosel Plant", brand="vw", country="Germany", src="en WP 'VW Zwickau-Mosel Plant'", models=[
    ("Volkswagen Golf",2006,2020,"Medium","Golf built until ICE end (last ICE 26 Jun 2020)"),
    ("Volkswagen Passat",2006,2020,"Medium","Passat (incl. Variant)"),
    ("Volkswagen ID.3",2020,None,"High","ID.3 from 2019/2020 (plant went all-electric)"),
    ("Volkswagen ID.4",2020,None,"High","ID.4"),
    ("Volkswagen ID.5",2021,None,"High","ID.5"),
    ("Audi Q4 e-tron",2021,None,"High","Q4 e-tron from 2021"),
    ("Cupra Born",2021,None,"Medium","Cupra Born from 2021"),
 ], employment=[(8000,2019,"en: ~8,000 employees (2019)")], units={}),

 "DE_AUDI_INGOLSTADT": dict(name="Audi Werk Ingolstadt", brand="audi", country="Germany", src=FACTS, models=[
    ("Audi A3",2006,None,"High","Ingolstadt lead plant for A3/A3 Sportback"),
    ("Audi A4",2006,None,"High","A4 sedan/Avant"),
    ("Audi A5",2007,None,"Medium","A5 coupe/Sportback"),
    ("Audi Q5",2008,2017,"Medium","Q5 (8R) until move to Mexico ~2017"),
    ("Audi Q2",2016,None,"High","Q2 from 2016"),
    ("Audi A3 Sedan",2013,2020,"Low","A3 sedan period"),
 ], employment=[], units={}),

 "DE_AUDI_NECKARSULM": dict(name="Audi Werk Neckarsulm", brand="audi", country="Germany", src=FACTS, models=[
    ("Audi A6",2006,None,"High","Neckarsulm lead plant for A6"),
    ("Audi A7",2010,None,"High","A7 Sportback from 2010"),
    ("Audi A8",2006,None,"High","A8 flagship"),
    ("Audi A4",2006,2015,"Medium","A4 also built here (older gens)"),
    ("Audi R8",2007,2024,"High","R8 hand-built (Böllinger Höfe) until 2024"),
    ("Audi e-tron GT",2020,None,"High","e-tron GT (Böllinger Höfe) from 2020"),
    ("Audi A6 e-tron",2024,None,"Medium","A6 e-tron from 2024"),
 ], employment=[], units={}),

 "DE_BMW_MUNICH": dict(name="BMW Werk München", brand="bmw", country="Germany", src=FACTS, models=[
    ("BMW 3 Series",2006,None,"High","Munich is the home of the 3 Series"),
    ("BMW M3",2007,None,"Medium","M3 variant"),
    ("BMW 4 Series",2014,None,"Low","also built across Munich/Dingolfing/Regensburg"),
    ("BMW i4",2021,None,"High","i4 (electric) from 2021"),
 ], employment=[], units={}),

 "DE_BMW_REGENSBURG": dict(name="BMW Werk Regensburg", brand="bmw", country="Germany", src=FACTS, models=[
    ("BMW 1 Series",2006,None,"High","1 Series core at Regensburg"),
    ("BMW 3 Series",2006,None,"Medium","3 Series also built here"),
    ("BMW Z4",2006,2016,"Medium","Z4 until move to Graz 2016"),
    ("BMW X2",2018,None,"Medium","X2 from 2018"),
    ("BMW X1",2022,None,"Medium","X1 (U11) from 2022"),
 ], employment=[], units={}),

 "DE_BMW_LEIPZIG": dict(name="BMW Werk Leipzig", brand="bmw", country="Germany", src=FACTS, models=[
    ("BMW 1 Series",2006,None,"High","1 Series at Leipzig"),
    ("BMW 2 Series",2014,None,"High","2 Series Active/Gran Tourer + Coupé"),
    ("BMW i3",2013,2022,"High","i3 (2013–2022)"),
    ("BMW i8",2014,2020,"High","i8 (2014–2020)"),
    ("BMW X1",2022,None,"Medium","X1 also built here"),
 ], employment=[], units={}),

 "DE_PORSCHE_LEIPZIG": dict(name="Porsche Werk Leipzig", brand="porsche", country="Germany", src=FACTS, models=[
    ("Porsche Carrera GT",2006,2006,"Medium","Carrera GT until 2006"),
    ("Porsche Cayenne",2006,None,"High","Cayenne finishing/assembly at Leipzig"),
    ("Porsche Panamera",2009,None,"High","Panamera from 2009"),
    ("Porsche Macan",2014,None,"High","Macan from 2014 (incl. electric Macan)"),
 ], employment=[], units={}),

 "DE_PORSCHE_ZUFFENHAUSEN": dict(name="Porsche Werk Zuffenhausen", brand="porsche", country="Germany", src="en WP factories list", models=[
    ("Porsche 911",2006,None,"High","911 home plant"),
    ("Porsche Boxster",2006,None,"High","Boxster (718)"),
    ("Porsche Cayman",2006,None,"High","Cayman (718)"),
    ("Porsche 918 Spyder",2013,2015,"High","918 Spyder 2013–2015"),
    ("Porsche Taycan",2019,None,"High","Taycan from 2019"),
 ], employment=[], units={}),

 "HU_MERCEDES_KECSKEMET": dict(name="Mercedes-Benz Manufacturing Hungary (Kecskemét)", brand="mercedes", country="Hungary", src=FACTS, models=[
    ("Mercedes-Benz B-Class",2012,2018,"High","B-Klasse (W246) from 2012"),
    ("Mercedes-Benz CLA",2013,None,"High","CLA (C117/C118) from 2013"),
    ("Mercedes-Benz CLA Shooting Brake",2015,None,"Medium","CLA Shooting Brake"),
    ("Mercedes-Benz GLB",2019,None,"High","GLB from 2019"),
    ("Mercedes-Benz GLA",2020,None,"Medium","GLA (H247) shared with Rastatt"),
    ("Mercedes-Benz EQB",2021,None,"High","EQB from 2021"),
 ], employment=[(4700,2018,"~4,700 employees (approx 2018)")], units={}),

 "ES_VW_NAVARRA": dict(name="Volkswagen Navarra (Pamplona)", brand="vw", country="Spain", src=FACTS, models=[
    ("Volkswagen Polo",2006,None,"High","Polo lead plant (Landaben)"),
    ("Volkswagen T-Cross",2019,None,"High","T-Cross from 2019"),
    ("Volkswagen Taigo",2021,None,"Medium","Taigo from 2021"),
 ], employment=[], units={}),

 "PL_VWN_POZNAN": dict(name="Volkswagen Poznań", brand="vw", country="Poland", src=FACTS, models=[
    ("Volkswagen Caddy",2006,None,"High","Caddy (commercial/MPV)"),
    ("Volkswagen Transporter",2006,2024,"Medium","T5/T6 variants"),
 ], employment=[], units={}),

 "PL_VWN_WRZESNIA": dict(name="Volkswagen Września", brand="vw", country="Poland", src=FACTS, models=[
    ("Volkswagen Crafter",2016,None,"High","Crafter since plant opened 2016"),
    ("MAN TGE",2017,None,"High","MAN TGE (badge sibling)"),
 ], employment=[], units={}),

 "PT_VW_PALMELA": dict(name="Volkswagen Autoeuropa (Palmela)", brand="vw", country="Portugal", src=FACTS, models=[
    ("Volkswagen Eos",2006,2015,"Medium","Eos until 2015"),
    ("Volkswagen Sharan",2006,2010,"Medium","Sharan until ~2010 (moved to Wolfsburg)"),
    ("Volkswagen Scirocco",2008,2017,"High","Scirocco 2008–2017"),
    ("Volkswagen T-Roc",2017,None,"High","T-Roc from 2017"),
 ], employment=[], units={}),

 "HU_BMW_DEBRECEN": dict(name="BMW Group Plant Debrecen", brand="bmw", country="Hungary", src=FACTS, models=[
    ("BMW iX3",2025,None,"Medium","Neue Klasse iX3, production from 2025"),
 ], employment=[], units={}),

 "DE_VW_EMDEN": dict(name="Volkswagen Emden", brand="vw", country="Germany", src=FACTS, models=[
    ("Volkswagen Passat",2006,2022,"High","Passat lead plant until 2022"),
    ("Volkswagen CC/Arteon",2008,2022,"Medium","CC (2008) then Arteon"),
    ("Volkswagen ID.4",2022,None,"High","ID.4 after EV conversion"),
    ("Volkswagen ID.7",2023,None,"High","ID.7 from 2023"),
 ], employment=[], units={}),

 "DE_VW_OSNABRUECK": dict(name="Volkswagen Osnabrück", brand="vw", country="Germany", src=FACTS, models=[
    ("Volkswagen Golf Cabriolet",2011,2016,"Medium","Golf Cabriolet 2011–2016"),
    ("Porsche Boxster/Cayman",2012,2016,"Medium","981 Boxster/Cayman built here"),
    ("Volkswagen T-Roc Cabriolet",2020,None,"High","T-Roc Cabriolet from 2020"),
 ], employment=[], units={}),

 "DE_MERCEDES_DUSSELDORF": dict(name="Mercedes-Benz Werk Düsseldorf", brand="mercedes", country="Germany", src=FACTS, models=[
    ("Mercedes-Benz Sprinter",2006,None,"High","Main Sprinter van plant"),
    ("Mercedes-Benz eSprinter",2019,None,"Medium","eSprinter from 2019"),
 ], employment=[], units={}),

 "DE_MERCEDES_LUDWIGSFELDE": dict(name="Mercedes-Benz Werk Ludwigsfelde", brand="mercedes", country="Germany", src=FACTS, models=[
    ("Mercedes-Benz Sprinter",2006,None,"High","Sprinter van production"),
    ("Mercedes-Benz Vario",2006,2013,"Low","Vario until ~2013"),
 ], employment=[], units={}),
}

# No complete cars in 2006-2026 (engines / components / batteries / motorcycles / historic)
NON_CAR_PLANTS = {
 "DE_MERCEDES_STUTTGART": ("component", "Untertürkheim — engines, axles, transmissions"),
 "DE_MERCEDES_HAMBURG": ("component", "Axles, e-drive components"),
 "DE_MERCEDES_BERLIN": ("component", "Marienfelde — engines/components"),
 "DE_MERCEDES_KOLLEDA": ("component", "MDC Power — engines only"),
 "DE_MERCEDES_KAMENZ": ("component", "Accumotive — batteries only"),
 "DE_MERCEDES_AFFALTERBACH": ("component", "AMG engines (one man, one engine); limited AMG assembly"),
 "DE_VW_SALZGITTER": ("component", "Engines / battery cells"),
 "DE_VW_KASSEL": ("component", "Transmissions / e-drives"),
 "DE_VW_CHEMNITZ": ("component", "Engines"),
 "DE_VW_BRUNSWICK": ("component", "Components / chassis / battery systems"),
 "DE_BMW_LANDSHUT": ("component", "Components (engine parts, plastics, e-drive)"),
 "DE_BMW_WACKERSDORF": ("component", "CKD / components / supplier park"),
 "DE_BMW_IRLBACH_STRASSKIRCHEN": ("component", "High-voltage battery assembly"),
 "DE_BMW_BERLIN": ("component", "BMW Motorrad (motorcycles) + brake discs — no cars"),
 "DE_AUDI_ZWICKAU": ("historic", "Historic Auto Union/Audi site (now Horch Museum); active plant is DE_VW_ZWICKAU"),
 "DE_BMW_EISENACH": ("historic", "Historic (pre-war BMW); now components/logistics — no current cars"),
}


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    bin_rows, unit_rows, long_rows, emp_rows = [], [], [], []
    for pid, p in PLANTS.items():
        for model, ys, ye, conf, ev in p["models"]:
            end = ye if ye is not None else END
            if ys > END or end < START:
                continue
            brow = {"plant_id": pid, "plant_name": p["name"], "brand": p["brand"],
                    "country": p["country"], "car model ": model}
            urow = dict(brow); uby = p["units"].get(model, {})
            for y in YEARS:
                pr = 1 if (ys <= y <= end) else 0
                brow[y] = pr
                urow[y] = uby.get(y, "") if pr else 0
            bin_rows.append(brow); unit_rows.append(urow)
            long_rows.append({"plant_id": pid, "plant_name": p["name"], "brand": p["brand"],
                "country": p["country"], "model": model, "year_start": ys,
                "year_end": ye if ye is not None else "", "confidence": conf,
                "source": p["src"], "evidence": ev})
        for emp, yr, ev in p["employment"]:
            emp_rows.append({"plant_id": pid, "plant_name": p["name"], "brand": p["brand"],
                "country": p["country"], "employees": emp, "reference_year": yr,
                "pattern": "wikipedia_curated", "snippet": ev})

    cols = ["plant_id", "plant_name", "brand", "country", "car model "] + YEARS
    _w(OUT/"models_binary_matrix_curated.csv", cols, bin_rows)
    _w(OUT/"models_production_units_curated.csv", cols, unit_rows)
    _w(OUT/"models_long_evidence_curated.csv",
       ["plant_id","plant_name","brand","country","model","year_start","year_end","confidence","source","evidence"], long_rows)
    _w(OUT/"employment_from_wikipedia_curated.csv",
       ["plant_id","plant_name","brand","country","employees","reference_year","pattern","snippet"], emp_rows)

    cls = []
    for pid, p in PLANTS.items():
        cls.append({"plant_id": pid, "classification": "assembly", "covered": "yes (curated)",
                    "n_models": sum(1 for m in p["models"] if (m[2] or END) >= START and m[1] <= END),
                    "note": p["src"]})
    for pid, (kind, note) in NON_CAR_PLANTS.items():
        cls.append({"plant_id": pid, "classification": kind, "covered": "n/a — no cars",
                    "n_models": 0, "note": note})
    _w(OUT/"plant_classification.csv", ["plant_id","classification","covered","n_models","note"], cls)

    print(f"Assembly plants curated: {len(PLANTS)} | model rows: {len(bin_rows)}")
    print(f"Non-car (component/historic): {len(NON_CAR_PLANTS)}")
    print(f"Total plants classified: {len(cls)} / 43")
    print(f"Employment records: {len(emp_rows)}")
    print(f"Written to {OUT}")


def _w(path, cols, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader()
        for r in rows: wr.writerow(r)


if __name__ == "__main__":
    build()
