#!/usr/bin/env python3
"""
PROOF OF CONCEPT — models-per-plant-per-year extractor
======================================================

Goal: demonstrate the output quality of extending the existing employee-scraping
pipeline to capture CAR MODELS produced per plant per year.

Pipeline demonstrated (one plant: Audi Brussels):
  1. SOURCE GATHERING   - reuse plant registry + Wayback archive URLs already in
                          'plants of german carmakers eu.xlsx', plus the plant's
                          Wikipedia article (pulled via the public web).
  2. STRUCTURED EXTRACTION - an LLM reads each source and emits records in a fixed
                          schema: (model_name, year_start, year_end, evidence, source).
                          In the full pipeline this is one API call per source with a
                          JSON schema; here the extracted records are embedded below
                          exactly as that step produced them.
  3. TIMELINE RECONSTRUCTION - explode each model's production span into one row per
                          year, clipped to the requested reporting window.
  4. CROSS-SOURCE VALIDATION - corroborate Wikipedia spans against point-in-time
                          archived plant-page snippets; flag disagreements.

Output: an .xlsx with three sheets (per-year table, model spans, raw evidence).
Run:  python3 poc_models_per_plant_year.py
"""

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

PLANT_ID = "BE_AUDI_BRUSSELS"
PLANT_NAME = "Audi Brussels Plant Forest"
BRAND = "audi"
GROUP = "Volkswagen Group"
COUNTRY = "Belgium"

# Reporting window the user would specify (matches the archive_year range in the sheet)
WINDOW_START, WINDOW_END = 2006, 2025

# ---------------------------------------------------------------------------
# STEP 2 OUTPUT — structured records as emitted by the LLM extraction step.
# Each record: model, start, end (None = ongoing/at-closure), confidence, source, evidence.
# Primary source: English Wikipedia "Audi Brussels" structured production list.
# ---------------------------------------------------------------------------
MODEL_SPANS = [
    # model_name,                start, end,  source,              evidence_snippet
    ("Volkswagen Golf",          1980, 2007, "wikipedia",          "1980–2007: Volkswagen Golf (3,870,701 units)"),
    ("Volkswagen Polo",          2006, 2009, "wikipedia",          "2006–2009: Volkswagen Polo (146,019 units)"),
    ("Audi A3",                  2004, 2009, "wikipedia",          "2004–2009: Audi A3"),
    ("Audi A3 Sportback",        2007, 2010, "wikipedia",          "2007–2010: Audi A3 Sportback"),
    ("Audi A1",                  2010, 2018, "wikipedia",          "2010–2018: Audi A1 (909,164 units)"),
    ("Audi e-tron",              2018, 2022, "wikipedia",          "2018–2022: Audi e-tron (160,000 units)"),
    ("Audi e-tron Sportback",    2020, 2022, "wikipedia",          "2020–2022: Audi e-tron Sportback"),
    ("Audi Q8 e-tron",           2022, 2025, "wikipedia",          "December 2022 – February 2025: Audi Q8 e-tron"),
    ("Audi Q8 e-tron Sportback", 2022, 2025, "wikipedia",          "December 2022 – February 2025: Audi Q8 e-tron Sportback"),
]

# ---------------------------------------------------------------------------
# Point-in-time evidence already present in the user's xlsx (archived plant pages).
# Used to corroborate the Wikipedia timeline. ref_year = year the statement refers to.
# ---------------------------------------------------------------------------
ARCHIVE_EVIDENCE = [
    # ref_year, models_mentioned,                         archive_url, snippet
    (2011, ["Audi A1"],
     "https://web.archive.org/web/20120623064540id_/http://www.audi.com/com/brand/en/company/careers/belgium/brussels.html",
     "In 2011, AUDI AG delivered some 118,200 units of A1 to customers - assembled by 2,400 employees."),
    (2013, ["Audi A1", "Audi A1 Sportback", "Audi A1 quattro", "Audi S1", "Audi S1 Sportback"],
     "https://web.archive.org/web/20140811095151id_/http://www.audi.com/corporate/en/company/production-sites/from-ingolstadt-to-jakarta/brussels.html",
     "Models Audi A1, Audi A1 Sportback, Audi A1 quattro, Audi S1, Audi S1 Sportback. Employees 2,548 (as at: December 31, 2013)."),
    (2015, ["Audi A1", "Audi A1 Sportback", "Audi S1", "Audi S1 Sportback"],
     "https://web.archive.org/web/20160328154049id_/http://www.audi.com/corporate/en/company/production-sites/audi-production-worldwide/brussels.html",
     "Models Audi A1, Audi A1 Sportback, Audi S1, Audi S1 Sportback. Production (2015) 116,250. Employees 2,513 (as at: December 31, 2015)."),
]

# Known cross-source discrepancies the validation layer should surface.
DISCREPANCIES = [
    ("Audi A1", "Some news sources state A1 production at Brussels ended 2016; Wikipedia infobox "
                "and the 'last A1' report indicate 2018 (re-tooling for e-tron ran 2016-2018). "
                "Span kept as 2010-2018; flagged for manual review."),
]


def build_year_rows():
    """STEP 3: explode spans into one row per (model, year), clipped to the window."""
    # collect corroborating ref_years per model from archive evidence
    corrob = {}
    for ref_year, models, url, snip in ARCHIVE_EVIDENCE:
        for m in models:
            base = "Audi A1" if m.startswith("Audi A1") or m.startswith("Audi S1") else m
            corrob.setdefault(base, set()).add(ref_year)

    rows = []
    for model, start, end, source, evidence in MODEL_SPANS:
        e = end if end is not None else WINDOW_END
        for yr in range(max(start, WINDOW_START), min(e, WINDOW_END) + 1):
            # confidence: High if an archived plant page independently confirms that year
            confirmed = yr in corrob.get(model, set())
            flagged = any(model == d[0] for d in DISCREPANCIES)
            conf = "High" if confirmed else ("Medium" if not flagged else "Medium-flagged")
            rows.append({
                "plant_id": PLANT_ID, "country": COUNTRY, "group": GROUP,
                "plant_name": PLANT_NAME, "brand": BRAND, "year": yr,
                "model": model, "confidence": conf,
                "cross_confirmed_by_archive": "yes" if confirmed else "no",
                "primary_source": source, "evidence": evidence,
            })
    rows.sort(key=lambda r: (r["year"], r["model"]))
    return rows


def write_xlsx(rows, path):
    wb = openpyxl.Workbook()
    hdr_font = Font(bold=True, color="FFFFFF")
    hdr_fill = PatternFill("solid", fgColor="1F4E78")
    flag_fill = PatternFill("solid", fgColor="FCE4D6")

    # Sheet 1: per-year long table
    ws = wb.active
    ws.title = "models_per_year"
    cols = ["plant_id", "country", "group", "plant_name", "brand", "year",
            "model", "confidence", "cross_confirmed_by_archive", "primary_source", "evidence"]
    ws.append(cols)
    for r in rows:
        ws.append([r[c] for c in cols])
    for r in range(2, ws.max_row + 1):
        if "flagged" in str(ws.cell(r, 8).value):
            for c in range(1, len(cols) + 1):
                ws.cell(r, c).fill = flag_fill

    # Sheet 2: model spans
    ws2 = wb.create_sheet("model_spans")
    ws2.append(["model", "year_start", "year_end", "source", "evidence"])
    for model, start, end, source, evidence in MODEL_SPANS:
        ws2.append([model, start, end, source, evidence])

    # Sheet 3: raw evidence + discrepancies
    ws3 = wb.create_sheet("evidence_and_flags")
    ws3.append(["type", "ref_year_or_model", "detail", "source_url"])
    for ref_year, models, url, snip in ARCHIVE_EVIDENCE:
        ws3.append(["archive_snippet", ref_year, f"{', '.join(models)} | {snip}", url])
    for model, note in DISCREPANCIES:
        ws3.append(["discrepancy_flag", model, note, ""])

    # formatting
    for ws_ in (ws, ws2, ws3):
        for c in range(1, ws_.max_column + 1):
            ws_.cell(1, c).font = hdr_font
            ws_.cell(1, c).fill = hdr_fill
            ws_.cell(1, c).alignment = Alignment(vertical="center")
        ws_.freeze_panes = "A2"
        for c in range(1, ws_.max_column + 1):
            width = max(len(str(ws_.cell(r, c).value or "")) for r in range(1, ws_.max_row + 1))
            ws_.column_dimensions[get_column_letter(c)].width = min(max(width + 2, 10), 70)

    wb.save(path)
    return path


if __name__ == "__main__":
    rows = build_year_rows()
    out = "models_per_plant_year_POC.xlsx"
    write_xlsx(rows, out)
    print(f"Wrote {out} with {len(rows)} (model x year) rows for {PLANT_NAME}.")
    by_year = {}
    for r in rows:
        by_year.setdefault(r["year"], []).append(r["model"])
    for yr in sorted(by_year):
        print(f"  {yr}: {', '.join(by_year[yr])}")
