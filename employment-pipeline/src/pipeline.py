#!/usr/bin/env python3
"""
Reproducible employment-merge pipeline
======================================
Raw plant-level employment scrapes  ->  one clean value per plant x year.

Run:  python src/pipeline.py         (from the employment-pipeline/ folder)

Everything the pipeline does is driven by the JSON/CSV files in config/.
No LLM is used here: the LLMs (Mistral/Haiku) only produced the RAW input CSVs
upstream (the `employees` + `snippet` in each file). This code is deterministic
and re-running it on the frozen data/raw/ inputs reproduces the outputs exactly.

Stages
------
1. load      - every raw source -> one long candidate table
2. contaminate - rule-based contamination filters (config/contamination_rules.json)
3. outlier   - per-plant robust outlier rule (config/parameters.json)
4. blocks    - curated manual blocklist (config/manual_blocks.csv)
5. consolidate - one value per (plant, archive_year): archived agreement/priority,
                 then press, then curated additions  (non-wiki layer)
6. wiki      - the single final Wikipedia file, placed on reference_year, gap-fill only
7. edits     - analyst manual edits (config/manual_edits.csv)
8. grid      - full plant x year grid 2006-2025
9. derive    - stability observations (carry a value forward only if a later snapshot
               reports the identical number) + full time-series grid
"""
import json, re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "data" / "raw"
CFG  = ROOT / "config"
OUT  = ROOT / "outputs"
DER  = ROOT / "data" / "derived"
OUT.mkdir(exist_ok=True); DER.mkdir(exist_ok=True)

def load_json(name): return json.loads((CFG / name).read_text())
REG   = load_json("source_registry.json")
PAR   = load_json("parameters.json")
CONTAM= load_json("contamination_rules.json")
META  = ["plant_id","country","group","plant_name","city","brand"]
Y0, Y1 = PAR["period"]["start_year"], PAR["period"]["end_year"]
def num(s): return pd.to_numeric(s, errors="coerce")

# ---------------------------------------------------------------- 1. LOAD
def _extract_method(status):
    s = str(status)
    if s in ("found_llm","found_llm_stuck","found_llm_partial"): return "llm"
    if s in ("found","found_stuck"): return "regex"
    return "unknown"

def _read_any(path):
    return pd.read_excel(path, dtype=str) if str(path).endswith(".xlsx") else pd.read_csv(path, dtype=str)

def load_source(label, filename, family, priority):
    d = _read_any(RAW / filename)
    d["emp"] = num(d["employees"])
    d["ref"] = num(d.get("reference_year"))
    d["arch"]= num(d.get("archive_year"))
    d["yrcol"] = num(d.get("year"))          # press files carry the applicable year here
    d = d[d["emp"].notna()].copy()
    if "extraction_method" in d and family == "press":
        d["extraction"] = d["extraction_method"]
    else:
        d["extraction"] = d["status"].map(_extract_method) if "status" in d else "unknown"
    # unify url + snippet column names across the source variants (coalesce whatever exists)
    url = pd.Series("", index=d.index)
    for c in ("archive_url","source_url","registry_url","original_url","original_current_url"):
        if c in d.columns:
            url = url.where(url.astype(str).str.len().gt(0), d[c].fillna(""))
    d["source_url"] = url
    d["snippet"] = d.get("snippet", "")
    d["source"], d["family"], d["priority"] = label, family, priority
    keep = ["plant_id"]+META[1:]+["arch","ref","emp","yrcol","source","family","priority","extraction","source_url","snippet"]
    for c in keep:
        if c not in d: d[c] = np.nan
    return d[keep]

def load_all():
    parts = []
    for grp in ("archived","press"):
        for s in REG[grp]:
            parts.append(load_source(s["label"], s["file"], grp, s["priority"]))
    C = pd.concat(parts, ignore_index=True)
    # placement year: archived -> archive_year ; press -> the file's `year` column (contemporaneous)
    C["place_year"] = C["arch"]
    press = C["family"].eq("press")
    C.loc[press, "place_year"] = C.loc[press, "yrcol"]
    # press files have no reference_year; the applicable year IS the reference year
    C.loc[press & C["ref"].isna(), "ref"] = C.loc[press & C["ref"].isna(), "yrcol"]
    C["place_year"] = num(C["place_year"])
    return C

# ---------------------------------------------------------------- 2. CONTAMINATION
def contamination_mask(C):
    sn  = C["snippet"].astype(str).str.lower()
    cty = C["city"].astype(str).str.lower()
    pid = C["plant_id"].astype(str)
    reason = pd.Series("", index=C.index)

    def setr(mask, name):
        take = mask & reason.eq("")
        reason.loc[take] = name

    # cross-plant duplicate snippet (keep the own-city match).
    # NOTE: applied ONLY to the Wikipedia source. Archived scrapes reuse marketing
    # boilerplate that legitimately names one plant (e.g. Munich's "Werk Munchen"
    # blurb); those are handled by the city_mismatch rule instead. Running this rule
    # on archived rows would drop the genuine Munich value.
    if CONTAM["cross_plant_duplicate"]["enabled"] and (C["family"] == "wikipedia").any():
        n = PAR["snippet_key_chars"]
        key = C["snippet"].astype(str).str[:n]
        dup = C.assign(_k=key).groupby("_k")["plant_id"].transform("nunique").gt(1)
        def own(i):
            s = sn[i].replace("ü","u").replace("ö","o"); c = cty[i].replace("ü","u").replace("ö","o")
            return bool(c) and c in s
        ownm = pd.Series([own(i) for i in C.index], index=C.index)
        setr((C["family"] == "wikipedia") & dup & ~ownm, "cross_plant_duplicate")

    # wrong-entity keyword rules
    for rule in CONTAM["wrong_entity"]["rules"]:
        m = pd.Series(True, index=C.index)
        for tok in rule["tokens"]:
            m &= sn.str.contains(re.escape(tok))
        if "require_absent_in_plant_id" in rule:
            m &= ~pid.str.contains(rule["require_absent_in_plant_id"])
        setr(m, "wrong_entity")

    # city-mismatch (curated list; see config note)
    for e in CONTAM["city_mismatch"]["entries"]:
        toks = "|".join(map(re.escape, e["tokens"]))
        m = sn.str.contains(toks, regex=True) & ~cty.str.contains(e["home_city"])
        setr(m, "city_mismatch")

    # production units next to the value
    words = "|".join(CONTAM["production_units"]["words"])
    def units(r):
        v = int(r["emp"]); s = str(r["snippet"]).lower()
        cand = {str(v), f"{v:,}", f"{v:,}".replace(",","."), (f"{v//1000}.{v%1000:03d}" if v>=1000 else str(v))}
        return any(re.search(re.escape(p)+r"\s*("+words+")", s) for p in cand)
    setr(C.apply(units, axis=1), "production_units")

    # job announcement / historical / artifact
    setr(sn.str.contains(CONTAM["job_announcement"]["regex"], regex=True, na=False), "job_announcement")
    setr(sn.str.contains("|".join(CONTAM["historical"]["words"]), regex=True, na=False), "historical")
    setr(sn.str.contains(CONTAM["artifact"]["regex"], regex=True, na=False), "artifact")
    return reason

def wiki_contamination_mask(W):
    ws = CONTAM["wiki_specific"]; sn = W["snippet"].astype(str).str.lower()
    reason = contamination_mask(W)  # reuse the shared rules on the wiki rows
    for i in W.index:
        if reason[i]: continue
        pid, v, ry = str(W.at[i,"plant_id"]), int(W.at[i,"emp"]), W.at[i,"ref"]
        s = str(W.at[i,"snippet"]).lower()
        if any(k in pid for k in ws["company_level_plant_id_contains"]): reason[i]="wiki_company_level"; continue
        if v < ws["subunit_max_value"] or any(k in s for k in ws["subunit_keywords"]): reason[i]="wiki_subunit"; continue
        if ws["wrzesnia_plant_id_contains"] in pid and ry==ry and ry < ws["wrzesnia_min_year"]: reason[i]="wiki_wrzesnia_pre_open"; continue
    return reason

# ---------------------------------------------------------------- 3. OUTLIER
def outlier_mask(C):
    p = PAR["outlier_rule"]; flag = pd.Series(False, index=C.index)
    for _, g in C.groupby("plant_id"):
        med = g["emp"].median()
        if len(g) >= p["min_values_per_plant"] and med and med==med:
            r = g["emp"]/med
            m = ((r>p["ratio_high"]) | (r<1/p["ratio_low_divisor"])) & ((g["emp"]-med).abs() > p["abs_gap_min"])
            flag.loc[g.index[m]] = True
    return flag

# ---------------------------------------------------------------- 4. BLOCKS
def block_mask(C):
    blocks = pd.read_csv(CFG/"manual_blocks.csv")
    m = pd.Series(False, index=C.index)
    for b in blocks.itertuples():
        m |= C["plant_id"].str.contains(b.plant_id_contains, regex=False) & (C["emp"]==b.employee_value)
    return m

# ---------------------------------------------------------------- 5. CONSOLIDATE (non-wiki)
def consolidate_non_wiki(C):
    C = C[C["place_year"].between(Y0, Y1)].copy()
    rows = []
    for (pid, yr), g in C.groupby(["plant_id","place_year"]):
        meta = g.iloc[0][META].to_dict()
        arch = g[g["family"]=="archived"]
        if len(arch):
            vc = arch["emp"].value_counts()
            if vc.iloc[0] >= 2:                      # 2+ archived sources agree
                pick = arch[arch["emp"]==vc.index[0]].sort_values("priority").iloc[0]; basis=f"archived_agree_{int(vc.iloc[0])}"
            else:
                pick = arch.sort_values("priority").iloc[0]; basis="archived_"+pick["source"]
        else:                                         # press gap-fill (priority order)
            pick = g.sort_values("priority").iloc[0]; basis="press_"+pick["source"]
        rows.append({**meta, "plant_id":pid, "year":int(yr), "employees":int(pick["emp"]),
                     "value_source":pick["source"], "selection_basis":basis, "extraction_method":pick["extraction"],
                     "reference_year": int(pick["ref"]) if pick["ref"]==pick["ref"] else int(yr),
                     "source_tier":"", "source_url":pick["source_url"], "snippet":pick["snippet"]})
    return pd.DataFrame(rows)

def add_curated(V, plants_meta):
    cur = pd.read_csv(CFG/"curated_additions.csv")
    have = set(zip(V["plant_id"], V["year"]))
    add = []
    for c in cur.itertuples():
        if (c.plant_id, int(c.reference_year)) in have: continue
        m = plants_meta.get(c.plant_id)
        if not m: continue
        add.append({**m, "plant_id":c.plant_id, "year":int(c.reference_year), "employees":int(c.employees),
                    "value_source":c.source_label, "selection_basis":"curated_addition", "extraction_method":"llm",
                    "reference_year":int(c.reference_year), "source_tier":"", "source_url":c.source_url, "snippet":c.snippet})
    return pd.concat([V, pd.DataFrame(add)], ignore_index=True) if add else V

def add_manual(V, plants_meta):
    """Analyst-entered press figures that have no raw scrape file (provenance in the URLs)."""
    ma = pd.read_csv(CFG/"manual_additions.csv")
    have = set(zip(V["plant_id"], V["year"]))
    add = []
    for c in ma.itertuples():
        if (c.plant_id, int(c.reference_year)) in have: continue
        m = plants_meta.get(c.plant_id)
        if not m: continue
        add.append({**m, "plant_id":c.plant_id, "year":int(c.reference_year), "employees":int(c.employees),
                    "value_source":"press_archive", "selection_basis":"analyst_manual", "extraction_method":"manual",
                    "reference_year":int(c.reference_year), "source_tier":"press", "source_url":c.source_url, "snippet":c.snippet})
    return pd.concat([V, pd.DataFrame(add)], ignore_index=True) if add else V

# ---------------------------------------------------------------- 6. WIKI
def add_wiki(V, plants_meta):
    w = REG["wikipedia"]
    W = load_source(w["label"], w["file"], "wikipedia", 99)
    W = W[W["ref"].notna()].copy()
    reason = wiki_contamination_mask(W)
    W = W[reason.eq("")].copy()
    # one value per plant x reference_year: prefer regex+llm, then most contemporaneous
    W["pri"] = W["extraction"].map({"regex+llm":0,"llm":1}).fillna(2)
    W = W.sort_values(["pri"]).drop_duplicates(["plant_id","ref"])
    W = W[W["ref"].between(Y0, Y1)]
    have = set(zip(V["plant_id"], V["year"]))
    add = []
    for r in W.itertuples():
        if (r.plant_id, int(r.ref)) in have: continue
        m = plants_meta.get(r.plant_id)
        if not m: continue
        add.append({**m, "plant_id":r.plant_id, "year":int(r.ref), "employees":int(r.emp),
                    "value_source":"wiki_final", "selection_basis":"wiki_reference_year", "extraction_method":r.extraction,
                    "reference_year":int(r.ref), "source_tier":"wikipedia", "source_url":r.source_url, "snippet":r.snippet})
    return pd.concat([V, pd.DataFrame(add)], ignore_index=True) if add else V

# ---------------------------------------------------------------- 7. MANUAL EDITS
def apply_edits(V):
    ed = pd.read_csv(CFG/"manual_edits.csv")
    for e in ed.itertuples():
        if e.action == "remove_plant":
            V = V[V["plant_id"] != e.plant_id]
        elif e.action == "remove_cell":
            V = V[~((V["plant_id"]==e.plant_id) & (V["year"]==int(e.year)) & (V["employees"]==int(e.employee_value)))]
    return V

# ---------------------------------------------------------------- 9. STABILITY DERIVATION
def stability_observations(V, C_clean):
    """Fresh value at each reference_year; carry forward only if a later snapshot
    reports the identical value (see METHODS)."""
    snaps = C_clean[["plant_id","arch","ref","emp"]].dropna(subset=["arch","ref"]).copy()
    snaps[["arch","ref","emp"]] = snaps[["arch","ref","emp"]].astype(int)
    rows = []
    for pid, vg in V.groupby("plant_id"):
        resolved = {int(r.reference_year): int(r.employees) for r in vg.itertuples()}
        occ = {ry: (val, 0) for ry, val in resolved.items()}          # fresh at reference_year
        for s in snaps[snaps["plant_id"]==pid].itertuples():          # carried where value repeats
            if s.arch > s.ref and s.ref in resolved and s.emp == resolved[s.ref] and s.arch not in occ:
                occ[s.arch] = (s.emp, s.arch - s.ref)
        m = vg.iloc[0]
        for oy,(val,carr) in occ.items():
            base = vg[vg["reference_year"]==(oy if carr==0 else oy-carr)].iloc[0]
            rows.append({"plant_id":pid,"country":m["country"],"group":m["group"],"plant_name":m["plant_name"],
                         "city":m["city"],"brand":m["brand"],"obs_year":oy,"employees":val,
                         "reference_year":oy-carr,"carried_years":carr,"kind":"fresh" if carr==0 else "carried",
                         "value_source":base["value_source"],"source_url":base["source_url"],"snippet":base["snippet"]})
    O = pd.DataFrame(rows).drop_duplicates(["plant_id","obs_year"])
    return O.sort_values(["group","plant_name","obs_year"]).reset_index(drop=True)

def full_grid(O):
    plants = O.drop_duplicates("plant_id")[META]
    grid = plants.assign(k=1).merge(pd.DataFrame({"obs_year":range(Y0,Y1+1),"k":1}), on="k").drop(columns="k")
    G = grid.merge(O.drop(columns=[c for c in META if c!="plant_id"]), on=["plant_id","obs_year"], how="left")
    pre = O[~O["obs_year"].between(Y0,Y1)]                            # keep pre-2006 obs as extra rows
    G = pd.concat([G, pre], ignore_index=True)
    G["employees"] = num(G["employees"]).astype("Int64")
    return G.sort_values(["group","plant_name","obs_year"]).reset_index(drop=True)

# ---------------------------------------------------------------- MAIN
def main():
    C = load_all()
    reason = contamination_mask(C)
    C.assign(reason=reason)[reason.ne("")][["plant_id","plant_name","place_year","emp","source","snippet"]]\
        .assign(reason=reason[reason.ne("")]).to_csv(OUT/"contamination_removed.csv", index=False)
    C = C[reason.eq("")].copy()
    C = C[~outlier_mask(C)].copy()
    C = C[~block_mask(C)].copy()
    C_clean = C.copy()

    # full plant metadata (every plant in the registry, incl. plants with no scraped value)
    reg = _read_any(RAW / REG["archived"][0]["file"]).drop_duplicates("plant_id")
    plants_meta = reg.set_index("plant_id")[META[1:]].to_dict("index")
    plants_meta = {k: {"plant_id":k, **v} for k,v in plants_meta.items()}

    V = consolidate_non_wiki(C)
    V = add_curated(V, plants_meta)
    V = add_manual(V, plants_meta)
    V = add_wiki(V, plants_meta)
    V = apply_edits(V)
    V = V.drop_duplicates(["plant_id","year"]).sort_values(["group","plant_name","year"]).reset_index(drop=True)

    cols = META+["year","employees","value_source","selection_basis","extraction_method","reference_year","source_tier","source_url","snippet"]
    V[cols].to_csv(OUT/"employment_final.csv", index=False)

    O = stability_observations(V, C_clean)
    O.to_csv(OUT/"employment_observations_stability.csv", index=False)
    full_grid(O).to_csv(OUT/"employment_timeseries_full.csv", index=False)

    print(f"final values      : {len(V)}")
    print(f"  non-wiki        : {(V['value_source']!='wiki_final').sum()}")
    print(f"  wiki_final      : {(V['value_source']=='wiki_final').sum()}")
    print(f"stability obs     : {len(O)}  ({(O['kind']=='carried').sum()} carried)")
    print(f"plants            : {V['plant_id'].nunique()}")
    return V

if __name__ == "__main__":
    main()
