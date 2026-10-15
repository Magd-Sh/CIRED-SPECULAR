#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 GAP FILLER — Tavily + Mistral   [timeline-aware, per-plant, multi-year]
================================================================================
Tavily sibling of fill_gaps_gdelt.py. Same anti-hallucination contract
(search -> LLM JSON -> PROGRAMMATIC verify -> fill blank cells only, never
overwrite), but the discovery layer is the Tavily Search API.

Why this exists next to fill_employment_gaps_press_tavily.py:
  - reads the NEW grid master (employment_merged_archive_final.xlsx)
  - searches ONCE PER PLANT (not per year) and lets ONE Mistral call pull EVERY
    year-tagged figure it can find -> far fewer API calls
  - extracts employment + production + models in a single pass
  - is timeline-aware (skips pre-existence / post-closure blanks)

Unlike GDELT Cloud, Tavily returns article text inline (raw_content), so there
is no separate page-fetch step and coverage is not limited to recent news.

PIPELINE (per plant that has >=1 real-gap year)
  1. Tavily search:  "<plant> <city>" employees production  (include_raw_content)
  2. ONE Mistral call extracts ALL year-tagged figures (employment/production/
     models), each with a verbatim quote + the result URL it came from.
  3. PROGRAMMATIC verification (identical to the other scrapers):
        - quote literal substring of the result's text
        - number appears inside that quote (separator-agnostic)
        - plant city / name within +-120 chars of the quote
        - employees in [30 .. PLANT_EMP_MAX]   (rejects group/worldwide totals)
        - reference year in [START_YEAR .. END_YEAR]
        - source_url is one Tavily actually returned
  4. Keep a verified figure only if (plant_id, reference_year) is a BLANK
     real-gap cell (pre-existence / post-closure years are skipped).

OUTPUTS (tavily_gap_output/)
  blank_classification.csv     every master blank labelled
  tavily_gap_employment.csv     verified employee counts (your employment schema)
  tavily_gap_production.csv     verified annual production figures
  tavily_gap_models.csv         models named as produced (cross-check w/ wiki)

  Feed the *_employment.csv into merge_press_into_master.py.

USAGE
  export TAVILY_API_KEY=tvly-...
  export MISTRAL_API_KEY=...
  MASTER_CSV='/Users/.../employment_merged_archive_final.xlsx' python3 fill_gaps_tavily.py

  CLASSIFY_ONLY=1 ...  -> only write blank_classification.csv (spends no API)
Reuses the press scraper's Tavily cache (press_gap_output/.cache) so overlapping
searches cost nothing.
================================================================================
"""
from __future__ import annotations
import os, re, sys, json, time, pickle, hashlib, threading
from pathlib import Path
from collections import deque
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# --- portability shim, added when this project was archived to INARI ---------
# The originals carried absolute paths from the machine they were written on.
# specular_paths resolves everything from the project root instead, so these
# scripts run wherever the folder is put. Set SPECULAR_ROOT to override.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from specular_paths import P as _P
# ---------------------------------------------------------------------------


try:
    import pandas as pd
except ImportError:
    sys.exit("pip install pandas openpyxl")

# ============================================================ CONFIG
MASTER_CSV = os.environ.get("MASTER_CSV",
    str(_P.scraper_out / "employment_merged_archive_final.xlsx"))
OUTPUT_DIR = Path(__file__).resolve().parent / "tavily_gap_output"
SHARED_TAVILY_CACHE = Path(__file__).resolve().parent / "press_gap_output" / ".cache" / "tavily.pkl"
START_YEAR, END_YEAR = 2006, 2026
PLANT_EMP_MAX = int(os.environ.get("PLANT_EMP_MAX", "120000"))

# --- Tavily -------------------------------------------------------------------
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
TAVILY_URL = "https://api.tavily.com/search"
TAVILY_MAX_RESULTS = int(os.environ.get("TAVILY_MAX_RESULTS", "6"))
TAVILY_SEARCH_DEPTH = os.environ.get("TAVILY_SEARCH_DEPTH", "basic")  # basic | advanced
TAVILY_RPM = int(os.environ.get("TAVILY_RPM", "60"))
MAX_TAVILY_CALLS = int(os.environ.get("MAX_TAVILY_CALLS", "300"))     # credit budget guard

# --- Mistral ------------------------------------------------------------------
LLM_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "mistral-small-2506")
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
LLM_MAX_OUTPUT_TOKENS = 1400
LLM_MAX_RPM = int(os.environ.get("LLM_MAX_RPM", "200"))
LLM_REQUEST_TIMEOUT = 60
LLM_MAX_RETRIES = 4

MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))
REQUEST_TIMEOUT = 40
PAGE_MAX_CHARS = 2800          # chars of each result kept for the blob
BLOB_MAX_CHARS = 16000

SOCIAL = ("facebook.", "linkedin.", "twitter.", "x.com", "instagram.", "pinterest.", "youtube.", "reddit.")
EXCLUDE_DOMAINS = list(filter(None, os.environ.get("EXCLUDE_DOMAINS",
    "facebook.com,linkedin.com,twitter.com,x.com,instagram.com,pinterest.com,youtube.com,reddit.com").split(",")))
COUNTRY_LANG = {"Germany": "de", "Belgium": "fr", "Spain": "es", "Italy": "it", "Hungary": "hu",
                "Slovakia": "sk", "Poland": "pl", "Portugal": "pt", "Netherlands": "nl", "Austria": "de"}
EMP_WORD = {"de": "Mitarbeiter Beschäftigte", "fr": "employés effectif", "es": "empleados plantilla",
            "it": "dipendenti", "hu": "dolgozó", "sk": "zamestnancov", "pl": "pracowników",
            "pt": "trabalhadores", "nl": "werknemers"}

# --- Operational timeline: plant_id -> (start_year, end_year_or_None). --------
PLANT_OPERATIONAL = {
    "BE_AUDI_BRUSSELS": (1949, 2025), "DE_AUDI_INGOLSTADT": (1949, None),
    "DE_AUDI_NECKARSULM": (1969, None), "DE_AUDI_ZWICKAU": (1904, 1991),
    "DE_BMW_BERLIN": (1939, None), "DE_BMW_DINGOLFING": (1967, None),
    "DE_BMW_EISENACH": (1898, 1991), "DE_BMW_IRLBACH_STRASSKIRCHEN": (2023, None),
    "DE_BMW_LANDSHUT": (1984, None), "DE_BMW_LEIPZIG": (2005, None),
    "DE_BMW_MUNICH": (1922, None), "DE_BMW_REGENSBURG": (1986, None),
    "DE_BMW_WACKERSDORF": (1990, None), "DE_MERCEDES_AFFALTERBACH": (1967, None),
    "DE_MERCEDES_BERLIN": (1902, None), "DE_MERCEDES_BREMEN": (1938, None),
    "DE_MERCEDES_DUSSELDORF": (1962, None), "DE_MERCEDES_HAMBURG": (1935, None),
    "DE_MERCEDES_KAMENZ": (2012, None), "DE_MERCEDES_KOLLEDA": (2003, None),
    "DE_MERCEDES_LUDWIGSFELDE": (1991, None), "DE_MERCEDES_RASTATT": (1992, None),
    "DE_MERCEDES_SINDELFINGEN": (1915, None), "DE_MERCEDES_STUTTGART": (1904, None),
    "DE_PORSCHE_LEIPZIG": (2002, None), "DE_PORSCHE_ZUFFENHAUSEN": (1938, None),
    "DE_VW_BRUNSWICK": (1938, None), "DE_VW_CHEMNITZ": (1990, None),
    "DE_VW_DRESDEN": (2002, 2025), "DE_VW_EMDEN": (1964, None),
    "DE_VW_KASSEL": (1958, None), "DE_VW_OSNABRUECK": (1953, None),
    "DE_VW_SALZGITTER": (1970, None), "DE_VW_WOLFSBURG": (1938, None),
    "DE_VW_ZWICKAU": (1990, None), "ES_VW_NAVARRA": (1965, None),
    "HU_AUDI_GYOR": (1993, None), "HU_BMW_DEBRECEN": (2025, None),
    "HU_MERCEDES_KECSKEMET": (2012, None), "PL_VWN_POZNAN": (1993, None),
    "PL_VWN_WRZESNIA": (2016, None), "PT_VW_PALMELA": (1995, None),
    "SK_VW_BRATISLAVA": (1991, None),
}


def classify_source(url):
    net = urlparse(str(url or "")).netloc.lower().replace("www.", "")
    if not net:
        return "unknown"
    if net.endswith("wikipedia.org"):
        return "wikipedia"
    if any(x in net for x in SOCIAL):
        return "social"
    if any(x in net for x in ("bmwgroup", "mercedes-benz.com", "mercedes-benz.", "daimler", "volkswagen-newsroom",
                              "volkswagenag", "audi-mediacenter", "audi.com", "porsche.com", "vwpress", "seat.", "skoda")):
        return "official_company"
    if any(x in net for x in ("blogspot", "wordpress", "medium.com", "-fans", "fans.", "blog", "forum")):
        return "blog_forum"
    if any(x in net for x in ("zeitung", "nachrichten", "news", "presse", "press", "faz.", "spiegel", "handelsblatt",
                              "sueddeutsche", "stuttgarter", "merkur", "idowa", "pnp.", "rtbf", "lesoir", "lecho",
                              "sme.sk", "gazeta", "wyborcza", "publico", "expresso", "kurier", "derstandard",
                              "krone", "heute", "rnd.", "tagesschau", "wdr", "ndr", "reuters", "bloomberg",
                              "automobilwoche", "automotive", "autonews")):
        return "press"
    return "other"


CACHE_DIR = OUTPUT_DIR / ".cache"
_HEADERS = {"User-Agent": "PlantGapResearch/1.0"}

# ============================================================ infra
_tl = threading.local()
_tav_cache: dict = {}
_llm_cache: dict = {}
_locks = {k: threading.Lock() for k in ("tav", "llm", "tav_rate", "llm_rate", "stats", "count")}
_tav_times: deque = deque(); _llm_times: deque = deque()
_stats = {"tavily": 0, "llm": 0, "emp": 0, "prod": 0, "models": 0, "no_result": 0, "last_error": ""}
_tav_call_count = [0]


def sess():
    s = getattr(_tl, "s", None)
    if s is None:
        s = requests.Session(); s.headers.update(_HEADERS); _tl.s = s
    return s


def _gate(times, rpm, lock):
    if rpm <= 0:
        return
    while True:
        with lock:
            now = time.time()
            while times and now - times[0] > 60:
                times.popleft()
            if len(times) < rpm:
                times.append(now); return
            wait = 60 - (now - times[0]) + 0.05
        time.sleep(min(max(wait, 0.1), 5))


def _rec(k, n=1, err=""):
    with _locks["stats"]:
        _stats[k] = _stats.get(k, 0) + n
        if err:
            _stats["last_error"] = err


def load_caches():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for path in (SHARED_TAVILY_CACHE, CACHE_DIR / "tavily.pkl"):
        if path.exists():
            try:
                _tav_cache.update(pickle.load(open(path, "rb")))
            except Exception:
                pass
    p = CACHE_DIR / "llm.pkl"
    if p.exists():
        try:
            _llm_cache.update(pickle.load(open(p, "rb")))
        except Exception:
            pass


def save_caches():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with _locks["tav"]:
        pickle.dump(_tav_cache, open(CACHE_DIR / "tavily.pkl", "wb"))
    with _locks["llm"]:
        pickle.dump(_llm_cache, open(CACHE_DIR / "llm.pkl", "wb"))


# ============================================================ Tavily
def tavily_search(query):
    with _locks["tav"]:
        if query in _tav_cache:
            return _tav_cache[query]
    with _locks["count"]:
        if _tav_call_count[0] >= MAX_TAVILY_CALLS:
            return None
        _tav_call_count[0] += 1
    _gate(_tav_times, TAVILY_RPM, _locks["tav_rate"])
    _rec("tavily")
    try:
        r = sess().post(TAVILY_URL, timeout=REQUEST_TIMEOUT,
                        headers={"Authorization": f"Bearer {TAVILY_API_KEY}", "Content-Type": "application/json"},
                        json={"query": query, "max_results": TAVILY_MAX_RESULTS, "include_raw_content": True,
                              "search_depth": TAVILY_SEARCH_DEPTH, "exclude_domains": EXCLUDE_DOMAINS})
        if r.status_code != 200:
            _rec("stats", 0, f"tavily {r.status_code}: {r.text[:120]}"); return None
        results = r.json().get("results", []) or []
    except Exception as e:
        _rec("stats", 0, f"tavily exc {str(e)[:100]}"); results = None
    if results is not None:
        with _locks["tav"]:
            _tav_cache[query] = results
    return results


# ============================================================ Mistral
_PROMPT = """You are reading several web search results (local press / official rooms) about ONE car plant. Extract EVERY year-specific fact you can find about THIS plant.

Plant: {name} in {city} ({brand}), {country}

Return VALID JSON ONLY:
{{"employment": [{{"employees": <int>, "year": "YYYY", "quote": "<verbatim>", "source_url": "<the [n] url it came from>"}}],
  "production": [{{"units": <int>, "year": "YYYY", "quote": "<verbatim>", "source_url": "<url>"}}],
  "models":     [{{"model": "<name>", "year": "YYYY" or null, "quote": "<verbatim>", "source_url": "<url>"}}]}}

Rules:
- Everything must concern THIS plant in {city} ONLY. NEVER the company/group/worldwide total.
- employees = plant headcount (30..400000). units = vehicles BUILT in a single year at this plant (not cumulative).
- Attach the correct year to each figure (the year the figure refers to). If a result gives a figure "as of <year>", use that year.
- Each *quote MUST be copied VERBATIM (exact characters) from the result it came from, and must contain the number.
- source_url MUST be the [n] URL of the result the quote is in. Use [] for any list with nothing found.

RESULTS:
---
{blob}
---
"""


def _llm(prompt):
    _gate(_llm_times, LLM_MAX_RPM, _locks["llm_rate"])
    _rec("llm")
    for a in range(LLM_MAX_RETRIES + 1):
        try:
            r = sess().post(MISTRAL_URL, timeout=LLM_REQUEST_TIMEOUT,
                            headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"},
                            json={"model": LLM_MODEL, "temperature": 0, "max_tokens": LLM_MAX_OUTPUT_TOKENS,
                                  "response_format": {"type": "json_object"},
                                  "messages": [{"role": "user", "content": prompt}]})
            if r.status_code in (429, 503):
                time.sleep(min(4 * 2 ** a, 40)); continue
            if r.status_code != 200:
                _rec("stats", 0, f"mistral {r.status_code}: {r.text[:100]}"); return None
            txt = (((r.json().get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
            txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.I | re.M).strip()
            try:
                return json.loads(txt)
            except Exception:
                m = re.search(r"\{.*\}", txt, re.S)
                return json.loads(m.group(0)) if m else None
        except Exception as e:
            _rec("stats", 0, f"mistral exc {str(e)[:100]}"); time.sleep(1.5 * (a + 1))
    return None


# ============================================================ verification helpers
def _norm(t):
    return re.sub(r"\s+", " ", str(t or "")).strip()


def _num_in(n, text):
    compact = re.sub(r"(?<=\d)[ .,   ](?=\d)", "", text or "")
    return re.search(r"(?<!\d)" + str(int(n)) + r"(?!\d)", compact) is not None


def _ctx_has_plant(page, quote, name, city):
    i = page.find(quote)
    if i < 0:
        return False
    ctx = page[max(0, i - 120): i + len(quote) + 120].lower()
    toks = [city.lower()] + [w.lower() for w in re.findall(r"[A-Za-zÀ-ÿ]{4,}", name)
                             if w.lower() not in ("plant", "werk", "group", "audi", "bmw", "volkswagen", "mercedes", "porsche")]
    return any(t and t in ctx for t in toks)


def _year_ok(y):
    try:
        yy = int(str(y)[:4])
        return START_YEAR <= yy <= END_YEAR, yy
    except (TypeError, ValueError):
        return False, None


# ============================================================ data / classification
def read_table(path):
    p = str(path)
    if p.lower().endswith((".xlsx", ".xlsm", ".xls")):
        xl = pd.ExcelFile(path)
        sheet = "By archive year" if "By archive year" in xl.sheet_names else xl.sheet_names[0]
        return pd.read_excel(xl, sheet_name=sheet)
    return pd.read_csv(path)


def load_master():
    if not Path(MASTER_CSV).exists():
        sys.exit(f"[!] master not found: {MASTER_CSV}  (set MASTER_CSV=...)")
    df = read_table(MASTER_CSV)
    emp = pd.to_numeric(df.get("employees"), errors="coerce")
    yc = "year" if "year" in df.columns else "reference_year"
    filled, meta = set(), {}
    for i in df.index:
        pid = str(df.at[i, "plant_id"])
        meta.setdefault(pid, {"plant_name": df.at[i, "plant_name"], "city": df.at[i, "city"],
                              "brand": df.at[i, "brand"], "country": df.at[i, "country"],
                              "group": df.at[i, "group"] if "group" in df.columns else ""})
        if pd.notna(emp.at[i]):
            y = pd.to_numeric(df.at[i, yc], errors="coerce")
            if pd.notna(y):
                filled.add((pid, int(y)))
    return filled, meta


def classify_blanks(filled, meta):
    rows, gap_years = [], {}
    for pid, m in meta.items():
        start, end = PLANT_OPERATIONAL.get(pid, (START_YEAR, None))
        end = end if end is not None else END_YEAR
        for y in range(START_YEAR, END_YEAR + 1):
            if (pid, y) in filled:
                continue
            if y < start:
                label = "pre_existence"
            elif y > end:
                label = "post_closure"
            else:
                label = "real_gap"; gap_years.setdefault(pid, set()).add(y)
            rows.append({"plant_id": pid, "plant_name": m["plant_name"], "year": y, "classification": label,
                         "operational_start": start, "operational_end": end if end != END_YEAR else ""})
    return rows, gap_years


# ============================================================ per-plant worker
def process(pid, m, gap_years):
    lang = COUNTRY_LANG.get(m["country"], "en")
    extra = EMP_WORD.get(lang, "")
    query = f'"{m["plant_name"]}" {m["city"]} employees jobs production {extra}'.strip()
    digest = hashlib.sha1(("TAV|" + query).encode()).hexdigest()[:16]
    with _locks["llm"]:
        if digest in _llm_cache:
            return _llm_cache[digest]

    out = {"emp": [], "prod": [], "models": []}
    results = tavily_search(query)
    if not results:
        _rec("no_result")
        with _locks["llm"]:
            _llm_cache[digest] = out
        return out

    # build url -> text map and the LLM blob
    pages = {}
    for res in results:
        u = (res.get("url") or "").strip()
        txt = _norm((res.get("raw_content") or "") + " " + (res.get("content") or ""))
        if u and txt:
            pages[u] = txt
    blob = "\n\n".join(f"[{i}] {u}\n{pages[u][:PAGE_MAX_CHARS]}"
                       for i, u in enumerate(pages))[:BLOB_MAX_CHARS]
    rec = _llm(_PROMPT.format(name=m["plant_name"], city=m["city"], brand=m["brand"],
                              country=m["country"], blob=blob))
    if not rec:
        with _locks["llm"]:
            _llm_cache[digest] = out
        return out

    gy = gap_years.get(pid, set())

    def page_for(url):
        return pages.get((url or "").strip())

    for it in (rec.get("employment") or []):
        try:
            e = int(it.get("employees"))
        except (TypeError, ValueError):
            continue
        ok, ey = _year_ok(it.get("year"))
        q = _norm(it.get("quote")); url = (it.get("source_url") or "").strip()
        page = page_for(url)
        if (page and 30 <= e <= PLANT_EMP_MAX and ok and ey in gy and q and q in page
                and _num_in(e, q) and _ctx_has_plant(page, q, m["plant_name"], m["city"])):
            out["emp"].append({"employees": e, "year": ey, "quote": q[:300], "url": url, "tier": classify_source(url)})
            _rec("emp")
    for it in (rec.get("production") or []):
        try:
            p = int(it.get("units"))
        except (TypeError, ValueError):
            continue
        ok, py = _year_ok(it.get("year"))
        q = _norm(it.get("quote")); url = (it.get("source_url") or "").strip()
        page = page_for(url)
        if (page and 100 <= p <= 2_000_000 and ok and q and q in page
                and _num_in(p, q) and _ctx_has_plant(page, q, m["plant_name"], m["city"])):
            out["prod"].append({"production": p, "year": py, "quote": q[:300], "url": url, "tier": classify_source(url)})
            _rec("prod")
    for it in (rec.get("models") or []):
        nm = _norm(it.get("model")); q = _norm(it.get("quote")); url = (it.get("source_url") or "").strip()
        ok, my = _year_ok(it.get("year"))
        page = page_for(url)
        if nm and page and q and q in page and _ctx_has_plant(page, q, m["plant_name"], m["city"]):
            out["models"].append({"model": nm, "year": my if ok else "", "quote": q[:200],
                                   "url": url, "tier": classify_source(url)})
            _rec("models")

    with _locks["llm"]:
        _llm_cache[digest] = out
    return out


def run():
    if not TAVILY_API_KEY:
        sys.exit("[!] set TAVILY_API_KEY (your tvly-... key)")
    if not LLM_API_KEY and os.environ.get("CLASSIFY_ONLY", "0") == "0":
        sys.exit("[!] set MISTRAL_API_KEY (or run with CLASSIFY_ONLY=1)")
    load_caches()
    filled, meta = load_master()
    cls_rows, gap_years = classify_blanks(filled, meta)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cls_rows).to_csv(OUTPUT_DIR / "blank_classification.csv", index=False)
    import collections
    cc = collections.Counter(r["classification"] for r in cls_rows)
    plants_to_chase = [pid for pid in gap_years if gap_years[pid]]
    print(f"[run] blanks: {dict(cc)} | plants with real gaps: {len(plants_to_chase)} | "
          f"total real-gap years: {sum(len(v) for v in gap_years.values())} | model={LLM_MODEL}")
    if os.environ.get("CLASSIFY_ONLY", "0") != "0":
        print(f"[run] CLASSIFY_ONLY: wrote blank_classification.csv only (no API spent). -> {OUTPUT_DIR}")
        return

    emp_rows, prod_rows, mdl_rows = [], [], []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(process, pid, meta[pid], gap_years): pid for pid in plants_to_chase}
        done = 0
        for fut in as_completed(futs):
            pid = futs[fut]; m = meta[pid]
            try:
                o = fut.result()
            except Exception as e:
                _rec("stats", 0, str(e)[:100]); o = {"emp": [], "prod": [], "models": []}
            for r in o["emp"]:
                emp_rows.append({"plant_id": pid, "country": m["country"], "group": m["group"],
                    "plant_name": m["plant_name"], "city": m["city"], "brand": m["brand"], "year": r["year"],
                    "employees": r["employees"], "value_source": "tavily_press", "extraction_method": "press_llm",
                    "pattern": "tavily_mistral", "source_url": r["url"], "snippet": r["quote"],
                    "source_tier": r["tier"], "reference_year": r["year"]})
            for r in o["prod"]:
                prod_rows.append({"plant_id": pid, "plant_name": m["plant_name"], "brand": m["brand"],
                    "country": m["country"], "year": r["year"], "production_units": r["production"],
                    "value_source": "tavily_press", "source_url": r["url"], "snippet": r["quote"], "source_tier": r["tier"]})
            for r in o["models"]:
                mdl_rows.append({"plant_id": pid, "plant_name": m["plant_name"], "brand": m["brand"],
                    "country": m["country"], "year": r["year"], "model": r["model"],
                    "value_source": "tavily_press", "source_url": r["url"], "snippet": r["quote"], "source_tier": r["tier"]})
            done += 1
            if done % 10 == 0:
                save_caches()
                print(f"  ...{done}/{len(plants_to_chase)} | emp={_stats['emp']} prod={_stats['prod']} "
                      f"models={_stats['models']} tavily_used={_tav_call_count[0]}")
    save_caches()

    def dedupe(rows, keys):
        seen, out = set(), []
        for r in rows:
            k = tuple(r[x] for x in keys)
            if k not in seen:
                seen.add(k); out.append(r)
        return out
    emp_rows = dedupe(emp_rows, ["plant_id", "year", "employees"])
    prod_rows = dedupe(prod_rows, ["plant_id", "year", "production_units"])
    mdl_rows = dedupe(mdl_rows, ["plant_id", "year", "model"])
    pd.DataFrame(emp_rows).to_csv(OUTPUT_DIR / "tavily_gap_employment.csv", index=False)
    pd.DataFrame(prod_rows).to_csv(OUTPUT_DIR / "tavily_gap_production.csv", index=False)
    pd.DataFrame(mdl_rows).to_csv(OUTPUT_DIR / "tavily_gap_models.csv", index=False)
    print(f"\n[run] employment: {len(emp_rows)} | production: {len(prod_rows)} | models: {len(mdl_rows)}")
    print(f"[run] tavily credits used: {_tav_call_count[0]}")
    print(f"[run] stats: {_stats}")
    print(f"[run] written to {OUTPUT_DIR}")
    print("[next] merge: MASTER_CSV='%s' PRESS_CSV='%s' python3 merge_press_into_master.py"
          % (MASTER_CSV, OUTPUT_DIR / "tavily_gap_employment.csv"))


if __name__ == "__main__":
    run()
