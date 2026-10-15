#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 EMPLOYMENT GAP-FILLER — LOCAL PRESS via TAVILY + Mistral (small-2506)
================================================================================
Fills the MISSING (plant, year) employee cells in your merged dataset by
searching local/regional press with the Tavily API, extracting a plant-specific
headcount with an LLM, and then VERIFYING it programmatically (the dominant risk
with press + LLM is fabricated citations, so a second LLM is not enough).

PIPELINE per missing (plant, year):
  1. Tavily search (local-language query) -> real result URLs + page text.
  2. Mistral reads the results and returns STRICT JSON:
        {employees, reference_year, exact_quote, source_url}
  3. PROGRAMMATIC verification (deterministic, not another LLM):
        - exact_quote is a literal substring of the fetched page text
        - the employees number actually appears inside exact_quote
        - the plant city/name appears in/near the quote (not a group total)
        - 30 <= employees <= PLANT_EMP_MAX (rejects group/worldwide totals)
        - source_url is one Tavily actually returned
        - reference_year in 2006-2026 AND that (plant, year) cell is still blank
  Only rows passing ALL checks are written. Existing values are never touched.

OUTPUT (same schema as employment_merged_comprehensive.csv, so it concatenates):
  press_gap_fills.csv   value_source=press_tavily, source_url + snippet(=quote) audit trail

USAGE
  export TAVILY_API_KEY=tvly-...
  export MISTRAL_API_KEY=...
  python3 fill_employment_gaps_press_tavily.py
  # limits: MAX_TAVILY_CALLS caps credit use; run again to resume (cached).
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
# Path to your merged dataset. Override with:  MERGED_CSV=/full/path.csv python3 ...
MERGED_CSV = os.environ.get("MERGED_CSV", str(_P.scraper_out / "employment_merged_comprehensive.csv"))
OUTPUT_DIR = Path(__file__).resolve().parent / "press_gap_output"
OUT_CSV = OUTPUT_DIR / "press_gap_fills.csv"
START_YEAR, END_YEAR = 2006, 2026
PLANT_EMP_MAX = int(os.environ.get("PLANT_EMP_MAX", "120000"))   # above this = group/worldwide total

# --- Tavily (search) ---
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
TAVILY_URL = "https://api.tavily.com/search"
TAVILY_MAX_RESULTS = 5
TAVILY_RPM = int(os.environ.get("TAVILY_RPM", "60"))
MAX_TAVILY_CALLS = int(os.environ.get("MAX_TAVILY_CALLS", "900"))   # protect the ~1000 free credits

# --- Mistral (extraction) ---
LLM_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "mistral-small-2506")
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
LLM_MAX_OUTPUT_TOKENS = 700
LLM_MAX_RPM = int(os.environ.get("LLM_MAX_RPM", "200"))
LLM_REQUEST_TIMEOUT = 60
LLM_MAX_RETRIES = 4

MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))
REQUEST_TIMEOUT = 40

# Plants to skip (pure component/engine/battery sites rarely get headcount press;
# saves credits). Empty the set to attempt every plant.
SKIP_PLANTS = set(filter(None, os.environ.get("SKIP_PLANTS", "").split(",")))

COUNTRY_LANG = {"Germany": "de", "Belgium": "fr", "France": "fr", "Spain": "es", "Italy": "it",
                "Netherlands": "nl", "Austria": "de", "Hungary": "hu", "Czechia": "cs",
                "Slovakia": "sk", "Poland": "pl", "Portugal": "pt", "United Kingdom": "en"}
EMP_WORD = {"de": "Mitarbeiter", "fr": "salariés", "es": "empleados", "it": "dipendenti",
            "hu": "alkalmazott", "pl": "pracownicy", "cs": "zaměstnanci", "sk": "zamestnanci",
            "pt": "funcionários", "nl": "werknemers", "en": "employees"}
# second query angle (different terms, no year) to catch gaps the first query missed
EMP_WORD2 = {"de": "Beschäftigte Arbeitsplätze", "fr": "emplois effectif", "es": "empleos plantilla",
             "it": "occupati dipendenti", "hu": "munkahely dolgozó", "pl": "zatrudnienie miejsca pracy",
             "cs": "zaměstnanci pracovní místa", "sk": "zamestnanci pracovné miesta",
             "pt": "empregos postos de trabalho", "nl": "banen werknemers", "en": "jobs workforce"}
# don't spend credits on social media (low authority); override via EXCLUDE_DOMAINS env
EXCLUDE_DOMAINS = list(filter(None, os.environ.get("EXCLUDE_DOMAINS",
    "facebook.com,linkedin.com,twitter.com,x.com,instagram.com,pinterest.com,youtube.com,reddit.com").split(",")))


def classify_source(url):
    net = urlparse(str(url or "")).netloc.lower().replace("www.", "")
    if not net:
        return "unknown"
    if net.endswith("wikipedia.org"):
        return "wikipedia"
    if any(x in net for x in ("facebook.", "linkedin.", "twitter.", "x.com", "instagram.", "pinterest.", "youtube.", "reddit.")):
        return "social"
    if any(x in net for x in ("bmwgroup", "mercedes-benz.com", "mercedes-benz.", "daimler", "volkswagen-newsroom",
                              "volkswagenag", "audi-mediacenter", "audi.com", "porsche.com", "vwpress", "seat.", "skoda")):
        return "official_company"
    if any(x in net for x in ("blogspot", "wordpress", "medium.com", "-fans", "fans.", "blog", "forum")):
        return "blog_forum"
    if any(x in net for x in ("zeitung", "nachrichten", "news", "presse", "press", "faz.", "spiegel", "handelsblatt",
                              "sueddeutsche", "stuttgarter", "merkur", "idowa", "pnp.", "rtbf", "lesoir", "lecho",
                              "sme.sk", "gazeta", "wyborcza", "publico", "expresso", "kurier", "derstandard",
                              "krone", "heute", "rnd.", "tagesschau", "wdr", "ndr")):
        return "press"
    return "other"

CACHE_DIR = OUTPUT_DIR / ".cache"
_HEADERS = {"User-Agent": "PlantEmploymentResearch/1.0"}

# ============================================================ infra
_tl = threading.local()
_tav_cache: dict = {}
_llm_cache: dict = {}
_locks = {k: threading.Lock() for k in ("tav", "llm", "tav_rate", "llm_rate", "stats", "count")}
_tav_times: deque = deque(); _llm_times: deque = deque()
_stats = {"tavily": 0, "llm": 0, "verified": 0, "rejected": 0, "no_result": 0, "last_error": ""}
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
    for path, tgt in ((CACHE_DIR / "tavily.pkl", "_tav_cache"), (CACHE_DIR / "llm.pkl", "_llm_cache")):
        if path.exists():
            try:
                globals()[tgt] = pickle.load(open(path, "rb"))
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
            return None                                   # credit guard
        _tav_call_count[0] += 1
    _gate(_tav_times, TAVILY_RPM, _locks["tav_rate"])
    _rec("tavily")
    try:
        r = sess().post(TAVILY_URL, timeout=REQUEST_TIMEOUT,
                        headers={"Authorization": f"Bearer {TAVILY_API_KEY}", "Content-Type": "application/json"},
                        json={"query": query, "max_results": TAVILY_MAX_RESULTS,
                              "include_raw_content": True, "search_depth": "basic",
                              "exclude_domains": EXCLUDE_DOMAINS})
        if r.status_code != 200:
            _rec("stats", 0, f"tavily HTTP {r.status_code}: {r.text[:120]}")
            return None
        results = r.json().get("results", []) or []
    except Exception as e:
        _rec("stats", 0, f"tavily exc {str(e)[:120]}")
        results = None
    if results is not None:
        with _locks["tav"]:
            _tav_cache[query] = results
    return results


# ============================================================ Mistral extraction
_PROMPT = """You read web search results (often local/regional press) to find the EMPLOYEE HEADCOUNT of ONE specific car plant.

Plant: {name} in {city} ({brand}), {country}
Target year: {year} (accept any year 2006-2026 if the text clearly states a figure for this plant)

Rules:
- Only the headcount of THIS plant in {city}. NEVER the company/group/worldwide total, NEVER a different plant, NEVER an area (m2) or a production/vehicle count.
- exact_quote MUST be copied VERBATIM (character-for-character) from the text below, a short sentence containing the number.
- source_url MUST be one of the result URLs listed below.
- reference_year = the year the figure refers to (from the quote/context).
- If there is no reliable plant-specific employee count, return employees = null.

Return VALID JSON ONLY:
{{"employees": <int or null>, "reference_year": <"YYYY" or null>, "exact_quote": "<verbatim sentence>", "source_url": "<one of the URLs>"}}

SEARCH RESULTS:
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
                _rec("stats", 0, f"mistral HTTP {r.status_code}: {r.text[:120]}"); return None
            txt = (((r.json().get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
            txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.I | re.M).strip()
            try:
                return json.loads(txt)
            except Exception:
                m = re.search(r"\{.*\}", txt, re.S)
                return json.loads(m.group(0)) if m else None
        except Exception as e:
            _rec("stats", 0, f"mistral exc {str(e)[:120]}"); time.sleep(1.5 * (a + 1))
    return None


# ============================================================ verification helpers
def _norm(t):
    return re.sub(r"\s+", " ", (t or "")).strip()


def _num_in(emp, text):
    """True if the integer `emp` appears in `text`, ignoring thousands separators."""
    compact = re.sub(r"(?<=\d)[ .,  ](?=\d)", "", text or "")
    return re.search(r"(?<!\d)" + str(emp) + r"(?!\d)", compact) is not None


def verify(rec, results, name, city):
    """Deterministic check of the LLM's claim against the real fetched text."""
    try:
        emp = int(rec.get("employees"))
    except (TypeError, ValueError):
        return None
    if emp < 30 or emp > PLANT_EMP_MAX:
        return None
    ry = rec.get("reference_year")
    try:
        ry = int(str(ry)[:4])
    except (TypeError, ValueError):
        return None
    if not (START_YEAR <= ry <= END_YEAR):
        return None
    quote = _norm(rec.get("exact_quote"))
    url = (rec.get("source_url") or "").strip()
    if not quote or not url:
        return None
    # source_url must be one Tavily returned; the quote must be in THAT page text
    page = None
    for res in results:
        if res.get("url") == url:
            page = _norm((res.get("raw_content") or "") + " " + (res.get("content") or ""))
            break
    if page is None:                                      # url not among results -> fabricated
        return None
    if quote not in page:                                 # quote not verbatim on the page -> fabricated
        return None
    if not _num_in(emp, quote):                           # the number isn't in the quote
        return None
    # plant context: city or a distinctive name token in the quote or its neighbourhood
    ctx = page[max(0, page.find(quote) - 120): page.find(quote) + len(quote) + 120].lower()
    toks = [city.lower()] + [w.lower() for w in re.findall(r"[A-Za-zÀ-ÿ]{4,}", name) if w.lower() not in
                             ("plant", "werk", "group", "audi", "bmw", "volkswagen", "mercedes", "porsche")]
    if not any(t and t in ctx for t in toks):
        return None
    return {"employees": emp, "reference_year": ry, "exact_quote": quote[:300], "source_url": url}


# ============================================================ data
def read_table(path):
    """Read .csv or .xlsx (picks the 'By archive year' sheet if present)."""
    p = str(path)
    if p.lower().endswith((".xlsx", ".xlsm", ".xls")):
        xl = pd.ExcelFile(path)
        sheet = "By archive year" if "By archive year" in xl.sheet_names else xl.sheet_names[0]
        return pd.read_excel(xl, sheet_name=sheet)
    return pd.read_csv(path)


def load_gaps():
    if not Path(MERGED_CSV).exists():
        sys.exit(f"[!] merged CSV not found: {MERGED_CSV}\n"
                 f"    Put employment_merged_comprehensive.csv in the script folder, or run with\n"
                 f"    MERGED_CSV=/full/path/to/employment_merged_comprehensive.csv python3 {Path(__file__).name}")
    df = read_table(MERGED_CSV)
    emp = pd.to_numeric(df.get("employees"), errors="coerce")
    yc = "year" if "year" in df.columns else "reference_year"
    filled = set()
    meta = {}
    for i in df.index:
        pid = str(df.at[i, "plant_id"])
        meta.setdefault(pid, {"plant_name": df.at[i, "plant_name"], "city": df.at[i, "city"],
                              "brand": df.at[i, "brand"], "country": df.at[i, "country"],
                              "group": df.at[i, "group"]})
        if pd.notna(emp.at[i]):
            y = pd.to_numeric(df.at[i, yc], errors="coerce")
            if pd.notna(y):
                filled.add((pid, int(y)))
    gaps = []
    for pid, m in meta.items():
        if pid in SKIP_PLANTS:
            continue
        for y in range(START_YEAR, END_YEAR + 1):
            if (pid, y) not in filled:
                gaps.append((pid, y))
    return gaps, filled, meta


def _row(pid, m, rec):
    return {"plant_id": pid, "country": m["country"], "group": m["group"], "plant_name": m["plant_name"],
            "city": m["city"], "brand": m["brand"], "year": rec["reference_year"], "employees": rec["employees"],
            "value_source": "press_tavily", "selection_basis": "press_verified", "extraction_method": "press_llm",
            "pattern": "tavily_mistral", "n_distinct_values": 1, "sources_present": "press", "value_spread": 0,
            "archive_year": rec["reference_year"], "source_url": rec["source_url"], "snippet": rec["exact_quote"],
            "source_tier": classify_source(rec["source_url"])}


# ============================================================ per-gap worker
def _run_query(q, year, m):
    """One query (cached by its own digest). Returns list of verified records (pre gap-filter)."""
    digest = hashlib.sha1(q.encode()).hexdigest()[:16]
    with _locks["llm"]:
        cached = _llm_cache.get(digest, "__miss__")
    if cached != "__miss__":
        return cached
    results = tavily_search(q)
    out = []
    if results:
        blob = "\n\n".join(f"[{i}] {r.get('url','')}\n{_norm((r.get('raw_content') or r.get('content') or ''))[:2500]}"
                           for i, r in enumerate(results))[:14000]
        rec = _llm(_PROMPT.format(name=m["plant_name"], city=m["city"], brand=m["brand"],
                                 country=m["country"], year=year, blob=blob))
        if rec:
            v = verify(rec, results, m["plant_name"], m["city"])
            if v:
                out.append(v)
            elif rec.get("employees") is not None:
                _rec("rejected")
    else:
        _rec("no_result")
    with _locks["llm"]:
        _llm_cache[digest] = out
    return out


def process_gap(pid, year, m, filled):
    lang = COUNTRY_LANG.get(m["country"], "en")
    queries = [
        f'"{m["plant_name"]}" {m["city"]} {EMP_WORD.get(lang, "employees")} {year}',   # precise (cached from pass 1)
        f'{m["brand"]} {m["city"]} {EMP_WORD2.get(lang, "jobs workforce")}',            # broader 2nd angle (no year)
    ]
    for q in queries:
        out = _run_query(q, year, m)
        good = [r for r in out if (pid, r["reference_year"]) not in filled]   # only blanks; never overwrite
        if good:
            for _ in good:
                _rec("verified")
            return [_row(pid, m, r) for r in good]
    return []


def run():
    if not TAVILY_API_KEY or not LLM_API_KEY:
        print("[!] set TAVILY_API_KEY and MISTRAL_API_KEY");
    load_caches()
    gaps, filled, meta = load_gaps()
    plants_with_data = {p for (p, _y) in filled}
    # best-first: plants that already have data (proven press coverage) first, recent years first
    gaps.sort(key=lambda g: (0 if g[0] in plants_with_data else 1, -g[1]))
    print(f"[run] {len(gaps)} blank (plant,year) cells to attempt | Tavily cap={MAX_TAVILY_CALLS} | model={LLM_MODEL}")
    rows = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(process_gap, pid, y, meta[pid], filled): (pid, y) for pid, y in gaps}
        done = 0
        for fut in as_completed(futs):
            try:
                rows += fut.result()
            except Exception as e:
                _rec("stats", 0, str(e)[:120])
            done += 1
            if done % 25 == 0:
                save_caches()
                print(f"  ...{done}/{len(gaps)} | verified={_stats['verified']} rejected={_stats['rejected']} "
                      f"tavily_used={_tav_call_count[0]}")
    save_caches()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["plant_id", "country", "group", "plant_name", "city", "brand", "year", "employees",
            "value_source", "selection_basis", "extraction_method", "pattern", "n_distinct_values",
            "sources_present", "value_spread", "archive_year", "source_url", "snippet", "source_tier", "suspect"]
    # one row per (plant, year); if several, keep first verified
    seen, dedup = set(), []
    for r in rows:
        k = (r["plant_id"], r["year"])
        if k not in seen:
            seen.add(k); dedup.append(r)
    # flag a value attributed to >=3 plants (hallucinated round number / shared total)
    vp = {}
    for r in dedup:
        vp.setdefault(r["employees"], set()).add(r["plant_id"])
    for r in dedup:
        r["suspect"] = "cross_plant_repeat" if len(vp.get(r["employees"], ())) >= 3 else ""
    pd.DataFrame(dedup, columns=cols).to_csv(OUT_CSV, index=False)
    print(f"\n[run] verified gap-fills: {len(dedup)}  (tavily calls used: {_tav_call_count[0]})")
    print(f"[run] stats: {_stats}")
    print(f"[run] written to {OUT_CSV}")


if __name__ == "__main__":
    run()
