#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 GAP FILLER — GDELT Cloud (gdeltcloud.com)  +  Mistral   [timeline-aware]
================================================================================
A GDELT-Cloud sibling of fill_established_plant_gaps.py / fill_employment_gaps_
press_tavily.py. Same anti-hallucination contract (search -> LLM JSON ->
PROGRAMMATIC verify -> fill blank cells only, never overwrite), but the discovery
layer is GDELT Cloud's clustered-Stories API instead of Tavily.

PIPELINE (per plant that has >=1 real-gap year)
  1. GET /api/v2/stories?search=<plant> <city> employees production  (Bearer key)
     -> Story cards; each card carries up to 3 inline `top_articles` (URL+title).
     (optional GDELT_DEEP=1 also pulls /stories/{id}/articles for more URLs.)
  2. Fetch each article URL ourselves (GDELT returns links, not body text),
     strip HTML to text.
  3. ONE Mistral call per plant extracts ALL year-tagged figures it can find:
     employment[], production[], models[] — each with a verbatim quote + the
     article URL it came from.
  4. PROGRAMMATIC verification (identical contract to the other scrapers):
        - quote must be a literal substring of the fetched page
        - the number must appear inside that quote (separator-agnostic)
        - plant city / name must sit within +-120 chars of the quote
        - employees in [30 .. PLANT_EMP_MAX]  (rejects group/worldwide totals)
        - reference year in [START_YEAR .. END_YEAR]
        - source_url must be one we actually fetched
  5. A verified figure is kept only if (plant_id, reference_year) is a BLANK
     real-gap cell in the master (pre-existence / post-closure years are skipped).

OUTPUTS (gdelt_gap_output/)
  blank_classification.csv     every master blank labelled (real_gap/pre_existence/post_closure)
  gdelt_gap_employment.csv      verified employee counts (your employment schema)
  gdelt_gap_production.csv      verified annual production figures
  gdelt_gap_models.csv          models named as produced (press-sourced; cross-check w/ wiki)

  Feed the *_employment.csv into merge_press_into_master.py (fills blanks in place).

IMPORTANT — coverage caveat
  GDELT Cloud is a RECENT-news product (story corpus is recent; date windows are
  short). In practice it fills recent gap years (≈2023-2026) well and is weak for
  2006-2015. The timeline classifier still lists every real gap, but expect the
  verified fills to cluster in recent years. That's expected, not a bug.

USAGE
  export GDELT_API_KEY=gdelt_sk_...        # GDELT Cloud key (Bearer)
  export MISTRAL_API_KEY=...
  MASTER_CSV='/Users/.../employment_merged_archive_final.xlsx' python3 fill_gaps_gdelt.py

  CLASSIFY_ONLY=1 ...   -> only write blank_classification.csv (spends no API)
  GDELT_DEEP=1 ...      -> also pull full article lists per top story (more Query Units)
================================================================================
"""
from __future__ import annotations
import os, re, sys, json, time, html, pickle, hashlib, threading
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
OUTPUT_DIR = Path(__file__).resolve().parent / "gdelt_gap_output"
START_YEAR, END_YEAR = 2006, 2026
PLANT_EMP_MAX = int(os.environ.get("PLANT_EMP_MAX", "120000"))

# --- GDELT Cloud --------------------------------------------------------------
GDELT_API_KEY = os.environ.get("GDELT_API_KEY", "")
GDELT_BASE = os.environ.get("GDELT_BASE", "https://gdeltcloud.com/api/v2")
GDELT_STORIES = GDELT_BASE + "/stories"
GDELT_SORT = os.environ.get("GDELT_SORT", "significance")   # significance | recent
GDELT_LIMIT = int(os.environ.get("GDELT_LIMIT", "25"))      # stories per search (max 100)
GDELT_DEEP = os.environ.get("GDELT_DEEP", "0") != "0"       # also pull /stories/{id}/articles
GDELT_DEEP_STORIES = int(os.environ.get("GDELT_DEEP_STORIES", "5"))  # top N stories to expand
GDELT_RPM = int(os.environ.get("GDELT_RPM", "60"))
MAX_GDELT_CALLS = int(os.environ.get("MAX_GDELT_CALLS", "300"))  # Query-Unit budget guard
ARTICLES_PER_PLANT = int(os.environ.get("ARTICLES_PER_PLANT", "14"))  # URLs fetched per plant

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
PAGE_FETCH_TIMEOUT = 25
PAGE_MAX_CHARS = 6000          # chars of each fetched article kept for the blob
BLOB_MAX_CHARS = 16000

SOCIAL = ("facebook.", "linkedin.", "twitter.", "x.com", "instagram.", "pinterest.", "youtube.", "reddit.")
EXCLUDE_DOMAINS = set(filter(None, os.environ.get("EXCLUDE_DOMAINS",
    "facebook.com,linkedin.com,twitter.com,x.com,instagram.com,pinterest.com,youtube.com,reddit.com").split(",")))
COUNTRY_LANG = {"Germany": "de", "Belgium": "fr", "Spain": "es", "Italy": "it", "Hungary": "hu",
                "Slovakia": "sk", "Poland": "pl", "Portugal": "pt", "Netherlands": "nl", "Austria": "de"}
EMP_WORD = {"de": "Mitarbeiter Beschäftigte", "fr": "employés effectif", "es": "empleados plantilla",
            "it": "dipendenti", "hu": "dolgozó", "sk": "zamestnancov", "pl": "pracowników",
            "pt": "trabalhadores", "nl": "werknemers"}

# --- Operational timeline: plant_id -> (start_year, end_year_or_None). --------
# Same conservative map as fill_established_plant_gaps.py. Used ONLY to decide
# which blanks are real gaps worth chasing vs legitimately empty.
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
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; PlantGapResearch/1.0; +research)"}

# ============================================================ infra
_tl = threading.local()
_stories_cache: dict = {}
_page_cache: dict = {}
_llm_cache: dict = {}
_locks = {k: threading.Lock() for k in ("st", "pg", "llm", "st_rate", "llm_rate", "stats", "count")}
_st_times: deque = deque(); _llm_times: deque = deque()
_stats = {"gdelt": 0, "pages": 0, "llm": 0, "emp": 0, "prod": 0, "models": 0,
          "no_story": 0, "no_page": 0, "last_error": ""}
_gdelt_call_count = [0]
_quota_hit = [False]


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
    for name, target in (("stories.pkl", _stories_cache), ("pages.pkl", _page_cache), ("llm.pkl", _llm_cache)):
        p = CACHE_DIR / name
        if p.exists():
            try:
                target.update(pickle.load(open(p, "rb")))
            except Exception:
                pass


def save_caches():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with _locks["st"]:
        pickle.dump(_stories_cache, open(CACHE_DIR / "stories.pkl", "wb"))
    with _locks["pg"]:
        pickle.dump(_page_cache, open(CACHE_DIR / "pages.pkl", "wb"))
    with _locks["llm"]:
        pickle.dump(_llm_cache, open(CACHE_DIR / "llm.pkl", "wb"))


# ============================================================ GDELT Cloud
def _gdelt_get(url, params):
    """GET against GDELT Cloud with Bearer auth; honour 429 Retry-After + quota stop."""
    if _quota_hit[0]:
        return None
    with _locks["count"]:
        if _gdelt_call_count[0] >= MAX_GDELT_CALLS:
            return None
        _gdelt_call_count[0] += 1
    _gate(_st_times, GDELT_RPM, _locks["st_rate"])
    _rec("gdelt")
    for a in range(4):
        try:
            r = sess().get(url, params=params, timeout=REQUEST_TIMEOUT,
                           headers={"Authorization": f"Bearer {GDELT_API_KEY}"})
            if r.status_code == 429:
                code = ""
                try:
                    code = (r.json().get("error") or {}).get("code", "")
                except Exception:
                    pass
                if code == "QUOTA_EXCEEDED":
                    _quota_hit[0] = True
                    _rec("stats", 0, "GDELT QUOTA_EXCEEDED"); return None
                wait = float(r.headers.get("Retry-After", 5))
                time.sleep(min(max(wait, 1), 30)); continue
            if r.status_code != 200:
                _rec("stats", 0, f"gdelt {r.status_code}: {r.text[:120]}"); return None
            return r.json()
        except Exception as e:
            _rec("stats", 0, f"gdelt exc {str(e)[:100]}"); time.sleep(1.5 * (a + 1))
    return None


def gdelt_search_urls(query, country=None):
    """Run a Story search and return a de-duped, filtered list of article URLs."""
    ck = ("S", query, country, GDELT_SORT, GDELT_LIMIT, GDELT_DEEP)
    with _locks["st"]:
        if ck in _stories_cache:
            return _stories_cache[ck]
    params = {"search": query, "limit": GDELT_LIMIT, "sort": GDELT_SORT}
    if country:
        params["country"] = country
    j = _gdelt_get(GDELT_STORIES, params)
    urls, seen = [], set()
    if j and isinstance(j, dict):
        stories = j.get("data") or []
        for n, st in enumerate(stories):
            for art in (st.get("top_articles") or []):
                u = (art.get("url") or "").strip()
                if u and u not in seen and not _excluded(u):
                    seen.add(u); urls.append(u)
            # optionally expand the most significant stories to their full article list
            if GDELT_DEEP and n < GDELT_DEEP_STORIES and st.get("id"):
                more = _gdelt_get(f"{GDELT_STORIES}/{st['id']}/articles", {"limit": 25})
                for art in ((more or {}).get("data") or []):
                    u = (art.get("url") or "").strip()
                    if u and u not in seen and not _excluded(u):
                        seen.add(u); urls.append(u)
    urls = urls[:ARTICLES_PER_PLANT]
    with _locks["st"]:
        _stories_cache[ck] = urls
    return urls


def _excluded(url):
    net = urlparse(url).netloc.lower().replace("www.", "")
    return any(net == d or net.endswith("." + d) for d in EXCLUDE_DOMAINS)


# ============================================================ page fetch + html->text
_TAG_RE = re.compile(r"(?is)<(script|style|noscript|template|svg)[^>]*>.*?</\1>")
_ANY_TAG = re.compile(r"(?s)<[^>]+>")


def fetch_text(url):
    with _locks["pg"]:
        if url in _page_cache:
            return _page_cache[url]
    txt = ""
    try:
        r = sess().get(url, timeout=PAGE_FETCH_TIMEOUT, allow_redirects=True)
        ct = r.headers.get("Content-Type", "")
        if r.status_code == 200 and "html" in ct.lower():
            h = r.text
            h = _TAG_RE.sub(" ", h)
            h = _ANY_TAG.sub(" ", h)
            txt = _norm(html.unescape(h))[:PAGE_MAX_CHARS]
    except Exception as e:
        _rec("stats", 0, f"page exc {str(e)[:80]}")
    with _locks["pg"]:
        _page_cache[url] = txt
    _rec("pages")
    return txt


# ============================================================ Mistral
_PROMPT = """You are reading several recent news articles about ONE car plant. Extract EVERY year-specific fact you can find about THIS plant.

Plant: {name} in {city} ({brand}), {country}

Return VALID JSON ONLY:
{{"employment": [{{"employees": <int>, "year": "YYYY", "quote": "<verbatim>", "source_url": "<the [n] url it came from>"}}],
  "production": [{{"units": <int>, "year": "YYYY", "quote": "<verbatim>", "source_url": "<url>"}}],
  "models":     [{{"model": "<name>", "year": "YYYY" or null, "quote": "<verbatim>", "source_url": "<url>"}}]}}

Rules:
- Everything must concern THIS plant in {city} ONLY. NEVER the company/group/worldwide total.
- employees = plant headcount (30..400000). units = vehicles BUILT in a single year at this plant (not cumulative).
- Attach the correct year to each figure (the year the figure refers to). If an article gives a figure "as of <year>", use that year.
- Each *quote MUST be copied VERBATIM (exact characters) from the article it came from, and must contain the number.
- source_url MUST be the [n] URL of the article the quote is in. Use [] for any list with nothing found.

ARTICLES:
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
    compact = re.sub(r"(?<=\d)[ .,   ](?=\d)", "", text or "")
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
    """One GDELT search + page fetches + one LLM extraction; keep only verified
    figures whose reference year is a blank real-gap for this plant."""
    lang = COUNTRY_LANG.get(m["country"], "en")
    extra = EMP_WORD.get(lang, "")
    query = f'{m["plant_name"]} {m["city"]} employees jobs workforce production {extra}'.strip()
    digest = hashlib.sha1(("GDELT|" + query).encode()).hexdigest()[:16]
    with _locks["llm"]:
        if digest in _llm_cache:
            return _llm_cache[digest]

    out = {"emp": [], "prod": [], "models": []}
    urls = gdelt_search_urls(query, country=m["country"])
    if not urls:
        _rec("no_story")
        with _locks["llm"]:
            _llm_cache[digest] = out
        return out

    pages = {}
    for u in urls:
        t = fetch_text(u)
        if t and len(t) > 200:
            pages[u] = t
    if not pages:
        _rec("no_page")
        with _locks["llm"]:
            _llm_cache[digest] = out
        return out

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

    # employment
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
    # production
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
    # models
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
    if not GDELT_API_KEY:
        sys.exit("[!] set GDELT_API_KEY (your gdelt_sk_... GDELT Cloud key)")
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
                    "employees": r["employees"], "value_source": "gdelt_press", "extraction_method": "gdelt_llm",
                    "pattern": "gdeltcloud_mistral", "source_url": r["url"], "snippet": r["quote"],
                    "source_tier": r["tier"], "reference_year": r["year"]})
            for r in o["prod"]:
                prod_rows.append({"plant_id": pid, "plant_name": m["plant_name"], "brand": m["brand"],
                    "country": m["country"], "year": r["year"], "production_units": r["production"],
                    "value_source": "gdelt_press", "source_url": r["url"], "snippet": r["quote"], "source_tier": r["tier"]})
            for r in o["models"]:
                mdl_rows.append({"plant_id": pid, "plant_name": m["plant_name"], "brand": m["brand"],
                    "country": m["country"], "year": r["year"], "model": r["model"],
                    "value_source": "gdelt_press", "source_url": r["url"], "snippet": r["quote"], "source_tier": r["tier"]})
            done += 1
            if done % 10 == 0:
                save_caches()
                print(f"  ...{done}/{len(plants_to_chase)} | emp={_stats['emp']} prod={_stats['prod']} "
                      f"models={_stats['models']} gdelt_calls={_gdelt_call_count[0]} pages={_stats['pages']}")
            if _quota_hit[0]:
                print("  [!] GDELT quota exhausted — stopping new searches, finishing in-flight work.")
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
    pd.DataFrame(emp_rows).to_csv(OUTPUT_DIR / "gdelt_gap_employment.csv", index=False)
    pd.DataFrame(prod_rows).to_csv(OUTPUT_DIR / "gdelt_gap_production.csv", index=False)
    pd.DataFrame(mdl_rows).to_csv(OUTPUT_DIR / "gdelt_gap_models.csv", index=False)
    print(f"\n[run] employment: {len(emp_rows)} | production: {len(prod_rows)} | models: {len(mdl_rows)}")
    print(f"[run] gdelt Query Units used: {_gdelt_call_count[0]} | pages fetched: {_stats['pages']} | "
          f"quota_hit={_quota_hit[0]}")
    print(f"[run] stats: {_stats}")
    print(f"[run] written to {OUTPUT_DIR}")
    print("[next] merge: MASTER_CSV='%s' PRESS_CSV='%s' python3 merge_press_into_master.py"
          % (MASTER_CSV, OUTPUT_DIR / "gdelt_gap_employment.csv"))


if __name__ == "__main__":
    run()
