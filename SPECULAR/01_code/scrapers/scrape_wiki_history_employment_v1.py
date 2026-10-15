#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 WIKIPEDIA-HISTORY EMPLOYMENT + PRODUCTION SCRAPER  (v1)
================================================================================
Same spirit as your old Wayback employment scraper, but the "time machine" is
Wikipedia's own revision history (complete, fast, no archive.org throttling).

For every plant:
  1. Resolve its Wikipedia article (MediaWiki search) and the 5 most common
     language editions + the plant's country language (configurable; via langlinks).
  2. For each (plant, language) and each YEAR 2006-2026, fetch the article exactly
     as it existed at that year-end, via the revision API (rvdir=older,
     rvstart={year}-12-31). Consecutive years that share the same revision are
     extracted only once (dedup by revid).
  3. From each revision extract, with Claude Haiku (one call per unique revision):
        - the plant EMPLOYEE headcount + the reference year it refers to
        - any ANNUAL plant-level PRODUCTION figure (vehicles built in a year)
  4. Emit rows in your EXISTING employment schema (so you can concatenate onto
     your current dataset) plus a separate production table.

  LLM backend: Anthropic Claude (Haiku) via the Messages API.

OUTPUTS (wiki_history_output/)
  employment_from_wikipedia_history.csv   your employment schema + source_lang/revid
  production_from_wikipedia_history.csv    plant_id, year, units, lang, evidence, revid
  wiki_articles_resolved.csv               plant_id -> resolved article + #languages

USAGE
  export ANTHROPIC_API_KEY=sk-ant-...
  python3 scrape_wiki_history_employment_v1.py
  # no-LLM dry run (regex-only employment from infoboxes, no key needed):
  LLM_ENABLED=0 python3 scrape_wiki_history_employment_v1.py
  # pick a different Claude model / throttle:
  LLM_MODEL=claude-haiku-4-5-20251001 LLM_MAX_RPM=50 python3 scrape_wiki_history_employment_v1.py
================================================================================
"""
from __future__ import annotations
import os, re, sys, json, time, pickle, hashlib, threading
import html as _htmlmod
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
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
PLANTS_INPUT_PATH = str(_P.scrapers / "inputs" / "plants of german carmakers eu.xlsx")
PLANTS_INPUT_SHEET = "automotive_employment_long_eu_2"
WIKI_OVERRIDE_COL = "wikipedia_url"          # optional manual override column

OUTPUT_DIR = Path(__file__).resolve().parent / "wiki_history_output"
START_YEAR, END_YEAR = 2006, 2026
YEARS = list(range(START_YEAR, END_YEAR + 1))

ALL_LANGUAGES = False                         # False => use LANGUAGES below (+ country language)
# The five most common editions for this dataset. The plant's own country
# language is added automatically (so Győr still gets 'hu', Bratislava 'sk', etc.).
# Set ALL_LANGUAGES = True to follow every langlink instead.
LANGUAGES = ["en", "de", "fr", "es", "it"]
COUNTRY_LANG = {"Germany": "de", "Belgium": "fr", "France": "fr", "Spain": "es",
                "Italy": "it", "Netherlands": "nl", "Austria": "de", "Hungary": "hu",
                "Czechia": "cs", "Slovakia": "sk", "Poland": "pl", "Portugal": "pt",
                "United Kingdom": "en"}

MAX_WORKERS = 3
REQUEST_TIMEOUT = 40
HTTP_RETRIES = 3
# Feed the LLM the RENDERED HTML of each revision (parse API) instead of raw
# wikitext. Captures Wikidata-populated infoboxes and rendered production tables.
USE_RENDERED_HTML = os.environ.get("USE_RENDERED_HTML", "1") != "0"
# Also mine the Wayback Machine snapshots of each Wikipedia URL (second source).
USE_WAYBACK = os.environ.get("USE_WAYBACK", "1") != "0"
WAYBACK_MAX_SNAPSHOTS_PER_LANG = int(os.environ.get("WAYBACK_MAX_SNAPSHOTS_PER_LANG", "15"))

# --- LLM backend: Anthropic Claude (Haiku) via the Messages API ---
LLM_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
LLM_ENABLED = os.environ.get("LLM_ENABLED", "1") != "0" and bool(LLM_API_KEY)
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001")
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
LLM_MAX_INPUT_CHARS = 14_000
LLM_MAX_OUTPUT_TOKENS = 4096
LLM_REQUEST_TIMEOUT = 60
LLM_MAX_RETRIES = 5
# Anthropic rate limits are tier-based (Tier 1 Haiku ~50 req/min); raise if higher.
LLM_MAX_RPM = int(os.environ.get("LLM_MAX_RPM", "50"))

CACHE_DIR = OUTPUT_DIR / ".cache"
RESUME = True

_HEADERS = {"User-Agent": "PlantHistoryResearchBot/1.0 (academic research; contact: researcher@example.com)"}

# ============================================================ HTTP + caches + rate limit
_tl = threading.local()
_rev_cache: dict = {}        # (lang,title,year) -> revid  (which revision was current)
_content_cache: dict = {}    # (lang,revid) -> wikitext
_html_cache: dict = {}       # (lang,revid) -> rendered HTML (parse API)
_llm_cache: dict = {}        # (lang,revid) -> extraction dict
_locks = {k: threading.Lock() for k in ("rev", "content", "llm", "rate", "stats")}
_rate_times: deque = deque()
_stats = {"calls": 0, "ok": 0, "empty": 0, "rate_limited": 0, "http_error": 0, "exception": 0, "last_error": ""}


def sess():
    s = getattr(_tl, "s", None)
    if s is None:
        s = requests.Session(); s.headers.update(_HEADERS); _tl.s = s
    return s


def _get(url, params=None):
    for a in range(HTTP_RETRIES):
        try:
            r = sess().get(url, params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 503):
                time.sleep(2 * (a + 1)); continue
            return None
        except requests.RequestException:
            time.sleep(1.5 * (a + 1))
    return None


def _rate_gate():
    if LLM_MAX_RPM <= 0:
        return
    while True:
        with _locks["rate"]:
            now = time.time()
            while _rate_times and now - _rate_times[0] > 60:
                _rate_times.popleft()
            if len(_rate_times) < LLM_MAX_RPM:
                _rate_times.append(now); return
            wait = 60 - (now - _rate_times[0]) + 0.05
        time.sleep(min(max(wait, 0.1), 5))


def _rec(k, err=""):
    with _locks["stats"]:
        _stats[k] = _stats.get(k, 0) + 1
        if err:
            _stats["last_error"] = err


def load_caches():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name, tgt in (("rev", "_rev_cache"), ("content", "_content_cache"),
                      ("html", "_html_cache"), ("llm", "_llm_cache")):
        p = CACHE_DIR / f"{name}.pkl"
        if p.exists():
            try:
                globals()[tgt] = pickle.load(open(p, "rb"))
            except Exception:
                pass


def save_caches():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name, src in (("rev", _rev_cache), ("content", _content_cache),
                      ("html", _html_cache), ("llm", _llm_cache)):
        try:
            pickle.dump(src, open(CACHE_DIR / f"{name}.pkl", "wb"))
        except Exception:
            pass


# ============================================================ MediaWiki helpers
def api(lang):
    return f"https://{lang}.wikipedia.org/w/api.php"


PLANT_WORDS = ("plant", "factory", "werk", "fabrik", "usine", "fábrica", "fabrica",
               "stabilimento", "fabriek", "manufactur", "assembly", "gyár", "závod",
               "zakłady", "montaj", "automobil")
# Titles that look like the right place but are NOT a production plant.
VENUE_WORDS = ("forum", "museum", "arena", "aréna", "villa", "memorial", "denkmal",
               "stadion", "stadium", "gigafactory", "hotel", "kirche", "church", "park",
               "theater", "theatre", "universit", "list of", "circuit", "aréna")
# Carmaker tokens used to reject cross-brand matches (e.g. Tesla page for a Mercedes plant).
_BRAND_ALIASES = {"vw": "volkswagen", "volkswagen": "volkswagen", "audi": "audi", "bmw": "bmw",
                  "mercedes": "mercedes", "mercedes-benz": "mercedes", "daimler": "mercedes",
                  "porsche": "porsche", "seat": "seat", "cupra": "seat", "skoda": "skoda",
                  "škoda": "skoda", "opel": "opel"}
_OTHER_CARMAKERS = ["tesla", "ford", "toyota", "renault", "fiat", "stellantis", "nissan",
                    "hyundai", "kia", "peugeot", "citroen", "citroën", "honda", "volvo", "tata"]


def resolve_article(name, city, brand, country, override=""):
    """Strict resolver: returns (lang, title) only for a genuine plant article for
    THIS brand & city, else None (no weak fallback — better no data than wrong data)."""
    if override:
        m = re.search(r"https?://(\w+)\.wikipedia\.org/wiki/([^?#]+)", override)
        if m:
            return m.group(1), requests.utils.unquote(m.group(2)).replace("_", " ")

    brand_tok = (brand or "").lower().split()[0] if brand else ""
    my_brand = _BRAND_ALIASES.get(brand_tok, brand_tok)
    queries = [name, f"{brand} {city} Werk", f"{brand} {city} plant",
               f"{brand} {city} factory", f"{city} {brand}", f"{brand} {city}"]

    for lang in dict.fromkeys(("en", COUNTRY_LANG.get(country, "en"))):
        for q in queries:
            r = _get(api(lang), {"action": "query", "list": "search", "srsearch": q,
                                 "srlimit": 6, "format": "json", "formatversion": 2})
            if not r:
                continue
            try:
                hits = r.json().get("query", {}).get("search", [])
            except Exception:
                continue
            for h in hits:
                if _acceptable_article(h.get("title", ""), h.get("snippet", ""), my_brand, city):
                    return lang, h["title"]
    return None


def _acceptable_article(title, snippet, my_brand, city):
    """True only if `title` is a genuine production-plant article for this brand+city."""
    t = title.lower()
    blob = (title + " " + snippet).lower()
    city_l = (city or "").lower().strip()
    if any(w in t for w in VENUE_WORDS):                       # museum / arena / villa / gigafactory / "list of"...
        return False
    for b in _OTHER_CARMAKERS:                                 # a different carmaker named in the title
        if re.search(r"\b" + re.escape(b) + r"\b", t):
            return False
    for b, canon in _BRAND_ALIASES.items():                    # a different German brand named in the title
        if canon != my_brand and re.search(r"\b" + re.escape(b) + r"\b", t):
            return False
    if city_l and city_l not in blob:                          # the plant's city must appear
        return False
    if any(w in t for w in PLANT_WORDS):                       # plant word in title -> accept
        return True
    if my_brand and my_brand in t and city_l and city_l in blob:   # brand in title + city present
        return True
    if city_l and city_l in t and my_brand and my_brand in blob:   # city in title + brand present
        return True
    return False


def langlinks(lang, title):
    out = {lang: title}
    r = _get(api(lang), {"action": "query", "prop": "langlinks", "lllimit": "max",
                         "titles": title, "format": "json", "formatversion": 2})
    if r:
        try:
            pages = r.json().get("query", {}).get("pages", [])
            for ll in (pages[0].get("langlinks", []) if pages else []):
                out[ll["lang"]] = ll["title"]
        except Exception:
            pass
    return out


def revision_asof(lang, title, year):
    """Return (revid, timestamp, wikitext) for the article as it existed at the
    END of `year`, or None if the article didn't exist yet."""
    key = (lang, title, year)
    with _locks["rev"]:
        cached_revid = _rev_cache.get(key, "__miss__")
    if cached_revid == "__miss__":
        r = _get(api(lang), {"action": "query", "prop": "revisions", "titles": title,
                             "rvlimit": 1, "rvdir": "older", "rvstart": f"{year}-12-31T23:59:59Z",
                             "rvprop": "ids|timestamp", "format": "json", "formatversion": 2})
        revid = ts = None
        if r:
            try:
                revs = r.json().get("query", {}).get("pages", [{}])[0].get("revisions", [])
                if revs:
                    revid, ts = revs[0]["revid"], revs[0]["timestamp"]
            except Exception:
                pass
        with _locks["rev"]:
            _rev_cache[key] = (revid, ts)
        cached_revid = (revid, ts)
    revid, ts = cached_revid
    if not revid:
        return None
    wt = fetch_revision_content(lang, revid)
    return (revid, ts, wt) if wt else None


def fetch_revision_content(lang, revid):
    ck = (lang, revid)
    with _locks["content"]:
        if ck in _content_cache:
            return _content_cache[ck]
    r = _get(api(lang), {"action": "query", "prop": "revisions", "revids": revid,
                         "rvprop": "content", "rvslots": "main", "format": "json", "formatversion": 2})
    wt = None
    if r:
        try:
            wt = r.json()["query"]["pages"][0]["revisions"][0]["slots"]["main"]["content"]
        except Exception:
            wt = None
    if wt:
        with _locks["content"]:
            _content_cache[ck] = wt
    return wt


def fetch_revision_html(lang, revid):
    """Rendered HTML of a specific revision via the parse API (templates expanded,
    tables rendered). Note: an old revision's wikitext is rendered with CURRENT
    template/Wikidata state, so wikitext-embedded prose/tables are historically
    accurate, but a purely Wikidata-sourced number reflects today's value."""
    ck = (lang, revid)
    with _locks["content"]:
        if ck in _html_cache:
            return _html_cache[ck]
    r = _get(api(lang), {"action": "parse", "oldid": revid, "prop": "text",
                         "format": "json", "formatversion": 2,
                         "disablelimitreport": 1, "disableeditsection": 1})
    html = None
    if r:
        try:
            html = r.json()["parse"]["text"]
        except Exception:
            html = None
    if html:
        with _locks["content"]:
            _html_cache[ck] = html
    return html


def html_to_text(html):
    """Strip HTML to text but keep TABLE structure (rows -> lines, cells -> ' | '),
    so production tables like '2015 | 324,131' survive for the LLM."""
    h = re.sub(r"<(script|style)\b.*?</\1>", " ", html or "", flags=re.S | re.I)
    h = re.sub(r"<sup\b[^>]*class=\"[^\"]*reference[^\"]*\"[^>]*>.*?</sup>", " ", h, flags=re.S | re.I)
    h = re.sub(r"</tr\s*>", "\n", h, flags=re.I)
    h = re.sub(r"</t[dh]\s*>", " | ", h, flags=re.I)
    h = re.sub(r"<[^>]+>", " ", h)
    h = _htmlmod.unescape(h)                 # &#228; -> ä, &nbsp; -> space, etc.
    h = h.replace("\xa0", " ")
    h = re.sub(r"[ \t]+", " ", h)
    h = re.sub(r"\n[ \t|]*\n+", "\n", h)
    return h.strip()


# ============================================================ parsing
_NUM = re.compile(r"\d{1,3}(?:[ ., ]\d{3})+|\d{3,7}")
# Infobox employee-count field names across the common Wikipedia editions.
_EMP_FIELD = re.compile(
    r"\|\s*(?:num_employees|number_of_employees|employees|"                  # en
    r"mitarbeiter|mitarbeiterzahl|besch[äa]ftigte|belegschaft|"              # de
    r"effectifs?|"                                                           # fr
    r"empleados|plantilla|n[úu]mero_de_empleados|"                          # es
    r"dipendenti|addetti|"                                                   # it
    r"funcion[áa]rios|empregados|colaboradores|"                            # pt
    r"werknemers|medewerkers|personeel|aantal_werknemers|"                  # nl
    r"pracownic\w*|zatrudnienie|liczba_pracownik\w*|"                       # pl
    r"zaměstnanc\w*|počet_zaměstnanc\w*|"                                   # cs
    r"zamestnanc\w*|"                                                       # sk
    r"alkalmazott\w*|foglalkoztatott\w*|dolgoz[óo]k|"                       # hu
    r"anställda|antal_anställda|"                                           # sv
    r"ansatte|"                                                            # da/no
    r"työntekij[äa]t?|henkilöstö|"                                         # fi
    r"angajați|num[ăa]r_angajați|"                                         # ro
    r"çalışan\w*|"                                                          # tr
    r"сотрудник\w*|работник\w*|персонал|число_сотрудник\w*"                  # ru/uk
    r")\s*=\s*(.+)", re.I)

# Prose triggers — if any appears we call the LLM on that revision.
_EMP_KEYWORDS = re.compile(
    r"employ|workforce|staff|worker|"                                       # en
    r"mitarbeiter|besch[äa]ftigt|belegschaft|arbeitnehmer|angestellt|"      # de
    r"effectif|salari[ée]|personnel|ouvrier|emploi|"                       # fr
    r"emplead|plantilla|trabajador|"                                       # es
    r"dipendenti|addetti|lavoratori|occupati|"                            # it
    r"empregad|funcion[áa]ri|trabalhador|colaborador|"                    # pt
    r"werknemer|medewerker|personeel|werkgelegenheid|"                    # nl
    r"pracownik|zatrudni|za[łl]oga|personel|"                             # pl
    r"zaměstnan|"                                                          # cs
    r"zamestnan|"                                                          # sk
    r"alkalmazott|dolgoz|munkavállal|foglalkoztat|munkás|"                # hu
    r"anställd|medarbetare|"                                              # sv
    r"ansatte|medarbejder|"                                               # da
    r"työntekij|henkilöstö|"                                              # fi
    r"angajat|salariat|"                                                  # ro
    r"çalışan|işçi|personel|"                                             # tr
    r"treballador|empleat|"                                               # ca
    r"сотрудник|работник|персонал|занят|"                                  # ru
    r"працівник|співробітник|"                                            # uk
    r"従業員|社員|"                                                          # ja
    r"员工|雇员|職工", re.I)

_PROD_KEYWORDS = re.compile(
    r"produc|manufactur|assembl|vehicles|cars|units|output|built|"          # en
    r"produktion|produzier|fahrzeuge|st[üu]ck|gefertigt|hergestellt|gebaut|"  # de
    r"production|produit|v[ée]hicules|fabriqu|unit[ée]s|"                   # fr
    r"producci[óo]n|producid|veh[íi]culos|fabricad|unidades|ensambl|"      # es
    r"produzione|prodotti|veicoli|fabbricat|unit[àa]|assemblat|"           # it
    r"produ[çc][ãa]o|produzid|ve[íi]culos|fabricad|montad|"                # pt
    r"productie|voertuigen|geproduceerd|vervaardigd|eenheden|"             # nl
    r"produkcj|produkowan|pojazd|wytwarz|sztuk|zmontowan|"                 # pl
    r"v[ýy]rob|vozidel|voz[ůy]|vozidl|kus[ůy]|mont[áa]ž|"                  # cs/sk
    r"gy[áa]rt|termel|j[áa]rm[űu]|g[ée]pj[áa]rm[űu]|darab|összeszerel|"    # hu
    r"tillverk|fordon|bilar|enheter|"                                      # sv
    r"k[øo]ret[øo]jer|"                                                    # da
    r"tuotanto|valmist|ajoneuvo|"                                          # fi
    r"producție|vehicule|fabricat|asamblat|"                              # ro
    r"üretim|ara[çc]|otomobil|adet|montaj|"                               # tr
    r"производ|автомобил|выпуск|штук|собран|"                              # ru
    r"виробн|випуск|"                                                      # uk
    r"生産|製造|車両|"                                                       # ja
    r"产量|生产|制造|汽车", re.I)


def parse_number(raw):
    raw = raw.strip()
    d = re.sub(r"[ ., ]", "", raw)
    try:
        return int(d)
    except ValueError:
        return None


def infobox_employment(wt):
    out = []
    for m in _EMP_FIELD.finditer(wt or ""):
        chunk = re.sub(r"\[\[|\]\]|\{\{|\}\}|<[^>]+>|&nbsp;", " ", m.group(1).split("\n")[0])
        n = _NUM.search(chunk)
        if not n:
            continue
        emp = parse_number(n.group(0))
        if emp is None or emp < 30 or emp > 400_000:
            continue
        # guard: a lone 1900-2026 value that equals the only year is a
        # founding/reference year, not a headcount.
        if 1900 <= emp <= END_YEAR and chunk.strip().replace(",", "").replace(".", "").replace(" ", "").isdigit():
            continue
        yr = re.search(r"(19|20)\d{2}", chunk)
        out.append((emp, int(yr.group(0)) if yr else None, "infobox: " + chunk.strip()[:160]))
    return out


def wikitext_excerpt(wt):
    """Light wikitext->text for the LLM (keep infobox + first chunks)."""
    t = re.sub(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", " ", wt or "", flags=re.S | re.I)
    t = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"'''?|<[^>]+>|&nbsp;", " ", t)
    t = re.sub(r"[ \t ]+", " ", t)
    return t[:LLM_MAX_INPUT_CHARS]


# ============================================================ Claude Haiku (combined emp + production)
_PROMPT = """From this Wikipedia article (wikitext) about ONE automobile plant, extract two things for THIS plant only.

Plant: {name}
City: {city}
Brand: {brand}

Return VALID JSON ONLY (no markdown):
{{"employment":[{{"employees":<int>,"reference_year":<"YYYY" or null>,"evidence":"<short quote>"}}],
  "production":[{{"year":"<YYYY>","units":<int>,"evidence":"<short quote>"}}]}}

Rules:
- employment: the employee headcount for THIS plant ({city}). NOT the group/worldwide total.
  Capture the year it refers to (e.g. "as of 2018", "Stand 2014", "fin 2022") else null.
  Plausible plant headcount 30..400000. The number must appear in the text.
- production: ANNUAL number of vehicles built at this plant in a given year
  (e.g. "324,131 vehicles in 2015"). One entry per year stated. Not cumulative totals.
- If nothing applies, return empty lists.

ARTICLE:
---
{text}
---
"""


def llm_call(name, city, brand, text):
    """Call Claude (Haiku) and return the parsed JSON dict, or None."""
    if not (LLM_ENABLED and text):
        return None
    prompt = _PROMPT.format(name=name, city=city, brand=brand, text=text)
    headers = {"x-api-key": LLM_API_KEY, "anthropic-version": ANTHROPIC_VERSION,
               "content-type": "application/json"}
    max_toks = LLM_MAX_OUTPUT_TOKENS
    _rec("calls")
    for a in range(LLM_MAX_RETRIES + 1):
        _rate_gate()
        body = {"model": LLM_MODEL, "max_tokens": max_toks, "temperature": 0,
                "messages": [{"role": "user", "content": prompt}]}
        try:
            r = sess().post(ANTHROPIC_URL, headers=headers, json=body, timeout=LLM_REQUEST_TIMEOUT)
            if r.status_code in (429, 529):                 # rate limited / overloaded
                _rec("rate_limited")
                ra = r.headers.get("retry-after", "")
                time.sleep(float(ra) if ra.replace(".", "", 1).isdigit() else min(5 * 2 ** a, 60)); continue
            if r.status_code != 200:
                _rec("http_error", f"HTTP {r.status_code}: {r.text[:200]}")
                return None                                  # 401 bad key / 400 bad request: don't retry
            data = r.json()
            txt = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()
            stop = data.get("stop_reason")
            if not txt:
                _rec("empty", f"no text (stop={stop})")
                if stop == "max_tokens" and a < LLM_MAX_RETRIES:
                    max_toks = min(max_toks * 2, 16384); continue
                return None
            txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.I | re.M).strip()
            try:
                parsed = json.loads(txt)
            except Exception:
                m = re.search(r"\{.*\}", txt, re.S)
                parsed = json.loads(m.group(0)) if m else None
            if parsed is not None:
                _rec("ok"); return parsed
            _rec("empty", "unparseable JSON")
            if stop == "max_tokens" and a < LLM_MAX_RETRIES:
                max_toks = min(max_toks * 2, 16384); continue
            return None
        except Exception as e:
            _rec("exception", str(e)[:200]); time.sleep(1.5 * (a + 1))
    return None


def extract_text(cache_key, llm_text, wt, name, city, brand, lang, src):
    """Generic extractor used by BOTH the revision-history and Wayback paths.
    `wt` = wikitext for the infobox regex (None for Wayback snapshots, which are
    rendered HTML only). `llm_text` = cleaned text fed to the LLM. `src` = 'rev'|'wb'.
    Cached per cache_key."""
    with _locks["llm"]:
        if cache_key in _llm_cache:
            c = _llm_cache[cache_key]
            return c["emp"], c["prod"]
    emp, prod = [], []
    if wt:                                       # regex infobox (historically exact where present)
        for e, y, ev in infobox_employment(wt):
            emp.append({"employees": e, "reference_year": y, "evidence": ev, "how": f"infobox_{src}_{lang}"})
    if LLM_ENABLED and llm_text and (_EMP_KEYWORDS.search(llm_text) or _PROD_KEYWORDS.search(llm_text)):
        data = llm_call(name, city, brand, llm_text[:LLM_MAX_INPUT_CHARS]) or {}
        for r in data.get("employment", []):
            try:
                e = int(r["employees"])
            except Exception:
                continue
            if 30 <= e <= 400_000:
                ry = r.get("reference_year")
                ry = int(ry) if (ry and re.match(r"^\d{4}$", str(ry))) else None
                emp.append({"employees": e, "reference_year": ry,
                            "evidence": str(r.get("evidence", ""))[:200], "how": f"llm_{src}_{lang}"})
        for r in data.get("production", []):
            try:
                u = int(r["units"]); y = int(str(r["year"])[:4])
            except Exception:
                continue
            if 100 <= u <= 2_000_000 and START_YEAR <= y <= END_YEAR:
                prod.append({"year": y, "units": u, "evidence": str(r.get("evidence", ""))[:200], "how": f"llm_{src}_{lang}"})
    with _locks["llm"]:
        _llm_cache[cache_key] = {"emp": emp, "prod": prod}
    return emp, prod


def _revision_llm_text(lang, revid, wt):
    if LLM_ENABLED and USE_RENDERED_HTML:
        h = fetch_revision_html(lang, revid)
        if h:
            return html_to_text(h)[:LLM_MAX_INPUT_CHARS]
    return wikitext_excerpt(wt) if LLM_ENABLED else ""


# ============================================================ Wayback Machine (second historical source)
def wayback_snapshots(url):
    """~one archived snapshot per year for the Wikipedia URL, via the CDX API."""
    if not USE_WAYBACK:
        return []
    r = _get("http://web.archive.org/cdx/search/cdx",
             {"url": url, "output": "json", "from": f"{START_YEAR}0101", "to": f"{END_YEAR}1231",
              "fl": "timestamp,original", "filter": "statuscode:200", "collapse": "timestamp:4"})
    if not r:
        return []
    try:
        rows = r.json()
    except Exception:
        return []
    return [(row[0], row[1]) for row in rows[1:]] if len(rows) > 1 else []


def fetch_wayback_html(ts, original):
    ck = ("wb", ts, original)
    with _locks["content"]:
        if ck in _html_cache:
            return _html_cache[ck]
    r = _get(f"https://web.archive.org/web/{ts}id_/{original}")
    htm = r.text if (r is not None and r.status_code == 200) else None
    if htm:
        with _locks["content"]:
            _html_cache[ck] = htm
    return htm


# ============================================================ per-plant pipeline
def process_plant(p):
    name, city, brand, country = p["plant_name"], p.get("city", ""), p["brand"], p["country"]
    resolved = resolve_article(name, city, brand, country, p.get(WIKI_OVERRIDE_COL, "") or "")
    emp_rows, prod_rows = [], []
    info = {"plant_id": p["plant_id"], "resolved": "", "languages": 0}
    if not resolved:
        return emp_rows, prod_rows, info
    base_lang, base_title = resolved
    lmap = langlinks(base_lang, base_title)
    if not ALL_LANGUAGES:
        want = set(LANGUAGES) | {COUNTRY_LANG.get(country, "en"), base_lang}
        lmap = {k: v for k, v in lmap.items() if k in want}
    info["resolved"] = f"[{base_lang}] {base_title}"
    info["languages"] = len(lmap)

    def _prod_row(pr, snap_year, ts, archive_url, source):
        return {"plant_id": p["plant_id"], "plant_name": name, "brand": brand, "country": country,
                "year": pr["year"], "production_units": pr["units"], "source": source,
                "source_lang": lang, "snapshot_year": snap_year, "evidence": pr["evidence"],
                "method": pr["how"], "archive_url": archive_url, "rev_timestamp": ts}

    for lang, title in lmap.items():
        wiki_url = f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"

        # --- A) Wikipedia revision history (year-end revision of each year) ---
        seen_revids = set()
        for year in YEARS:
            rv = revision_asof(lang, title, year)
            if not rv:
                continue
            revid, ts, wt = rv
            if revid in seen_revids:
                continue
            seen_revids.add(revid)
            llm_text = _revision_llm_text(lang, revid, wt)
            emp, prod = extract_text((lang, "rev", revid), llm_text, wt, name, city, brand, lang, "rev")
            permalink = f"https://{lang}.wikipedia.org/w/index.php?oldid={revid}"
            for e in emp:
                emp_rows.append(_emp_row(p, e, lang, year, revid, ts, permalink, wiki_url, "revision"))
            for pr in prod:
                prod_rows.append(_prod_row(pr, year, ts, permalink, "revision"))

        # --- B) Wayback Machine snapshots of the same Wikipedia URL ---
        if USE_WAYBACK:
            for ts, original in wayback_snapshots(wiki_url)[:WAYBACK_MAX_SNAPSHOTS_PER_LANG]:
                snap_year = int(ts[:4]) if ts[:4].isdigit() else None
                if snap_year is None or not (START_YEAR <= snap_year <= END_YEAR):
                    continue
                htm = fetch_wayback_html(ts, original)
                if not htm:
                    continue
                llm_text = html_to_text(htm)[:LLM_MAX_INPUT_CHARS]
                emp, prod = extract_text((lang, "wb", ts), llm_text, None, name, city, brand, lang, "wb")
                arch_url = f"https://web.archive.org/web/{ts}/{original}"
                for e in emp:
                    emp_rows.append(_emp_row(p, e, lang, snap_year, "", ts, arch_url, wiki_url, "wayback"))
                for pr in prod:
                    prod_rows.append(_prod_row(pr, snap_year, ts, arch_url, "wayback"))
    return emp_rows, prod_rows, info


def _emp_row(p, e, lang, snap_year, revid, ts, permalink, wiki_url, source):
    """Match the existing employment long schema as closely as possible."""
    ry = e["reference_year"]
    return {
        "plant_id": p["plant_id"], "country": p.get("country", ""), "group": p.get("group", ""),
        "plant_name": p.get("plant_name", ""), "city": p.get("city", ""), "brand": p.get("brand", ""),
        "archive_year": snap_year,                 # the year the snapshot/revision represents
        "registry_url": wiki_url, "url_kind": "WIKIPEDIA",
        "archive_timestamp": ts, "archive_url": permalink,
        "status": "found_wikipedia", "employees": e["employees"],
        "reference_year": ry if ry is not None else "",
        "snippet": e["evidence"], "pattern": e["how"],
        "source": source,                          # 'revision' or 'wayback'
        "source_lang": lang, "revid": revid, "error": "",
    }


# ============================================================ load + run
def load_plants():
    p = Path(PLANTS_INPUT_PATH)
    df = pd.read_excel(p, sheet_name=PLANTS_INPUT_SHEET) if p.suffix.lower().startswith(".xls") else pd.read_csv(p)
    keep = [c for c in ["plant_id", "country", "group", "plant_name", "city", "brand", WIKI_OVERRIDE_COL] if c in df.columns]
    return df[keep].drop_duplicates(subset=["plant_id"]).fillna("").to_dict("records")


def _dedupe_emp(rows):
    """Keep one employment row per (plant, reference_year) — prefer one with a real
    reference_year and the most recent revision; collapses duplicates across langs."""
    best = {}
    for r in rows:
        ry = r["reference_year"] or r["archive_year"]
        k = (r["plant_id"], ry, r["employees"])
        if k not in best or (r["reference_year"] and not best[k]["reference_year"]):
            best[k] = r
    return list(best.values())


def run():
    load_caches()
    plants = load_plants()
    print(f"[run] {len(plants)} plants | ALL_LANGUAGES={ALL_LANGUAGES} | LLM={LLM_ENABLED} | model={LLM_MODEL}")
    emp_all, prod_all, info_all = [], [], []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(process_plant, p): p for p in plants}
        done = 0
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                emp, prod, info = fut.result()
                emp_all += emp; prod_all += prod; info_all.append(info)
                print(f"  ✓ {p['plant_id']:26} langs={info['languages']:2} emp={len(emp):3} prod={len(prod):3}  {info['resolved'][:40]}")
            except Exception as e:
                print(f"  ✗ {p['plant_id']}: {e}")
            done += 1
            if done % 5 == 0:
                save_caches()
    save_caches()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    emp_dedup = _dedupe_emp(emp_all)
    emp_cols = ["plant_id", "country", "group", "plant_name", "city", "brand", "archive_year",
                "registry_url", "url_kind", "archive_timestamp", "archive_url", "status",
                "employees", "reference_year", "snippet", "pattern", "source", "source_lang", "revid", "error"]
    pd.DataFrame(emp_dedup, columns=emp_cols).to_csv(OUTPUT_DIR / "employment_from_wikipedia_history.csv", index=False)
    pd.DataFrame(prod_all).to_csv(OUTPUT_DIR / "production_from_wikipedia_history.csv", index=False)
    pd.DataFrame(info_all).to_csv(OUTPUT_DIR / "wiki_articles_resolved.csv", index=False)
    print(f"\n[run] employment rows: {len(emp_dedup)} (from {len(emp_all)} raw) | production rows: {len(prod_all)}")
    print(f"[run] LLM stats: {_stats}")
    if _stats['calls'] and _stats['rate_limited'] > _stats['ok']:
        print(f"[run] WARNING: heavy throttling — lower LLM_MAX_RPM (now {LLM_MAX_RPM}) and rerun (cache kept).")
    print(f"[run] written to {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
