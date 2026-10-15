#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 MODELS-PER-PLANT-PER-YEAR + EMPLOYMENT-FROM-WIKIPEDIA SCRAPER  (v1)
================================================================================
Companion to `scrape_european_2006_2026_v3_1_gemini.py`.

WHAT IT DOES
------------
For every plant in your registry it:
  1. Resolves the plant's Wikipedia article (MediaWiki API search; no manual
     URL list required, but an optional `wikipedia_url` override column is used
     if present).
  2. Follows langlinks and pulls the article in MULTIPLE LANGUAGES
     (en, de, fr, nl, es, it, + the plant country's language). Different
     language editions list different models / years / production figures, so
     we union them and keep per-language provenance.
  3. Extracts the MODEL PRODUCTION TIMELINE from each language edition:
       - a structured parser for the "models produced" list + infobox, then
       - a loosened Gemini pass as a fallback / corroboration.
     Canonical nameplates AND variants are kept as distinct models.
  4. Optionally corroborates against the Wayback snapshots your employee
     pipeline already fetched (reuses the `archive_url` column of your existing
     employment dataset — no re-crawl of archive.org needed).
  5. Extracts EMPLOYMENT (headcount + reference year) from each article and
     writes it in your EXISTING employment schema so you can concatenate it
     onto your current employment dataset.

OUTPUTS  (written to OUTPUT_DIR)
--------------------------------
  models_binary_matrix.csv      Wide 0/1 matrix matching your example file:
                                rows = (plant, model), cols = 2006..2026.
                                FULL GRID WITH ZEROS (panel-ready).
  models_production_units.csv   Same layout; cells = ANNUAL units produced
                                where a year-specific figure was found (else
                                blank). Cumulative totals are NOT spread here.
  models_long_evidence.csv      Tidy provenance: one row per
                                (plant, model, source_lang) with year span,
                                units (annual & cumulative), evidence snippet,
                                confidence, and cross-source agreement flags.
  employment_from_wikipedia.csv Employment in your existing long schema
                                (pattern="wikipedia"), ready to consolidate.

USAGE
-----
  export GEMINI_API_KEY=...            # same key your other scraper uses
  python3 scrape_models_wikipedia_v1.py

  # structured-only dry run (no LLM, no key needed):
  LLM_ENABLED=0 python3 scrape_models_wikipedia_v1.py

The LLM prompts here are deliberately LESS strict than the employee scraper's
(no headcount range guard, no employment-keyword proximity guard) because model
names need looser matching. Tune the CONFIG block to taste.
================================================================================
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import pickle
import hashlib
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import deque
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
    sys.exit("pandas is required:  pip install pandas openpyxl")

# ============================================================
# 1. CONFIG
# ============================================================

# --- Where the plant list comes from -------------------------------------
# Easiest: point this at the employment dataset your other scraper produced
# (any sheet/CSV that has plant_id, plant_name, city, brand, country, group).
# A registry xlsx also works as long as those columns exist.
PLANTS_INPUT_PATH = str(_P.scrapers / "inputs" / "plants of german carmakers eu.xlsx")
PLANTS_INPUT_SHEET = "automotive_employment_long_eu_2"   # set None for CSV

# Optional column in the input that pins a specific Wikipedia URL for a plant.
WIKIPEDIA_OVERRIDE_COL = "wikipedia_url"   # ignored if the column is absent

OUTPUT_DIR = Path(str(_P.scraper_out / "models_output"))

START_YEAR = 2006
END_YEAR = 2026
YEARS = list(range(START_YEAR, END_YEAR + 1))

# Languages to read for every plant. The plant's country language is added
# automatically (e.g. Hungarian for a Hungarian plant).
LANGUAGES = ["en", "de", "fr", "nl", "es", "it"]
COUNTRY_LANG = {
    "Germany": "de", "Belgium": "fr", "France": "fr", "Spain": "es",
    "Italy": "it", "Netherlands": "nl", "Austria": "de", "Hungary": "hu",
    "Czechia": "cs", "Czech Republic": "cs", "Slovakia": "sk", "Poland": "pl",
    "Portugal": "pt", "United Kingdom": "en",
}

MAX_WORKERS = 3            # fewer parallel plants => fewer simultaneous LLM bursts
REQUEST_TIMEOUT = 40
HTTP_RETRIES = 3

# Reuse the Wayback snapshots already discovered by the employee scraper.
MODELS_USE_WAYBACK = True
WAYBACK_ARCHIVE_URL_COL = "archive_url"      # column in PLANTS_INPUT_PATH
WAYBACK_MAX_PAGES_PER_PLANT = 8              # cap LLM calls on archive pages

# --- LLM (Gemini) --------------------------------------------------------
LLM_ENABLED = os.environ.get("LLM_ENABLED", "1") != "0" and bool(os.environ.get("GEMINI_API_KEY"))
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-3.5-flash")   # match your other scraper
LLM_MAX_INPUT_CHARS = 16_000
LLM_MAX_OUTPUT_TOKENS = 8192          # generous: model lists can be long
LLM_REQUEST_TIMEOUT = 60
LLM_MAX_RETRIES = 5                   # 429s/timeouts are common; retry harder
# Global request-rate cap (requests/min) across ALL workers. Free Gemini tiers
# are ~10-15 RPM; set higher if your key has more quota (env: LLM_MAX_RPM).
LLM_MAX_RPM = int(os.environ.get("LLM_MAX_RPM", "12"))

# --- caches / checkpoint -------------------------------------------------
CACHE_DIR = OUTPUT_DIR / ".cache"
WIKI_CACHE_PATH = CACHE_DIR / "wiki_text.pkl"
LLM_CACHE_PATH = CACHE_DIR / "llm_models.pkl"
RESUME = True

_HEADERS = {
    "User-Agent": "PlantModelResearchBot/1.0 (academic research; contact: researcher@example.com)",
    "Accept-Language": "en,de,fr,nl,es,it;q=0.8",
}

# ============================================================
# 2. HTTP SESSION + CACHES
# ============================================================

_thread_local = threading.local()
_wiki_cache: dict[str, str] = {}
_llm_cache: dict[str, dict] = {}
_wiki_lock = threading.Lock()
_llm_lock = threading.Lock()

# Diagnostics: run one plant verbosely with  DEBUG_PLANT=DE_VW_WOLFSBURG python3 ...
DEBUG_PLANT = os.environ.get("DEBUG_PLANT", "").strip()
VERBOSE = bool(DEBUG_PLANT)
_llm_stats = {"calls": 0, "ok": 0, "empty": 0, "rate_limited": 0,
              "http_error": 0, "exception": 0, "last_error": ""}
_stats_lock = threading.Lock()


def _dbg(*a):
    if VERBOSE:
        print("   [debug]", *a)


def get_session() -> requests.Session:
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update(_HEADERS)
        _thread_local.session = s
    return s


def _http_get(url: str, params: Optional[dict] = None) -> Optional[requests.Response]:
    for attempt in range(HTTP_RETRIES):
        try:
            r = get_session().get(url, params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 503):
                time.sleep(2.0 * (attempt + 1))
                continue
            return None
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
    return None


def _load_caches() -> None:
    global _wiki_cache, _llm_cache
    for path, target in ((WIKI_CACHE_PATH, "_wiki_cache"), (LLM_CACHE_PATH, "_llm_cache")):
        if path.exists():
            try:
                with path.open("rb") as f:
                    globals()[target] = pickle.load(f)
                print(f"[cache] loaded {len(globals()[target])} entries from {path.name}")
            except Exception as e:
                print(f"[cache] load failed {path.name}: {e}")


def _save_caches() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with _wiki_lock:
        with WIKI_CACHE_PATH.open("wb") as f:
            pickle.dump(_wiki_cache, f)
    with _llm_lock:
        with LLM_CACHE_PATH.open("wb") as f:
            pickle.dump(_llm_cache, f)


# ============================================================
# 3. MEDIAWIKI HELPERS
# ============================================================

def _api(lang: str) -> str:
    return f"https://{lang}.wikipedia.org/w/api.php"


def resolve_article(plant_name: str, city: str, brand: str, country: str,
                    override_url: str = "") -> Optional[tuple[str, str]]:
    """Return (lang, title) for the best English-or-native article, or None.

    If override_url is given (……/wiki/Title), it is parsed directly.
    """
    if override_url:
        m = re.search(r"https?://(\w+)\.wikipedia\.org/wiki/([^?#]+)", override_url)
        if m:
            return m.group(1), requests.utils.unquote(m.group(2)).replace("_", " ")

    queries = [
        f"{plant_name}",
        f"{brand} {city} plant",
        f"{brand} {city} factory",
        f"{brand} plant {city}",
        f"{city} {brand} Werk",
        f"{brand} {city}",
    ]
    plant_words = ("plant", "factory", "werk", "usine", "fabrik", "fábrica",
                   "stabilimento", "fabriek", "manufactur", "assembly", "gyár",
                   "montaż", "závod", "továreň", "fábriho")
    # titles we must NEVER accept: the bare brand or the bare city
    brand_l = (brand or "").lower()
    city_l = (city or "").lower()
    brand_blocklist = {brand_l, "audi", "bmw", "mercedes-benz", "mercedes",
                       "volkswagen", "volkswagen group", "vw", "porsche", "seat",
                       "škoda", "skoda", "opel", "bmw group", "audi ag",
                       "mercedes-benz group", "porsche ag", "bayerische motoren werke"}

    def _title_ok(title: str) -> tuple[bool, bool]:
        """(acceptable, strong). 'strong' = title names a plant AND the city."""
        t = title.lower()
        if t in brand_blocklist or t == city_l:
            return (False, False)
        has_plant = any(w in t for w in plant_words)
        has_city = city_l and city_l in t
        # a dedicated plant article almost always has a plant word + the city
        if has_plant and has_city:
            return (True, True)
        # accept "<Brand> <City>"-style plant titles (e.g. "Audi Brussels")
        if has_city and brand_l and brand_l.split()[0] in t:
            return (True, True)
        if has_plant and (brand_l.split()[0] in t if brand_l else False):
            return (True, False)
        return (False, False)

    weak: Optional[tuple[str, str]] = None
    for lang in dict.fromkeys(("en", COUNTRY_LANG.get(country, "en"))):
        for q in queries:
            r = _http_get(_api(lang), {
                "action": "query", "list": "search", "srsearch": q,
                "srlimit": 6, "format": "json", "formatversion": 2,
            })
            if not r:
                continue
            try:
                hits = r.json().get("query", {}).get("search", [])
            except Exception:
                continue
            for h in hits:
                title = h.get("title", "")
                ok, strong = _title_ok(title)
                if ok and strong:
                    return lang, title
                if ok and weak is None:
                    weak = (lang, title)
    # Only fall back to a weak match — NEVER to a bare brand/city article,
    # which is what produced CEO/mayor lists in v1.
    return weak


def get_langlinks(lang: str, title: str) -> dict[str, str]:
    """Return {lang_code: title} for all language editions of the article."""
    out: dict[str, str] = {lang: title}
    r = _http_get(_api(lang), {
        "action": "query", "prop": "langlinks", "lllimit": "max",
        "titles": title, "format": "json", "formatversion": 2,
    })
    if not r:
        return out
    try:
        pages = r.json().get("query", {}).get("pages", [])
        for ll in (pages[0].get("langlinks", []) if pages else []):
            out[ll["lang"]] = ll["title"]
    except Exception:
        pass
    return out


def fetch_article_text(lang: str, title: str) -> Optional[str]:
    """Plain-text extract of an article (cached)."""
    key = f"{lang}|{title}"
    with _wiki_lock:
        if key in _wiki_cache:
            return _wiki_cache[key]
    r = _http_get(_api(lang), {
        "action": "query", "prop": "extracts", "explaintext": 1,
        "titles": title, "format": "json", "formatversion": 2,
    })
    text = None
    if r:
        try:
            pages = r.json().get("query", {}).get("pages", [])
            if pages:
                text = pages[0].get("extract")
        except Exception:
            text = None
    if text:
        with _wiki_lock:
            _wiki_cache[key] = text
    return text


def fetch_wikitext(lang: str, title: str) -> Optional[str]:
    """Raw wikitext (used to read the infobox)."""
    r = _http_get(_api(lang), {
        "action": "query", "prop": "revisions", "rvprop": "content",
        "rvslots": "main", "titles": title, "format": "json", "formatversion": 2,
    })
    if not r:
        return None
    try:
        pages = r.json().get("query", {}).get("pages", [])
        return pages[0]["revisions"][0]["slots"]["main"]["content"] if pages else None
    except Exception:
        return None


# ============================================================
# 4. NUMBER + MODEL PARSING (structured, language-agnostic)
# ============================================================

# European + English number formats: 1 143 664 / 909.164 / 909,164 / 116250
_NUM_RE = re.compile(r"\d{1,3}(?:[ .,]\d{3})+|\d{4,7}")
_UNITS_WORD = r"(?:units?|unit[ée]s?|Stück|St\.|exemplaires|vehicles|cars|wagen|wagons)"
_PRESENT = ("present", "today", "heute", "aujourd'hui", "actuel", "ongoing", "—", "")

# A production-list line, e.g.
#   "1980 – 2007: Volkswagen Golf (3,870,701 units)"
#   "December 2022 – February 2025: Audi Q8 e-tron"
#   "2018 – 2022 : Audi e-tron Quattro (160 000 unités)"
_LINE_RE = re.compile(
    r"""(?:[A-Za-zÀ-ÿ]+\s+)?            # optional month word
        (?P<start>(?:19|20)\d{2})          # start year
        \s*[–\-—]\s*
        (?:(?:[A-Za-zÀ-ÿ]+\s+)?            # optional end month word
           (?P<end>(?:19|20)\d{2})         # end year ...
         | (?P<present>present|today|heute|aujourd'hui|actuel))?
        \s*[:：]\s*
        (?P<rest>.*)$""",
    re.VERBOSE | re.IGNORECASE,
)


# A line is only a car model if it names a marque or a recognisable nameplate.
# This is what stops executive-tenure / mayor lists (people + year ranges) from
# being mistaken for models.
_MODEL_TOKEN_RE = re.compile(
    r"\b(audi|bmw|mini|mercedes|benz|maybach|volkswagen|vw|porsche|seat|cupra|"
    r"sko?da|škoda|opel|bentley|lamborghini|bugatti|smart|"
    r"e-?tron|i\d|q\d|a\d|s\d|rs\d|x\d|m\d|z\d|"  # model codes A1,Q8,X5,i3,M3...
    r"golf|passat|polo|tiguan|touran|touareg|beetle|coccinelle|k[äa]fer|lupo|fox|"
    r"caddy|transporter|amarok|arteon|jetta|bora|sharan|toledo|le[oó]n|ibiza|altea|"
    r"iltis|cayenne|macan|panamera|taycan|boxster|cayman|911|912|914|718|"
    r"\d{3}\s?(?:i|d|e)\b)",                      # 320i, 530d ...
    re.IGNORECASE,
)
# Birth/death or "(born ...)" markers => the line is a PERSON, not a model.
_PERSON_RE = re.compile(r"\(\s*(?:\*\s*)?\d{4}\s*[–\-—]\s*\d{4}\s*\)|\(\s*born\b|\(\s*geb\.|\(\s*né\b", re.I)


def looks_like_model(name: str, raw_line: str) -> bool:
    if _PERSON_RE.search(raw_line):
        return False
    return bool(_MODEL_TOKEN_RE.search(name))


def parse_number(raw: str) -> Optional[int]:
    raw = raw.strip()
    if not raw:
        return None
    digits = re.sub(r"[ .,]", "", raw)
    try:
        return int(digits)
    except ValueError:
        return None


@dataclass
class ModelSpan:
    model: str
    year_start: int
    year_end: Optional[int]          # None = ongoing / until closure
    units_total: Optional[int] = None      # cumulative figure if stated
    units_by_year: dict[int, int] = field(default_factory=dict)  # annual figures
    source_lang: str = ""
    source_url: str = ""
    evidence: str = ""
    method: str = "structured"       # structured | llm | wayback_llm


def _clean_model_name(rest: str) -> tuple[str, Optional[int]]:
    """Split 'Audi A1 (909,164 units)' -> ('Audi A1', 909164)."""
    units = None
    m = re.search(r"\(([^)]*\d[^)]*)\)", rest)
    if m:
        inside = m.group(1)
        if re.search(_UNITS_WORD, inside, re.I) or _NUM_RE.search(inside):
            num = _NUM_RE.search(inside)
            if num:
                units = parse_number(num.group(0))
            rest = rest[:m.start()] + rest[m.end():]
    # strip footnote markers, brackets, trailing punctuation
    name = re.sub(r"\[\d+\]|\[[a-z]\]", "", rest)
    name = re.sub(r"\s{2,}", " ", name).strip(" :–-—.,;")
    return name, units


def parse_production_list(text: str, lang: str) -> list[ModelSpan]:
    """Find production-timeline lines in a plain-text article extract.

    Handles both single-line entries ("1980-2007: VW Golf (...)") and the
    split form where the year range ends with ':' and the model name is on
    the following line(s) — common in MediaWiki's plaintext export.
    """
    spans: list[ModelSpan] = []
    lines = [ln.strip().lstrip("•*-–").strip() for ln in text.splitlines()]
    for i, line in enumerate(lines):
        if not line or (":" not in line and "：" not in line):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        start = int(m.group("start"))
        end = int(m.group("end")) if m.group("end") else None
        rest = (m.group("rest") or "").strip()
        # model name spilled onto the next non-empty line(s)
        if not re.search(r"[A-Za-z]", rest):
            for j in range(i + 1, min(i + 3, len(lines))):
                if lines[j] and not _LINE_RE.match(lines[j]):
                    rest = (rest + " " + lines[j]).strip()
                    if re.search(r"[A-Za-z]", rest):
                        break
        name, units = _clean_model_name(rest)
        if not name or len(name) > 80 or not re.search(r"[A-Za-z]", name):
            continue
        # skip lines that are clearly ownership periods, not models
        if re.search(r"\b(AG|GmbH|D[’']Ieteren|Volkswagen AG|Audi AG)\b$", name):
            continue
        # v1.1: reject executive/mayor lists — require an actual marque/nameplate
        # and reject person entries (birth-year parentheticals).
        if not looks_like_model(name, line):
            continue
        spans.append(ModelSpan(
            model=name, year_start=start, year_end=end, units_total=units,
            source_lang=lang, evidence=line[:300], method="structured",
        ))
    return spans


_INFOBOX_PROD_RE = re.compile(r"\|\s*production\s*=\s*(.+?)(?:\n\|)", re.I | re.S)
_INFOBOX_EMP_RE = re.compile(
    r"\|\s*(?:num_employees|employees|effectif|mitarbeiter|besch[äa]ftigte)\s*=\s*(.+?)(?:\n\|)",
    re.I | re.S,
)


def parse_infobox_employment(wikitext: str, lang: str) -> list[tuple[int, Optional[int]]]:
    """Return [(employees, reference_year)] found in the infobox."""
    out: list[tuple[int, Optional[int]]] = []
    if not wikitext:
        return out
    for m in _INFOBOX_EMP_RE.finditer(wikitext):
        chunk = m.group(1)
        num = _NUM_RE.search(re.sub(r"\[\[|\]\]", "", chunk))
        if not num:
            continue
        emp = parse_number(num.group(0))
        if emp is None or emp < 30 or emp > 300_000:
            continue
        yr_m = re.search(r"(20\d{2}|19\d{2})", chunk)
        out.append((emp, int(yr_m.group(1)) if yr_m else None))
    return out


# ============================================================
# 5. GEMINI (loosened prompts) — models + employment
# ============================================================

_LLM_MODELS_PROMPT = """You read an encyclopedic / archived web page about ONE specific automobile plant and list which car MODELS were produced there and in which YEARS.

Plant: {plant_name}
City: {city}
Brand: {brand}
Source language: {lang}

Rules:
- List every distinct car model assembled AT THIS PLANT that the text supports.
- Keep canonical nameplates AND named variants as separate entries
  (e.g. "Audi A1", "Audi A1 Sportback", "Audi S1" are three entries).
- For each model give the production start year and end year AT THIS PLANT.
  Use null for end_year if production was ongoing / until the plant closed.
- If the text gives a production volume, put it in units_total (a cumulative
  figure, e.g. "909,164 units" over the whole run) OR units_by_year
  (a map of {{"YYYY": number}}) when the figure is tied to a specific year.
- Only use facts present in the text. Do not invent models or years.

Return VALID JSON ONLY, no markdown, exactly:
{{"models": [{{"model": "<name>", "year_start": <int>, "year_end": <int or null>,
  "units_total": <int or null>, "units_by_year": {{}}, "evidence": "<short quote>"}}]}}

PAGE TEXT (truncated):
---
{page_text}
---
"""

_LLM_EMP_PROMPT = """From the text about the automobile plant below, extract every EMPLOYEE HEADCOUNT for THIS plant together with the year it refers to.

Plant: {plant_name}
City: {city}
Brand: {brand}

Rules:
- Only the count for THIS plant ({city}). Never the brand/group worldwide total.
- The number must appear in the text. Capture the reference year if stated
  (e.g. "end of 2022", "Stand 2018"); else null.
- Plausible plant headcount is 30 to 300000.

Return VALID JSON ONLY:
{{"records": [{{"employees": <int>, "reference_year": <"YYYY" or null>, "evidence": "<short quote>"}}]}}

PAGE TEXT (truncated):
---
{page_text}
---
"""


def _record(key: str, err: str = "") -> None:
    with _stats_lock:
        _llm_stats[key] = _llm_stats.get(key, 0) + 1
        if err:
            _llm_stats["last_error"] = err


# --- global token-bucket rate limiter shared by all worker threads ----------
_rate_lock = threading.Lock()
_rate_times: deque = deque()


def _rate_gate() -> None:
    """Block until issuing another request stays under LLM_MAX_RPM (per 60s)."""
    if LLM_MAX_RPM <= 0:
        return
    while True:
        with _rate_lock:
            now = time.time()
            while _rate_times and now - _rate_times[0] > 60:
                _rate_times.popleft()
            if len(_rate_times) < LLM_MAX_RPM:
                _rate_times.append(now)
                return
            sleep_for = 60 - (now - _rate_times[0]) + 0.05
        time.sleep(min(max(sleep_for, 0.1), 5))


def _gemini_json(prompt: str) -> Optional[dict]:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None
    _record("calls")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{LLM_MODEL}:generateContent"
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": LLM_MAX_OUTPUT_TOKENS,
            "temperature": 0,
            "responseMimeType": "application/json",
            # Gemini 2.5/3 Flash "think" by default and those hidden tokens are
            # charged against maxOutputTokens, returning EMPTY responses
            # (finishReason=MAX_TOKENS, no parts). Disable thinking so the whole
            # budget goes to the actual answer.
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    for attempt in range(LLM_MAX_RETRIES + 1):
        _rate_gate()
        try:
            r = get_session().post(url, params={"key": api_key},
                                   headers={"Content-Type": "application/json"},
                                   json=body, timeout=LLM_REQUEST_TIMEOUT)
            if r.status_code == 429:
                _record("rate_limited")
                ra = r.headers.get("Retry-After", "")
                wait = float(ra) if ra.replace(".", "", 1).isdigit() else min(5 * (2 ** attempt), 60)
                _dbg(f"429 rate-limited; sleeping {wait:.0f}s (attempt {attempt+1})")
                time.sleep(wait)
                continue
            if r.status_code != 200:
                msg = f"HTTP {r.status_code}: {r.text[:300]}"
                _record("http_error", msg)
                _dbg("Gemini", msg)
                # If thinkingConfig is rejected by an older model, retry without it
                if "thinkingConfig" in r.text or "thinking_config" in r.text:
                    body["generationConfig"].pop("thinkingConfig", None)
                    continue
                return None
            cand = r.json().get("candidates", [{}])[0]
            parts = cand.get("content", {}).get("parts", [])
            if not parts:
                fr = cand.get("finishReason", "")
                _record("empty", f"no parts (finishReason={fr})")
                _dbg(f"Gemini returned no parts (finishReason={fr}):", str(r.json())[:300])
                if fr == "MAX_TOKENS":
                    body["generationConfig"]["maxOutputTokens"] = min(
                        body["generationConfig"]["maxOutputTokens"] * 2, 32768)
                    continue
                return None
            txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", parts[0].get("text", "").strip(),
                         flags=re.I | re.M).strip()
            try:
                _record("ok")
                return json.loads(txt)
            except Exception:
                m = re.search(r"\{.*\}", txt, re.S)
                if m:
                    _record("ok")
                    return json.loads(m.group(0))
                _record("empty")
                _dbg("Gemini text not JSON-parseable:", txt[:200])
                return None
        except Exception as e:
            _record("exception", str(e)[:200])
            _dbg("Gemini exception:", str(e)[:200])
            time.sleep(1.5 * (attempt + 1))
    return None


def llm_extract_models(text: str, plant_name: str, city: str, brand: str,
                       lang: str, url: str) -> list[ModelSpan]:
    if not LLM_ENABLED or not text:
        return []
    digest = hashlib.sha1((text[:LLM_MAX_INPUT_CHARS] + "|m").encode()).hexdigest()[:20]
    with _llm_lock:
        cached = _llm_cache.get(digest)
    if cached is None:
        prompt = _LLM_MODELS_PROMPT.format(plant_name=plant_name, city=city, brand=brand,
                                           lang=lang, page_text=text[:LLM_MAX_INPUT_CHARS])
        raw = _gemini_json(prompt)
        if raw is None:
            return []          # call FAILED — do not cache, so it retries next run
        cached = raw if isinstance(raw, dict) else {"models": []}
        with _llm_lock:
            _llm_cache[digest] = cached
    spans: list[ModelSpan] = []
    for m in cached.get("models", []):
        try:
            ys = int(m["year_start"])
        except Exception:
            continue
        ye = m.get("year_end")
        ye = int(ye) if ye not in (None, "", "null") else None
        uby = {}
        for k, v in (m.get("units_by_year") or {}).items():
            try:
                uby[int(k)] = int(v)
            except Exception:
                pass
        spans.append(ModelSpan(
            model=str(m.get("model", "")).strip(), year_start=ys, year_end=ye,
            units_total=m.get("units_total") if isinstance(m.get("units_total"), int) else None,
            units_by_year=uby, source_lang=lang, source_url=url,
            evidence=str(m.get("evidence", ""))[:300],
            method="wayback_llm" if "web.archive.org" in url else "llm",
        ))
    return [s for s in spans if s.model and re.search(r"[A-Za-z]", s.model)]


def llm_extract_employment(text: str, plant_name: str, city: str, brand: str,
                           lang: str) -> list[tuple[int, Optional[int], str]]:
    if not LLM_ENABLED or not text:
        return []
    digest = hashlib.sha1((text[:LLM_MAX_INPUT_CHARS] + "|e").encode()).hexdigest()[:20]
    with _llm_lock:
        cached = _llm_cache.get(digest)
    if cached is None:
        prompt = _LLM_EMP_PROMPT.format(plant_name=plant_name, city=city, brand=brand,
                                        page_text=text[:LLM_MAX_INPUT_CHARS])
        raw = _gemini_json(prompt)
        if raw is None:
            return []          # call FAILED — do not cache, so it retries next run
        cached = raw if isinstance(raw, dict) else {"records": []}
        with _llm_lock:
            _llm_cache[digest] = cached
    out = []
    for rec in cached.get("records", []):
        try:
            emp = int(rec["employees"])
        except Exception:
            continue
        if emp < 30 or emp > 300_000:
            continue
        ry = rec.get("reference_year")
        ry = int(ry) if (ry and re.match(r"^\d{4}$", str(ry))) else None
        out.append((emp, ry, str(rec.get("evidence", ""))[:300]))
    return out


# ============================================================
# 6. MERGE ACROSS LANGUAGES / SOURCES
# ============================================================

_BRAND_PREFIX = re.compile(r"^(projekt|project|audi|volkswagen|vw|bmw|mercedes(?:-benz)?|porsche|opel|seat|skoda|škoda)\s+", re.I)
_VARIANT_NOISE = re.compile(r"\b(quattro|gen\.?|generation|mk\s?\d|GE|\(?\d{4}\)?)\b", re.I)


def normalize_key(model: str) -> str:
    """Loose key so cross-language surface forms merge.
    'Audi e-tron Quattro' / 'e-tron' -> 'e-tron'. Keeps variant words like
    'sportback', 'a1', 'q8' which distinguish real variants."""
    s = model.lower().strip()
    s = _BRAND_PREFIX.sub("", s)
    s = _VARIANT_NOISE.sub("", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


@dataclass
class MergedModel:
    display: str
    surfaces: set[str]
    year_start: int
    year_end: Optional[int]
    starts_seen: set
    ends_seen: set
    units_total: Optional[int]
    units_by_year: dict[int, int]
    langs: set[str]
    methods: set[str]
    evidence: str
    year_conflict: bool = False


def merge_spans(spans: list[ModelSpan]) -> list[MergedModel]:
    groups: dict[str, list[ModelSpan]] = {}
    for s in spans:
        if not s.model:
            continue
        groups.setdefault(normalize_key(s.model), []).append(s)

    merged: list[MergedModel] = []
    for key, items in groups.items():
        starts = {i.year_start for i in items}
        ends = {i.year_end for i in items if i.year_end is not None}
        # pick widest supported span; flag if sources disagree
        y_start = min(starts)
        y_end = None if any(i.year_end is None for i in items) else (max(ends) if ends else None)
        # display name = prefer the English surface form, else most frequent
        surfaces = [i.model for i in items]
        en_surfaces = [i.model for i in items if i.source_lang == "en"]
        pool = en_surfaces or surfaces
        display = max(sorted(set(pool)), key=lambda x: (sum(1 for s in pool if s == x), len(x)))
        uby: dict[int, int] = {}
        for i in items:
            uby.update(i.units_by_year)
        totals = [i.units_total for i in items if i.units_total]
        merged.append(MergedModel(
            display=display, surfaces=set(surfaces),
            year_start=y_start, year_end=y_end,
            starts_seen=starts, ends_seen=ends or {None},
            units_total=max(totals) if totals else None, units_by_year=uby,
            langs={i.source_lang for i in items}, methods={i.method for i in items},
            evidence=next((i.evidence for i in items if i.evidence), ""),
            year_conflict=(len(starts) > 1 or len(ends) > 1),
        ))
    return merged


# ============================================================
# 7. WAYBACK CORROBORATION (reuse existing archive_url column)
# ============================================================

def _html_to_text(html: str) -> str:
    html = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;|&amp;|&#\d+;", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def wayback_model_spans(archive_urls: list[str], plant_name: str, city: str,
                        brand: str) -> list[ModelSpan]:
    spans: list[ModelSpan] = []
    if not (MODELS_USE_WAYBACK and LLM_ENABLED):
        return spans
    for url in archive_urls[:WAYBACK_MAX_PAGES_PER_PLANT]:
        r = _http_get(url)
        if not r:
            continue
        text = _html_to_text(r.text)
        if len(text) < 200:
            continue
        spans.extend(llm_extract_models(text, plant_name, city, brand, "archived", url))
    return spans


# ============================================================
# 8. PER-PLANT PIPELINE
# ============================================================

@dataclass
class PlantResult:
    plant_info: dict
    models: list[MergedModel]
    employment: list[dict]
    wikipedia_urls: dict[str, str]


def process_plant(plant_info: dict, archive_urls: list[str]) -> PlantResult:
    name = plant_info.get("plant_name", "")
    city = plant_info.get("city", "")
    brand = plant_info.get("brand", "")
    country = plant_info.get("country", "")
    override = plant_info.get(WIKIPEDIA_OVERRIDE_COL, "") or ""

    resolved = resolve_article(name, city, brand, country, override)
    wiki_urls: dict[str, str] = {}
    all_spans: list[ModelSpan] = []
    employment_rows: list[dict] = []

    if not resolved:
        _dbg(f"NO ARTICLE RESOLVED for {name} / {city} / {brand}")
    if resolved:
        base_lang, base_title = resolved
        _dbg(f"resolved -> [{base_lang}] '{base_title}'")
        langmap = get_langlinks(base_lang, base_title)
        wanted = set(LANGUAGES) | {COUNTRY_LANG.get(country, "en"), base_lang}
        for lang in sorted(l for l in langmap if l in wanted):
            title = langmap[lang]
            wiki_urls[lang] = f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
            text = fetch_article_text(lang, title)
            if not text:
                _dbg(f"[{lang}] '{title}' -> NO TEXT")
                continue
            # models: structured + LLM
            struct = parse_production_list(text, lang)
            all_spans.extend(struct)
            llm_models = llm_extract_models(text, name, city, brand, lang, wiki_urls[lang])
            all_spans.extend(llm_models)
            _dbg(f"[{lang}] '{title}' text={len(text)}c  structured={len(struct)}  llm={len(llm_models)}")
            # employment: infobox + LLM
            wt = fetch_wikitext(lang, title)
            for emp, ry in parse_infobox_employment(wt or "", lang):
                employment_rows.append(_employment_row(plant_info, emp, ry,
                                                        "infobox", lang, wiki_urls[lang], ""))
            for emp, ry, ev in llm_extract_employment(text, name, city, brand, lang):
                employment_rows.append(_employment_row(plant_info, emp, ry,
                                                        "wikipedia_llm", lang, wiki_urls[lang], ev))

    # Wayback corroboration
    for s in wayback_model_spans(archive_urls, name, city, brand):
        all_spans.append(s)

    merged = merge_spans(all_spans)
    return PlantResult(plant_info, merged, employment_rows, wiki_urls)


def _employment_row(plant_info: dict, employees: int, ref_year: Optional[int],
                    pattern: str, lang: str, url: str, evidence: str) -> dict:
    """Employment record in the SAME schema as the employee scraper output."""
    return {
        "plant_id": plant_info.get("plant_id", ""),
        "country": plant_info.get("country", ""),
        "group": plant_info.get("group", ""),
        "plant_name": plant_info.get("plant_name", ""),
        "city": plant_info.get("city", ""),
        "brand": plant_info.get("brand", ""),
        "archive_year": ref_year or "",
        "registry_url": url,
        "url_kind": "WIKIPEDIA",
        "status": "found_wikipedia",
        "employees": employees,
        "reference_year": ref_year or "",
        "snippet": evidence,
        "pattern": f"{pattern}_{lang}",
        "archive_url": url,
        "source_lang": lang,
    }


# ============================================================
# 9. OUTPUT BUILDERS
# ============================================================

def build_outputs(results: list[PlantResult]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    binary_rows, units_rows, long_rows, emp_rows = [], [], [], []

    for res in results:
        pid = res.plant_info.get("plant_id", "")
        pname = res.plant_info.get("plant_name", "")
        brand = res.plant_info.get("brand", "")
        country = res.plant_info.get("country", "")

        for mm in sorted(res.models, key=lambda m: m.year_start):
            end = mm.year_end if mm.year_end is not None else END_YEAR
            # Only keep models in production at some point during 2006-2026.
            if mm.year_start > END_YEAR or end < START_YEAR:
                continue
            # binary row (full grid with zeros across the whole window)
            brow = {"plant_id": pid, "plant_name": pname, "brand": brand,
                    "country": country, "car model ": mm.display}
            urow = dict(brow)
            for y in YEARS:
                produced = 1 if (mm.year_start <= y <= end) else 0
                brow[y] = produced
                u = mm.units_by_year.get(y)
                urow[y] = u if (produced and u) else ("" if produced else 0)
            binary_rows.append(brow)
            units_rows.append(urow)

            long_rows.append({
                "plant_id": pid, "plant_name": pname, "brand": brand, "country": country,
                "model": mm.display, "model_surfaces": " | ".join(sorted(mm.surfaces)),
                "year_start": mm.year_start, "year_end": mm.year_end,
                "units_total_cumulative": mm.units_total,
                "units_by_year": json.dumps(mm.units_by_year) if mm.units_by_year else "",
                "source_langs": ",".join(sorted(mm.langs)),
                "methods": ",".join(sorted(mm.methods)),
                "year_conflict": int(mm.year_conflict),
                "starts_seen": ",".join(str(s) for s in sorted(mm.starts_seen)),
                "ends_seen": ",".join(str(e) for e in sorted(x for x in mm.ends_seen if x is not None)),
                "evidence": mm.evidence,
            })
        # employment: keep only records whose reference year is in-window
        # (or undated). This also drops garbage like a 1936 founding year.
        for er in res.employment:
            try:
                ryi = int(er.get("reference_year"))
            except (TypeError, ValueError):
                ryi = None
            if ryi is not None and not (START_YEAR <= ryi <= END_YEAR):
                continue
            emp_rows.append(er)

    cols = ["plant_id", "plant_name", "brand", "country", "car model "] + YEARS
    pd.DataFrame(binary_rows, columns=cols).to_csv(OUTPUT_DIR / "models_binary_matrix.csv", index=False)
    pd.DataFrame(units_rows, columns=cols).to_csv(OUTPUT_DIR / "models_production_units.csv", index=False)
    pd.DataFrame(long_rows).to_csv(OUTPUT_DIR / "models_long_evidence.csv", index=False)
    pd.DataFrame(emp_rows).to_csv(OUTPUT_DIR / "employment_from_wikipedia.csv", index=False)

    print(f"\n[output] {len(binary_rows)} (plant,model) rows  |  {len(emp_rows)} employment records")
    print(f"[output] written to {OUTPUT_DIR}")


# ============================================================
# 10. LOAD PLANTS + RUN
# ============================================================

def load_plants() -> tuple[list[dict], dict[str, list[str]]]:
    p = Path(PLANTS_INPUT_PATH)
    if p.suffix.lower() in (".xlsx", ".xlsm"):
        df = pd.read_excel(p, sheet_name=PLANTS_INPUT_SHEET) if PLANTS_INPUT_SHEET else pd.read_excel(p)
    else:
        df = pd.read_csv(p)
    keep = ["plant_id", "country", "group", "plant_name", "city", "brand"]
    if WIKIPEDIA_OVERRIDE_COL in df.columns:
        keep.append(WIKIPEDIA_OVERRIDE_COL)
    present = [c for c in keep if c in df.columns]
    plants_df = df[present].drop_duplicates(subset=["plant_id"]).fillna("")
    plants = plants_df.to_dict("records")

    archive_map: dict[str, list[str]] = {}
    if MODELS_USE_WAYBACK and WAYBACK_ARCHIVE_URL_COL in df.columns:
        for pid, grp in df.groupby("plant_id"):
            urls = [u for u in grp[WAYBACK_ARCHIVE_URL_COL].dropna().tolist()
                    if isinstance(u, str) and "web.archive.org" in u]
            archive_map[str(pid)] = list(dict.fromkeys(urls))
    return plants, archive_map


def run() -> None:
    _load_caches()
    plants, archive_map = load_plants()
    if DEBUG_PLANT:
        plants = [p for p in plants if str(p.get("plant_id", "")) == DEBUG_PLANT]
        print(f"[debug] single-plant mode: {DEBUG_PLANT}  ({len(plants)} match)")
        print(f"[debug] LLM_ENABLED={LLM_ENABLED}  model={LLM_MODEL}  "
              f"key_present={bool(os.environ.get('GEMINI_API_KEY'))}")
        for pl in plants:
            res = process_plant(pl, archive_map.get(str(pl.get("plant_id", "")), []))
            print(f"\n[debug] RESULT models={len(res.models)} employment={len(res.employment)}")
            for m in sorted(res.models, key=lambda x: x.year_start):
                print(f"    {m.display:30} {m.year_start}-{m.year_end}  langs={sorted(m.langs)} methods={sorted(m.methods)}")
        print(f"\n[debug] LLM stats: {_llm_stats}")
        _save_caches()
        return
    print(f"[run] {len(plants)} plants | LLM_ENABLED={LLM_ENABLED} | model={LLM_MODEL}")
    print(f"[run] languages: {LANGUAGES} (+ country language)\n")

    results: list[PlantResult] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(process_plant, pl, archive_map.get(str(pl.get("plant_id", "")), [])): pl
                for pl in plants}
        done = 0
        for fut in as_completed(futs):
            pl = futs[fut]
            try:
                res = fut.result()
                results.append(res)
                print(f"  ✓ {pl.get('plant_id',''):28} models={len(res.models):3d} "
                      f"emp={len(res.employment):2d} langs={sorted(res.wikipedia_urls)}")
            except Exception as e:
                print(f"  ✗ {pl.get('plant_id','')}: {e}")
            done += 1
            if done % 10 == 0:
                _save_caches()

    _save_caches()
    print(f"\n[run] LLM stats: {_llm_stats}")
    ok = _llm_stats["ok"]; rl = _llm_stats["rate_limited"]
    if _llm_stats["calls"] and ok == 0:
        print("[run] WARNING: every Gemini call failed — check GEMINI_API_KEY and LLM_MODEL "
              f"(currently '{LLM_MODEL}'). last_error: {_llm_stats['last_error']}")
    elif rl > ok:
        print(f"[run] WARNING: heavy rate-limiting ({rl} 429s vs {ok} ok). Lower LLM_MAX_RPM "
              f"(now {LLM_MAX_RPM}) and/or MAX_WORKERS (now {MAX_WORKERS}), then re-run — "
              "cached successes are kept, only the throttled plants are retried.")
    build_outputs(results)


if __name__ == "__main__":
    run()
