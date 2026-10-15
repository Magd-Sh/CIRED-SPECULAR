#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 WIKIPEDIA PLANT SCRAPER v2  —  MODELS + EMPLOYMENT + PRODUCTION  (Mistral AI)
================================================================================
One run, three deliverables, for every plant in the registry:

  • models_binary_matrix.csv      each MODEL built per plant per YEAR 2006-2026 (0/1)
  • models_long.csv               model spans + evidence/source/confidence
  • employment_from_wikipedia.csv  every employee count + reference year (your schema)
  • production_from_wikipedia.csv  annual plant production figures (deduped)
  • wiki_articles_resolved.csv     resolution log

HOW IT WORKS
  1. Resolve each plant's Wikipedia article (strict) + the 5 most common language
     editions + the plant's country language.
  2. Walk the article through time via the REVISION HISTORY (old AND new): the
     year-end revision of each year 2006-2026.
  3. If an article/edition has NO revisions (deleted/never existed), FALL BACK to
     the Wayback Machine for that URL.
  4. REGEX-FIRST to keep the paid API cheap:
        - infobox employee field          (per revision, wikitext)
        - prose employee figure + year     (per revision, multilingual)
        - production TABLE rows            (rendered HTML "YEAR | NUMBER")
        - structured model lists           ("2004-2009: Audi A3", brand-filtered)
     The LLM (Mistral) is called only on a few sampled revisions per language
     (newest + oldest by default) to get the full model timeline and mop up prose.

USAGE
  export MISTRAL_API_KEY=...
  python3 scrape_wiki_full_v2_mistral.py
  LLM_ENABLED=0 python3 scrape_wiki_full_v2_mistral.py        # regex-only, no key/cost
  LLM_SAMPLE_MAX=4 LLM_MAX_RPM=30 python3 scrape_wiki_full_v2_mistral.py
================================================================================
"""
from __future__ import annotations
import os, re, sys, json, time, pickle, threading
import html as _htmlmod
from pathlib import Path
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from urllib.parse import quote
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
WIKI_OVERRIDE_COL = "wikipedia_url"

# Existing employment dataset: ONLY fill (plant, year) employment that is missing
# here; models & production extraction are unchanged (full). Set "" to disable.
EXISTING_EMPLOYMENT_PATH = str(_P.scraper_out / "automotive_employment_long_european_2006_2026_v3_1.csv")

# Output + cache folder is tagged with the model, so running large and small in
# parallel can't collide or share a cache (cache is keyed by revision, not model).
_MODEL_TAG = re.sub(r"[^a-z0-9]+", "-", os.environ.get("LLM_MODEL", "mistral-small-2506").lower())
OUTPUT_DIR = Path(__file__).resolve().parent / ("wiki_v2_output_" + _MODEL_TAG)
START_YEAR, END_YEAR = 1950, 2026
YEARS = list(range(START_YEAR, END_YEAR + 1))
# No single plant exceeds this headcount; figures above it are group/worldwide
# totals (e.g. Daimler ~380k on the Kamenz page) and are rejected outright.
PLANT_EMP_MAX = int(os.environ.get("PLANT_EMP_MAX", "120000"))

ALL_LANGUAGES = False
LANGUAGES = ["en", "de", "fr", "es", "it"]            # 5 most common (+ country lang auto-added)
COUNTRY_LANG = {"Germany": "de", "Belgium": "fr", "France": "fr", "Spain": "es",
                "Italy": "it", "Netherlands": "nl", "Austria": "de", "Hungary": "hu",
                "Czechia": "cs", "Slovakia": "sk", "Poland": "pl", "Portugal": "pt",
                "United Kingdom": "en"}

MAX_WORKERS = 3
REQUEST_TIMEOUT = 40
HTTP_RETRIES = 3
USE_WAYBACK_FALLBACK = True                           # Wayback only when revisions are missing
WAYBACK_MAX_SNAPSHOTS = 15

# --- LLM backend: Mistral AI via the chat/completions API ---
LLM_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
LLM_ENABLED = os.environ.get("LLM_ENABLED", "1") != "0" and bool(LLM_API_KEY)
LLM_MODEL = os.environ.get("LLM_MODEL", "mistral-small-2506")   # best free-tier throughput (2.25M TPM, 5 req/s)
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"
# (Mistral uses Bearer auth; no version header needed)
LLM_MAX_INPUT_CHARS = 16_000
LLM_MAX_OUTPUT_TOKENS = 4096
LLM_REQUEST_TIMEOUT = 90
LLM_MAX_RETRIES = 5
LLM_MAX_RPM = int(os.environ.get("LLM_MAX_RPM", "250"))   # mistral-small-2506 allows ~300/min; headroom left
# Default: sample a few revisions per language for the LLM (newest + oldest +
# interior years), which keeps Mistral cost down; per-year infobox regex still
# fills the in-between years. Set LLM_ALL_REVISIONS=1 for every revision (priciest).
LLM_ALL_REVISIONS = os.environ.get("LLM_ALL_REVISIONS", "0") != "0"   # 0 = sample (LLM_SAMPLE_MAX); set 1 for every revision
LLM_SAMPLE_MAX = int(os.environ.get("LLM_SAMPLE_MAX", "10"))   # newest + oldest + ~2 interior

CACHE_DIR = OUTPUT_DIR / ".cache"

_HEADERS = {"User-Agent": "PlantResearchBot/2.0 (academic research; researcher@example.com)"}

# ============================================================ HTTP / caches / rate limit
_tl = threading.local()
_rev_cache: dict = {}
_content_cache: dict = {}
_html_cache: dict = {}
_llm_cache: dict = {}
_locks = {k: threading.Lock() for k in ("rev", "content", "llm", "rate", "stats")}
_rate_times: deque = deque()
_stats = {"calls": 0, "ok": 0, "empty": 0, "rate_limited": 0, "http_error": 0, "exception": 0,
          "emp_unverified": 0, "prod_unverified": 0, "last_error": ""}


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


# ============================================================ MediaWiki
def api(lang):
    return f"https://{lang}.wikipedia.org/w/api.php"


PLANT_WORDS = ("plant", "factory", "werk", "fabrik", "usine", "fábrica", "fabrica", "stabilimento",
               "fabriek", "manufactur", "assembly", "gyár", "závod", "zakłady", "montaj", "automobil")
VENUE_WORDS = ("forum", "museum", "arena", "aréna", "villa", "memorial", "denkmal", "stadion",
               "stadium", "gigafactory", "hotel", "kirche", "church", "park", "theater", "theatre",
               "universit", "list of", "circuit")
_BRAND_ALIASES = {"vw": "volkswagen", "volkswagen": "volkswagen", "audi": "audi", "bmw": "bmw",
                  "mercedes": "mercedes", "mercedes-benz": "mercedes", "daimler": "mercedes",
                  "porsche": "porsche", "seat": "seat", "cupra": "seat", "skoda": "skoda",
                  "škoda": "skoda", "opel": "opel"}
_OTHER_CARMAKERS = ["tesla", "ford", "toyota", "renault", "fiat", "stellantis", "nissan",
                    "hyundai", "kia", "peugeot", "citroen", "citroën", "honda", "volvo", "tata"]


def _acceptable_article(title, snippet, my_brand, city):
    t = title.lower(); blob = (title + " " + snippet).lower(); city_l = (city or "").lower().strip()
    if any(w in t for w in VENUE_WORDS):
        return False
    for b in _OTHER_CARMAKERS:
        if re.search(r"\b" + re.escape(b) + r"\b", t):
            return False
    for b, canon in _BRAND_ALIASES.items():
        if canon != my_brand and re.search(r"\b" + re.escape(b) + r"\b", t):
            return False
    if city_l and city_l not in blob:
        return False
    if any(w in t for w in PLANT_WORDS):
        return True
    if my_brand and my_brand in t and city_l and city_l in blob:
        return True
    if city_l and city_l in t and my_brand and my_brand in blob:
        return True
    return False


def resolve_article(name, city, brand, country, override=""):
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
    key = (lang, title, year)
    with _locks["rev"]:
        cached = _rev_cache.get(key, "__miss__")
    if cached == "__miss__":
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
        cached = (revid, ts)
        with _locks["rev"]:
            _rev_cache[key] = cached
    return cached


def fetch_wikitext(lang, revid):
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
    ck = (lang, revid)
    with _locks["content"]:
        if ck in _html_cache:
            return _html_cache[ck]
    r = _get(api(lang), {"action": "parse", "oldid": revid, "prop": "text", "format": "json",
                         "formatversion": 2, "disablelimitreport": 1, "disableeditsection": 1})
    h = None
    if r:
        try:
            h = r.json()["parse"]["text"]
        except Exception:
            h = None
    if h:
        with _locks["content"]:
            _html_cache[ck] = h
    return h


# ============================================================ Wayback (fallback only)
def wayback_snapshots(url):
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


# ============================================================ text helpers
def html_to_text(html):
    h = re.sub(r"<(script|style)\b.*?</\1>", " ", html or "", flags=re.S | re.I)
    h = re.sub(r"<sup\b[^>]*class=\"[^\"]*reference[^\"]*\"[^>]*>.*?</sup>", " ", h, flags=re.S | re.I)
    h = re.sub(r"</tr\s*>", "\n", h, flags=re.I)
    h = re.sub(r"</t[dh]\s*>", " | ", h, flags=re.I)
    h = re.sub(r"<[^>]+>", " ", h)
    h = _htmlmod.unescape(h).replace("\xa0", " ")
    h = re.sub(r"[ \t]+", " ", h)
    h = re.sub(r"\n[ \t|]*\n+", "\n", h)
    return h.strip()


def wikitext_excerpt(wt):
    t = re.sub(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", " ", wt or "", flags=re.S | re.I)
    t = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"'''?|<[^>]+>|&nbsp;", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t[:LLM_MAX_INPUT_CHARS]


def parse_number(raw):
    try:
        return int(re.sub(r"[ ., ]", "", raw.strip()))
    except ValueError:
        return None


_NUM = re.compile(r"\d{1,3}(?:[ ., ]\d{3})+|\d{3,7}")

def _norm_ws(t):
    return re.sub(r"\s+", " ", str(t or "")).strip()


def _num_in_text(n, text):
    """True iff the integer n appears in text, ignoring thousands separators
    (space / non-breaking / thin space / dot / comma / apostrophe between digits).
    This is the 'can I open the page and see this number?' check."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return False
    compact = re.sub(r"(?<=\d)[\s.,'   ](?=\d)", "", text or "")
    return re.search(r"(?<!\d)" + str(n) + r"(?!\d)", compact) is not None


# ============================================================ REGEX EXTRACTORS (cheap, multilingual)
_EMP_FIELD = re.compile(
    r"\|\s*(?:num_employees|number_of_employees|employees|mitarbeiter|mitarbeiterzahl|"
    r"besch[äa]ftigte|belegschaft|effectifs?|empleados|plantilla|dipendenti|addetti|"
    r"funcion[áa]rios|empregados|werknemers|medewerkers|personeel|pracownic\w*|"
    r"zatrudnienie|zaměstnanc\w*|zamestnanc\w*|alkalmazott\w*|foglalkoztatott\w*|"
    r"anställda|ansatte|työntekij[äa]t?|angajați|çalışan\w*|сотрудник\w*|работник\w*)\s*=\s*(.+)", re.I)

_EMP_WORDS = re.compile(
    r"employ|workforce|staff|worker|mitarbeiter|besch[äa]ftigt|belegschaft|arbeitnehmer|"
    r"angestellt|effectif|salari[ée]|personnel|ouvrier|emplead|plantilla|trabajador|"
    r"dipendenti|addetti|lavoratori|occupati|empregad|funcion[áa]ri|trabalhador|collaborateur|colaborador|"
    r"werknemer|medewerker|personeel|pracownik|zatrudni|zaměstnan|zamestnan|alkalmazott|"
    r"dolgoz|munkavállal|foglalkoztat|munkás|anställd|ansatte|työntekij|henkilöstö|"
    r"angajat|salariat|çalışan|işçi|treballador|сотрудник|работник|персонал|従業員|社員|员工", re.I)

_PROD_WORDS = re.compile(
    r"produc|manufactur|assembl|vehicle|cars|units|output|built|produktion|produzier|"
    r"fahrzeuge|gefertigt|hergestellt|gebaut|véhicules|fabriqu|veh[íi]culos|fabricad|"
    r"veicoli|fabbricat|ve[íi]culos|voertuigen|geproduceerd|pojazd|wyprodukowan|vozidel|"
    r"vyrobeno|gyárt|jármű|gépjármű|fordon|ajoneuvo|vehicule|araç|otomobil|производ|автомобил", re.I)

# rendered-HTML table row:  "2015 | 324,131"
_TABLE_ROW = re.compile(r"(?:^|\n)\s*((?:19|20)\d{2})\s*\|\s*([\d][\d.,  ]{2,12}\d)")
# prose production: "324,131 vehicles ... 2015"  or  "120,000 A1s in 2011"
_YEAR = re.compile(r"(19|20)\d{2}")


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
        if 1900 <= emp <= END_YEAR and chunk.strip().replace(",", "").replace(".", "").replace(" ", "").isdigit():
            continue
        yr = _YEAR.search(chunk)
        out.append((emp, int(yr.group(0)) if yr else None, "infobox: " + chunk.strip()[:150]))
    return out


def prose_employment(text):
    """A number within ~45 chars of an employee keyword, with a nearby year.
    Deduped by (employees, reference_year)."""
    best = {}
    for m in _EMP_WORDS.finditer(text or ""):
        win = text[max(0, m.start() - 45): m.end() + 45]
        nm = _NUM.search(win)
        if not nm:
            continue
        emp = parse_number(nm.group(0))
        if emp is None or emp < 30 or emp > 400_000:
            continue
        yctx = text[max(0, m.start() - 80): m.end() + 80]
        yr = _YEAR.search(yctx)
        ry = int(yr.group(0)) if yr else None
        if ry is not None and emp == ry:                 # number is the year itself, not a count
            continue
        key = (emp, ry)
        if key not in best:
            best[key] = (emp, ry, "prose: " + win.strip()[:150])
    return list(best.values())


def table_production(text):
    out = []
    for m in _TABLE_ROW.finditer(text or ""):
        y = int(m.group(1)); u = parse_number(m.group(2))
        if u and 100 <= u <= 2_000_000 and START_YEAR <= y <= END_YEAR:
            out.append((y, u, f"table: {m.group(1)} | {m.group(2)}"))
    return out


# ---- structured model-list parser (brand/nameplate filtered) ----
_MODEL_TOKEN = re.compile(
    r"\b(audi|bmw|mini|mercedes|benz|maybach|volkswagen|vw|porsche|seat|cupra|sko?da|škoda|opel|"
    r"bentley|lamborghini|bugatti|smart|e-?tron|i\d|q\d|a\d|s\d|rs\d|x\d|m\d|z\d|golf|passat|polo|"
    r"tiguan|touran|touareg|beetle|coccinelle|k[äa]fer|lupo|caddy|transporter|amarok|arteon|jetta|"
    r"bora|sharan|toledo|le[oó]n|ibiza|altea|iltis|cayenne|macan|panamera|taycan|boxster|cayman|"
    r"911|912|914|718|sprinter|crafter|t-?roc|t-?cross|taigo|tarraco|tayron|terramar)", re.I)
_PERSON = re.compile(r"\(\s*(?:\*\s*)?\d{4}\s*[–\-—]\s*\d{4}\s*\)|\(\s*born\b|\(\s*geb\.|\(\s*né\b", re.I)
_LINE = re.compile(
    r"""(?:[A-Za-zÀ-ÿ]+\s+)?((?:19|20)\d{2})\s*[–\-—]\s*
        (?:(?:[A-Za-zÀ-ÿ]+\s+)?((?:19|20)\d{2})|present|today|heute|aujourd'hui|actuel)?
        \s*[:：]\s*(.*)$""", re.VERBOSE | re.I)
_UNITS = r"(?:units?|unit[ée]s?|Stück|St\.|exemplaires|vehicles|cars|St[üu]ck)"


def _clean_model(rest):
    units = None
    mm = re.search(r"\(([^)]*\d[^)]*)\)", rest)
    if mm and (re.search(_UNITS, mm.group(1), re.I) or _NUM.search(mm.group(1))):
        num = _NUM.search(mm.group(1))
        if num:
            units = parse_number(num.group(0))
        rest = rest[:mm.start()] + rest[mm.end():]
    name = re.sub(r"\[\d+\]|\[[a-z]\]", "", rest)
    return re.sub(r"\s{2,}", " ", name).strip(" :–-—.,;"), units


def model_list(text, lang):
    spans = []
    lines = [ln.strip().lstrip("•*-–").strip() for ln in (text or "").splitlines()]
    for i, line in enumerate(lines):
        if not line or (":" not in line and "：" not in line):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        rest = (m.group(3) or "").strip()
        if not re.search(r"[A-Za-z]", rest):
            for j in range(i + 1, min(i + 3, len(lines))):
                if lines[j] and not _LINE.match(lines[j]):
                    rest = (rest + " " + lines[j]).strip()
                    if re.search(r"[A-Za-z]", rest):
                        break
        name, units = _clean_model(rest)
        if not name or len(name) > 80 or _PERSON.search(line) or not _MODEL_TOKEN.search(name):
            continue
        ys = int(m.group(1)); ye = int(m.group(2)) if m.group(2) else None
        spans.append({"model": name, "year_start": ys, "year_end": ye,
                      "units": units, "evidence": line[:160], "how": f"regex_{lang}"})
    return spans


# ============================================================ Mistral AI
_PROMPT = """From this Wikipedia article about ONE automobile plant, extract three things for THIS plant only.

Plant: {name}
City: {city}
Brand: {brand}

Return VALID JSON ONLY (no markdown):
{{"models":[{{"model":"<name>","year_start":<int>,"year_end":<int or null>,"evidence":"<short quote>"}}],
  "production":[{{"year":"<YYYY>","units":<int>,"evidence":"<short quote>"}}],
  "employment":[{{"employees":<int>,"reference_year":<"YYYY" or null>,"evidence":"<short quote>"}}]}}

Rules:
- models: every car model assembled AT THIS PLANT, canonical nameplates AND named
  variants as separate entries (e.g. "Audi A1","Audi A1 Sportback","Audi S1").
  year_start/year_end = production years at this plant; year_end null if ongoing.
- production: ANNUAL vehicles built at this plant in a given year (not cumulative totals).
- employment: the TOTAL employee headcount for THIS plant ({city}) ONLY.
  * NEVER the group/company/worldwide total; NEVER a single department/line sub-figure.
  * NEVER an area (m2 / square metres / hectares) or a production/vehicle count — those are not employees.
  * reference_year = the year the count refers to ("Stand 2018","as of 2022","fin 2022").
    ALWAYS try to give a year; use null only if the text truly states none. Plausible 30..400000.
- Use only facts in the text. Empty lists if nothing applies.

ARTICLE:
---
{text}
---
"""


def llm_call(name, city, brand, text):
    """Call Mistral (chat/completions) and return the parsed JSON dict, or None."""
    if not (LLM_ENABLED and text):
        return None
    prompt = _PROMPT.format(name=name, city=city, brand=brand, text=text)
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    _rec("calls")
    for a in range(LLM_MAX_RETRIES + 1):
        _rate_gate()
        body = {"model": LLM_MODEL, "temperature": 0, "max_tokens": LLM_MAX_OUTPUT_TOKENS,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": prompt}]}
        try:
            r = sess().post(MISTRAL_URL, headers=headers, json=body, timeout=LLM_REQUEST_TIMEOUT)
            if r.status_code in (429, 503):
                _rec("rate_limited")
                ra = r.headers.get("retry-after", "")
                time.sleep(float(ra) if ra.replace(".", "", 1).isdigit() else min(5 * 2 ** a, 60)); continue
            if r.status_code != 200:
                _rec("http_error", f"HTTP {r.status_code}: {r.text[:200]}")
                return None
            data = r.json()
            choice = (data.get("choices") or [{}])[0]
            txt = ((choice.get("message") or {}).get("content") or "").strip()
            if not txt:
                _rec("empty", "no content"); return None
            txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.I | re.M).strip()
            try:
                parsed = json.loads(txt)
            except Exception:
                m = re.search(r"\{.*\}", txt, re.S)
                parsed = json.loads(m.group(0)) if m else None
            if parsed is not None:
                _rec("ok"); return parsed
            _rec("empty", "unparseable JSON"); return None
        except Exception as e:
            _rec("exception", str(e)[:200]); time.sleep(1.5 * (a + 1))
    return None


_VALIDATE_PROMPT = """Classify the number for the plant below. Answer with ONE word only: plant, group, or other.
  plant = the employee headcount of THIS specific plant.
  group = a company/brand/group/worldwide total (more than just this one plant).
  other = not an employee count at all (area in m2/hectares, vehicles produced, money, a year, etc.).

Plant: {name} in {city} (brand: {brand})
Number: {number}
Context: "{snippet}"
Answer:"""


def llm_validate_employment(name, city, brand, number, snippet):
    """Cheap LLM check on a regex-found figure: 'plant' | 'group' | 'other' | 'uncertain'.
    Cached by (number, snippet) so repeats across revisions don't re-call."""
    if not LLM_ENABLED:
        return "plant"                                    # dry run: don't filter
    key = ("val", int(number), (snippet or "")[:80])
    with _locks["llm"]:
        if key in _llm_cache:
            return _llm_cache[key]
    prompt = _VALIDATE_PROMPT.format(name=name, city=city, brand=brand, number=number, snippet=(snippet or "")[:200])
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    verdict = "uncertain"
    _rec("calls")
    for a in range(2):
        _rate_gate()
        try:
            r = sess().post(MISTRAL_URL, headers=headers, timeout=LLM_REQUEST_TIMEOUT,
                            json={"model": LLM_MODEL, "max_tokens": 8, "temperature": 0,
                                  "messages": [{"role": "user", "content": prompt}]})
            if r.status_code in (429, 503):
                _rec("rate_limited"); time.sleep(min(5 * 2 ** a, 30)); continue
            if r.status_code != 200:
                _rec("http_error", f"HTTP {r.status_code}: {r.text[:120]}"); break
            t = (((r.json().get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip().lower()
            _rec("ok")
            verdict = "plant" if "plant" in t else "group" if "group" in t else "other" if "other" in t else "uncertain"
            break
        except Exception as e:
            _rec("exception", str(e)[:120]); time.sleep(1.5 * (a + 1))
    with _locks["llm"]:
        _llm_cache[key] = verdict
    return verdict


def llm_extract(cache_key, text, name, city, brand, lang, src):
    with _locks["llm"]:
        if cache_key in _llm_cache:
            return _llm_cache[cache_key]
    models, prod, emp = [], [], []
    if LLM_ENABLED and text and (_EMP_WORDS.search(text) or _PROD_WORDS.search(text) or _MODEL_TOKEN.search(text)):
        data = llm_call(name, city, brand, text[:LLM_MAX_INPUT_CHARS]) or {}
        for r in data.get("models", []):
            try:
                ys = int(r["year_start"])
            except Exception:
                continue
            ye = r.get("year_end"); ye = int(ye) if ye not in (None, "", "null") else None
            mn = str(r.get("model", "")).strip()
            if mn and re.search(r"[A-Za-z]", mn):
                models.append({"model": mn, "year_start": ys, "year_end": ye, "units": None,
                               "evidence": str(r.get("evidence", ""))[:160], "how": f"llm_{src}_{lang}"})
        for r in data.get("production", []):
            try:
                u = int(r["units"]); y = int(str(r["year"])[:4])
            except Exception:
                continue
            if 100 <= u <= 2_000_000 and START_YEAR <= y <= END_YEAR:
                prod.append((y, u, str(r.get("evidence", ""))[:160], f"llm_{src}_{lang}"))
        for r in data.get("employment", []):
            try:
                e = int(r["employees"])
            except Exception:
                continue
            if 30 <= e <= 400_000:
                ry = r.get("reference_year"); ry = int(ry) if (ry and re.match(r"^\d{4}$", str(ry))) else None
                emp.append((e, ry, str(r.get("evidence", ""))[:160], f"llm_{src}_{lang}"))
    res = {"models": models, "prod": prod, "emp": emp}
    with _locks["llm"]:
        _llm_cache[cache_key] = res
    return res


# ============================================================ merge helpers
_BRAND_PREFIX = re.compile(r"^(projekt|project|audi|volkswagen|vw|bmw|mercedes(?:-benz)?|porsche|opel|seat|skoda|škoda)\s+", re.I)
_VARIANT_NOISE = re.compile(r"\b(quattro|gen\.?|generation|mk\s?\d|GE|\(?\d{4}\)?)\b", re.I)
_BARE_BRANDS = {"audi", "bmw", "mercedes", "mercedes benz", "volkswagen", "vw", "porsche",
                "seat", "skoda", "opel", "daimler", "mini", "cupra", "bentley", "lamborghini"}


def _norm_model(model):
    s = re.sub(r"\([^)]*\)", " ", model.lower().strip())        # drop (C5),(B7),(Typ 8E),(2018)...
    s = _VARIANT_NOISE.sub("", _BRAND_PREFIX.sub("", s))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


def merge_models(spans):
    groups = {}
    for s in spans:
        nk = _norm_model(s["model"])
        if s["model"] and nk and nk not in _BARE_BRANDS:    # drop empty / bare-brand names ("Porsche")
            groups.setdefault(nk, []).append(s)
    out = []
    for items in groups.values():
        starts = {i["year_start"] for i in items}
        ends = {i["year_end"] for i in items if i["year_end"] is not None}
        surfaces = [i["model"] for i in items]
        en = [i["model"] for i in items if i["how"].endswith("_en")]
        pool = en or surfaces
        display = max(sorted(set(pool)), key=lambda x: (sum(1 for s in pool if s == x), len(x)))
        display = re.sub(r"\s*\([^)]*\)", "", display).strip()   # strip (C5)/(B7)-type codes from the label
        out.append({"display": display, "surfaces": sorted(set(surfaces)),
                    "year_start": min(starts),
                    "year_end": None if any(i["year_end"] is None for i in items) else (max(ends) if ends else None),
                    "langs": sorted({i["how"].split("_")[-1] for i in items}),
                    "methods": sorted({i["how"] for i in items}),
                    "year_conflict": int(len(starts) > 1 or len(ends) > 1),
                    "evidence": next((i["evidence"] for i in items if i["evidence"]), "")})
    return out


# ============================================================ per-plant pipeline
def process_plant(p):
    name, city, brand, country = p["plant_name"], p.get("city", ""), p["brand"], p["country"]
    resolved = resolve_article(name, city, brand, country, p.get(WIKI_OVERRIDE_COL, "") or "")
    info = {"plant_id": p["plant_id"], "resolved": "", "languages": 0}
    model_spans, emp_rows, prod_recs = [], [], []
    if not resolved:
        return model_spans, emp_rows, prod_recs, info
    base_lang, base_title = resolved
    lmap = langlinks(base_lang, base_title)
    if not ALL_LANGUAGES:
        want = set(LANGUAGES) | {COUNTRY_LANG.get(country, "en"), base_lang}
        lmap = {k: v for k, v in lmap.items() if k in want}
    info["resolved"] = f"[{base_lang}] {base_title}"; info["languages"] = len(lmap)

    def add_emp(e, ry, ev, how, lang, snap_year, revid, ts, url, source, verify_text=""):
        if e > PLANT_EMP_MAX:
            return                                            # group/worldwide total, not a single plant
        # HARD REQUIREMENT: the figure MUST actually appear on the page that `url`
        # points to (this exact revision / Wayback snapshot). If you can't open the
        # link and see the number, the row is dropped instead of being emitted.
        if not _num_in_text(e, verify_text):
            _rec("emp_unverified")
            return
        if ry is not None and not (START_YEAR <= ry <= END_YEAR):
            return                                            # only 2006-2026 figures
        ry_src = "stated" if ry is not None else "revision_year"
        if ry is None:
            ry = snap_year                                    # guarantee a year for merging
        if _COVERED_EMP.get(p["plant_id"]) and ry in _COVERED_EMP[p["plant_id"]]:
            return                                            # gap-fill: skip years already known
        emp_rows.append({"plant_id": p["plant_id"], "country": country, "group": p.get("group", ""),
            "plant_name": name, "city": city, "brand": brand, "archive_year": snap_year,
            "registry_url": url, "url_kind": "WIKIPEDIA", "archive_timestamp": ts,
            "archive_url": url, "status": "found_wikipedia", "employees": e,
            "reference_year": ry, "ref_year_source": ry_src, "verified_on_page": "Y",
            "snippet": ev, "pattern": how,
            "extraction": "llm" if how.startswith("llm") else ("regex+llm" if how.startswith("regexllm") else "regex"),
            "source": source, "source_lang": lang, "revid": revid, "error": ""})

    def add_prod(y, u, ev, how, lang, source, url, verify_text=""):
        if verify_text and not _num_in_text(u, verify_text):
            _rec("prod_unverified"); return          # LLM figure not on the openable page
        prod_recs.append({"plant_id": p["plant_id"], "plant_name": name, "brand": brand, "country": country,
            "year": y, "production_units": u, "source": source, "source_lang": lang,
            "evidence": ev, "method": how, "archive_url": url})

    for lang, title in lmap.items():
        wiki_url = f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
        # year-end revisions (dedup by revid), newest first
        year_rev = {}
        for y in YEARS:
            revid, ts = revision_asof(lang, title, y)
            if revid:
                year_rev[y] = (revid, ts)

        if not year_rev:
            # ---- WAYBACK FALLBACK (no revisions for this edition) ----
            if USE_WAYBACK_FALLBACK:
                for ts, original in wayback_snapshots(wiki_url)[:WAYBACK_MAX_SNAPSHOTS]:
                    sy = int(ts[:4]) if ts[:4].isdigit() else None
                    if sy is None or not (START_YEAR <= sy <= END_YEAR):
                        continue
                    htm = fetch_wayback_html(ts, original)
                    if not htm:
                        continue
                    txt = html_to_text(htm)
                    arch = f"https://web.archive.org/web/{ts}/{original}"
                    for y, u, ev in table_production(txt):
                        add_prod(y, u, ev, f"regex_wb_{lang}", lang, "wayback", arch)
                    for s in model_list(txt, lang):
                        s["how"] = f"regex_wb_{lang}"; model_spans.append(s)
                    res = llm_extract(("wb", lang, ts), txt, name, city, brand, lang, "wb")
                    model_spans.extend(res["models"])
                    wb_verify = _norm_ws(txt)
                    for y, u, ev, how in res["prod"]:
                        add_prod(y, u, ev, how, lang, "wayback", arch, verify_text=wb_verify)
                    for e, ry, ev, how in res["emp"]:
                        add_emp(e, ry, ev, how, lang, sy, "", ts, arch, "wayback", verify_text=wb_verify)
            continue

        # which revisions get an LLM call
        years_sorted = sorted(year_rev)
        revid_to_year = {}
        for y in years_sorted:
            revid_to_year.setdefault(year_rev[y][0], y)
        unique_years = sorted(revid_to_year.values())
        if LLM_ALL_REVISIONS:
            sample = set(unique_years)                          # full deployment: every revision
        else:
            sample = {unique_years[0], unique_years[-1]} if unique_years else set()
            if LLM_SAMPLE_MAX > 2 and len(unique_years) > 2:
                step = max(1, len(unique_years) // (LLM_SAMPLE_MAX - 1))
                sample |= set(unique_years[::step])

        seen_rev = set()
        for y in years_sorted:
            revid, ts = year_rev[y]
            if revid in seen_rev:
                continue
            seen_rev.add(revid)
            wt = fetch_wikitext(lang, revid) or ""
            # Clickable, checkable permalink to THIS exact revision (with title).
            permalink = f"https://{lang}.wikipedia.org/w/index.php?title={quote(title.replace(' ', '_'))}&oldid={revid}"
            # Rendered revision text = exactly what you see when you OPEN the permalink.
            # Every employment/production figure is verified against this before emit.
            html = fetch_revision_html(lang, revid)
            rendered = html_to_text(html) if html else ""
            txt = rendered or wikitext_excerpt(wt)
            verify_text = _norm_ws(rendered + " \n " + wikitext_excerpt(wt))
            # employment HYBRID: regex finds infobox figures (recall); the number must
            # be present on the openable page, THEN the LLM validates plant vs group.
            for e, ry, ev in infobox_employment(wt):
                if not _num_in_text(e, verify_text):
                    _rec("emp_unverified"); continue          # not on the openable page -> skip
                verdict = llm_validate_employment(name, city, brand, e, ev)
                if verdict in ("plant", "uncertain"):
                    add_emp(e, ry, ev, f"regexllm_rev_{lang}", lang, y, revid, ts, permalink, "revision", verify_text=verify_text)
            for s in model_list(wt, lang):
                s["how"] = f"regex_rev_{lang}"; model_spans.append(s)
            for yy, u, ev in table_production(txt):
                add_prod(yy, u, ev, f"regex_rev_{lang}", lang, "revision", permalink)
            if y in sample:
                res = llm_extract(("rev", lang, revid), txt, name, city, brand, lang, "rev")
                model_spans.extend(res["models"])
                for yy, u, ev, how in res["prod"]:
                    add_prod(yy, u, ev, how, lang, "revision", permalink, verify_text=verify_text)
                for e, ry, ev, how in res["emp"]:
                    add_emp(e, ry, ev, how, lang, y, revid, ts, permalink, "revision", verify_text=verify_text)
    return model_spans, emp_rows, prod_recs, info


# ============================================================ output
def build_outputs(all_models, emp_rows, prod_recs, infos):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # ---- models: merge per plant, binary matrix + long ----
    by_plant = {}
    plant_meta = {}
    for pid, name, brand, country, span in all_models:
        by_plant.setdefault(pid, []).append(span)
        plant_meta[pid] = (name, brand, country)
    bin_rows, long_rows = [], []
    for pid, spans in by_plant.items():
        name, brand, country = plant_meta[pid]
        for mm in sorted(merge_models(spans), key=lambda x: x["year_start"]):
            end = mm["year_end"] if mm["year_end"] is not None else END_YEAR
            if mm["year_start"] > END_YEAR or end < START_YEAR:
                continue
            row = {"plant_id": pid, "plant_name": name, "brand": brand, "country": country, "car model ": mm["display"]}
            for y in YEARS:
                row[y] = 1 if (mm["year_start"] <= y <= end) else 0
            bin_rows.append(row)
            long_rows.append({"plant_id": pid, "plant_name": name, "brand": brand, "country": country,
                "model": mm["display"], "model_surfaces": " | ".join(mm["surfaces"]),
                "year_start": mm["year_start"], "year_end": mm["year_end"] if mm["year_end"] is not None else "",
                "source_langs": ",".join(mm["langs"]), "methods": ",".join(mm["methods"]),
                "year_conflict": mm["year_conflict"], "evidence": mm["evidence"]})
    cols = ["plant_id", "plant_name", "brand", "country", "car model "] + YEARS
    pd.DataFrame(bin_rows, columns=cols).to_csv(OUTPUT_DIR / "models_binary_matrix.csv", index=False)
    pd.DataFrame(long_rows).to_csv(OUTPUT_DIR / "models_long.csv", index=False)

    # ---- employment: dedupe by (plant, ref_year|archive_year, employees) ----
    best = {}
    for r in emp_rows:
        ry = r["reference_year"]
        # dated figures keyed by their reference year; undated repeats collapse to one per (plant, count)
        kkey = ry if str(ry) not in ("", "nan", "None") else "NA"
        k = (r["plant_id"], kkey, r["employees"])
        if k not in best:
            best[k] = r
    # Flag a headcount value attributed to >=3 different plants: almost always a
    # hallucinated round number or a shared division/group total, not a real plant figure.
    _val_plants = {}
    for r in best.values():
        _val_plants.setdefault(r["employees"], set()).add(r["plant_id"])
    for r in best.values():
        r["suspect"] = "cross_plant_repeat" if len(_val_plants.get(r["employees"], ())) >= 3 else ""
    emp_cols = ["plant_id", "country", "group", "plant_name", "city", "brand", "archive_year",
                "registry_url", "url_kind", "archive_timestamp", "archive_url", "status", "employees",
                "reference_year", "ref_year_source", "verified_on_page", "snippet", "pattern",
                "extraction", "suspect", "source", "source_lang", "revid", "error"]
    pd.DataFrame(list(best.values()), columns=emp_cols).to_csv(OUTPUT_DIR / "employment_from_wikipedia.csv", index=False)

    # ---- production: dedupe by (plant, year, units) ----
    pbest = {}
    for r in prod_recs:
        k = (r["plant_id"], r["year"], r["production_units"])
        if k not in pbest:
            pbest[k] = r
    pd.DataFrame(list(pbest.values())).to_csv(OUTPUT_DIR / "production_from_wikipedia.csv", index=False)

    pd.DataFrame(infos).to_csv(OUTPUT_DIR / "wiki_articles_resolved.csv", index=False)
    print(f"\n[out] models: {len(bin_rows)} (plant,model) rows over {len(by_plant)} plants")
    print(f"[out] employment: {len(best)} rows | production: {len(pbest)} rows")
    print(f"[out] written to {OUTPUT_DIR}")


def load_plants():
    p = Path(PLANTS_INPUT_PATH)
    df = pd.read_excel(p, sheet_name=PLANTS_INPUT_SHEET) if p.suffix.lower().startswith(".xls") else pd.read_csv(p)
    keep = [c for c in ["plant_id", "country", "group", "plant_name", "city", "brand", WIKI_OVERRIDE_COL] if c in df.columns]
    return df[keep].drop_duplicates(subset=["plant_id"]).fillna("").to_dict("records")


_COVERED_EMP: dict = {}     # plant_id -> {years that already have an employment figure}


def load_existing_employment():
    """Build {plant_id: set(years)} that already have an employment number, so the
    scraper only fills the gaps. Years come from reference_year and archive_year."""
    if not EXISTING_EMPLOYMENT_PATH or not Path(EXISTING_EMPLOYMENT_PATH).exists():
        return {}
    df = pd.read_csv(EXISTING_EMPLOYMENT_PATH)
    if "employees" not in df.columns or "plant_id" not in df.columns:
        return {}
    emp = pd.to_numeric(df["employees"], errors="coerce")
    cov = {}
    for i in df.index:
        if pd.isna(emp.at[i]):
            continue
        pid = str(df.at[i, "plant_id"])
        for col in ("reference_year", "archive_year"):
            if col in df.columns:
                y = pd.to_numeric(df.at[i, col], errors="coerce")
                if pd.notna(y) and START_YEAR <= int(y) <= END_YEAR:
                    cov.setdefault(pid, set()).add(int(y))
    return cov


def run():
    global _COVERED_EMP
    load_caches()
    plants = load_plants()
    _COVERED_EMP = load_existing_employment()
    if _COVERED_EMP:
        n = sum(len(v) for v in _COVERED_EMP.values())
        print(f"[run] gap-fill: {n} (plant,year) employment pairs already known across "
              f"{len(_COVERED_EMP)} plants -> skipping those, filling only the gaps")
    print(f"[run] {len(plants)} plants | LLM={LLM_ENABLED} model={LLM_MODEL} | langs={LANGUAGES}+country | sample={LLM_SAMPLE_MAX}/lang")
    all_models, emp_all, prod_all, infos = [], [], [], []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(process_plant, pl): pl for pl in plants}
        done = 0
        for fut in as_completed(futs):
            pl = futs[fut]
            try:
                spans, emp, prod, info = fut.result()
                for s in spans:
                    all_models.append((pl["plant_id"], pl["plant_name"], pl["brand"], pl["country"], s))
                emp_all += emp; prod_all += prod; infos.append(info)
                print(f"  ✓ {pl['plant_id']:26} langs={info['languages']:2} models~{len(spans):3} emp={len(emp):3} prod={len(prod):3}  {info['resolved'][:34]}")
            except Exception as e:
                print(f"  ✗ {pl['plant_id']}: {e}")
            done += 1
            if done % 5 == 0:
                save_caches()
    save_caches()
    build_outputs(all_models, emp_all, prod_all, infos)
    print(f"[run] LLM stats: {_stats}")
    if _stats["calls"] and _stats["rate_limited"] > _stats["ok"]:
        print(f"[run] WARNING: heavy throttling — lower LLM_MAX_RPM (now {LLM_MAX_RPM}) and rerun (cache kept).")


if __name__ == "__main__":
    run()
