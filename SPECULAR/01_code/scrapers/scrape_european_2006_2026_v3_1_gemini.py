"""

# --- portability shim, added when this project was archived to INARI ---------
# The originals carried absolute paths from the machine they were written on.
# specular_paths resolves everything from the project root instead, so these
# scripts run wherever the folder is put. Set SPECULAR_ROOT to override.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from specular_paths import P as _P
# ---------------------------------------------------------------------------

Wayback employment scraper v3.1 (EUROPEAN) — automotive plants 2006–2026.

European variant
----------------
Same engine as the German Haiku scraper but expanded to all countries in the
Cleaned_Wayback_Registry: Germany, Belgium, Hungary, Poland, Portugal,
Slovakia, Spain (43 plants total). Also adds Porsche as a 5th brand.

Specific changes vs. the German variant:
  • Country filter removed — every plant in the registry is scraped.
  • Brand detector recognises Porsche from `_PORSCHE_` plant_ids and the
    porsche / newsroom.porsche.com hosts.
  • Porsche regex pack (strict + loose) added; reuses VW newsroom-shaped
    structure since Porsche Newsroom layouts are similar.
  • _CITY_VARIANTS extended with Brussels, Győr, Debrecen, Kecskemét,
    Poznań, Września, Palmela, Bratislava, Pamplona/Navarra,
    Zuffenhausen/Stuttgart, etc.
  • `volkswagen-poznan.pl` is classified GROUP_OVERVIEW so the two Polish
    plants (Poznań + Września) that share this URL each get city-anchored
    extraction.
  • `porsche.com/.../porsche-ag/` is GROUP_OVERVIEW (it lists all Porsche
    sites). Newsroom URLs are PLANT_SPECIFIC.
  • HUGE_ARCHIVE_HOSTS includes newsroom.porsche.com and porsche.com so
    those use city-filtered prefix queries rather than bare host_prefix.
  • All output files + caches use an `_european_` suffix so they live
    alongside but don't overwrite the German scraper's outputs/caches.

What changed vs. v3
-------------------
v3 was finding things but missing valid snapshots (Audi Ingolstadt 2015 page,
which contains "Employees 40,027 (as at: December 31, 2014)", was reported as
no_archive) and producing LLM hallucinations (BMW Berlin showed paired round
numbers 2000/2000/2100/2100/2200/2200/2400/2400 with no reference year,
Audi Ingolstadt 2006=31337 — leet pattern). v3.1 patches three things:

  1. **CDX host_prefix queries got truncated on big domains.** Wayback caps
     CDX results at ~150k and orders ascending by timestamp by default, so
     a host-wide query on audi.com returned the OLDEST snapshots and lost
     2015+. v3.1 replaces the bare host_prefix on HUGE_ARCHIVE_HOSTS with a
     server-side city-filtered prefix query
     (`url=audi.com matchType=prefix filter=original:.*ingolstadt.*`),
     which surgically returns just plant-relevant snapshots. Bare-domain
     hosts like `werk-bremen.mercedes-benz.com` still use the host_prefix
     fallback because their archives are small.

  2. **LLM acceptance is now strict.** LLM hits MUST include a reference
     year. The "round-number-no-ref" failure mode (2000, 2100, 2200…)
     where the LLM was clearly rounding-up vague mentions is now rejected
     at the source.

  3. **Proximity check in LLM guard.** The returned number must appear
     within 150 characters of an employment keyword (Mitarbeiter,
     Beschäftigte, employees, workforce, Stand). Otherwise it could be a
     postal code, telephone number, vehicle count, etc.

What changed vs. v2
-------------------
v2's architecture is correct (URL classification, group-page mode, strict/loose
patterns, no matchType=domain) — confirmed by the 36/39 genuine hits in the
Book20 run. The problems in v2 were RECALL bugs, not structural bugs:

  1. **Empty CDX results were cached.** A single rate-limited CDX response
     pickled an empty {} for that URL, and the 30-day TTL meant it never got
     re-tried. v3 ONLY caches non-empty results. Rate-limited URLs get a
     fresh chance on the next run.

  2. **CDX timeouts were too aggressive.** v2's 20s timeout + 2 retries gave
     up too fast on slow Wayback responses. v3 uses 30s + 3 retries + 2.0s
     backoff (between v1's "too patient" and v2's "too impatient").

  3. **`no_archive` conflated two failures.** Couldn't tell "Wayback really
     has no snapshots" from "every CDX query failed/timed-out." v3 emits
     new status values: `cdx_empty` (verified empty after all retries) vs
     `cdx_failed` (every attempt errored/timed-out — retry-worthy).

  4. **Snapshot validation was too strict.** v2 rejected host-prefix matches
     unless the snapshot path contained the exact city slug. v3 also accepts
     paths containing 'werk', 'standort', 'daten', 'fakten', 'plant' — these
     are the legacy BMW Werk-Daten and Mercedes legacy host URL shapes.

  5. **LLM fallback for `not_found` pages.** When a page is successfully
     fetched but regex returns nothing, v3 calls Anthropic's Haiku model
     with a tight extraction prompt. Results are validated (the returned
     number must actually appear in the page) and cached by digest so
     identical pages are only billed once. Status `found_llm` distinguishes
     LLM hits from regex hits. Disable by leaving GEMINI_API_KEY unset.

Setup
-----
  export GEMINI_API_KEY=AIza...                            # required for LLM fallback
  python3 scrape_european_2006_2026_v3_1_gemini.py         # resumes from checkpoint

Before first run, delete v2 cache (it has poisoned empty entries):
  rm "<output_dir>/.cdx_cache_v2.pkl"
v3 uses a fresh cache file (.cdx_cache_v3.pkl) so v2 results aren't reused.

Expected: 30–50% recall regex-only, 70–90% recall with LLM fallback.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# 0. CONFIG
# ============================================================

INPUT_PATH = str(_P.scrapers / "inputs" / "Cleaned_Wayback_Registry.xlsx")
# European registry uses sheet 'Cleaned Registry'. Older variants kept as
# backward-compatible fallbacks in case the user points at an older file.
INPUT_SHEET_CANDIDATES = ["Cleaned Registry", "Wayback Registry", "URL Registry"]
INPUT_SHEET = INPUT_SHEET_CANDIDATES[0]   # used in error messages only

OUTPUT_DIR = Path(str(_P.scraper_out / "all_german_scraper"))
OUTPUT_LONG = OUTPUT_DIR / "automotive_employment_long_european_2006_2026_v3_1_gemini.csv"
OUTPUT_WIDE_SNAPSHOT = OUTPUT_DIR / "automotive_employment_wide_by_snapshot_year_european_v3_1_gemini.csv"
OUTPUT_WIDE_REFERENCE = OUTPUT_DIR / "automotive_employment_wide_by_reference_year_european_v3_1_gemini.csv"
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint_employment_european_v3_1_gemini.csv"
# CDX and text caches are LLM-agnostic — same Wayback snapshots, same page
# text — so they SHARE files with the Haiku scraper to avoid re-doing the
# slow Wayback discovery + fetch work.
CDX_CACHE_PATH = OUTPUT_DIR / ".cdx_cache_european_v3_1.pkl"
TEXT_CACHE_PATH = OUTPUT_DIR / ".text_cache_european_v3_1.pkl"
# LLM cache is MODEL-specific — keep Gemini results in a separate file so
# Haiku and Gemini extractions don't intermix.
LLM_CACHE_PATH = OUTPUT_DIR / ".llm_cache_european_v3_1_gemini.pkl"

# v3.1: hosts where a bare host_prefix CDX returns too much (Wayback caps the
# result and we lose later years). For these we use a city-filtered prefix
# query instead of the bare host_prefix.
HUGE_ARCHIVE_HOSTS = {
    "audi.com", "www.audi.com",
    "volkswagenag.com", "www.volkswagenag.com",
    "volkswagen-newsroom.com", "www.volkswagen-newsroom.com",
    "volkswagen-group.com", "www.volkswagen-group.com",
    "bmwgroup.com", "www.bmwgroup.com",
    "audi-mediacenter.com", "www.audi-mediacenter.com",
    "group.mercedes-benz.com",
    "mercedes-benz.com", "www.mercedes-benz.com",
    # European scraper additions:
    "newsroom.porsche.com",
    "porsche.com", "www.porsche.com",
}

START_YEAR = 2006
END_YEAR = 2026

MAX_SNAPSHOTS_PER_URL_PER_YEAR = 5   # v3.1.1: bumped from 3 so the right snapshot is tried even when forum/museum URLs outnumber plant-info URLs
MAX_WORKERS = 8
REQUEST_TIMEOUT = 100                 # v2 was 20 (too short), v1 was 60 (too long)
CDX_RETRY_ATTEMPTS = 6               # v2 was 2
CDX_RETRY_BACKOFF = 6              # v2 was 1.5
CHECKPOINT_EVERY = 50
RESUME_FROM_CHECKPOINT = True
CDX_CACHE_TTL_DAYS = 30

# ---- LLM fallback (Gemini variant) ----
LLM_ENABLED = bool(os.environ.get("GEMINI_API_KEY"))
LLM_MODEL = "gemini-3.5-flash"   # user-specified Google model
LLM_MAX_INPUT_CHARS = 12_000              # truncate cleaned page text to this
LLM_REQUEST_TIMEOUT = 30
LLM_MAX_RETRIES = 2


# ============================================================
# 1. HTTP — one Session per thread (thread-safe, pooled)
# ============================================================

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
    )
}
_thread_local = threading.local()


def _build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=2,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=MAX_WORKERS,
        pool_maxsize=MAX_WORKERS,
    )
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update(_HEADERS)
    return s


def get_session() -> requests.Session:
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = _build_session()
        _thread_local.session = s
    return s


# ============================================================
# 2. TEXT HELPERS
# ============================================================

def clean_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = str(text).replace("\xa0", " ").replace(" ", " ")
    return re.sub(r"\s+", " ", text).strip()


def to_int(raw: str) -> Optional[int]:
    if not raw:
        return None
    digits = re.sub(r"\D", "", raw)
    if not digits:
        return None
    if len(digits) == 4 and not re.search(r"[.,]", raw):
        return None
    val = int(digits)
    if val < 50 or val > 250_000:
        return None
    return val


_PLAUSIBLE_YEAR_RE = re.compile(r"(20(?:0[6-9]|1[0-9]|2[0-9]|30))")


def parse_reference_year(stand_text: str, snapshot_year: Optional[str] = None) -> Optional[str]:
    if not stand_text:
        return None
    s = stand_text
    anchor_re = re.compile(
        r"(?:Stand|im\s+Jahr|as\s+(?:of|at))\b[^.)]{0,40}?(20(?:0[6-9]|1[0-9]|2[0-9]|30))",
        re.I,
    )
    m = anchor_re.search(s)
    if m:
        return m.group(1)
    candidates = _PLAUSIBLE_YEAR_RE.findall(s)
    if not candidates:
        return None
    if snapshot_year and snapshot_year.isdigit():
        sy = int(snapshot_year)
        candidates.sort(key=lambda y: abs(int(y) - sy))
    return candidates[0]


# ============================================================
# 3. BRAND / URL CLASSIFICATION
# ============================================================

def detect_brand(url: str, plant_id: str = "") -> str:
    u = (url or "").lower()
    pid = (plant_id or "").upper()
    if "_VW_" in pid or "_VWN_" in pid:    return "vw"     # VWN = VW Nutzfahrzeuge (commercial)
    if "_AUDI_" in pid:      return "audi"
    if "_MERCEDES_" in pid:  return "mercedes"
    if "_BMW_" in pid:       return "bmw"
    if "_PORSCHE_" in pid:   return "porsche"              # European scraper addition
    if "porsche" in u:       return "porsche"
    if "volkswagen" in u:    return "vw"
    if "audi" in u:          return "audi"
    if "mercedes" in u or "daimler" in u or "amg" in u: return "mercedes"
    if "bmw" in u:           return "bmw"
    return "unknown"


GROUP_LANDING_PATTERNS = [
    re.compile(r"bmwgroup\.com/(?:en|de)/company/locations(?:\.html)?$", re.I),
    re.compile(r"volkswagen-group\.com/(?:en|de)/volkswagen-group-sites-", re.I),
    re.compile(r"audi-mediacenter\.com/(?:en|de)/(?:sites|standorte)-\d+", re.I),
    re.compile(r"group\.mercedes-benz\.com/company/production/procuction-network", re.I),
    re.compile(r"group\.mercedes-benz\.com/unternehmen/produktion/produktionsnetzwerk/?$", re.I),
]


GROUP_OVERVIEW_PATTERNS = [
    re.compile(r"volkswagenag\.com/.*production_plants\.html", re.I),
    re.compile(r"volkswagenag\.com/.*portrait-and-production-plants\.html", re.I),
    re.compile(r"volkswagen-sachsen\.de/(?:de|en)\.html?$", re.I),
    re.compile(r"audi\.com/corporate/(?:en|de)/company/production-sites(?:\.html)?$", re.I),
    # European scraper additions:
    # volkswagen-poznan.pl is shared between PL_VWN_POZNAN and PL_VWN_WRZESNIA —
    # forces city-anchored extraction so each plant's data is isolated.
    re.compile(r"^https?://(?:www\.)?volkswagen-poznan\.pl/?$", re.I),
    # porsche.com/locations-and-contact lists all Porsche plants — group overview.
    re.compile(r"porsche\.com/(?:germany|international)/locations-and-contact/porsche-ag/?$", re.I),
]


def classify_url(url: str) -> str:
    if not url:
        return "PLANT_SPECIFIC"
    for p in GROUP_LANDING_PATTERNS:
        if p.search(url):
            return "GROUP_LANDING"
    for p in GROUP_OVERVIEW_PATTERNS:
        if p.search(url):
            return "GROUP_OVERVIEW"
    return "PLANT_SPECIFIC"


def extract_page_id(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    m = re.search(r"-(\d{3,8})$", path)
    return m.group(1) if m else ""


def _normalise_host(netloc: str) -> str:
    return netloc.replace(":443", "").replace(":80", "")


# ============================================================
# 4. CDX SNAPSHOT DISCOVERY
# ============================================================

def build_cdx_queries(url: str, city_slugs: Optional[list[str]] = None) -> list[dict]:
    """v3.1: takes optional city_slugs for HUGE_ARCHIVE_HOSTS, which use a
    server-side city-filtered prefix query instead of a bare host_prefix.
    Bare host_prefix is kept for small bare-domain hosts (werk-*.de etc.).
    """
    parsed = urlparse(url)
    host = _normalise_host(parsed.netloc)
    path = parsed.path.rstrip("/")
    page_id = extract_page_id(url)

    queries: list[dict] = []
    queries.append({"label": "exact", "params": {"url": url.rstrip("/")}})

    if host.startswith("www."):
        alt_host = host[4:]
    else:
        alt_host = f"www.{host}"
    if alt_host != host:
        alt_url = f"{parsed.scheme or 'https'}://{alt_host}{path or '/'}"
        if parsed.query:
            alt_url += f"?{parsed.query}"
        queries.append({"label": "alt_www", "params": {"url": alt_url.rstrip("/")}})

    if "/de/" in path:
        queries.append({"label": "alt_lang",
                        "params": {"url": url.replace("/de/", "/en/", 1).rstrip("/")}})
    elif "/en/" in path:
        queries.append({"label": "alt_lang",
                        "params": {"url": url.replace("/en/", "/de/", 1).rstrip("/")}})

    if page_id:
        for lang_path in ("/de/", "/en/"):
            queries.append({
                "label": f"prefix_{lang_path.strip('/')}",
                "params": {"url": f"{host}{lang_path}", "matchType": "prefix"},
                "filter_page_id": page_id,
            })

    if classify_url(url) == "PLANT_SPECIFIC":
        is_huge = host in HUGE_ARCHIVE_HOSTS or alt_host in HUGE_ARCHIVE_HOSTS
        for h in (host, alt_host):
            if h in HUGE_ARCHIVE_HOSTS:
                # City-filtered prefix instead of bare host_prefix: surgically
                # finds URLs containing the plant city, won't hit Wayback's
                # internal result cap that lost 2015+ snapshots in v3.
                if city_slugs:
                    for cs in city_slugs:
                        if not cs:
                            continue
                        queries.append({
                            "label": f"host_prefix_city_{cs.lower().replace(' ', '_')}",
                            "params": {"url": h, "matchType": "prefix"},
                            "filter_city_slug": cs,
                        })
            else:
                # Small bare-domain host — bare host_prefix is fine and often
                # the only way to find anything (e.g. werk-bremen.mercedes-benz.com).
                queries.append({
                    "label": f"host_prefix_{'www' if h.startswith('www.') else 'nowww'}",
                    "params": {"url": h, "matchType": "prefix"},
                })

    seen, out = set(), []
    for q in queries:
        k = (q["params"].get("url"), q["params"].get("matchType"))
        if k in seen:
            continue
        seen.add(k)
        out.append(q)
    return out


def _cdx_cache_key(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:16]


_cdx_cache_lock = threading.Lock()
_cdx_cache: dict = {}
_cdx_stats_lock = threading.Lock()
_cdx_stats: dict[str, dict] = {}     # per-URL summary for end-of-run report


def _record_cdx_stat(url: str, label: str, outcome: str) -> None:
    """outcome ∈ {'rows', 'empty', 'failed'}."""
    with _cdx_stats_lock:
        s = _cdx_stats.setdefault(url, {"rows": 0, "empty": 0, "failed": 0,
                                        "labels_tried": set(), "labels_succeeded": set()})
        s[outcome] = s.get(outcome, 0) + 1
        s["labels_tried"].add(label)
        if outcome == "rows":
            s["labels_succeeded"].add(label)


def _load_cdx_cache() -> None:
    global _cdx_cache
    if CDX_CACHE_PATH.exists():
        try:
            with CDX_CACHE_PATH.open("rb") as f:
                raw = pickle.load(f)
            now = time.time()
            ttl = CDX_CACHE_TTL_DAYS * 86400
            # v3: drop expired AND drop any empty-by-year entries from older runs
            kept = {}
            for k, v in raw.items():
                if now - v.get("_t", 0) >= ttl:
                    continue
                by_year = v.get("by_year", {})
                total = sum(len(snaps) for snaps in by_year.values())
                if total == 0:
                    continue
                kept[k] = v
            _cdx_cache = kept
            print(f"[cache] loaded {len(_cdx_cache)} non-empty CDX cache entries")
        except Exception as e:
            print(f"[cache] failed to load CDX cache ({e}); starting fresh")
            _cdx_cache = {}


def _save_cdx_cache() -> None:
    try:
        CDX_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _cdx_cache_lock:
            with CDX_CACHE_PATH.open("wb") as f:
                pickle.dump(_cdx_cache, f)
    except Exception as e:
        print(f"[cache] save failed: {e}")


def _query_cdx_once(query: dict) -> tuple[list[dict], str]:
    """Returns (rows, outcome_info). Outcome strings start with 'rows' / 'empty' / 'failed:...'."""
    # v3.1.2: accept 200, 301, 302. Wayback often indexes the canonical
    # https variant as a 301/302 from the http URL, and CDX with statuscode:200
    # only was missing those. (BMW Dingolfing 2017 case.)
    filters = ["!statuscode:[45]\\d{2}"]
    if query.get("filter_page_id"):
        pid = query["filter_page_id"]
        filters.append(rf"original:.*-{pid}(?:/.*|\?.*|#.*)?$")
    if query.get("filter_city_slug"):
        # v3.1: server-side filter restricts host-prefix CDX query to
        # URLs containing the city slug. Avoids Wayback's 150k cap that
        # was truncating later years.
        slug = re.escape(query["filter_city_slug"].lower())
        filters.append(rf"original:.*{slug}.*")

    base = {
        "from": str(START_YEAR),
        "to": str(END_YEAR),
        "output": "json",
        "fl": "timestamp,original,statuscode,mimetype,digest",
        "filter": filters,
        "collapse": "digest",
    }
    _drop = {"filter_page_id", "filter_city_slug"}
    params = {**base, **{k: v for k, v in query["params"].items() if k not in _drop}}
    r = get_session().get(
        "https://web.archive.org/cdx/search/cdx",
        params=params,
        timeout=REQUEST_TIMEOUT,
    )
    if r.status_code in (429, 503):
        return [], f"failed: rate-limited (HTTP {r.status_code})"
    r.raise_for_status()
    body = r.text
    if not body.strip():
        return [], f"empty: HTTP {r.status_code} no body"
    if not body.lstrip().startswith("["):
        # v3 treats non-JSON as a failure (likely rate-limit HTML), not as empty
        return [], f"failed: non-JSON (HTTP {r.status_code})"
    try:
        data = r.json()
    except Exception as e:
        return [], f"failed: json parse {e}"
    if len(data) <= 1:
        return [], "empty: result set"

    filter_id = query.get("filter_page_id", "")
    rows = []
    for row in data[1:]:
        if len(row) < 5:
            continue
        timestamp, original, statuscode, mimetype, digest = row
        if mimetype and not mimetype.lower().startswith(
            ("text/html", "text/plain", "application/xhtml", "unk", "warc/revisit")
        ):
            continue
        if filter_id and not re.search(rf"-{filter_id}(?:[/?#]|$)", original):
            continue
        rows.append({
            "timestamp": timestamp,
            "year": timestamp[:4],
            "original": original,
            "digest": digest,
            "cdx_query": query["label"],
            "archive_url": f"https://web.archive.org/web/{timestamp}id_/{original}",
        })
    if not rows:
        return [], "empty: all filtered out"
    return rows, f"rows: {len(rows)}"


def _query_cdx_with_retry(query: dict, url_for_stat: str) -> tuple[list[dict], str]:
    """Returns (rows, final_outcome). Last outcome wins, but we report
    'failed' only when EVERY attempt failed; one empty result is recorded
    as 'empty' (legit, don't keep retrying)."""
    last_outcome = "failed: never ran"
    for i in range(CDX_RETRY_ATTEMPTS):
        try:
            rows, info = _query_cdx_once(query)
            last_outcome = info
            if rows:
                _record_cdx_stat(url_for_stat, query["label"], "rows")
                return rows, info
            if info.startswith("empty"):
                # Verified empty after a successful HTTP exchange — don't retry.
                _record_cdx_stat(url_for_stat, query["label"], "empty")
                return [], info
            # otherwise it's a 'failed' outcome — sleep & retry
        except Exception as e:
            last_outcome = f"failed: exception {e}"
        time.sleep(CDX_RETRY_BACKOFF * (i + 1))
    _record_cdx_stat(url_for_stat, query["label"], "failed")
    return [], last_outcome


# v3.1.2: cap availability calls per URL to bound runtime
_MAX_AVAILABILITY_CALLS_PER_URL = 12

# Cache for availability API results (per (url, year))
_avail_cache_lock = threading.Lock()
_avail_cache: dict[tuple[str, int], Optional[dict]] = {}


def _query_availability_api(url: str, year: int) -> Optional[dict]:
    """Ask Wayback's availability API for the nearest snapshot of `url`
    near YYYY-06-01. Returns a snapshot dict in the same shape as CDX rows,
    or None if no snapshot is available.

    URL-encoding is critical here — the `:` and `/` in the embedded URL
    must be percent-encoded or the server returns empty body. (This was
    why the earlier standalone diagnostic looked like it was failing.)
    """
    cache_key = (url, year)
    with _avail_cache_lock:
        if cache_key in _avail_cache:
            return _avail_cache[cache_key]
    target_ts = f"{year}0601000000"
    try:
        r = get_session().get(
            "https://archive.org/wayback/available",
            params={"url": url, "timestamp": target_ts},
            timeout=REQUEST_TIMEOUT,
        )
        if r.status_code != 200 or not r.text.strip():
            result = None
        else:
            try:
                data = r.json()
            except Exception:
                result = None
            else:
                closest = data.get("archived_snapshots", {}).get("closest")
                if not closest or not closest.get("available"):
                    result = None
                else:
                    snap_ts = closest.get("timestamp", "")
                    snap_year = snap_ts[:4]
                    # Only accept if the returned snapshot is actually in the year
                    # we asked about — otherwise we'd be inserting a wrong-year
                    # snapshot into the bucket.
                    if snap_year != str(year):
                        result = None
                    else:
                        original = closest.get("url") or url
                        # Strip Wayback's web/* prefix if present
                        original_match = re.search(r"https?://[^?]+$", original)
                        original = original_match.group(0) if original_match else original
                        result = {
                            "timestamp": snap_ts,
                            "year": snap_year,
                            "original": original,
                            "digest": f"avail_{snap_ts}",  # synthesised; no digest from availability API
                            "cdx_query": "availability_api",
                            "archive_url": f"https://web.archive.org/web/{snap_ts}id_/{original}",
                        }
    except Exception:
        result = None
    with _avail_cache_lock:
        _avail_cache[cache_key] = result
    return result


def _fill_year_gaps_via_availability(url: str, by_year: dict[str, list[dict]]) -> None:
    """For each year in 2006-2026 where by_year has 0 snapshots, ask the
    availability API. Capped at _MAX_AVAILABILITY_CALLS_PER_URL per URL so
    a URL that's broadly missing doesn't burn forever on this fallback.
    """
    missing_years = [y for y in range(START_YEAR, END_YEAR + 1)
                     if str(y) not in by_year]
    if not missing_years:
        return
    # Prefer middle years (more likely to have snapshots) over extremes
    missing_years.sort(key=lambda y: abs(y - 2018))
    calls = 0
    for year in missing_years:
        if calls >= _MAX_AVAILABILITY_CALLS_PER_URL:
            break
        calls += 1
        snap = _query_availability_api(url, year)
        if snap is not None:
            by_year.setdefault(str(year), []).append(snap)


def discover_snapshots_all_years(url: str,
                                 city_slugs: Optional[list[str]] = None
                                 ) -> tuple[dict[str, list[dict]], str]:
    """One CDX call per query covering all years; bucket by year client-side.
    v3.1: city_slugs drives the host_prefix-with-city-filter queries on
    HUGE_ARCHIVE_HOSTS. Empty results are still not cached.
    Returns (by_year, status) where status ∈ {'rows', 'empty', 'failed'}.
    """
    # Cache key includes the city slugs so different plants on the same host
    # don't share each other's city-filtered cache entry.
    slugs_key = "|".join(sorted(city_slugs or []))
    key = _cdx_cache_key(f"{url}::{slugs_key}")
    with _cdx_cache_lock:
        cached = _cdx_cache.get(key)
    if cached is not None:
        by_year = cached["by_year"]
        # v3.1.3: ALSO run the availability fallback on cached entries.
        # If the original cache build missed certain years (Wayback CDX
        # has gaps, or an earlier scraper version had buggy queries), this
        # fills them in. The cached dict is mutated in place so the cache
        # gets better over time. Availability API calls are themselves
        # memoized so cost stays bounded.
        _fill_year_gaps_via_availability(url, by_year)
        return by_year, "cached"

    all_rows: list[dict] = []
    all_outcomes: list[str] = []
    queries = build_cdx_queries(url, city_slugs=city_slugs)
    for q in queries:
        try:
            rows, outcome = _query_cdx_with_retry(q, url)
        except Exception as e:
            outcome = f"failed: exception {e}"
            rows = []
        all_outcomes.append(outcome)
        all_rows.extend(rows)
        if q["label"] == "exact" and len(all_rows) >= 50:
            break
        if len(all_rows) >= 200:
            break

    seen, unique = set(), []
    for r in all_rows:
        k = (r["timestamp"], r["original"])
        if k in seen:
            continue
        seen.add(k)
        unique.append(r)

    by_year: dict[str, list[dict]] = {}
    for snap in unique:
        by_year.setdefault(snap["year"], []).append(snap)
    for y in by_year:
        by_year[y].sort(key=lambda x: x["timestamp"], reverse=True)

    # v3.1.2: availability-API fallback for year gaps.
    # If CDX returned at least *some* snapshots for this URL, but specific
    # years are missing, ask the Wayback availability API "is there a
    # snapshot of this URL near YYYY-06-01?" That fills gaps that CDX has
    # but doesn't index well (BMW Dingolfing 2017/2023 case).
    if unique:
        _fill_year_gaps_via_availability(url, by_year)

    # v3: cache only non-empty results
    if unique:
        with _cdx_cache_lock:
            _cdx_cache[key] = {"by_year": by_year, "_t": time.time()}
        status = "rows"
    elif all(o.startswith("failed") for o in all_outcomes):
        status = "failed"
    else:
        status = "empty"
    return by_year, status


# v3: looser snapshot validation (city slug OR plant-page anchor words)
_PLANT_PAGE_ANCHORS = ("werk", "standort", "daten", "fakten", "plant",
                       "produktion", "production", "produktionsnetzwerk",
                       "production-network", "facts-and-figures")


def validate_snapshot_for_url(snap: dict, registry_url: str, city: str, plant_id: str) -> bool:
    original = snap.get("original", "").lower()
    reg_parsed = urlparse(registry_url)
    reg_path = reg_parsed.path.lower().rstrip("/")

    label = snap.get("cdx_query", "")
    if label in ("exact", "alt_www", "alt_lang", "availability_api") or label.startswith("prefix_"):
        return True

    # For host_prefix queries, require either the city slug OR a plant-page anchor
    city_slugs = city_variants(city)
    if reg_path and reg_path != "/" and reg_path in original:
        return True
    for cs in city_slugs:
        if cs and cs.lower() in original:
            return True
    for anchor in _PLANT_PAGE_ANCHORS:
        if anchor in original:
            return True
    return False


# Penalise URL paths that obviously aren't the plant-info page even when they
# match the city slug. The Audi /aola/ archive in 2015 had ~80 forum/museum/
# tour pages and 1 production_plants page; without this penalty the forum
# pages would outrank the production page by timestamp alone.
_NON_PLANT_PATH_WORDS = (
    "forum", "museum", "experience", "discovery", "tour", "tours",
    "event", "kundencenter", "customer_center", "customer-center",
    "rundgang", "besucher",
)


def _snapshot_relevance_score(snap: dict, registry_url: str) -> int:
    """v3.1.1: lower score = higher priority. Used to decide which snapshots
    to try first within a year's bucket when many candidates exist.

    Order of preference:
      0  cdx_query was "exact" — most authoritative
      1  cdx_query was "alt_www" or "alt_lang" — same page, different host/lang
      2  cdx_query was "prefix_de" / "prefix_en" — page-id filtered
      3  original URL contains a plant-info anchor (production, werk, daten…)
         AND no non-plant word (forum, museum, tour, event…)
      4  original URL contains a plant-info anchor and ALSO contains a
         non-plant word (e.g. production_plants AND audi_forum)
      5  city-slug-only match, no plant-info anchor present
      9  shouldn't normally occur (would have been rejected by validate)
    """
    label = snap.get("cdx_query", "")
    if label == "exact":
        return 0
    if label in ("alt_www", "alt_lang"):
        return 1
    if label.startswith("prefix_"):
        return 2
    if label == "availability_api":
        # Availability-API snapshots are exact-URL matches by definition
        # (the API returns the closest snapshot of THIS url), so rank them
        # alongside exact-CDX matches.
        return 0

    original = snap.get("original", "").lower()
    has_anchor = any(a in original for a in _PLANT_PAGE_ANCHORS)
    has_non_plant = any(w in original for w in _NON_PLANT_PATH_WORDS)
    if has_anchor and not has_non_plant:
        return 3
    if has_anchor and has_non_plant:
        return 4
    return 5


# ============================================================
# 5. REGEX EXTRACTION — same as v2 (correct; needs no changes)
# ============================================================

@dataclass
class Hit:
    employees: int
    reference_year: Optional[str]
    snippet: str
    pattern: str


NUM = r"\d{1,3}(?:[.,]\d{3})+|\d{3,6}"
QUAL = r"(?:rund|ca\.?|circa|etwa|mehr\s+als|über|ueber|knapp|around|approx\.?|approximately|more\s+than|over|about)"

PATTERNS_MERCEDES_STRICT = [
    ("mercedes_anzahl_im_jahr",
     re.compile(rf"(?P<num>{NUM})\s*(?:Anzahl\s+)?Mitarbeiter(?:innen)?\s+im\s+Jahr\s+(?P<year>20\d{{2}})", re.I)),
    ("mercedes_mitarbeiter_stand",
     re.compile(rf"Mitarbeiter(?:innen)?\s+(?P<num>{NUM})\s*\(\s*Stand[^)]*?(?P<year>20\d{{2}})\s*\)", re.I)),
    ("mercedes_am_standort",
     re.compile(rf"am\s+Standort\s+(?:{QUAL}\s+)?(?P<num>{NUM})\s+Mitarbeit(?:er|ende)", re.I)),
    ("mercedes_insgesamt_reversed",
     re.compile(rf"Insgesamt\s+sind\s+(?:{QUAL}\s+)?(?P<num>{NUM})\s+Mitarbeiter(?:innen|inne)?(?:\s+und\s+Mitarbeiter\w*)?\s+am\s+Standort", re.I)),
    ("mercedes_n_mitarbeiter_verb",
     re.compile(rf"(?:{QUAL}\s+)?(?P<num>{NUM})\s+Mitarbeit\w*(?:\s+und\s+Mitarbeit\w*)?\s+(?:fertigen|produzieren|montieren|arbeiten|bauen|sind\s+am\s+Standort)", re.I)),
    ("mercedes_verb_n_mitarbeiter",
     re.compile(rf"(?:fertigen|fertigten|produzieren|produzierten|montieren|bauen|arbeiten)\s+(?:hier\s+)?(?:{QUAL}\s+)?(?P<num>{NUM})\s+Mitarbeit\w*", re.I)),
    ("mercedes_derzeit",
     re.compile(rf"(?:sind|werden)\s+(?:derzeit\s+)?(?:{QUAL}\s+)?(?P<num>{NUM})\s+Mitarbeit\w*(?:\s+und\s+Mitarbeit\w*)?\s+(?:besch[äa]ftigt|t[äa]tig)", re.I)),
]
PATTERNS_MERCEDES_LOOSE = [
    ("mercedes_mit_n_mitarbeiter",
     re.compile(rf"Mit\s+(?:{QUAL}\s+)?(?P<num>{NUM})\s+Mitarbeiter(?:innen|inne)?(?:\s+und\s+Mitarbeiter\w*)?", re.I)),
    ("mercedes_label_colon",
     re.compile(rf"Mitarbeiter(?:innen)?\s*[:–\-]\s*(?:{QUAL}\s+)?(?P<num>{NUM})", re.I)),
]

PATTERNS_VW_STRICT = [
    ("vw_beschaeftigte_stand",
     re.compile(rf"Besch[äa]ftigte\s*[:\s]\s*(?:{QUAL}\s+)?(?P<num>{NUM})[^\.]{{0,140}}?Stand\s*:?\s*(?:\d{{1,2}}[./])?(?P<year>20\d{{2}})", re.I)),
    ("vw_beschaeftigte_date_paren",
     re.compile(rf"Besch[äa]ftigte\s+(?:{QUAL}\s+)?(?P<num>{NUM})\s*\(\s*\d{{1,2}}\.\d{{1,2}}\.(?P<year>20\d{{2}})\s*\)", re.I)),
    ("vw_beschaeftigte_year_paren",
     re.compile(rf"Besch[äa]ftigte\s*[:\s]\s*(?:{QUAL}\s+)?(?P<num>{NUM})\s*\(\s*(?P<year>20\d{{2}})\s*\)", re.I)),
    ("vw_employees_year",
     re.compile(rf"Employees\s*:?\s*(?:{QUAL}\s+)?(?P<num>{NUM})\s*\(\s*(?P<year>20\d{{2}})\s*\)", re.I)),
    ("vw_employees_date_paren",
     re.compile(rf"Employees\s*:?\s*(?:{QUAL}\s+)?(?P<num>{NUM})\s*\(\s*(?:[A-Za-z]+\s+\d{{1,2}},\s+)?(?P<year>20\d{{2}})\s*\)", re.I)),
]
PATTERNS_VW_LOOSE = [
    ("vw_beschaeftigte_plain",
     re.compile(rf"Besch[äa]ftigte\s*[:\s]\s*(?:{QUAL}\s+)?(?P<num>{NUM})", re.I)),
    ("vw_employees_plain",
     re.compile(rf"Employees\s*:?\s*(?:{QUAL}\s+)?(?P<num>{NUM})", re.I)),
    ("vw_am_standort_sentence",
     re.compile(rf"Am\s+Standort[^\.]{{0,80}}?(?:{QUAL}\s+)?(?P<num>{NUM})\s+Mitarbeiter(?:innen|inne)?", re.I)),
]

PATTERNS_AUDI_STRICT = [
    ("audi_arbeiten_stand",
     re.compile(rf"arbeite[nt]\s+(?:{QUAL}\s+)?(?P<num>{NUM})\s+(?:Menschen|Mitarbeiter\w*|Besch[äa]ftigte)\s*\(\s*Stand[^)]*?(?P<year>20\d{{2}})\s*\)", re.I)),
    ("audi_mit_stand",
     re.compile(rf"mit\s+(?:{QUAL}\s+)?(?P<num>{NUM})\s+Mitarbeit(?:ern|er|enden)\s*\(\s*Stand[^)]*?(?P<year>20\d{{2}})\s*\)", re.I)),
    ("audi_generic_stand",
     re.compile(rf"(?:{QUAL}\s+)?(?P<num>{NUM})\s+(?:Menschen|Mitarbeiter\w*|Besch[äa]ftigte)\s*\(\s*Stand[^)]*?(?P<year>20\d{{2}})\s*\)", re.I)),
    ("audi_corporate_employees_as_at",
     re.compile(rf"Employees\s+(?P<num>{NUM})\s*\(\s*as\s+at[: ]+[^)]*?(?P<year>20\d{{2}})\s*\)", re.I)),
    ("audi_corporate_biggest_employer",
     re.compile(rf"(?:biggest\s+employer[^.\n]{{0,30}}?with\s+)(?P<num>{NUM})\s+employees\s*\(\s*as\s+at[: ]+[^)]*?(?P<year>20\d{{2}})\s*\)", re.I)),
]
PATTERNS_AUDI_LOOSE = [
    ("audi_arbeiten_plain",
     re.compile(rf"arbeite[nt]\s+(?:{QUAL}\s+)?(?P<num>{NUM})\s+(?:Menschen|Mitarbeiter\w*)", re.I)),
]

PATTERNS_BMW_STRICT = [
    ("bmw_werk_mitarbeiter_insgesamt",
     re.compile(rf"Mitarbeiter\s+insgesamt\s+(?:{QUAL}\s+)?(?P<num>{NUM})", re.I)),
    ("bmw_insgesamt_zaehlt",
     re.compile(rf"(?:Insgesamt\s+)?z[äa]hlt\s+(?:der\s+Standort\s+)?(?:{QUAL}\s+)?(?P<num>{NUM})\s+Mitarbeit\w*", re.I)),
    ("bmw_am_standort",
     re.compile(rf"am\s+Standort\s+(?:{QUAL}\s+)?(?P<num>{NUM})\s+Mitarbeit\w*", re.I)),
    # European-scraper addition (from BMW Munich 2025 page):
    #   "6.500 Mitarbeitende im Werk München"
    ("bmw_mitarbeitende_im_werk",
     re.compile(rf"(?P<num>{NUM})\s+Mitarbeit(?:ende|er(?:innen)?)\s+im\s+Werk", re.I)),
]
PATTERNS_BMW_LOOSE = [
    ("bmw_arbeiten",
     re.compile(rf"arbeite[nt]\s+(?:{QUAL}\s+)?(?P<num>{NUM})\s+(?:Menschen|Mitarbeit\w*|Besch[äa]ftigte)", re.I)),
    ("bmw_n_mitarbeiter_und",
     re.compile(rf"(?:{QUAL}\s+)?(?P<num>{NUM})\s+Mitarbeit\w*\s+und\s+Mitarbeit\w*", re.I)),
    ("bmw_n_employees",
     re.compile(rf"(?:~|≈|approx\.?|about|around)?\s*(?P<num>{NUM})\s+employees(?:\s+(?:from|in|at))?", re.I)),
    # European-scraper addition (from BMW Munich 2017 page):
    #   "Leidenschaft seiner rund 7.800 Mitarbeiter aus über 50 Nationen"
    # The "aus N Nationen/Ländern" anchor is plant-flavoured (specific
    # workforce diversity statement, not a group total).
    ("bmw_n_mitarbeiter_aus_nationen",
     re.compile(rf"(?:{QUAL}\s+)?(?P<num>{NUM})\s+Mitarbeit\w*\s+aus\s+(?:{QUAL}\s+)?\d+\s+(?:Nation(?:en|s)|L[äa]nder(?:n)?|countries)", re.I)),
]

PATTERNS_GENERIC_STRICT = [
    ("label_mitarbeiter_stand",
     re.compile(rf"(?:Anzahl\s+)?Mitarbeiter(?:innen)?\s*[:\-]?\s*(?:{QUAL}\s+)?(?P<num>{NUM})\s*\(\s*Stand[^)]*?(?P<year>20\d{{2}})\s*\)", re.I)),
    ("label_mitarbeiter_im_jahr",
     re.compile(rf"(?P<num>{NUM})\s+(?:Anzahl\s+)?Mitarbeiter(?:innen)?\s+im\s+Jahr\s+(?P<year>20\d{{2}})", re.I)),
    ("employees_year_paren",
     re.compile(rf"(?:{QUAL}\s+)?(?P<num>{NUM})\s+(?:employees|workforce)\s*\(\s*(?P<year>20\d{{2}})\s*\)", re.I)),
]
PATTERNS_GENERIC_LOOSE = [
    ("generic_label_colon",
     re.compile(rf"Mitarbeiter(?:innen)?(?:\s+insgesamt)?\s*[:\-]?\s*(?:{QUAL}\s+)?(?P<num>{NUM})", re.I)),
]

# Porsche newsroom pages tend to look like VW newsroom (sibling brand). Until
# we have more concrete Porsche examples, reuse the generic Stand-anchored
# patterns plus a couple Porsche-flavored ones that follow common BMW/VW prose.
PATTERNS_PORSCHE_STRICT = [
    ("porsche_mitarbeiter_stand",
     re.compile(rf"Mitarbeiter(?:innen)?\s+(?P<num>{NUM})\s*\(\s*Stand[^)]*?(?P<year>20\d{{2}})\s*\)", re.I)),
    ("porsche_beschaeftigte_year_paren",
     re.compile(rf"Besch[äa]ftigte\s*[:\s]\s*(?:{QUAL}\s+)?(?P<num>{NUM})\s*\(\s*(?P<year>20\d{{2}})\s*\)", re.I)),
    ("porsche_employees_year_paren",
     re.compile(rf"Employees\s*:?\s*(?:{QUAL}\s+)?(?P<num>{NUM})\s*\(\s*(?P<year>20\d{{2}})\s*\)", re.I)),
    ("porsche_am_standort",
     re.compile(rf"am\s+Standort[^\.]{{0,80}}?(?:{QUAL}\s+)?(?P<num>{NUM})\s+Mitarbeit\w*", re.I)),
]
PATTERNS_PORSCHE_LOOSE = [
    ("porsche_beschaeftigte_plain",
     re.compile(rf"Besch[äa]ftigte\s*[:\s]\s*(?:{QUAL}\s+)?(?P<num>{NUM})", re.I)),
    ("porsche_arbeiten",
     re.compile(rf"arbeite[nt]\s+(?:{QUAL}\s+)?(?P<num>{NUM})\s+(?:Menschen|Mitarbeit\w*|Besch[äa]ftigte)", re.I)),
    ("porsche_employees_plain",
     re.compile(rf"Employees\s*:?\s*(?:{QUAL}\s+)?(?P<num>{NUM})", re.I)),
]

PATTERNS_STRICT_BY_BRAND = {
    "mercedes": PATTERNS_MERCEDES_STRICT,
    "vw":       PATTERNS_VW_STRICT,
    "audi":     PATTERNS_AUDI_STRICT,
    "bmw":      PATTERNS_BMW_STRICT,
    "porsche":  PATTERNS_PORSCHE_STRICT,
    "unknown":  PATTERNS_GENERIC_STRICT,
}
PATTERNS_LOOSE_BY_BRAND = {
    "mercedes": PATTERNS_MERCEDES_LOOSE,
    "vw":       PATTERNS_VW_LOOSE,
    "audi":     PATTERNS_AUDI_LOOSE,
    "bmw":      PATTERNS_BMW_LOOSE,
    "porsche":  PATTERNS_PORSCHE_LOOSE,
    "unknown":  PATTERNS_GENERIC_LOOSE,
}

_BAD_NEAR = re.compile(
    r"(?:Fahrzeuge?|Jahresproduktion|Tonnen|CO2|CO₂|"
    r"Quadratmeter|m²|Grundfl[äa]che|kWh|Umsatz|Euro|Bild[-\s]?Nr|"
    r"Tagesproduktion|Bremsscheiben|Konzern\s+insgesamt|"
    r"weltweit|worldwide|globally|über\s+100\s+L[äa]nder)",
    re.I,
)
# v3.1.3: broadened from v3.1's narrow set. The BMW Eisenach 124,729 case
# slipped through because "BMW Group" alone (without "had") wasn't matched,
# and the snippet window was too narrow (90 chars). Now matches any of the
# common group/brand-wide markers, and we use a wider 200-char window for
# this proximity check while still keeping the displayed snippet at 90.
# European-scraper refinement: the BMW Munich page reads "BMW Group Werk
# München" — the words "BMW Group" appear, but as part of the official plant
# name, NOT as a group-total reference. v3.1.3 was rejecting these as group
# totals. Negative lookahead on Werk/Plant/Plants now keeps plant names safe.
_GROUP_TOTAL_NEAR = re.compile(
    r"(?:"
    r"der\s+(?:Konzern|Volkswagen\s+Konzern|Audi\s+Konzern|Daimler|Mercedes-Benz\s+Group)"
    r"|"
    # "BMW Group" NOT followed by Werk/Plant — those are plant names
    r"BMW\s+Group(?!\s+(?:Werk\w*|Plant|Plants))"
    r"|"
    # Same protection for the other brand names
    r"Audi\s+(?:Konzern|AG|Group)(?!\s+(?:Werk\w*|Plant|Plants))"
    r"|"
    r"Mercedes-Benz\s+(?:Group|AG|Konzern)(?!\s+(?:Werk\w*|Plant|Plants))"
    r"|"
    r"Volkswagen\s+(?:Konzern|Group|AG)(?!\s+(?:Werk\w*|Plant|Plants))"
    r"|"
    r"Daimler(?:\s+(?:Konzern|AG|Group))?(?!\s+(?:Werk\w*|Plant|Plants))"
    r"|"
    r"workforce\s+of"
    r"|"
    r"weltweit"                                 # German "worldwide", just by itself
    r"|"
    r"globally|worldwide"
    r"|"
    r"in\s+(?:mehr\s+als\s+|über\s+|over\s+|more\s+than\s+)?\d+\s+(?:L[äa]ndern|countries)"
    r"|"
    r"Konzern\s+(?:insgesamt|weltweit|world|global)"
    r"|"
    r"the\s+(?:BMW|Audi|Mercedes-Benz|Volkswagen)\s+Group(?!\s+(?:Werk\w*|Plant|Plants))"
    r")",
    re.I,
)


def _snippet_around(text: str, start: int, end: int, pad: int = 90) -> str:
    return clean_text(text[max(0, start - pad): min(len(text), end + pad)])


def _is_group_total_snippet(snippet: str) -> bool:
    return bool(_GROUP_TOTAL_NEAR.search(snippet))


# v3.1.3: wider-window group-total check. We look at ±200 chars around the
# regex match (instead of just the displayed snippet's ±90) so the group
# total marker doesn't get scrolled out of frame when the number sits in
# the middle of a paragraph.
def _has_group_total_in_window(text: str, start: int, end: int, pad: int = 200) -> bool:
    window = text[max(0, start - pad): min(len(text), end + pad)]
    return bool(_GROUP_TOTAL_NEAR.search(window))


# European-scraper addition: phrases that strongly anchor the match to a
# specific plant. If the displayed snippet contains one of these, we trust
# it's plant-specific even when "BMW Group" appears elsewhere in the page.
# (Otherwise the wider-window check rejects BMW Munich's "6.500
# Mitarbeitende im Werk München" because the page header also says
# "BMW Group" 100+ chars away.)
_STRONG_PLANT_ANCHOR_NEAR = re.compile(
    r"(?:"
    r"im\s+Werk\b"          # German: "im Werk [Stadt]"
    r"|"
    r"am\s+Standort\b"      # German: "am Standort"
    r"|"
    r"in\s+the\s+plant\b"   # English: "in the plant"
    r"|"
    r"at\s+the\s+(?:plant|site)\b"
    r"|"
    r"Werk\s+[A-ZÄÖÜ]"      # German: "Werk Berlin"/"Werk München" — capitalised
    r"|"
    r"Plant\s+[A-Z]"        # English: "Plant Eisenach"
    r"|"
    r"Standort\s+[A-ZÄÖÜ]"  # German: "Standort Wolfsburg"
    r")",
    re.I,
)


def _snippet_has_strong_plant_anchor(snippet: str) -> bool:
    return bool(_STRONG_PLANT_ANCHOR_NEAR.search(snippet))


# European-scraper addition: a stricter version that checks whether the
# matched regex SUBSTRING (not the surrounding snippet) already contains an
# anchor that proves it's plant-specific. If yes, we skip the group-total
# proximity checks entirely — the match's own text is sufficient evidence.
#
# Examples that this catches:
#   "arbeiten 40.118 Menschen (Stand 31.12.2022)"  ← has "Stand" anchor
#   "6.500 Mitarbeitende im Werk München"          ← has "im Werk" anchor
#   "Employees 40,027 (as at: December 31, 2014)"  ← has "as at" anchor
# Examples it does NOT catch (still get group-total check):
#   "124,729 employees"                             ← no anchor in match
#   "85.000 Menschen"                               ← no anchor in match
_MATCH_HAS_OWN_ANCHOR = re.compile(
    r"(?:"
    # German "Stand" anchor — handle "Stand 2018", "Stand: 12/2018",
    # "Stand 31.12.2022", "Stand: Dezember 2024" by just requiring Stand
    # followed by punct/spaces then at least one digit OR a month name
    # within ~30 chars.
    r"Stand[\s:\.]*(?:\d|Januar|Februar|M[äa]rz|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember|January|February|March|April|May|June|July|August|September|October|November|December)"
    r"|"
    r"im\s+Werk\b"                              # German "im Werk"
    r"|"
    r"am\s+Standort\b"                          # German "am Standort"
    r"|"
    r"as\s+(?:of|at)\s*:?\s*[A-Za-z\d]"         # English "as of"/"as at"
    r"|"
    r"in\s+the\s+plant\b"
    r"|"
    r"at\s+the\s+(?:plant|site)\b"
    r"|"
    r"im\s+Jahr\s+\d{4}"                        # German "im Jahr YYYY"
    r")",
    re.I,
)


def _match_has_own_anchor(match_text: str) -> bool:
    return bool(_MATCH_HAS_OWN_ANCHOR.search(match_text))


# v3.1.3: set of pattern names that are loose enough to need LLM-validation
# when a candidate match looks suspiciously like a group total. Computed at
# import time from the loose pattern packs.
def _build_loose_pattern_names() -> set[str]:
    names: set[str] = set()
    for patterns in (PATTERNS_LOOSE_BY_BRAND.values()):
        for name, _ in patterns:
            names.add(name)
    return names


LOOSE_PATTERN_NAMES: set[str] = _build_loose_pattern_names()
# Numeric threshold above which a loose-pattern hit triggers LLM validation
# (or, if LLM is disabled, the broadened proximity check). Picked so that
# Wolfsburg's ~70k headcount still passes regex while BMW Group's ~120k+
# is reviewed.
LOOSE_SUSPECT_THRESHOLD = 50_000


def extract_plant_specific(text: str, brand: str, snapshot_year: str,
                           plant_name: str = "", city: str = "",
                           digest: str = "") -> Optional[Hit]:
    """v3.1.3: extract_plant_specific now takes plant_name/city/digest so it
    can ask the LLM to validate suspicious loose-pattern hits (e.g. when a
    BMW plant page mentions both BMW Group's 124,729 worldwide headcount
    and the actual plant's count of 450)."""
    text = clean_text(text)
    if not text:
        return None
    ordered = (list(PATTERNS_STRICT_BY_BRAND.get(brand, PATTERNS_GENERIC_STRICT))
               + list(PATTERNS_LOOSE_BY_BRAND.get(brand, PATTERNS_GENERIC_LOOSE))
               + list(PATTERNS_GENERIC_STRICT))
    seen_names: set[str] = set()
    for name, regex in ordered:
        if name in seen_names:
            continue
        seen_names.add(name)
        for m in regex.finditer(text):
            employees = to_int(m.group("num"))
            if employees is None:
                continue
            snippet = _snippet_around(text, m.start(), m.end())
            snip_lc = snippet.lower()
            if (_BAD_NEAR.search(snippet)
                    and "mitarbeit" not in snip_lc
                    and "employee" not in snip_lc
                    and "workforce" not in snip_lc
                    and "besch" not in snip_lc):
                continue
            # European scraper: skip group-total rejection entirely when the
            # MATCH itself contains a self-anchor (Stand YYYY, im Werk, am
            # Standort, as at, as of, etc.). These anchors are plant-specific
            # by construction — the page is explicitly reporting a count
            # tied to this plant at this date. Bypasses the false-positive
            # cases like Audi 40,118 (page header has "AUDI AG") and BMW
            # Munich 6,500 (page header has "BMW Group").
            match_self_anchored = _match_has_own_anchor(m.group(0))
            if not match_self_anchored:
                # v3.1.3 — narrow-window check (existing, displayed snippet)
                if _is_group_total_snippet(snippet):
                    continue
                # European refinement: still skip the wider-window check
                # when the displayed snippet has a strong plant anchor
                # close-by (e.g. "im Werk München").
                if not _snippet_has_strong_plant_anchor(snippet):
                    # v3.1.3 — wider 200-char proximity check. Catches
                    # "BMW Group" mentions just outside the 90-char snippet.
                    if _has_group_total_in_window(text, m.start(), m.end()):
                        continue
            # v3.1.3 — for loose-pattern hits with suspiciously large numbers,
            # ask the LLM to confirm this is the PLANT count and not a group
            # total that slipped past the regex guards.
            is_loose = name in LOOSE_PATTERN_NAMES
            if is_loose and employees >= LOOSE_SUSPECT_THRESHOLD:
                if LLM_ENABLED:
                    verdict = llm_validate_plant_specific(
                        text, plant_name, city, brand, employees, snippet, digest
                    )
                    if verdict in ("group", "uncertain"):
                        continue
                # If LLM is disabled, the broadened regex above is our only
                # guard. We've already passed it, so accept the hit. (We
                # don't reject silently — that would hurt recall too much
                # for users who can't run the LLM step.)
            ref_year: Optional[str] = None
            if "year" in regex.groupindex:
                try:
                    ref_year = m.group("year")
                except IndexError:
                    ref_year = None
            if not ref_year:
                ref_year = parse_reference_year(snippet, snapshot_year) or None
            return Hit(
                employees=employees,
                reference_year=ref_year or snapshot_year,
                snippet=snippet,
                pattern=name,
            )
    return None


def extract_group_overview(text: str, city: str, brand: str, snapshot_year: str) -> Optional[Hit]:
    text = clean_text(text)
    if not text:
        return None
    variants = city_variants(city)
    alt = "|".join(re.escape(v) for v in variants if v)
    if not alt:
        return None
    pat = re.compile(rf"\b({alt})\b", re.I)
    strict_patterns = (list(PATTERNS_STRICT_BY_BRAND.get(brand, PATTERNS_GENERIC_STRICT))
                       + list(PATTERNS_GENERIC_STRICT))
    for m in pat.finditer(text):
        win_start = max(0, m.start() - 200)
        win_end = min(len(text), m.end() + 1300)
        block = text[win_start:win_end]
        if _is_group_total_snippet(block) and block.lower().count(m.group(0).lower()) < 2:
            continue
        for name, regex in strict_patterns:
            for hm in regex.finditer(block):
                employees = to_int(hm.group("num"))
                if employees is None:
                    continue
                snip = _snippet_around(block, hm.start(), hm.end())
                if _BAD_NEAR.search(snip):
                    continue
                if _is_group_total_snippet(snip):
                    continue
                ref_year: Optional[str] = None
                if "year" in regex.groupindex:
                    try:
                        ref_year = hm.group("year")
                    except IndexError:
                        ref_year = None
                if not ref_year:
                    ref_year = parse_reference_year(snip, snapshot_year) or None
                return Hit(
                    employees=employees,
                    reference_year=ref_year or snapshot_year,
                    snippet=snip,
                    pattern=f"group:{name}",
                )
    return None


_CITY_VARIANTS = {
    "wolfsburg":   ["wolfsburg"],
    "brunswick":   ["brunswick", "braunschweig"],
    "chemnitz":    ["chemnitz"],
    "dresden":     ["dresden"],
    "emden":       ["emden"],
    "kassel":      ["kassel"],
    "osnabrück":   ["osnabrück", "osnabrueck"],
    "salzgitter":  ["salzgitter"],
    "zwickau":     ["zwickau"],
    "ingolstadt":  ["ingolstadt"],
    "neckarsulm":  ["neckarsulm"],
    "sindelfingen":["sindelfingen"],
    "bremen":      ["bremen"],
    "rastatt":     ["rastatt"],
    "düsseldorf":  ["düsseldorf", "duesseldorf", "dusseldorf"],
    "berlin":      ["berlin"],
    "hamburg":     ["hamburg"],
    "kamenz":      ["kamenz", "accumotive"],
    "stuttgart":   ["stuttgart", "untertürkheim", "untertuerkheim"],
    "ludwigsfelde":["ludwigsfelde"],
    "affalterbach":["affalterbach", "mercedes-amg", "amg"],
    "kölleda":     ["kölleda", "koelleda", "kolleda", "mdc-power"],
    "dingolfing":  ["dingolfing"],
    "eisenach":    ["eisenach"],
    "landshut":    ["landshut"],
    "leipzig":     ["leipzig"],
    "munich":      ["munich", "münchen", "muenchen"],
    "regensburg":  ["regensburg"],
    "wackersdorf": ["wackersdorf"],
    "irlbach-strasskirchen": ["irlbach", "strasskirchen", "straßkirchen"],
    # ---- European scraper additions ----
    # Porsche-related cities. "stuttgart" already exists above for Mercedes;
    # Porsche Zuffenhausen is in the Stuttgart-Zuffenhausen registry entry.
    "zuffenhausen": ["zuffenhausen", "stuttgart-zuffenhausen", "stuttgart"],
    # Belgium
    "brussels":    ["brussels", "bruxelles", "brussel", "vorst", "forest"],
    # Hungary
    "győr":        ["győr", "gyor", "ungarn", "hungary"],
    "debrecen":    ["debrecen"],
    "kecskemét":   ["kecskemét", "kecskemet"],
    # Poland
    "poznań":      ["poznań", "poznan"],
    "września":    ["września", "wrzesnia"],
    # Portugal
    "palmela":     ["palmela", "autoeuropa"],
    # Slovakia
    "bratislava":  ["bratislava", "slovakia", "slovensko"],
    # Spain
    "pamplona":    ["pamplona", "navarra"],
}


def city_variants(city: str) -> list[str]:
    c = (city or "").strip().lower()
    if c in _CITY_VARIANTS:
        return _CITY_VARIANTS[c]
    repl = (c.replace("ü", "ue").replace("ö", "oe").replace("ä", "ae").replace("ß", "ss"))
    return list({c, repl}) if c else []


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for sel in [
        "#wm-ipp-base", "#wm-ipp", "#wm-ipp-print",
        "script", "style", "noscript",
        "nav", "header", "footer",
        ".cookie", ".privacy", ".search-overlay",
    ]:
        for el in soup.select(sel):
            el.decompose()
    parts: list[str] = []
    for sel in ['meta[name="description"]',
                'meta[property="og:description"]',
                'meta[name="twitter:description"]']:
        el = soup.select_one(sel)
        if el and el.get("content"):
            parts.append(el["content"])
    main = soup.select_one("main") or soup.select_one("article") or soup.body or soup
    parts.append(main.get_text(" ", strip=True))
    return clean_text(" ".join(parts))


# ============================================================
# 6. LLM FALLBACK — Haiku via Anthropic API
# ============================================================

_llm_cache_lock = threading.Lock()
_llm_cache: dict[str, dict] = {}    # digest → {'employees', 'reference_year', 'rationale'} or {'employees': None}


def _load_llm_cache() -> None:
    global _llm_cache
    if LLM_CACHE_PATH.exists():
        try:
            with LLM_CACHE_PATH.open("rb") as f:
                _llm_cache = pickle.load(f)
            print(f"[cache] loaded {len(_llm_cache)} LLM cache entries")
        except Exception as e:
            print(f"[cache] LLM cache load failed ({e})")
            _llm_cache = {}


def _save_llm_cache() -> None:
    try:
        with _llm_cache_lock:
            with LLM_CACHE_PATH.open("wb") as f:
                pickle.dump(_llm_cache, f)
    except Exception as e:
        print(f"[cache] LLM cache save failed: {e}")


_LLM_PROMPT_TMPL = """You are extracting one specific piece of structured data from a Wayback-archived web page for a German automotive plant.

Plant: {plant_name}
City: {city}
Brand: {brand}
Snapshot year: {snapshot_year}

Task: Find the SPECIFIC employee headcount for THIS plant (city: {city}) on the page below.

CRITICAL rules:
- Return ONLY the number for the plant in {city}. NOT the group total, NOT the brand worldwide total, NOT a different plant on the same page.
- The number MUST literally appear in the page text. Do not guess or compute.
- If you cannot find a plant-specific count for {city} on this page, return employees=null.
- If the page mentions the count as of a specific reference year (e.g. "Stand 2018", "as of December 2014", "im Jahr 2019"), capture that year in reference_year. If no explicit reference year is mentioned but a count is clearly stated, set reference_year to the snapshot year ({snapshot_year}) — the data is "as of" the time the snapshot was taken.
- Headcount should be 50–250,000 (filter out years, dates, monetary amounts, square-metre figures, vehicle counts).
- If multiple plant-specific counts appear (e.g. "X total, Y in production"), prefer the TOTAL or "insgesamt" / "Mitarbeiter insgesamt" / "employees" figure.

Output VALID JSON ONLY in this exact shape (no markdown, no explanation, no code fences):
{{"employees": <int or null>, "reference_year": <"YYYY" or null>, "rationale": "<one short sentence quoting the matched phrase>"}}

PAGE TEXT (truncated to {max_chars} chars):
---
{page_text}
---
"""


def _llm_extract_raw(page_text: str, plant_name: str, city: str, brand: str, snapshot_year: str) -> Optional[dict]:
    """Call Gemini once. Returns parsed JSON dict or None on failure.

    Gemini variant: uses Google's generateContent endpoint instead of
    Anthropic's /v1/messages. API key is passed as a query parameter.
    Response shape: candidates[0].content.parts[0].text.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None
    prompt = _LLM_PROMPT_TMPL.format(
        plant_name=plant_name, city=city, brand=brand,
        snapshot_year=snapshot_year, max_chars=LLM_MAX_INPUT_CHARS,
        page_text=page_text[:LLM_MAX_INPUT_CHARS],
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{LLM_MODEL}:generateContent"
    body = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}],
        }],
        "generationConfig": {
            "maxOutputTokens": 256,
            "temperature": 0,
        },
    }
    headers = {"Content-Type": "application/json"}
    last_exc: Optional[Exception] = None
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            r = get_session().post(
                url,
                params={"key": api_key},
                headers=headers, json=body, timeout=LLM_REQUEST_TIMEOUT,
            )
            if r.status_code == 429:
                time.sleep(2.0 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return None
            content = candidates[0].get("content") or {}
            parts = content.get("parts") or []
            if not parts:
                return None
            txt = (parts[0].get("text") or "").strip()
            # Strip possible code fence
            txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.I | re.M).strip()
            try:
                parsed = json.loads(txt)
            except Exception:
                # Try to find JSON in the response
                m = re.search(r"\{.*\}", txt, re.S)
                if not m:
                    return None
                parsed = json.loads(m.group(0))
            return parsed
        except Exception as e:
            last_exc = e
            time.sleep(1.5 * (attempt + 1))
    return None


def llm_extract(page_text: str, plant_name: str, city: str, brand: str,
                snapshot_year: str, digest: str) -> Optional[Hit]:
    """Cache-aware LLM extraction with strict v3.1 guards:
      1. Number must literally appear in the page text.
      2. Number must be within 150 chars of an employment keyword.
      3. Reference year MUST be present (rejects round-number hallucinations).
      4. Rationale must not contain group-total markers (Konzern, worldwide).
    """
    if not LLM_ENABLED:
        return None
    if not page_text or not page_text.strip():
        return None

    cache_key = f"{digest}|{city}" if digest else f"{hashlib.sha1(page_text.encode()).hexdigest()[:16]}|{city}"
    with _llm_cache_lock:
        cached = _llm_cache.get(cache_key)
    if cached is not None:
        if cached.get("employees") is None:
            return None
        if not _number_appears_in_text(cached["employees"], page_text):
            return None
        return Hit(
            employees=int(cached["employees"]),
            reference_year=cached.get("reference_year"),
            snippet=cached.get("rationale", "")[:300],
            pattern="llm",
        )

    parsed = _llm_extract_raw(page_text, plant_name, city, brand, snapshot_year)
    result_to_cache = {"employees": None, "reference_year": None, "rationale": ""}
    if parsed and parsed.get("employees") is not None:
        try:
            emp = int(parsed["employees"])
        except Exception:
            emp = None
        ry = parsed.get("reference_year")
        if ry is not None and not re.match(r"^20\d{2}$", str(ry)):
            ry = None
        rationale = str(parsed.get("rationale", ""))[:300]

        # v3.1 acceptance rules — fail-fast, all must pass
        rejected = False
        if emp is None or emp < 50 or emp > 250_000:
            rejected = True
        elif not _number_appears_in_text(emp, page_text):
            rejected = True  # number isn't in the page → hallucination
        elif _rationale_looks_like_group_total(rationale):
            rejected = True  # quoted phrase smells like group total
        elif ry is None:
            # European/Gemini fix: previously we rejected hits with null
            # reference_year. That over-rejected valid Gemini extractions
            # — Gemini follows the prompt literally and returns null when
            # the page doesn't explicitly tie the count to a specific
            # year. (Probe confirmed: Gemini correctly extracts the number
            # then honestly says ref_year=null per prompt instructions.)
            # The original purpose of this guard was rejecting hallucinated
            # round numbers like "BMW Berlin: 2,000" with no anchor — those
            # are already filtered by _number_appears_in_text and
            # _number_near_employment_keyword. So when ref_year is null,
            # fall back to the snapshot year as a reasonable proxy.
            ry = snapshot_year
            # Re-check the proximity guard AFTER setting the fallback —
            # the original elif chain skipped this when ry was None.
            if not _number_near_employment_keyword(emp, page_text):
                rejected = True
        elif not _number_near_employment_keyword(emp, page_text):
            rejected = True  # v3.1 proximity: must be near Mitarbeiter/employees/etc.

        if not rejected:
            result_to_cache = {
                "employees": emp,
                "reference_year": str(ry),
                "rationale": rationale,
            }

    with _llm_cache_lock:
        _llm_cache[cache_key] = result_to_cache

    if result_to_cache["employees"] is None:
        return None
    return Hit(
        employees=int(result_to_cache["employees"]),
        reference_year=result_to_cache["reference_year"],
        snippet=result_to_cache["rationale"],
        pattern="llm",
    )


# ============================================================
# 6b. LLM VALIDATION — sanity-check suspicious regex hits
# ============================================================

_LLM_VALIDATE_PROMPT_TMPL = """You are checking whether an extracted employee count from a webpage refers to a SPECIFIC plant or to the brand's GROUP / worldwide total.

Plant: {plant_name}
City: {city}
Brand: {brand}
Extracted number: {number}
Where it appeared (quoted snippet): "{snippet}"

Examples to ground your judgment:
- "BMW Group employs 124,000 people worldwide" → group
- "Im BMW Werk Eisenach arbeiten 450 Menschen" → plant
- "Audi Konzern beschäftigt 85,000 Menschen" → group
- "Audi Ingolstadt employs 40,000 (Stand 2022)" → plant
- "more than 124,729 employees, the BMW Group is the world's leading…" → group

Reply with EXACTLY one of these words: plant, group, or uncertain.

Page text (truncated):
---
{page_text}
---
"""


# Cache for validation verdicts (per digest + number + city)
_llm_validate_cache_lock = threading.Lock()
_llm_validate_cache: dict[str, str] = {}


def _parse_validation_verdict(text: str) -> str:
    """Returns 'plant', 'group', or 'uncertain'."""
    if not text:
        return "uncertain"
    first = text.strip().split()[0].lower().rstrip(".,;:!?")
    if first in ("plant", "group", "uncertain"):
        return first
    if "group" in text.lower() or "konzern" in text.lower() or "weltweit" in text.lower():
        return "group"
    if "plant" in text.lower() or "werk" in text.lower() or "standort" in text.lower():
        return "plant"
    return "uncertain"


def llm_validate_plant_specific(page_text: str, plant_name: str, city: str, brand: str,
                                number: int, snippet: str, digest: str = "") -> str:
    """Returns 'plant' / 'group' / 'uncertain'. Cached by (digest, number, city)
    so the same page+number is validated only once even if many plants share
    a page or many years share a snapshot.
    """
    if not LLM_ENABLED:
        return "uncertain"
    key = f"{digest or hashlib.sha1(page_text.encode()).hexdigest()[:16]}|{number}|{city.lower()}"
    with _llm_validate_cache_lock:
        cached = _llm_validate_cache.get(key)
    if cached is not None:
        return cached

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return "uncertain"
    prompt = _LLM_VALIDATE_PROMPT_TMPL.format(
        plant_name=plant_name or "(unknown plant)",
        city=city or "(unknown city)",
        brand=brand,
        number=number,
        snippet=snippet[:300],
        page_text=page_text[:LLM_MAX_INPUT_CHARS],
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{LLM_MODEL}:generateContent"
    body = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}],
        }],
        "generationConfig": {
            "maxOutputTokens": 16,             # one word — tiny budget on purpose
            "temperature": 0,
        },
    }
    headers = {"Content-Type": "application/json"}
    verdict = "uncertain"
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            r = get_session().post(
                url,
                params={"key": api_key},
                headers=headers, json=body, timeout=LLM_REQUEST_TIMEOUT,
            )
            if r.status_code == 429:
                time.sleep(2.0 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            candidates = data.get("candidates", [])
            if candidates:
                content = candidates[0].get("content") or {}
                parts = content.get("parts") or []
                if parts:
                    txt = parts[0].get("text", "")
                    verdict = _parse_validation_verdict(txt)
            break
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    with _llm_validate_cache_lock:
        _llm_validate_cache[key] = verdict
    return verdict


def _load_llm_validate_cache() -> None:
    """Validation cache is persisted alongside the extraction cache."""
    global _llm_validate_cache
    path = LLM_CACHE_PATH.with_suffix(".validate.pkl")
    if path.exists():
        try:
            with path.open("rb") as f:
                _llm_validate_cache = pickle.load(f)
            print(f"[cache] loaded {len(_llm_validate_cache)} LLM-validation cache entries")
        except Exception:
            _llm_validate_cache = {}


def _save_llm_validate_cache() -> None:
    path = LLM_CACHE_PATH.with_suffix(".validate.pkl")
    try:
        with _llm_validate_cache_lock:
            with path.open("wb") as f:
                pickle.dump(_llm_validate_cache, f)
    except Exception as e:
        print(f"[cache] LLM-validation cache save failed: {e}")


_EMPLOYMENT_KEYWORDS_RE = re.compile(
    r"(?:Mitarbeit\w*|Besch[äa]ftigt\w*|employees?|workforce|staff|Stand\s*:|as\s+(?:at|of)|im\s+Jahr|Personalbestand)",
    re.I,
)


def _number_near_employment_keyword(n: int, text: str, max_distance: int = 150) -> bool:
    """v3.1 proximity guard: the number must appear within max_distance
    characters of an employment-related keyword. Rejects matches where the
    LLM grabbed a postal code, telephone number, or vehicle count that
    happens to be in the page."""
    s = str(n)
    variants = {s}
    if len(s) > 3:
        groups = []
        rem = s
        while len(rem) > 3:
            groups.append(rem[-3:])
            rem = rem[:-3]
        groups.append(rem)
        joined = list(reversed(groups))
        variants.add(".".join(joined))
        variants.add(",".join(joined))
        variants.add(" ".join(joined))
    for v in variants:
        idx = text.find(v)
        while idx != -1:
            window_start = max(0, idx - max_distance)
            window_end = min(len(text), idx + len(v) + max_distance)
            if _EMPLOYMENT_KEYWORDS_RE.search(text[window_start:window_end]):
                return True
            idx = text.find(v, idx + 1)
    return False


def _number_appears_in_text(n: int, text: str) -> bool:
    """Returns True if the integer n shows up in the text in any standard
    German/English thousands-separator form."""
    if n < 50:
        return False
    s = str(n)
    # Try N, N with . separator, N with , separator, and N with space separator
    variants = {s}
    if len(s) > 3:
        # 70000 → 70.000 / 70,000 / 70 000
        groups = []
        rem = s
        while len(rem) > 3:
            groups.append(rem[-3:])
            rem = rem[:-3]
        groups.append(rem)
        joined = list(reversed(groups))
        variants.add(".".join(joined))
        variants.add(",".join(joined))
        variants.add(" ".join(joined))
    for v in variants:
        if v in text:
            return True
    return False


_LLM_RATIONALE_GROUP_RE = re.compile(
    r"(?:konzern|gruppe|group\s+(?:had|of|total)|worldwide|weltweit|globally)",
    re.I,
)


def _rationale_looks_like_group_total(rationale: str) -> bool:
    return bool(_LLM_RATIONALE_GROUP_RE.search(rationale or ""))


# ============================================================
# 7. SCRAPE ONE SNAPSHOT
# ============================================================

@dataclass
class ScrapeResult:
    status: str
    employees: Optional[int]
    reference_year: Optional[str]
    snippet: str
    pattern: str
    error: str = ""


_text_cache_lock = threading.Lock()
_text_cache: dict[str, str] = {}


def _load_text_cache() -> None:
    global _text_cache
    if TEXT_CACHE_PATH.exists():
        try:
            with TEXT_CACHE_PATH.open("rb") as f:
                _text_cache = pickle.load(f)
            print(f"[cache] loaded {len(_text_cache)} text cache entries")
        except Exception as e:
            print(f"[cache] text cache load failed ({e})")
            _text_cache = {}


def _save_text_cache() -> None:
    try:
        with _text_cache_lock:
            with TEXT_CACHE_PATH.open("wb") as f:
                pickle.dump(_text_cache, f)
    except Exception as e:
        print(f"[cache] text cache save failed: {e}")


def fetch_and_extract(archive_url: str, brand: str, snapshot_year: str,
                      city: str, plant_name: str, registry_url: str,
                      url_kind: str, digest: str = "") -> ScrapeResult:
    text: Optional[str] = None
    if digest:
        with _text_cache_lock:
            text = _text_cache.get(digest)

    if text is None:
        try:
            r = get_session().get(archive_url, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                return ScrapeResult("http_error", None, None, "", "", error=f"HTTP {r.status_code}")
            text = html_to_text(r.text)
            if digest:
                with _text_cache_lock:
                    _text_cache[digest] = text
        except requests.exceptions.Timeout:
            return ScrapeResult("timeout", None, None, "", "", error="timeout")
        except Exception as e:
            return ScrapeResult("error", None, None, "", "", error=str(e))

    # Step 1: regex
    if url_kind == "GROUP_OVERVIEW":
        hit = extract_group_overview(text, city, brand, snapshot_year)
    else:
        # v3.1.3: pass plant_name/city/digest so loose-pattern hits with
        # suspiciously large numbers can be validated via LLM.
        hit = extract_plant_specific(
            text, brand, snapshot_year,
            plant_name=plant_name, city=city, digest=digest,
        )

    if hit:
        return ScrapeResult("found", hit.employees, hit.reference_year, hit.snippet, hit.pattern)

    # Step 2: LLM fallback (only if enabled and not a GROUP_LANDING)
    # Skip group-overview for LLM too — it's risky on multi-plant pages.
    if LLM_ENABLED and url_kind == "PLANT_SPECIFIC":
        hit = llm_extract(text, plant_name, city, brand, snapshot_year, digest)
        if hit:
            return ScrapeResult("found_llm", hit.employees, hit.reference_year, hit.snippet, hit.pattern)

    return ScrapeResult("not_found", None, None, "", "")


# ============================================================
# 8. PER (PLANT, YEAR) JOB
# ============================================================

@dataclass
class UrlAttempt:
    url: str
    role: str
    registry_role: str
    confidence: str
    kind: str
    period_start: int
    period_end: int


def build_attempts(reg: pd.DataFrame) -> dict[str, list[UrlAttempt]]:
    out: dict[str, list[UrlAttempt]] = {}
    for _, row in reg.iterrows():
        pid = str(row.get("plant_id", "")).strip()
        if not pid:
            continue
        try:
            ps = int(row["period_start"])
            pe = int(row["period_end"])
        except Exception:
            continue
        role_text = clean_text(row.get("url_role", ""))
        conf = clean_text(row.get("url_confidence", ""))
        for slot, url_col in [("primary", "url"),
                              ("alt1", "alternate_url_1"),
                              ("alt2", "alternate_url_2")]:
            u = clean_text(row.get(url_col, ""))
            if not u or u.lower() == "nan":
                continue
            kind = classify_url(u)
            if kind == "GROUP_LANDING":
                continue
            out.setdefault(pid, []).append(
                UrlAttempt(u, slot, role_text, conf, kind, ps, pe)
            )
    for pid, atts in out.items():
        seen: set[str] = set()
        unique: list[UrlAttempt] = []
        for a in atts:
            if a.url in seen:
                continue
            seen.add(a.url)
            unique.append(a)
        out[pid] = unique
    return out


def scrape_plant(plant_info: dict, attempts: list[UrlAttempt],
                 already_found: set[tuple[str, str]]) -> list[dict]:
    plant_id = plant_info["plant_id"]
    brand = plant_info["brand"]
    city = plant_info.get("city", "")
    plant_name = plant_info.get("plant_name", "")

    url_snapshots: dict[str, dict[str, list[dict]]] = {}
    url_cdx_status: dict[str, str] = {}
    slugs = city_variants(city)

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, max(1, len(attempts)))) as ex:
        futs = {ex.submit(discover_snapshots_all_years, a.url, slugs): a for a in attempts}
        for fut in as_completed(futs):
            a = futs[fut]
            try:
                by_year, status = fut.result()
            except Exception as e:
                by_year, status = {}, f"failed_exception:{e}"
                print(f"  CDX error for {a.url[:80]}: {e}")
            url_snapshots[a.url] = by_year
            url_cdx_status[a.url] = status
            total = sum(len(v) for v in by_year.values())
            print(f"  CDX [{a.kind:<14}] {a.url[:80]:80}  →  {total} snapshots ({status})")

    out_records: list[dict] = []
    for year in range(START_YEAR, END_YEAR + 1):
        if (plant_id, str(year)) in already_found:
            continue
        applicable = [a for a in attempts if a.period_start <= year <= a.period_end] \
                     + [a for a in attempts if not (a.period_start <= year <= a.period_end)]
        seen_u: set[str] = set()
        ordered: list[UrlAttempt] = []
        for a in applicable:
            if a.url in seen_u:
                continue
            seen_u.add(a.url)
            ordered.append(a)

        last_row: Optional[dict] = None
        found = False
        for att in ordered:
            snaps = url_snapshots.get(att.url, {}).get(str(year), [])
            snaps = [s for s in snaps if validate_snapshot_for_url(s, att.url, city, plant_id)]
            if not snaps:
                cdx_st = url_cdx_status.get(att.url, "")
                if cdx_st == "failed":
                    status = "cdx_failed"
                elif cdx_st == "empty":
                    status = "cdx_empty"
                else:
                    status = "no_archive"
                last_row = _row(plant_info, year, att, None,
                                ScrapeResult(status, None, None, "", ""))
                continue
            # v3.1.1: relevance sort. When the city-filter host_prefix CDX
            # query returns 80+ snapshots that contain "ingolstadt" but are
            # really forum/museum/tour pages, we want the actual plant-info
            # page tried FIRST, even if its timestamp isn't the newest.
            snaps = sorted(snaps, key=lambda s: (
                _snapshot_relevance_score(s, att.url),
                -int(s.get("timestamp", "0") or "0"),
            ))
            for snap in snaps[:MAX_SNAPSHOTS_PER_URL_PER_YEAR]:
                result = fetch_and_extract(
                    snap["archive_url"], brand, str(year), city, plant_name,
                    att.url, att.kind, digest=snap.get("digest", "")
                )
                last_row = _row(plant_info, year, att, snap, result)
                if result.status in ("found", "found_llm"):
                    found = True
                    break
            if found:
                break

        if last_row is None:
            last_row = _row(plant_info, year, None, None,
                            ScrapeResult("no_registry_url", None, None, "", ""))
        out_records.append(last_row)
        if last_row["status"] == "found":
            print(f"  {year}: FOUND     {last_row['employees']}  ref={last_row['reference_year']}  [{last_row['pattern']}]  slot={last_row.get('url_slot')}")
        elif last_row["status"] == "found_llm":
            print(f"  {year}: FOUND_LLM {last_row['employees']}  ref={last_row['reference_year']}  slot={last_row.get('url_slot')}")
        else:
            print(f"  {year}: {last_row['status']:<14}  {last_row.get('registry_url','')[:60]}")
    return out_records


def _row(plant_info: dict, year: int,
         att: Optional[UrlAttempt],
         snap: Optional[dict],
         result: ScrapeResult) -> dict:
    return {
        **plant_info,
        "archive_year": year,
        "registry_url": att.url if att else "",
        "url_kind": att.kind if att else "",
        "registry_url_role": att.registry_role if att else "",
        "url_confidence": att.confidence if att else "",
        "url_slot": att.role if att else "",
        "archive_timestamp": snap["timestamp"] if snap else "",
        "archive_url": snap["archive_url"] if snap else "",
        "cdx_query_used": snap.get("cdx_query", "") if snap else "",
        "status": result.status,
        "employees": result.employees,
        "reference_year": result.reference_year or "",
        "snippet": result.snippet,
        "pattern": result.pattern,
        "error": result.error,
    }


# ============================================================
# 9. CHECKPOINT / WIDE OUTPUTS
# ============================================================

def load_checkpoint() -> pd.DataFrame:
    if RESUME_FROM_CHECKPOINT and CHECKPOINT_PATH.exists():
        return pd.read_csv(
            CHECKPOINT_PATH,
            dtype={"archive_timestamp": str, "archive_year": str, "reference_year": str},
        )
    return pd.DataFrame()


def save_checkpoint(records: list[dict]) -> None:
    if not records:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(CHECKPOINT_PATH, index=False)


def found_keys(df: pd.DataFrame) -> set[tuple[str, str]]:
    if df.empty or not {"plant_id", "archive_year", "status"}.issubset(df.columns):
        return set()
    return {(str(r["plant_id"]), str(r["archive_year"]))
            for _, r in df.iterrows() if str(r["status"]) in ("found", "found_llm")}


def make_wide(long_df: pd.DataFrame, year_col: str, fname: Path) -> None:
    if long_df.empty:
        return
    df = long_df.copy()
    df = df[df["employees"].notna()]
    if df.empty:
        return
    df[year_col] = df[year_col].astype(str).str.strip()
    df = df[df[year_col].str.match(r"^20\d{2}$", na=False)]
    if df.empty:
        return
    df = df.sort_values(["plant_id", year_col, "archive_timestamp"], ascending=[True, True, False])
    df = df.drop_duplicates(["plant_id", year_col], keep="first")
    index_cols = [c for c in
                  ["plant_id", "country", "group", "plant_name", "city", "brand", "original_current_url"]
                  if c in df.columns]
    wide = df.pivot_table(
        index=index_cols, columns=year_col, values="employees", aggfunc="first"
    ).reset_index()
    wide.to_csv(fname, index=False)


def flag_stuck_values(long_df: pd.DataFrame, min_run: int = 3) -> int:
    """v3.1.3: Cross-year stickiness pass. Finds rows where the same
    (plant_id, employees, reference_year) repeats across `min_run`+ consecutive
    archive_years for a single plant. That pattern almost always means a
    static fact on the page (or one false-positive snapshot that propagated
    through repeated scrapes), not a real year-by-year measurement.

    Mutates `status` in place — changes 'found'→'found_stuck' and
    'found_llm'→'found_llm_stuck' for the flagged rows. Returns the count
    flagged so the caller can report it.
    """
    if long_df.empty or "status" not in long_df.columns:
        return 0
    mask_found = long_df["status"].isin(["found", "found_llm"])
    if not mask_found.any():
        return 0
    flagged_idx: list = []
    for plant_id, grp in long_df[mask_found].groupby("plant_id"):
        # Group same-plant rows by (employees, reference_year)
        for (emp, ref), sub in grp.groupby(["employees", "reference_year"]):
            if len(sub) < min_run:
                continue
            years_sorted = sorted(sub["archive_year"].astype(int).tolist())
            # Find the longest run of consecutive years
            longest = current = 1
            run_indices: list = [sub[sub["archive_year"].astype(int) == years_sorted[0]].index[0]]
            best_run: list = list(run_indices)
            for i in range(1, len(years_sorted)):
                idx_of = sub[sub["archive_year"].astype(int) == years_sorted[i]].index[0]
                if years_sorted[i] - years_sorted[i - 1] == 1:
                    current += 1
                    run_indices.append(idx_of)
                else:
                    current = 1
                    run_indices = [idx_of]
                if current > longest:
                    longest = current
                    best_run = list(run_indices)
            if longest >= min_run:
                flagged_idx.extend(best_run)
    if not flagged_idx:
        return 0
    # Rewrite status — keep numeric value and snippet so user can audit
    for idx in flagged_idx:
        st = str(long_df.at[idx, "status"])
        if st == "found":
            long_df.at[idx, "status"] = "found_stuck"
        elif st == "found_llm":
            long_df.at[idx, "status"] = "found_llm_stuck"
    return len(flagged_idx)


# ============================================================
# 10. MAIN
# ============================================================

def run() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _load_cdx_cache()
    _load_text_cache()
    _load_llm_cache()
    _load_llm_validate_cache()

    if LLM_ENABLED:
        print(f"[llm] ENABLED via GEMINI_API_KEY  model={LLM_MODEL}")
    else:
        print("[llm] DISABLED (set GEMINI_API_KEY to enable)")

    # v3.1: try the new sheet name first, fall back to the old one
    reg = None
    last_exc = None
    for sheet_name in INPUT_SHEET_CANDIDATES:
        try:
            reg = pd.read_excel(INPUT_PATH, sheet_name=sheet_name)
            print(f"[input] loaded sheet '{sheet_name}' from {INPUT_PATH}")
            break
        except ValueError as e:
            last_exc = e
    if reg is None:
        raise ValueError(
            f"None of the candidate sheets {INPUT_SHEET_CANDIDATES} were found in "
            f"{INPUT_PATH}. Last error: {last_exc}"
        )
    required = {"plant_id", "country", "group", "plant_name", "city",
                "period_start", "period_end", "url"}
    missing = required - set(reg.columns)
    if missing:
        raise ValueError(f"Registry missing columns: {missing}")
    # European scraper: accept all countries present in the registry.
    # The country filter the German variant had is removed here.
    reg = reg.copy()
    print(f"Loaded Registry: {len(reg)} rows, "
          f"{reg['plant_id'].nunique()} plants, "
          f"{reg['country'].nunique()} countries  "
          f"({sorted(reg['country'].dropna().unique().tolist())})")

    plant_meta = (
        reg.sort_values(["plant_id", "period_start"])
           .drop_duplicates("plant_id", keep="first")[
               ["plant_id", "country", "group", "plant_name", "city", "original_current_url"]
           ]
           .reset_index(drop=True)
    )
    plant_meta["brand"] = plant_meta.apply(
        lambda r: detect_brand(r["original_current_url"], r["plant_id"]), axis=1
    )

    attempts_by_plant = build_attempts(reg)

    checkpoint_df = load_checkpoint()
    records: list[dict] = checkpoint_df.to_dict("records") if not checkpoint_df.empty else []
    already_found = found_keys(checkpoint_df)
    new_since_flush = 0
    if records:
        print(f"Resuming with {len(records)} existing records ({len(already_found)} found).")

    for _, prow in plant_meta.iterrows():
        plant_id = prow["plant_id"]
        plant_info = {
            "plant_id":             plant_id,
            "country":              prow["country"],
            "group":                prow["group"],
            "plant_name":           prow["plant_name"],
            "city":                 prow["city"],
            "brand":                prow["brand"],
            "original_current_url": prow["original_current_url"],
        }
        attempts = attempts_by_plant.get(plant_id, [])
        if not attempts:
            print(f"\nPLANT {plant_id}: no usable URLs (all GROUP_LANDING)")
            continue

        print(f"\n{'=' * 80}\nPLANT: {prow['plant_name']}  [{prow['brand']}]  {plant_id}  ({len(attempts)} URLs)")
        plant_records = scrape_plant(plant_info, attempts, already_found)
        records.extend(plant_records)
        new_since_flush += len(plant_records)
        if new_since_flush >= CHECKPOINT_EVERY:
            save_checkpoint(records)
            _save_cdx_cache()
            _save_text_cache()
            _save_llm_cache()
            _save_llm_validate_cache()
            new_since_flush = 0

    save_checkpoint(records)
    _save_cdx_cache()
    _save_text_cache()
    _save_llm_cache()
    _save_llm_validate_cache()

    long_df = pd.DataFrame(records)
    # v3.1.3: post-process pass — flag rows where the same employee value
    # repeats across 3+ consecutive years (stale fact on the page,
    # propagated false positive, etc.). Renamed status so user can filter
    # them in Excel without losing the underlying value.
    stuck_count = flag_stuck_values(long_df, min_run=3)
    if stuck_count:
        print(f"\n[stuck] Flagged {stuck_count} rows as 'found_stuck' / 'found_llm_stuck' "
              f"(same value repeated across 3+ consecutive years for one plant).")
    long_df.to_csv(OUTPUT_LONG, index=False)
    make_wide(long_df, "archive_year",   OUTPUT_WIDE_SNAPSHOT)
    make_wide(long_df, "reference_year", OUTPUT_WIDE_REFERENCE)

    # === End-of-run diagnostics ===
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(long_df["status"].value_counts().to_string())

    # Per-URL CDX summary
    print("\nPer-URL CDX outcomes (rows / empty / failed):")
    with _cdx_stats_lock:
        snap = dict(_cdx_stats)
    for url, s in sorted(snap.items()):
        print(f"  rows={s.get('rows',0):>3} empty={s.get('empty',0):>3} failed={s.get('failed',0):>3}  "
              f"succeeded_via={sorted(s.get('labels_succeeded', set()))}  {url[:90]}")

    # Sanity: duplicates across plants
    df_found = long_df[long_df["status"].isin(["found", "found_llm", "found_stuck", "found_llm_stuck"])]
    dup_warn = df_found.groupby("employees")["plant_id"].nunique()
    dup_warn = dup_warn[dup_warn >= 3]
    if len(dup_warn) > 0:
        print("\n[WARN] Employee values found in 3+ plants — likely group-total false positives:")
        for emp, n in dup_warn.items():
            plants = df_found[df_found["employees"] == emp]["plant_id"].unique().tolist()
            print(f"   employees={emp:>7}: {n} plants  ({plants})")

    print(f"\n  rows: {len(long_df)}  "
          f"found(regex): {(long_df['status']=='found').sum()}  "
          f"found(llm): {(long_df['status']=='found_llm').sum()}  "
          f"stuck: {(long_df['status'].isin(['found_stuck','found_llm_stuck'])).sum()}")
    print(f"  long:                 {OUTPUT_LONG}")
    print(f"  wide (snapshot year): {OUTPUT_WIDE_SNAPSHOT}")
    print(f"  wide (reference yr):  {OUTPUT_WIDE_REFERENCE}")
    print(f"  checkpoint:           {CHECKPOINT_PATH}")
    return long_df


if __name__ == "__main__":
    t0 = time.time()
    df = run()
    print(f"\nElapsed: {time.time() - t0:.1f}s")
