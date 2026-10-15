# -*- coding: utf-8 -*-
"""
================================================================================
 EEA COMMERCIAL NAME  ->  THESIS MODEL   CROSSWALK
 Preparation step for Module 03b (vehicle mass)
================================================================================

WHY THIS FILE EXISTS
--------------------
An exact join between our model panel and the EEA aggregate matches 48% of
plant-model-year rows. That is not because the data are missing; only 3.2% of
our rows are things that could never have a passenger-car mass (axles, plastic
components, Citaro buses, Transit vans). The ceiling is about 97%. The gap is
entirely a NAMING problem, and it runs in both directions.

    EEA `Cn` is free text typed by manufacturers. One model appears as
    dozens of strings carrying trim, engine, drivetrain and power:
        A 180 · A180 · A 180 CDI · A 180 D 4MATIC · A180CDI
        116 I · 116I · 116D XDRIVE · 116D EFFICIENT DYNAMICS
        "ID.3 PRO 150 KW" · "ID.3 GTX PERF. 240 KW"
    plus internal type codes (8P, 4G, 5N TIGUAN, 7HC), placeholders
    ("---", "1", "9") and outright typos (A1 SPORTBAK, A1 QUATRO, AUSTRIAL).

    OUR side carries generation codes the EEA never uses:
        GOLF MK2 · A3 8V · TT FV · Q3 F3 · A 180 W177 · MONDEO IVB

So the join cannot be an equality. It has to normalise both sides to a model
STEM and then match a thesis stem against every EEA string that begins with it,
aggregating those variants with registration weights. That is also the right
statistic: "the mass of the Golf" should be the weighted mean over all Golf
versions sold, which is exactly what stem matching produces.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
No fuzzy string distance. Edit-distance matching on names this short silently
pairs Renault 5 with BMW 5, Audi A3 with Audi A4, and 206 with 306. Every rule
here is explicit and auditable, and `match_report()` prints what each one buys
so a reviewer can see the cost of each decision rather than trusting a score.

Author: Magd Shamashan
"""

from __future__ import annotations

import re
import unicodedata
import warnings
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

import foundation as fd

__all__ = [
    "BRAND_TO_EEA_MAKES", "NON_CAR_PATTERNS",
    "normalise", "thesis_stem", "eea_stem", "is_non_car",
    "build_crosswalk", "match_report", "attach_mass", "MANUAL_ALIASES",
]


# ==============================================================================
# 1. BRAND FAMILIES
# ==============================================================================
# Our `brand` column is not always a marque. Several plant records carry the
# GROUP there instead, which is why "VOLKSWAGEN" owns Cayenne, Bentayga,
# Octavia, Citigo and Lamborghini Urus, why "AUDI" owns Golf and Polo, and why
# "STELLANTIS" — which is not a marque at all and appears nowhere in EEA — owns
# 206, 3008, 500 and 159. Constraining the join to an exact make would throw all
# of those away. Constraining it to nothing would let Renault 5 match BMW 5.
# So each of our brands maps to the set of EEA makes it may legitimately match.

BRAND_TO_EEA_MAKES: Dict[str, Set[str]] = {
    "STELLANTIS": {"PEUGEOT", "CITROEN", "CITROËN", "DS", "OPEL", "VAUXHALL",
                   "FIAT", "ALFA ROMEO", "LANCIA", "JEEP", "ABARTH",
                   "CHRYSLER", "MASERATI", "AUTOBIANCHI"},
    "VOLKSWAGEN": {"VOLKSWAGEN", "VW", "SEAT", "SKODA", "ŠKODA", "CUPRA",
                   "AUDI", "PORSCHE", "BENTLEY", "LAMBORGHINI",
                   "FORD"},   # AutoEuropa built the Galaxy alongside the Sharan
    "AUDI": {"AUDI", "VOLKSWAGEN", "VW"},
    "BMW": {"BMW", "MINI", "ROLLS-ROYCE"},
    "MERCEDES-BENZ": {"MERCEDES-BENZ", "MERCEDES", "SMART", "MAYBACH"},
    "RENAULT": {"RENAULT", "DACIA", "ALPINE", "NISSAN"},
    "FORD": {"FORD"},
    # Kolin (TPCA) built the Aygo / 107 / C1 triplets, so Toyota plant records
    # legitimately carry Peugeot and Citroen nameplates.
    "TOYOTA": {"TOYOTA", "LEXUS", "DAIHATSU", "PEUGEOT", "CITROEN", "CITROËN"},
    "PORSCHE": {"PORSCHE"},
}


# Hand-verified aliases for the few cases no rule can reach, kept explicit so a
# reviewer can check them one by one rather than trust a similarity score.
#   e-Golf   EEA registers it as plain GOLF with Ft = ELECTRIC, never "E-GOLF"
#   GT/GT R  Mercedes files the AMG GT under "AMG GT ..."
#   W222     an S-class chassis code our panel uses as if it were a model name
MANUAL_ALIASES: Dict[Tuple[str, str], str] = {
    ("VOLKSWAGEN", "E-GOLF"): "GOLF",
    ("MERCEDES-BENZ", "GT"): "AMG GT",
    ("MERCEDES-BENZ", "GT R"): "AMG GT R",
    ("MERCEDES-BENZ", "GT S"): "AMG GT S",
    ("MERCEDES-BENZ", "W222"): "S-CLASS",
    ("MERCEDES-BENZ", "W223"): "S-CLASS",
    ("AUDI", "Q4 E-TRON QUATTRO"): "Q4 E-TRON",
    ("AUDI", "SQ8 E-TRON QUATTRO"): "SQ8 E-TRON",
}


# Battery-electric nameplates whose EEA name is shared with combustion versions.
# The match finds the right nameplate but averages every powertrain under it:
# the e-Golf came out at the mass of all Golfs (0.9% electric), the Kangoo Z.E.
# at that of all Kangoos. For these the matched rows are restricted to the
# registrations of the given class in eea_model_mass_by_fuel.csv (7 Oct 2026).
FUEL_RESTRICTED: Dict[Tuple[str, str], str] = {
    ("VOLKSWAGEN", "E-GOLF"): "BEV",
    ("RENAULT", "KANGOO Z.E."): "BEV",
}
_FUEL_N: Dict[Tuple[str, str], str] = {}
_BY_FUEL: Dict[str, pd.DataFrame] = {}


def _restrict_fuel(hit: pd.DataFrame, cls: str) -> pd.DataFrame:
    if "f" not in _BY_FUEL:
        _BY_FUEL["f"] = pd.read_csv(fd.CONFIG.path("eea_model_mass_by_fuel.csv"))
    f = _BY_FUEL["f"]
    f = f[f["fuel_class"] == cls].drop(columns=["fuel_class"])
    keys = hit[["eea_make", "eea_model", "year"]].drop_duplicates()
    out = f.merge(keys, on=["eea_make", "eea_model", "year"], how="inner")
    out["bev_share"] = 1.0 if cls == "BEV" else 0.0
    return out


# Keys are normalised on load, because `normalise` deletes the hyphens and
# dots the literal keys are written with ("MERCEDES-BENZ" -> "MERCEDES BENZ",
# "e-Golf" -> "E GOLF"). Written the obvious way, none of them ever fired.
_ALIAS_N: Dict[Tuple[str, str], str] = {}


# Rows that cannot have a passenger-car mass. Excluded from the DENOMINATOR
# rather than counted as failures — reporting them as unmatched would overstate
# how much of the panel the crosswalk is missing.
NON_CAR_PATTERNS = (
    r"(AXLE|COMPONENT|ENGINE AND|CAST |GEARBOX|TRANSMISSION|STAMPING|FOUNDR"
    r"|STOSSF|STOFF|BUMPER|PRESSWERK|ELECTRIQUE|ELECTRIQ|^CARS?$)",
    r"\b(CITARO|CONECTO|TOURISMO|INTOURO|O\d{3})\b",
    r"\b(TRANSIT|DYNA|HIACE|MASTER|MOVANO|CRAFTER|INTERSTAR|PRIMASTAR"
    r"|NV\d00|EXPRESS|KUBISTAR|CITAN|E-TGE|ECRAFTER|V\.U"
    r"|VARIO|BOXER|JUMPER|DUCATO|SPRINTER|VITO|TRAFIC|TALENTO|SCUDO"
    r"|\bTGE\b|GRAND CALIFORNIA)\b",
)


# ==============================================================================
# 2. NORMALISATION
# ==============================================================================

# Generation / platform codes that appear on OUR side but never on EEA's.
_GEN_CODES = re.compile(
    r"\b("
    r"MK\s?\d+"                     # GOLF MK2
    r"|[WVZCS]\d{3}"                # A 180 W177, A 180 BERLINA V177, C216
    r"|\d[A-Z]{1,2}\b"              # A3 8V, Q3 F3, 4G, 8P
    r"|[A-Z]{1,2}\d(?=\s)"          # B7 A4, C5 RS6
    r"|I{2,3}V?B?|IVB|VIII|VII|VI|IV|III|II"   # MONDEO IVB, CLIO III
    r")\b")

# Chassis / platform codes, TRAILING ONLY. Position is the whole point: in
# "X5 E70" and "A 180 W177" the code follows a real name and is noise, but
# "E 63", "C 320" and "W222" ARE the name. Matching them anywhere reduced
# those three to an empty stem, which then matched EEA's "---" placeholder and
# assigned them a 2,649 kg mass. Caught by auditing the stem rule.
_TRAILING_GEN = re.compile(r"(?<=\S)\s+([EFG]\s?\d{2}|[WVZC]\s?\d{3})$")

# Marketing suffixes that only ever appear on our side.
_THESIS_TRIM = re.compile(
    r"\b(SPECIAL\s?EDITION|DTM\s?EDITION|CHAMPION\s?EDITION|BLACK\s?SERIES"
    r"|\d+TH\s?ANNIVERSARY|ANNIVERSARY|EDITION|SPECIAL)\b")

# Trim, engine, drivetrain and body noise on the EEA side.
#
# NOT stripped, deliberately: GT, GTS, RS, MAX, PLUS, LONG. These read like
# trim but are load-bearing parts of real names -- Mercedes AMG GT, Toyota
# GT86, Audi RS 6, Renault Clio RS, Ford S-MAX. Stripping "GT" sent
# "AMG GT 4-Door" to an AMG G 63, which an audit of backoff matches caught.
_EEA_NOISE = re.compile(
    r"\b("
    r"CDI|TDI|TSI|TFSI|FSI|HDI|DCI|BLUETEC|BLUEHDI|CRDI|D4D|VTI|THP"
    r"|4MATIC|QUATTRO|QUATRO|XDRIVE|X\s?DRIVE|4MOTION|ALLRAD|AWD|4WD|4X4"
    r"|EFFICIENT\s?DYNAMICS|EDRIVE|E-?TECH|BLUEMOTION|ECOBOOST|TCE|SCE"
    r"|HYBRID|HYBRIDE|MHEV|PHEV|PLUG-?IN|ELECTRIC|E-?POWER"
    r"|CABRIO(LET)?|COUPE|COUPÉ|ROADSTER|SPYDER|SW|AVANT|SPORTBACK|SPORTSBACK"
    r"|SPORTBAK|VARIANT|TOURER|ESTATE|BERLINA|SEDAN|SALOON|LIMOUSINE|KOMBI"
    r"|GRAN\s?TURISMO|GRAN\s?COUPE|ALLSTREET|CITY\s?CARVER|CITTCARVER|CITYCARVER"
    r"|PERFORMANCE|PERF\.?|COMPETITION|S\s?LINE|GTI|GTE|GTD|GTX|R-?LINE"
    r"|LWB|SWB|KR|LR|PRO|PURE|ADVANCE|TREND|STYLE|LIFE"
    r"|\d+\s?KW|\d+\s?PS|\d+\s?CH|\d+\s?AH|\d[.,]\d\s?[A-Z]*"
    r")\b")

_PUNCT = re.compile(r"[\"'`´(),;:/\\+*_]")


def normalise(s: str) -> str:
    """Upper-case, strip accents and punctuation, collapse whitespace."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.upper()
    s = _PUNCT.sub(" ", s)
    s = s.replace("-", " ").replace(".", " ")
    # Split letter/digit boundaries. Manufacturers type the same car as CL63 and
    # CL 63, A180 and A 180, SLK32 and SLK 32; without this the two never meet,
    # and Mercedes' AMG range alone accounted for ~400 unmatched rows.
    s = re.sub(r"(?<=[A-Z])(?=\d)", " ", s)
    s = re.sub(r"(?<=\d)(?=[A-Z])", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _init_aliases() -> None:
    _ALIAS_N.clear()
    for (b, m), v in MANUAL_ALIASES.items():
        _ALIAS_N[(normalise(b), normalise(m))] = v
    _FUEL_N.clear()
    for (b, m), v in FUEL_RESTRICTED.items():
        _FUEL_N[(normalise(b), normalise(m))] = v


def is_non_car(name: str) -> bool:
    """True for components, buses and vans — no passenger-car mass exists."""
    u = str(name).upper()
    return any(re.search(p, u) for p in NON_CAR_PATTERNS)


def _mercedes_class(s: str) -> Optional[str]:
    """'A CLASS' / 'CLASS A' -> 'A'. Mercedes' EEA names are 'A 180', not 'A-CLASS'."""
    m = re.fullmatch(r"([A-Z]{1,3})\s?CLASS", s) or re.fullmatch(r"CLASS\s([A-Z]{1,3})", s)
    return m.group(1) if m else None


def _bmw_series(s: str) -> Optional[str]:
    """'3 SERIES' / '3ER' / '3ER SERIE' -> '3'. EEA writes 318D, 320I."""
    m = (re.fullmatch(r"(\d)\s?(SERIES|SERIE|ER)", s)
         or re.fullmatch(r"(\d)\s?ER\s?SERIE", s))
    return m.group(1) if m else None


def thesis_stem(name: str) -> str:
    """Our model name reduced to the part EEA would recognise."""
    s = normalise(name)
    # Leading platform code: "B7 A4" is the B7-generation A4, "C5 RS6" the C5
    # RS6. Left in place, backoff strips the REAL name off the end and matches
    # on the platform letter, which is how "B7 A4" ended up pointing at an
    # Audi B8 and "C5 RS6" at a C4. Only stripped when something follows it.
    # Only B, C and D lead an Audi platform code (B7 A4, C5 RS6, D3 A8). The
    # unrestricted version ate the "A 3" out of "A3 8V" and left the stem as
    # "8 V", which matched nothing at all.
    s = re.sub(r"^([BCD]\s\d)\s+(?=\S)", "", s)
    full = s
    s = _TRAILING_GEN.sub("", s)
    s = _GEN_CODES.sub(" ", s)
    s = _THESIS_TRIM.sub(" ", s)
    s = _EEA_NOISE.sub(" ", s)       # QUATTRO, PERFORMANCE, S LINE ... on our side too
    s = re.sub(r"\s+", " ", s).strip()
    # Never let stripping consume the whole name: an empty or one-character
    # stem matches indiscriminately. Fall back to the unstripped name.
    return s if len(s.replace(" ", "")) >= 2 else full


def eea_stem(name: str) -> str:
    """An EEA commercial name stripped of trim, engine and body noise."""
    s = normalise(name)
    s = _EEA_NOISE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


# ==============================================================================
# 3. THE MATCHER
# ==============================================================================

def _tokens(s: str) -> Tuple[str, ...]:
    return tuple(t for t in s.split(" ") if t)


def _starts_with(hay: Tuple[str, ...], needle: Tuple[str, ...]) -> bool:
    """Token-boundary prefix test — so 'A3' never matches 'A30'."""
    return len(needle) > 0 and hay[:len(needle)] == needle


def build_crosswalk(model_panel: pd.DataFrame,
                    eea: Optional[pd.DataFrame] = None,
                    min_regs: int = 1,
                    by_year: bool = False) -> pd.DataFrame:
    """
    One row per (brand, model) in our panel, with the registration-weighted
    mass of every EEA commercial name that resolves to it.

    Matching is tried in a fixed order and the FIRST rule that fires wins, so
    every match carries the name of the rule that produced it and can be
    audited or overridden individually.

    by_year : collapse the matched EEA rows PER YEAR instead of over the whole
              period. The default (False) gives one mass per model for all
              years, which is fine for mass -- a Golf is about a Golf's weight
              throughout -- but wrong for CO2, which fell by a quarter over the
              window and is simply absent before 2018. Anything time-varying
              should use by_year=True.

        exact     normalised names identical
        stem      normalised stems identical
        prefix    EEA name begins with our stem  (GOLF <- GOLF VARIANT ...)
        class     Mercedes A-CLASS  <- A 180, A 200, A 250 ...
        series    BMW 3 SERIES      <- 318D, 320I, 330E ...
    """
    if eea is None:
        eea = pd.read_csv(fd.CONFIG.path("eea_model_mass.csv"))

    e = eea.copy()
    e["make_n"] = e["eea_make"].map(normalise)
    e["name_n"] = e["eea_model"].map(normalise)
    e["stem_n"] = e["eea_model"].map(eea_stem)
    e = e[e["registrations"] >= min_regs]
    # EEA placeholders: "---", "-", "1", "9", and names that noise-stripping
    # empties out ("AVANT", "COUPE", "HYBRID" with no model attached). They
    # carry 21 registrations in total and cannot identify anything.
    bad = (e["name_n"].str.replace(" ", "", regex=False).str.len() < 2) | \
          (e["name_n"].str.fullmatch(r"[-\s\d]{1,3}", na=False))
    if bad.any():
        e = e[~bad]
    e.loc[e["stem_n"].str.replace(" ", "", regex=False).str.len() < 2,
          "stem_n"] = e.loc[e["stem_n"].str.replace(" ", "", regex=False)
                            .str.len() < 2, "name_n"]
    e["_tok_name"] = e["name_n"].map(_tokens)
    e["_tok_stem"] = e["stem_n"].map(_tokens)

    pairs = (model_panel[["brand", "model"]].dropna().astype(str)
             .drop_duplicates().reset_index(drop=True))
    pairs["brand_n"] = pairs["brand"].map(normalise)
    pairs["stem"] = pairs["model"].map(thesis_stem)
    pairs["non_car"] = pairs["model"].map(is_non_car)

    # index EEA by make for speed
    by_make: Dict[str, pd.DataFrame] = {m: g for m, g in e.groupby("make_n")}

    rows = []
    for r in pairs.itertuples():
        if r.non_car:
            if by_year:
                continue
            rows.append(dict(brand=r.brand, model=r.model, rule="non-car", tokens_dropped=0,
                             n_eea=0, registrations=np.nan, mass_kg=np.nan,
                             bev_share=np.nan, co2_g_km=np.nan,
                             co2_coverage=np.nan, examples=""))
            continue

        makes = BRAND_TO_EEA_MAKES.get(r.brand_n, {r.brand_n})
        cand = [by_make[m] for m in makes if m in by_make]
        if not cand:
            if by_year:
                continue
            rows.append(dict(brand=r.brand, model=r.model, rule="no-make", tokens_dropped=0,
                             n_eea=0, registrations=np.nan, mass_kg=np.nan,
                             bev_share=np.nan, co2_g_km=np.nan,
                             co2_coverage=np.nan, examples=""))
            continue
        c = pd.concat(cand, ignore_index=True)

        name_n = normalise(r.model)
        alias = _ALIAS_N.get((r.brand_n, name_n))
        if alias:
            name_n = normalise(alias)
        stem = thesis_stem(alias) if alias else r.stem
        tok = _tokens(stem)
        hit, rule = None, None

        m_exact = c["name_n"] == name_n
        if m_exact.any():
            hit, rule = c[m_exact], "alias" if alias else "exact"

        if hit is None:
            m_stem = c["stem_n"] == stem
            if m_stem.any():
                hit, rule = c[m_stem], "stem"

        if hit is None and tok:
            m_pre = c["_tok_name"].map(lambda t: _starts_with(t, tok))
            m_pre |= c["_tok_stem"].map(lambda t: _starts_with(t, tok))
            if m_pre.any():
                hit, rule = c[m_pre], "prefix"

        if hit is None:
            # Class-level fallback: "S 55 AMG", "CLK 63 AMG", "C 36 AMG" are
            # 1990s AMG derivatives with no entry of their own in EEA. Their
            # class does have one, so fall back to it and say so.
            mm = re.match(r"^([A-Z]{1,4})\s\d{2,3}\b", stem)
            if mm:
                pat = rf"^{re.escape(mm.group(1))}\s?\d{{2,3}}\b"
                m_cf = c["name_n"].str.match(pat, na=False)
                if m_cf.any():
                    hit, rule = c[m_cf], "class"

        if hit is None:
            cls = _mercedes_class(stem)
            if cls:
                pat = rf"^{re.escape(cls)}\s?\d{{2,3}}\b"
                m_cls = c["name_n"].str.match(pat, na=False)
                if m_cls.any():
                    hit, rule = c[m_cls], "class"

        # BACKOFF -- drop trailing tokens until something matches.
        # This generalises what a pile of regexes was doing badly: "A 3 8 V"
        # -> "A 3", "TT FV" -> "TT", "SL 63 AMG" -> "SL 63" -> "SL". It is
        # tried LAST among the name rules and records how many tokens it had to
        # drop, because a two-token backoff is a much weaker claim than an exact
        # hit and the thesis should be able to tell them apart.
        n_dropped = 0
        if hit is None and len(tok) > 1:
            for k in range(len(tok) - 1, 0, -1):
                short = tok[:k]
                # never back off to a bare letter ("B", "C", "A"): it matches
                # every model of that make and is not evidence of anything
                if len(short) == 1 and len(short[0]) < 2:
                    break
                m_b = c["_tok_name"].map(lambda t: _starts_with(t, short))
                m_b |= c["_tok_stem"].map(lambda t: _starts_with(t, short))
                if m_b.any():
                    hit, rule = c[m_b], "backoff"
                    n_dropped = len(tok) - k
                    break

        if hit is None:
            ser = _bmw_series(stem)
            if ser:
                pat = rf"^{ser}\d{{2}}\b|^{ser}\s?(SERIES|SERIE|ER)\b"
                m_ser = c["name_n"].str.match(pat, na=False)
                if m_ser.any():
                    hit, rule = c[m_ser], "series"

        fuel_cls = _FUEL_N.get((r.brand_n, normalise(r.model)))
        if fuel_cls and hit is not None and len(hit):
            hit = _restrict_fuel(hit, fuel_cls)   # rule name kept, so match counts do not move

        if hit is None or not len(hit):
            if by_year:
                continue
            rows.append(dict(brand=r.brand, model=r.model, rule="unmatched", tokens_dropped=0,
                             n_eea=0, registrations=np.nan, mass_kg=np.nan,
                             bev_share=np.nan, co2_g_km=np.nan,
                             co2_coverage=np.nan, examples=""))
            continue

        if by_year:
            for yr, hy in hit.groupby("year"):
                wy = hy["registrations"].fillna(0)
                ty = float(wy.sum())
                hc = hy["co2_mean_g_km"].notna()
                wc = wy.where(hc, 0.0); tc = float(wc.sum())
                rows.append(dict(
                    brand=r.brand, model=r.model, year=int(yr), rule=rule,
                    n_eea=int(len(hy)), registrations=ty,
                    mass_kg=float((hy["mass_mean_kg"] * wy).sum() / ty) if ty else np.nan,
                    bev_share=(float((hy["bev_share"] * wy).sum() / ty)
                               if ty and "bev_share" in hy else np.nan),
                    co2_g_km=(float((hy["co2_mean_g_km"].fillna(0) * wc).sum() / tc)
                              if tc else np.nan),
                    co2_coverage=tc / ty if ty else np.nan))
            continue

        w = hit["registrations"].fillna(0)
        tot = float(w.sum())
        # CO2 keeps a separate denominator: Ewltp is absent before 2018, so a
        # model matched across 2010-2025 has a mass for every year but a CO2
        # figure only for the later ones. Weighting it by TOTAL registrations
        # would drag the mean towards zero in proportion to how much of the
        # model's history predates the measurement.
        has_c = hit["co2_mean_g_km"].notna() if "co2_mean_g_km" in hit else None
        if has_c is not None and has_c.any():
            wc = w.where(has_c, 0.0)
            totc = float(wc.sum())
            co2 = float((hit["co2_mean_g_km"].fillna(0) * wc).sum() / totc) if totc else np.nan
            co2_cov = totc / tot if tot else np.nan
        else:
            co2, co2_cov = np.nan, np.nan
        rows.append(dict(
            brand=r.brand, model=r.model, rule=rule,
            tokens_dropped=n_dropped, n_eea=int(len(hit)),
            registrations=tot,
            mass_kg=float((hit["mass_mean_kg"] * w).sum() / tot) if tot else np.nan,
            bev_share=(float((hit["bev_share"] * w).sum() / tot)
                       if tot and "bev_share" in hit else np.nan),
            co2_g_km=co2, co2_coverage=co2_cov,
            examples=" | ".join(hit.sort_values("registrations", ascending=False)
                                ["eea_model"].astype(str).head(3))))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # A match is only as good as the rule that made it. exact/stem/prefix
    # identify the model itself; class/series/backoff identify a FAMILY and
    # return its registration-weighted mean, which is systematically flatter
    # than the true variant (a Maybach Pullman priced into the S-class mean).
    # Module 03b should be able to exclude the coarse ones as a robustness check.
    # Coarse matches are treated as VALID (agreed with the author). The flag is
    # kept so a robustness check can drop them, not because they are excluded.
    PRECISE = {"exact", "alias", "stem", "prefix"}
    COARSE = {"class", "series", "backoff"}
    out["match_quality"] = np.where(out["rule"].isin(PRECISE), "precise",
                            np.where(out["rule"].isin(COARSE), "coarse", "none"))
    return out


def match_report(model_panel: pd.DataFrame, cw: pd.DataFrame) -> pd.DataFrame:
    """
    Coverage by rule, on rows (not on distinct names) and excluding the
    non-car records from the denominator, since they cannot be matched.
    """
    mp = model_panel.copy()
    key = cw.set_index(["brand", "model"])["rule"].to_dict()
    mp["rule"] = [key.get((str(b), str(m)), "unmatched")
                  for b, m in zip(mp["brand"], mp["model"])]
    car = mp[mp["rule"] != "non-car"]
    g = (mp.groupby("rule").size().rename("rows").to_frame()
         .assign(pct_of_car_rows=lambda d: 100 * d["rows"] / len(car))
         .sort_values("rows", ascending=False))
    matched = car[~car["rule"].isin(["unmatched", "no-make"])]
    precise = car[car["rule"].isin(["exact", "alias", "stem", "prefix"])]
    g.attrs["coverage_pct"] = 100 * len(matched) / len(car) if len(car) else 0.0
    g.attrs["precise_pct"] = 100 * len(precise) / len(car) if len(car) else 0.0
    g.attrs["n_precise"] = len(precise)
    g.attrs["n_car_rows"] = len(car)
    g.attrs["n_matched"] = len(matched)
    return g


def attach_mass(model_panel: pd.DataFrame,
                cw: Optional[pd.DataFrame] = None,
                by_year: bool = False) -> pd.DataFrame:
    """
    Left-join the crosswalk's mass onto the model panel.

    `by_year=True` joins on (brand, model, YEAR), so a model's mass and CO2 are
    the values for that year rather than a period average. Required for CO2.
    """
    if cw is None:
        cw = build_crosswalk(model_panel, by_year=by_year)
    if by_year or "year" in cw.columns:
        keep = cw[["brand", "model", "year", "mass_kg", "bev_share", "co2_g_km",
                   "co2_coverage", "registrations", "rule", "n_eea"]].rename(
            columns={"registrations": "eea_regs", "rule": "match_rule"})
        out = model_panel.merge(keep, on=["brand", "model", "year"], how="left")
        out["mass_source"] = np.where(out["mass_kg"].notna(), "EEA", "unmatched")
        return out
    keep = cw[["brand", "model", "mass_kg", "bev_share", "co2_g_km",
               "co2_coverage", "registrations", "rule", "n_eea"]].rename(
        columns={"registrations": "eea_regs", "rule": "match_rule"})
    out = model_panel.merge(keep, on=["brand", "model"], how="left")
    out["mass_source"] = np.where(out["mass_kg"].notna(), "EEA", "unmatched")
    return out


_init_aliases()
