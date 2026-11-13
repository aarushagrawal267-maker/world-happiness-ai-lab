"""Country-name standardisation, ISO-3 codes and WHR regional classification.

Different WHR editions spell countries differently (e.g. ``Turkey`` / ``Turkiye`` /
``Türkiye``; ``Congo (Kinshasa)`` / ``DR Congo``; WHR 2022 adds ``*`` to flag
small samples). This module maps every known variant onto one canonical name,
attaches an ISO-3 code for maps, and the WHR regional indicator.

The reference table lives in ``data/reference/countries.csv`` and is rebuilt from
the raw data on demand (``build_country_reference``). Nothing here invents
observations — it only harmonises identifiers.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import pandas as pd

from .config import REFERENCE_DIR, REGIONS

REFERENCE_FILE = REFERENCE_DIR / "countries.csv"

# variant (lower-case, normalised) -> canonical name
COUNTRY_ALIASES: dict[str, str] = {
    "côte d'ivoire": "Ivory Coast",
    "cote d'ivoire": "Ivory Coast",
    "côte d’ivoire": "Ivory Coast",
    "ivory coast": "Ivory Coast",
    "dr congo": "Congo (Kinshasa)",
    "democratic republic of the congo": "Congo (Kinshasa)",
    "congo, dem. rep.": "Congo (Kinshasa)",
    "congo": "Congo (Brazzaville)",
    "republic of the congo": "Congo (Brazzaville)",
    "congo, rep.": "Congo (Brazzaville)",
    "czech republic": "Czechia",
    "eswatini, kingdom of": "Eswatini",
    "swaziland": "Eswatini",
    "hong kong": "Hong Kong S.A.R. of China",
    "hong kong s.a.r., china": "Hong Kong S.A.R. of China",
    "hong kong sar of china": "Hong Kong S.A.R. of China",
    "hong kong sar, china": "Hong Kong S.A.R. of China",
    "lao pdr": "Laos",
    "lao people's democratic republic": "Laos",
    "macedonia": "North Macedonia",
    "north macedonia": "North Macedonia",
    "northern cyprus": "North Cyprus",
    "north cyprus": "North Cyprus",
    "palestinian territories": "State of Palestine",
    "palestine": "State of Palestine",
    "republic of korea": "South Korea",
    "korea, rep.": "South Korea",
    "korea": "South Korea",
    "republic of moldova": "Moldova",
    "russian federation": "Russia",
    "somaliland region": "Somaliland region",
    "somaliland": "Somaliland region",
    "trinidad & tobago": "Trinidad and Tobago",
    "turkey": "Türkiye",
    "turkiye": "Türkiye",
    "türkiye": "Türkiye",
    "viet nam": "Vietnam",
    "taiwan": "Taiwan Province of China",
    "taiwan province of china": "Taiwan Province of China",
    "united states of america": "United States",
    "usa": "United States",
    "uk": "United Kingdom",
    "iran, islamic rep.": "Iran",
    "egypt, arab rep.": "Egypt",
    "syrian arab republic": "Syria",
    "kyrgyz republic": "Kyrgyzstan",
    "slovak republic": "Slovakia",
    "gambia, the": "Gambia",
    "yemen, rep.": "Yemen",
    "venezuela, rb": "Venezuela",
}

INVALID_NAMES = {"", "xx", "nan", "none", "world", "total"}

# ISO-3 codes for canonical names (used when a source carries no code).
ISO3_OVERRIDES: dict[str, str | None] = {
    "Kosovo": "XKX",          # user-assigned code used by Plotly / World Bank
    "Somaliland region": None,  # not a separately mapped territory
    "North Cyprus": None,       # not a separately mapped territory
    "Türkiye": "TUR",
    "Hong Kong S.A.R. of China": "HKG",
    "Taiwan Province of China": "TWN",
    "State of Palestine": "PSE",
    "Ivory Coast": "CIV",
    "Congo (Kinshasa)": "COD",
    "Congo (Brazzaville)": "COG",
    "Laos": "LAO",
    "Russia": "RUS",
    "South Korea": "KOR",
    "Moldova": "MDA",
    "Vietnam": "VNM",
    "Czechia": "CZE",
    "Eswatini": "SWZ",
    "North Macedonia": "MKD",
    "Trinidad and Tobago": "TTO",
    "Iran": "IRN",
    "Syria": "SYR",
    "Bolivia": "BOL",
    "Venezuela": "VEN",
    "Tanzania": "TZA",
    "United States": "USA",
    "United Kingdom": "GBR",
}

# WHR regional indicator for countries that do not appear in the editions that publish it
# (WHR 2020 and WHR 2021). Assignments follow the WHR 10-region scheme.
REGION_OVERRIDES: dict[str, str] = {
    "Angola": "Sub-Saharan Africa",
    "Belize": "Latin America and Caribbean",
    "Bhutan": "South Asia",
    "Central African Republic": "Sub-Saharan Africa",
    "Cuba": "Latin America and Caribbean",
    "Djibouti": "Sub-Saharan Africa",
    "Eswatini": "Sub-Saharan Africa",
    "Guyana": "Latin America and Caribbean",
    "Oman": "Middle East and North Africa",
    "Puerto Rico": "Latin America and Caribbean",
    "Qatar": "Middle East and North Africa",
    "Somalia": "Sub-Saharan Africa",
    "Somaliland region": "Sub-Saharan Africa",
    "South Sudan": "Sub-Saharan Africa",
    "Sudan": "Sub-Saharan Africa",
    "Suriname": "Latin America and Caribbean",
    "Syria": "Middle East and North Africa",
    "Trinidad and Tobago": "Latin America and Caribbean",
    "Czechia": "Central and Eastern Europe",
    "Congo (Brazzaville)": "Sub-Saharan Africa",
    "Congo (Kinshasa)": "Sub-Saharan Africa",
    "Ivory Coast": "Sub-Saharan Africa",
    "State of Palestine": "Middle East and North Africa",
    "Türkiye": "Middle East and North Africa",
    "North Macedonia": "Central and Eastern Europe",
    "North Cyprus": "Western Europe",
    "Hong Kong S.A.R. of China": "East Asia",
    "Taiwan Province of China": "East Asia",
    "Laos": "Southeast Asia",
    "Moldova": "Commonwealth of Independent States",
    "Russia": "Commonwealth of Independent States",
    "South Korea": "East Asia",
    "Vietnam": "Southeast Asia",
}


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.rstrip("*").strip()
    return text


def standardize_country_name(name: object) -> str | None:
    """Return the canonical country name, or ``None`` for invalid entries."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    clean = _normalise(str(name))
    if clean.lower() in INVALID_NAMES:
        return None
    key = clean.lower()
    if key in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[key]
    return clean


def standardize_country_column(series: pd.Series) -> pd.Series:
    return series.map(standardize_country_name)


# --------------------------------------------------------------------------- reference table
def build_country_reference(
    panel: pd.DataFrame | None = None,
    region_sources: list[pd.DataFrame] | None = None,
    names: list[str] | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Build the canonical country reference table.

    * ISO-3 codes come from the panel's ``iso3`` column, then ``ISO3_OVERRIDES``,
      then (optionally) ``pycountry`` if installed.
    * Regions come from WHR editions that publish a ``region`` column, then
      ``REGION_OVERRIDES``.
    """
    iso: dict[str, str | None] = {}
    if panel is not None and "iso3" in panel:
        for c, i in panel[["country", "iso3"]].dropna().drop_duplicates().itertuples(index=False):
            iso[c] = i if isinstance(i, str) and len(i) == 3 else None
    regions: dict[str, str] = {}
    for src in region_sources or []:
        if "region" in src:
            for c, r in src[["country", "region"]].dropna().drop_duplicates().itertuples(index=False):
                if r in REGIONS:
                    regions.setdefault(c, r)
    all_names = set(names or []) | set(iso) | set(regions) | set(REGION_OVERRIDES)
    try:  # optional helper, only used at build time
        import pycountry  # type: ignore
    except Exception:  # pragma: no cover
        pycountry = None
    rows = []
    for c in sorted(n for n in all_names if n):
        code = ISO3_OVERRIDES.get(c, iso.get(c))
        if c not in ISO3_OVERRIDES and code is None and pycountry is not None:
            try:
                code = pycountry.countries.lookup(c).alpha_3
            except LookupError:
                code = None
        if c == "Kosovo":
            code = "XKX"
        region = REGION_OVERRIDES.get(c, regions.get(c))
        if c in regions:
            region = regions[c]
        rows.append({"country": c, "iso3": code, "region": region})
    ref = pd.DataFrame(rows)
    if save:
        ref.to_csv(REFERENCE_FILE, index=False)
    load_country_reference.cache_clear()
    return ref


@lru_cache(maxsize=1)
def load_country_reference() -> pd.DataFrame:
    if REFERENCE_FILE.exists():
        return pd.read_csv(REFERENCE_FILE, keep_default_na=False, na_values=[""])
    return pd.DataFrame(columns=["country", "iso3", "region"])


def attach_geography(df: pd.DataFrame, overwrite_region: bool = False) -> pd.DataFrame:
    """Add/complete ``iso3`` and ``region`` columns from the reference table."""
    ref = load_country_reference().set_index("country")
    out = df.copy()
    iso_map = ref["iso3"].to_dict()
    reg_map = ref["region"].to_dict()
    if "iso3" not in out:
        out["iso3"] = out["country"].map(iso_map)
    else:
        out["iso3"] = out["iso3"].where(out["iso3"].notna(), out["country"].map(iso_map))
        out.loc[out["country"] == "Kosovo", "iso3"] = "XKX"
        out.loc[out["country"].isin(["Somaliland region", "North Cyprus"]), "iso3"] = None
    if "region" not in out or overwrite_region:
        out["region"] = out["country"].map(reg_map)
    else:
        out["region"] = out["region"].where(out["region"].isin(REGIONS), out["country"].map(reg_map))
    return out
