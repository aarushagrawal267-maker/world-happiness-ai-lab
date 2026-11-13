"""Data cleaning: column-name harmonisation, type conversion, invalid values,
duplicate handling and year parsing.

WHR editions use very different headers for the same concept (``Life Ladder``,
``Ladder score``, ``Happiness Score``, ``Score``, ``Life evaluation (3-year
average)``) and even reuse the same header for different quantities: in the
Kaggle 2018/2019 files ``GDP per capita`` is the *explained-by contribution*,
not the raw indicator. ``standardize_columns`` maps headers to a canonical
schema and ``reclassify_contribution_columns`` uses value ranges to detect the
contribution case.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import VALID_RANGES
from .geography import standardize_country_column

# --------------------------------------------------------------------------- column aliases
CONTRIB_COLUMNS = [
    "exp_gdp",
    "exp_social_support",
    "exp_health",
    "exp_freedom",
    "exp_generosity",
    "exp_corruption",
]
RAW_TO_CONTRIB = {
    "log_gdp_per_capita": "exp_gdp",
    "social_support": "exp_social_support",
    "healthy_life_expectancy": "exp_health",
    "freedom": "exp_freedom",
    "generosity": "exp_generosity",
    "perceptions_of_corruption": "exp_corruption",
}


def _key(col: str) -> str:
    col = str(col).replace("﻿", "").lower()
    col = re.sub(r"[^a-z0-9]+", " ", col)
    return re.sub(r"\s+", " ", col).strip()


COLUMN_ALIASES: dict[str, str] = {
    # identifiers
    "country name": "country",
    "country": "country",
    "country or region": "country",
    "iso alpha": "iso3",
    "iso3": "iso3",
    "iso code": "iso3",
    "code": "iso3",
    "regional indicator": "region",
    "region": "region",
    "year": "year",
    "rank": "rank",
    "overall rank": "rank",
    "happiness rank": "rank",
    # target
    "life ladder": "life_ladder",
    "ladder score": "ladder_score",
    "happiness score": "ladder_score",
    "score": "ladder_score",
    "life evaluation 3 year average": "ladder_score",
    # uncertainty
    "standard error of ladder score": "ladder_se",
    "standard error": "ladder_se",
    "upperwhisker": "upper_whisker",
    "upper whisker": "upper_whisker",
    "whisker high": "upper_whisker",
    "upper confidence interval": "upper_whisker",
    "lowerwhisker": "lower_whisker",
    "lower whisker": "lower_whisker",
    "whisker low": "lower_whisker",
    "lower confidence interval": "lower_whisker",
    # raw indicators
    "logged gdp per capita": "log_gdp_per_capita",
    "log gdp per capita": "log_gdp_per_capita",
    "gdp per capita": "log_gdp_per_capita",
    "social support": "social_support",
    "healthy life expectancy": "healthy_life_expectancy",
    "healthy life expectancy at birth": "healthy_life_expectancy",
    "freedom to make life choices": "freedom",
    "generosity": "generosity",
    "perceptions of corruption": "perceptions_of_corruption",
    "positive affect": "positive_affect",
    "negative affect": "negative_affect",
    # explained-by contributions (always contribution scale)
    "explained by log gdp per capita": "exp_gdp",
    "explained by gdp per capita": "exp_gdp",
    "economy gdp per capita": "exp_gdp",
    "explained by social support": "exp_social_support",
    "family": "exp_social_support",
    "explained by healthy life expectancy": "exp_health",
    "health life expectancy": "exp_health",
    "explained by freedom to make life choices": "exp_freedom",
    "freedom": "freedom",  # ambiguous in Kaggle 2015-17: reclassified below
    "explained by generosity": "exp_generosity",
    "explained by perceptions of corruption": "exp_corruption",
    "trust government corruption": "exp_corruption",
    "dystopia residual": "dystopia_residual",
    "dystopia residual 2": "dystopia_residual",
    "ladder score in dystopia": "dystopia_benchmark",
}


def map_column(col: str) -> str | None:
    k = _key(col)
    if k in COLUMN_ALIASES:
        return COLUMN_ALIASES[k]
    if k.startswith("dystopia") and "residual" in k:
        return "dystopia_residual"
    return None


def standardize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename columns to canonical snake_case names. Unknown columns are kept
    (normalised to snake_case) so no information is silently dropped."""
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for col in df.columns:
        canon = map_column(col)
        if canon is None:
            canon = _key(col).replace(" ", "_") or "unnamed"
        if canon in used:  # avoid duplicate column names
            canon = f"{canon}__dup"
        used.add(canon)
        mapping[col] = canon
    return df.rename(columns=mapping), mapping


def reclassify_contribution_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect files where indicator headers actually hold explained-by
    contributions (Kaggle 2015–2019 style) and rename them accordingly.

    Raw log GDP lies in ~[6, 12] and raw healthy life expectancy in ~[30, 80];
    contributions are all below ~2.5. If either check fails the raw-range test,
    every raw indicator column in that file is treated as a contribution.
    """
    out = df.copy()
    looks_contrib = False
    for col, lo in (("log_gdp_per_capita", 4.0), ("healthy_life_expectancy", 10.0)):
        if col in out:
            vals = pd.to_numeric(out[col], errors="coerce")
            if vals.notna().any() and vals.max() < lo:
                looks_contrib = True
    if looks_contrib or (any(c in out for c in CONTRIB_COLUMNS) and "log_gdp_per_capita" not in out):
        rename = {raw: c for raw, c in RAW_TO_CONTRIB.items() if raw in out and c not in out}
        out = out.rename(columns=rename)
    return out


# --------------------------------------------------------------------------- values
def to_numeric_columns(df: pd.DataFrame, exclude: tuple[str, ...] = ("country", "iso3", "region", "source")) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col in exclude:
            continue
        if out[col].dtype == object or str(out[col].dtype).startswith(("string", "str")):
            cleaned = out[col].astype(str).str.replace(",", ".", regex=False).str.strip()
            cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "..": np.nan, "-": np.nan, "None": np.nan})
            converted = pd.to_numeric(cleaned, errors="coerce")
            # only convert if the column is predominantly numeric
            if converted.notna().sum() >= 0.8 * out[col].notna().sum():
                out[col] = converted
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


@dataclass
class CleaningLog:
    invalid_values: dict[str, int] = field(default_factory=dict)
    dropped_invalid_country: int = 0
    exact_duplicates_removed: int = 0
    conflicting_duplicates: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "invalid_values_set_to_nan": self.invalid_values,
            "rows_dropped_invalid_country": self.dropped_invalid_country,
            "exact_duplicates_removed": self.exact_duplicates_removed,
            "conflicting_duplicate_country_years": self.conflicting_duplicates,
            "notes": self.notes,
        }


def invalidate_out_of_range(df: pd.DataFrame, log: CleaningLog | None = None) -> pd.DataFrame:
    """Values outside the definitional ranges are impossible (e.g. a share > 1)
    and are set to NaN — never clipped or replaced with an invented value."""
    out = df.copy()
    for col, (lo, hi) in VALID_RANGES.items():
        if col in out and col != "year":
            vals = pd.to_numeric(out[col], errors="coerce")
            bad = vals.notna() & ((vals < lo) | (vals > hi))
            if bad.any():
                if log is not None:
                    log.invalid_values[col] = int(bad.sum())
                    for idx in out.index[bad][:20]:
                        who = out.at[idx, "country"] if "country" in out else idx
                        when = f" {int(out.at[idx, 'year'])}" if "year" in out and pd.notna(out.at[idx, "year"]) else ""
                        log.notes.append(
                            f"{col} = {vals[idx]:.3f} for {who}{when} is outside the plausible range [{lo}, {hi}] → set to missing."
                        )
                out.loc[bad, col] = np.nan
    return out


def parse_year(value: object) -> float:
    """Parse year formats seen in WHR material: 2019, '2019', 2019.0,
    '2017-2019' (takes the final survey year), 'WHR2020' (edition, kept as is)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, np.integer)):
        return float(value)
    if isinstance(value, (float, np.floating)):
        return float(int(value))
    if hasattr(value, "year"):
        return float(value.year)
    s = str(value).strip()
    years = re.findall(r"(19\d{2}|20\d{2})", s)
    if years:
        return float(years[-1])
    return np.nan


def parse_edition_from_filename(name: str) -> int | None:
    """``WHR20_Data_Figure_2.1.xls`` -> 2020; ``2015.csv`` -> 2015; ``WHR2023.csv`` -> 2023."""
    m = re.search(r"WHR[_ ]?(20\d{2})", name, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"WHR[_ ]?(\d{2})(?!\d)", name, re.I)
    if m:
        return 2000 + int(m.group(1))
    m = re.search(r"(20\d{2})", name)
    if m:
        return int(m.group(1))
    return None


def clean_frame(df: pd.DataFrame, log: CleaningLog | None = None) -> pd.DataFrame:
    """Standard cleaning applied to every source after column standardisation."""
    log = log or CleaningLog()
    out = df.copy()
    out = out.loc[:, ~out.columns.str.startswith("unnamed")]
    if "country" in out:
        before = len(out)
        out["country"] = standardize_country_column(out["country"])
        out = out[out["country"].notna()]
        log.dropped_invalid_country += before - len(out)
    if "year" in out:
        out["year"] = out["year"].map(parse_year)
    out = to_numeric_columns(out)
    if "year" in out:
        out = out[out["year"].notna()]
        out["year"] = out["year"].astype(int)
    out = invalidate_out_of_range(out, log)
    return out.reset_index(drop=True)


def deduplicate_country_year(df: pd.DataFrame, keys: tuple[str, ...] = ("country", "year"), log: CleaningLog | None = None) -> pd.DataFrame:
    """Remove *genuine* duplicates (identical rows). Conflicting duplicates
    (same country-year, different values) are resolved by keeping the row with
    the most non-missing values, and are counted so they can be reported."""
    log = log or CleaningLog()
    keys = tuple(k for k in keys if k in df)
    if not keys:
        return df
    before = len(df)
    out = df.drop_duplicates()
    log.exact_duplicates_removed += before - len(out)
    dup_mask = out.duplicated(list(keys), keep=False)
    if dup_mask.any():
        log.conflicting_duplicates += int(out[dup_mask].groupby(list(keys)).ngroups)
        out = (
            out.assign(_nn=out.notna().sum(axis=1))
            .sort_values("_nn", ascending=False)
            .drop_duplicates(list(keys), keep="first")
            .drop(columns="_nn")
        )
        log.notes.append(f"Resolved {log.conflicting_duplicates} conflicting duplicate {keys} groups by completeness.")
    return out.sort_values(list(keys)).reset_index(drop=True)
