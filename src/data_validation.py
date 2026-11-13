"""Dataset validation: schema detection, required columns, value ranges,
duplicate country-year records and missing values.

Validation never mutates data; it returns a structured ``ValidationReport``
that the app displays and the tests assert against.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from .config import ALL_NUMERIC, INDICATORS, VALID_RANGES
from .data_cleaning import CONTRIB_COLUMNS

DATASET_KINDS = {
    "annual_panel": "Annual country-year panel (Life Ladder + indicators, e.g. WHR DataForTable2.1)",
    "ranking_panel": "Multi-year ranking table (3-year averages + explained-by contributions)",
    "edition_raw": "Single-edition ranking table with raw indicator values",
    "edition_contrib": "Single-edition ranking table with explained-by contributions only",
    "unknown": "Unrecognised structure",
}


def detect_kind(df: pd.DataFrame) -> str:
    """Classify a *standardised* frame into one of ``DATASET_KINDS``."""
    cols = set(df.columns)
    if "country" not in cols:
        return "unknown"
    n_raw = len(cols & set(INDICATORS))
    has_year = "year" in cols and df["year"].nunique() > 1
    if has_year and ("life_ladder" in cols or "ladder_score" in cols) and n_raw >= 3:
        return "annual_panel"
    if has_year and "ladder_score" in cols:
        return "ranking_panel"
    if "ladder_score" in cols and n_raw >= 3:
        return "edition_raw"
    if "ladder_score" in cols and len(cols & set(CONTRIB_COLUMNS)) >= 3:
        return "edition_contrib"
    if "ladder_score" in cols:
        return "edition_contrib"
    return "unknown"


@dataclass
class ValidationIssue:
    level: str  # "error" | "warning" | "info"
    check: str
    message: str


@dataclass
class ValidationReport:
    name: str
    kind: str
    n_rows: int
    n_columns: int
    n_countries: int
    years: list[int]
    target_column: str | None
    missing_counts: dict[str, int] = field(default_factory=dict)
    missing_pct: dict[str, float] = field(default_factory=dict)
    duplicate_country_year: int = 0
    out_of_range: dict[str, int] = field(default_factory=dict)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(i.level == "error" for i in self.issues)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["is_valid"] = self.is_valid
        return d


def validate_dataset(df: pd.DataFrame, name: str = "dataset", kind: str | None = None) -> ValidationReport:
    kind = kind or detect_kind(df)
    target = "life_ladder" if "life_ladder" in df else ("ladder_score" if "ladder_score" in df else None)
    years = sorted(int(y) for y in df["year"].dropna().unique()) if "year" in df else []
    rep = ValidationReport(
        name=name,
        kind=kind,
        n_rows=len(df),
        n_columns=df.shape[1],
        n_countries=int(df["country"].nunique()) if "country" in df else 0,
        years=years,
        target_column=target,
    )
    if "country" not in df:
        rep.issues.append(ValidationIssue("error", "required_columns", "No country column detected."))
    if target is None:
        rep.issues.append(ValidationIssue("error", "required_columns", "No happiness score column (Life Ladder / Ladder score) detected."))
    if kind == "unknown":
        rep.issues.append(ValidationIssue("error", "schema", "Dataset structure not recognised."))

    # missing values
    for col in df.columns:
        n = int(df[col].isna().sum())
        if n:
            rep.missing_counts[col] = n
            rep.missing_pct[col] = round(100 * n / max(len(df), 1), 2)
    if target and df[target].isna().any():
        rep.issues.append(
            ValidationIssue("warning", "target_missing", f"{int(df[target].isna().sum())} rows have no happiness score; they are excluded from modelling (targets are never imputed).")
        )
    for col in ALL_NUMERIC:
        if col in rep.missing_pct and rep.missing_pct[col] > 20:
            rep.issues.append(ValidationIssue("warning", "missingness", f"{col} is missing in {rep.missing_pct[col]}% of rows."))

    # duplicates
    keys = [k for k in ("country", "year") if k in df]
    if keys:
        dup = int(df.duplicated(keys, keep=False).sum())
        rep.duplicate_country_year = dup
        if dup:
            rep.issues.append(ValidationIssue("error", "duplicates", f"{dup} rows share a {'/'.join(keys)} key."))

    # ranges
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df:
            vals = pd.to_numeric(df[col], errors="coerce")
            bad = int((vals.notna() & ((vals < lo) | (vals > hi))).sum())
            if bad:
                rep.out_of_range[col] = bad
                rep.issues.append(ValidationIssue("error", "range", f"{bad} values of {col} fall outside [{lo}, {hi}]."))

    if not rep.issues:
        rep.issues.append(ValidationIssue("info", "summary", "All checks passed."))
    return rep


def check_no_leakage(feature_columns: list[str]) -> list[str]:
    """Return any feature columns that are the target or target-derived."""
    from .config import LEAKAGE_COLUMNS

    return [c for c in feature_columns if c in LEAKAGE_COLUMNS or "ladder" in c or "dystopia" in c]
