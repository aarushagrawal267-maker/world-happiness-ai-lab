"""Exploratory data analysis computations (pure pandas; charts live in
``visualization.py``). Every statistic is computed from the loaded data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ALL_NUMERIC, INDICATORS, TARGET


def dataset_overview(panel: pd.DataFrame, target: str = TARGET) -> dict:
    feats = [c for c in ALL_NUMERIC if c in panel]
    s = panel[target]
    imax, imin = s.idxmax(), s.idxmin()
    return {
        "observations": int(len(panel)),
        "countries": int(panel["country"].nunique()),
        "years": int(panel["year"].nunique()),
        "year_min": int(panel["year"].min()),
        "year_max": int(panel["year"].max()),
        "features": len(feats),
        "mean_score": float(s.mean()),
        "median_score": float(s.median()),
        "std_score": float(s.std()),
        "max_score": float(s.max()),
        "max_country": str(panel.at[imax, "country"]),
        "max_year": int(panel.at[imax, "year"]),
        "min_score": float(s.min()),
        "min_country": str(panel.at[imin, "country"]),
        "min_year": int(panel.at[imin, "year"]),
        "missing_cells": int(panel[feats + [target]].isna().sum().sum()),
        "missing_pct": float(panel[feats + [target]].isna().mean().mean() * 100),
        "duplicate_country_years": int(panel.duplicated(["country", "year"]).sum()),
    }


def ranking_overview(rankings: pd.DataFrame, year: int | None = None) -> dict:
    year = year or int(rankings["year"].max())
    r = rankings[rankings["year"] == year]
    top, bottom = r.loc[r["ladder_score"].idxmax()], r.loc[r["ladder_score"].idxmin()]
    return {
        "year": year,
        "edition": year + 1,
        "countries": int(r["country"].nunique()),
        "mean_score": float(r["ladder_score"].mean()),
        "max_score": float(top["ladder_score"]),
        "max_country": str(top["country"]),
        "min_score": float(bottom["ladder_score"]),
        "min_country": str(bottom["country"]),
    }


def coverage_by_year(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.groupby("year").agg(countries=("country", "nunique"), mean_score=(TARGET, "mean")).reset_index()


def best_reporting_year(panel: pd.DataFrame, min_share: float = 0.8) -> int:
    """Latest panel year whose country coverage is at least ``min_share`` of the max coverage."""
    cov = panel.groupby("year")["country"].nunique()
    ok = cov[cov >= min_share * cov.max()]
    return int(ok.index.max())


def top_bottom(df: pd.DataFrame, score: str, n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.dropna(subset=[score])
    return d.nlargest(n, score), d.nsmallest(n, score)


def region_summary(df: pd.DataFrame, score: str) -> pd.DataFrame:
    return (
        df.dropna(subset=[score, "region"])
        .groupby("region")[score]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )


def correlation_matrix(df: pd.DataFrame, columns: list[str] | None = None, method: str = "pearson") -> pd.DataFrame:
    cols = columns or [c for c in [TARGET] + ALL_NUMERIC if c in df]
    return df[cols].corr(method=method)


def country_period_change(rankings: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Change in official ladder score between two ranking years."""
    a = rankings[rankings["year"] == start].set_index("country")[["ladder_score", "rank"]]
    b = rankings[rankings["year"] == end].set_index("country")[["ladder_score", "rank", "region"]]
    j = a.merge(b, left_index=True, right_index=True, suffixes=("_start", "_end"), how="inner")
    j["change"] = j["ladder_score_end"] - j["ladder_score_start"]
    j["rank_change"] = j["rank_start"] - j["rank_end"]  # positive = moved up
    return j.reset_index().sort_values("change", ascending=False)


def yoy_changes(panel_fe: pd.DataFrame) -> pd.DataFrame:
    """Year-over-year Life Ladder changes (consecutive observed years only)."""
    col = "ladder_yoy_change"
    return panel_fe.dropna(subset=[col])[["country", "region", "year", TARGET, col]].sort_values(col)


def trend_slopes(panel: pd.DataFrame, min_years: int = 8) -> pd.DataFrame:
    """Per-country linear trend in Life Ladder (points per year)."""
    rows = []
    for c, g in panel.dropna(subset=[TARGET]).groupby("country"):
        if g["year"].nunique() >= min_years:
            slope, intercept = np.polyfit(g["year"], g[TARGET], 1)
            rows.append({"country": c, "region": g["region"].iloc[0], "slope_per_year": slope, "n_years": g["year"].nunique(),
                         "first_year": int(g["year"].min()), "last_year": int(g["year"].max())})
    return pd.DataFrame(rows).sort_values("slope_per_year", ascending=False)


def distribution_stats(s: pd.Series) -> dict:
    from scipy import stats

    s = s.dropna()
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std()),
        "skewness": float(stats.skew(s)),
        "kurtosis": float(stats.kurtosis(s)),
        "q1": float(s.quantile(0.25)),
        "q3": float(s.quantile(0.75)),
    }


def gdp_growth_vs_happiness(panel_fe: pd.DataFrame) -> pd.DataFrame:
    """Pairs of (Δ log GDP, Δ Life Ladder) for consecutive years."""
    return panel_fe.dropna(subset=["gdp_growth_1y", "ladder_yoy_change"])[
        ["country", "region", "year", "gdp_growth_1y", "ladder_yoy_change"]
    ]


def indicator_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in INDICATORS if c in df]
