"""Country comparison utilities: side-by-side tables, percentile radar data,
trend series and data-availability reporting."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import INDICATORS, SHORT_LABELS, TARGET

COMPARE_FEATURES = [TARGET] + INDICATORS


def latest_observation(panel: pd.DataFrame, country: str, max_year: int | None = None) -> pd.Series | None:
    d = panel[panel["country"] == country]
    if max_year is not None:
        d = d[d["year"] <= max_year]
    if d.empty:
        return None
    return d.sort_values("year").iloc[-1]


def comparison_table(panel: pd.DataFrame, countries: list[str], year: int) -> pd.DataFrame:
    """Values for ``year`` (exact). Missing observations stay missing and are flagged."""
    rows = []
    for c in countries:
        r = panel[(panel["country"] == c) & (panel["year"] == year)]
        rec = {"country": c, "year": year, "observed": not r.empty}
        for f in COMPARE_FEATURES:
            rec[f] = float(r[f].iloc[0]) if not r.empty and pd.notna(r[f].iloc[0]) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def percentile_profile(panel: pd.DataFrame, countries: list[str], year: int) -> pd.DataFrame:
    """Percentile of each indicator among all countries observed in ``year``.

    For perceived corruption the scale is reversed ("lower perceived corruption"
    = higher percentile) so that outward on the radar always means the
    direction associated with higher life evaluations in the data.
    """
    yr = panel[panel["year"] == year]
    rows = []
    for c in countries:
        r = yr[yr["country"] == c]
        rec = {"country": c}
        for f in COMPARE_FEATURES:
            col = yr[f].dropna()
            if r.empty or pd.isna(r[f].iloc[0]) or col.empty:
                rec[f] = np.nan
                continue
            v = r[f].iloc[0]
            pct = (col < v).mean() * 100 + (col == v).mean() * 50
            rec[f] = 100 - pct if f == "perceptions_of_corruption" else pct
        rows.append(rec)
    return pd.DataFrame(rows)


def radar_axis_labels() -> list[str]:
    return [SHORT_LABELS[f] if f != "perceptions_of_corruption" else "Low perceived corruption" for f in COMPARE_FEATURES]


def trend_series(panel: pd.DataFrame, countries: list[str], feature: str = TARGET) -> pd.DataFrame:
    return panel[panel["country"].isin(countries)][["country", "year", feature]].dropna().sort_values(["country", "year"])


def availability(panel: pd.DataFrame, rankings: pd.DataFrame, countries: list[str]) -> pd.DataFrame:
    rows = []
    for c in countries:
        p = panel[panel["country"] == c]
        r = rankings[rankings["country"] == c] if not rankings.empty else pd.DataFrame()
        all_years = set(range(int(panel["year"].min()), int(panel["year"].max()) + 1))
        missing_years = sorted(all_years - set(p["year"])) if not p.empty else []
        rows.append({
            "country": c,
            "panel_years": f"{int(p['year'].min())}–{int(p['year'].max())}" if not p.empty else "none",
            "panel_observations": int(len(p)),
            "panel_missing_years": ", ".join(map(str, missing_years)) if missing_years else "—",
            "missing_indicator_cells": int(p[INDICATORS].isna().sum().sum()) if not p.empty else 0,
            "ranking_years": f"{int(r['year'].min())}–{int(r['year'].max())}" if len(r) else "none",
        })
    return pd.DataFrame(rows)


def common_years(panel: pd.DataFrame, countries: list[str]) -> list[int]:
    sets = [set(panel.loc[panel["country"] == c, "year"]) for c in countries]
    return sorted(set.intersection(*sets)) if sets else []
