"""Feature engineering.

Only features with a clear analytical meaning are created, and each one is
documented in ``FEATURE_DOCS`` (formula, whether it may be used as a model
input, and its leakage status).

Rules
-----
* Time-based features look **backwards only** (previous observed year), so a
  model never sees information from the future.
* Change features are computed only when the previous calendar year is
  actually observed for that country (no gap-bridging).
* Target-derived features (year-over-year ladder change, happiness category,
  ladder-vs-GDP gap) are descriptive only and are listed in
  ``config.LEAKAGE_COLUMNS`` so they can never enter a model.
* Quantile-based categories are computed from a *reference* frame (pass the
  training data) to avoid using test-set information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import INDICATORS, TARGET

HAPPINESS_BINS = [0, 4.5, 5.5, 6.5, 10.01]
HAPPINESS_LABELS = ["Low (<4.5)", "Lower-middle (4.5–5.5)", "Upper-middle (5.5–6.5)", "High (≥6.5)"]
QUARTILE_LABELS = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]

FEATURE_DOCS: dict[str, dict[str, str]] = {
    "year": {
        "formula": "Survey year of the observation.",
        "model_input": "yes (engineered set)",
        "rationale": "Captures common global shocks and slow trends shared across countries.",
    },
    "region": {
        "formula": "WHR regional indicator (10 regions), one-hot encoded inside the pipeline.",
        "model_input": "yes (engineered set)",
        "rationale": "Proxy for shared cultural / historical context and response styles.",
    },
    "gdp_growth_1y": {
        "formula": "log GDP pc(t) − log GDP pc(t−1), only when year t−1 is observed for the same country.",
        "model_input": "yes (engineered set) — backward-looking only",
        "rationale": "Approximates annual real growth (≈ % change / 100); tests whether growth adds information beyond level.",
    },
    "composite_index": {
        "formula": "Mean of z-scores of the six core indicators, perceived corruption sign-flipped; means/SDs fitted on training data (CompositeIndexTransformer).",
        "model_input": "optional (pipeline transformer)",
        "rationale": "Single interpretable socio-economic summary; justified because indicators are strongly inter-correlated.",
    },
    "gdp_category": {
        "formula": "Quartile of log GDP pc computed on the reference (training) frame.",
        "model_input": "no — descriptive",
        "rationale": "Readable grouping for EDA and group comparisons.",
    },
    "social_support_category": {"formula": "Quartile of social support (reference frame).", "model_input": "no — descriptive", "rationale": "EDA grouping."},
    "freedom_category": {"formula": "Quartile of freedom (reference frame).", "model_input": "no — descriptive", "rationale": "EDA grouping."},
    "health_category": {"formula": "Quartile of healthy life expectancy (reference frame).", "model_input": "no — descriptive", "rationale": "EDA grouping."},
    "happiness_category": {
        "formula": "Fixed bands of Life Ladder: <4.5, 4.5–5.5, 5.5–6.5, ≥6.5.",
        "model_input": "NO — target-derived (leakage)",
        "rationale": "Readable outcome bands for charts and error analysis.",
    },
    "ladder_yoy_change": {
        "formula": "Life Ladder(t) − Life Ladder(t−1), only for consecutive observed years.",
        "model_input": "NO — target-derived (leakage)",
        "rationale": "Identifies improvers and decliners.",
    },
    "ladder_vs_gdp_residual": {
        "formula": "Life Ladder minus the value predicted by a univariate linear fit on log GDP pc.",
        "model_input": "NO — target-derived (leakage)",
        "rationale": "Shows countries reporting higher/lower life evaluations than income alone would suggest.",
    },
}


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-looking change features for an annual panel."""
    out = df.sort_values(["country", "year"]).copy()
    g = out.groupby("country", sort=False)
    prev_year = g["year"].shift(1)
    consecutive = (out["year"] - prev_year) == 1
    if "log_gdp_per_capita" in out:
        out["gdp_growth_1y"] = np.where(consecutive, out["log_gdp_per_capita"] - g["log_gdp_per_capita"].shift(1), np.nan)
    if TARGET in out:
        out["ladder_yoy_change"] = np.where(consecutive, out[TARGET] - g[TARGET].shift(1), np.nan)
    out["years_observed_so_far"] = g.cumcount() + 1
    return out.reset_index(drop=True)


def add_categories(df: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """Quartile categories (thresholds from ``reference``) + fixed happiness bands."""
    out = df.copy()
    ref = reference if reference is not None else df
    for col, name in (
        ("log_gdp_per_capita", "gdp_category"),
        ("social_support", "social_support_category"),
        ("freedom", "freedom_category"),
        ("healthy_life_expectancy", "health_category"),
    ):
        if col in out and ref[col].notna().sum() >= 8:
            qs = ref[col].quantile([0.25, 0.5, 0.75]).to_numpy()
            edges = np.concatenate([[-np.inf], qs, [np.inf]])
            if np.all(np.diff(edges) > 0):
                out[name] = pd.cut(out[col], bins=edges, labels=QUARTILE_LABELS)
    if TARGET in out:
        out["happiness_category"] = pd.cut(out[TARGET], bins=HAPPINESS_BINS, labels=HAPPINESS_LABELS, right=False)
    return out


def add_gdp_gap(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """Descriptive: ladder relative to what log GDP alone predicts (OLS line)."""
    out = df.copy()
    m = out[[target, "log_gdp_per_capita"]].dropna()
    if len(m) >= 10:
        slope, intercept = np.polyfit(m["log_gdp_per_capita"], m[target], 1)
        out["ladder_vs_gdp_residual"] = out[target] - (intercept + slope * out["log_gdp_per_capita"])
    return out


def engineer_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Full descriptive feature set for an annual panel."""
    out = add_time_features(panel)
    out = add_categories(out)
    out = add_gdp_gap(out)
    return out


class CompositeIndexTransformer(BaseEstimator, TransformerMixin):
    """Appends a composite socio-economic index (mean z-score) to a numeric array.

    Parameters fitted on training data only. ``sign`` lets indicators where
    "higher = worse" (perceived corruption) be flipped.
    """

    def __init__(self, columns: list[str] | None = None, negative: tuple[str, ...] = ("perceptions_of_corruption",)):
        self.columns = columns
        self.negative = negative

    def fit(self, X, y=None):
        Xa = np.asarray(X, dtype=float)
        self.mean_ = np.nanmean(Xa, axis=0)
        self.std_ = np.nanstd(Xa, axis=0)
        self.std_[self.std_ == 0] = 1.0
        cols = self.columns or [f"x{i}" for i in range(Xa.shape[1])]
        self.sign_ = np.array([-1.0 if c in self.negative else 1.0 for c in cols])
        self.n_features_in_ = Xa.shape[1]
        return self

    def transform(self, X):
        Xa = np.asarray(X, dtype=float)
        z = (Xa - self.mean_) / self.std_ * self.sign_
        comp = np.nanmean(z, axis=1, keepdims=True)
        return np.hstack([Xa, comp])

    def get_feature_names_out(self, input_features=None):
        names = list(input_features) if input_features is not None else list(self.columns or [])
        return np.array(names + ["composite_index"], dtype=object)


def composite_index(df: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.Series:
    """Composite index for descriptive use (z-scores against ``reference``)."""
    cols = [c for c in INDICATORS if c in df]
    ref = reference if reference is not None else df
    z = (df[cols] - ref[cols].mean()) / ref[cols].std(ddof=0)
    if "perceptions_of_corruption" in z:
        z["perceptions_of_corruption"] *= -1
    return z.mean(axis=1, skipna=True)
