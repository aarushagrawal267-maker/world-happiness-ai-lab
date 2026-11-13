"""Preprocessing: leakage-safe scikit-learn pipelines, missing-value analysis,
imputation-strategy comparison and outlier analysis.

All learned transformations (imputation statistics, scaling parameters,
one-hot categories, composite-index z-scores) live inside a
``ColumnTransformer`` that is part of the model ``Pipeline``. They are therefore
fitted on training data / training folds only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from .config import CATEGORICAL_FEATURES, INDICATORS, LEAKAGE_COLUMNS, TARGET
from .feature_engineering import CompositeIndexTransformer


def split_feature_types(features: list[str]) -> tuple[list[str], list[str]]:
    cats = [f for f in features if f in CATEGORICAL_FEATURES]
    nums = [f for f in features if f not in cats]
    return nums, cats


def assert_no_leakage(features: list[str]) -> None:
    bad = [f for f in features if f in LEAKAGE_COLUMNS or "ladder" in f or "dystopia" in f]
    if bad:
        raise ValueError(f"Target leakage: {bad} cannot be used as model inputs.")


def build_preprocessor(
    features: list[str],
    scaler: str = "standard",
    imputer: str = "median",
    add_composite: bool = False,
) -> ColumnTransformer:
    """Numeric: impute → (composite) → scale. Categorical: impute → one-hot."""
    assert_no_leakage(features)
    nums, cats = split_feature_types(features)
    steps: list = [("impute", SimpleImputer(strategy=imputer))]
    if add_composite:
        steps.append(("composite", CompositeIndexTransformer(columns=nums)))
    if scaler == "standard":
        steps.append(("scale", StandardScaler()))
    elif scaler == "robust":
        steps.append(("scale", RobustScaler()))
    transformers = [("num", Pipeline(steps), nums)]
    if cats:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cats,
            )
        )
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


# --------------------------------------------------------------------------- missing values
def missing_summary(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    cols = columns or list(df.columns)
    s = pd.DataFrame(
        {
            "missing": df[cols].isna().sum(),
            "missing_pct": (df[cols].isna().mean() * 100).round(2),
            "non_missing": df[cols].notna().sum(),
        }
    )
    return s.sort_values("missing", ascending=False)


def missing_by_year(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return (df.groupby("year")[columns].apply(lambda g: g.isna().mean() * 100)).round(1)


def country_interpolate(df: pd.DataFrame, columns: list[str], method: str = "linear") -> pd.DataFrame:
    """Within-country imputation.

    * ``linear``  interpolates between a country's own observed years (uses
      later years → not appropriate for forecasting-style evaluation).
    * ``ffill``   carries the last *past* observation forward (backward-looking,
      safe for time-aware evaluation).
    Never fills the target and never borrows values from other countries.
    """
    out = df.sort_values(["country", "year"]).copy()
    cols = [c for c in columns if c in out and c != TARGET]
    for c in cols:
        if method == "linear":
            out[c] = out.groupby("country")[c].transform(lambda s: s.interpolate(method="linear", limit_area="inside"))
        elif method == "ffill":
            out[c] = out.groupby("country")[c].transform(lambda s: s.ffill())
        else:
            raise ValueError(method)
    return out


def compare_imputation_strategies(panel: pd.DataFrame, features: list[str] | None = None, folds: int = 5) -> pd.DataFrame:
    """Compare imputation strategies by data retained, distribution shift and
    grouped-CV error of a simple Ridge model (country-grouped folds)."""
    features = features or INDICATORS
    base = panel.dropna(subset=[TARGET]).copy()
    observed_mean = base[features].mean()
    results = []

    def _cv(frame: pd.DataFrame, impute: str | None) -> float:
        X, y, g = frame[features], frame[TARGET], frame["country"]
        steps = []
        if impute:
            steps.append(("impute", SimpleImputer(strategy=impute)))
        steps += [("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]
        cv = GroupKFold(n_splits=folds)
        scores = cross_val_score(Pipeline(steps), X, y, groups=g, cv=cv, scoring="neg_root_mean_squared_error")
        return float(-scores.mean())

    strategies = {
        "Complete cases only (drop rows)": (base.dropna(subset=features), None),
        "Median (fitted in pipeline)": (base, "median"),
        "Mean (fitted in pipeline)": (base, "mean"),
        "Country interpolation + median": (country_interpolate(base, features, "linear"), "median"),
        "Country past-only carry-forward + median": (country_interpolate(base, features, "ffill"), "median"),
    }
    for name, (frame, imp) in strategies.items():
        remaining_missing = int(frame[features].isna().sum().sum())
        shift = float(((frame[features].mean() - observed_mean) / base[features].std()).abs().mean())
        results.append(
            {
                "strategy": name,
                "rows_used": len(frame),
                "rows_lost": len(base) - len(frame),
                "countries_used": frame["country"].nunique(),
                "cells_filled_outside_pipeline": int(base[features].isna().sum().sum()) - remaining_missing if imp and "Country" in name else 0,
                "mean_shift_sd": round(shift, 4),
                "grouped_cv_rmse": round(_cv(frame, imp), 4),
            }
        )
    return pd.DataFrame(results)


# --------------------------------------------------------------------------- outliers
def outlier_analysis(df: pd.DataFrame, columns: list[str] | None = None, iqr_k: float = 1.5, z_thresh: float = 3.5) -> pd.DataFrame:
    """Flag unusual values with the IQR rule and a robust (median/MAD) z-score.

    Flags are informational: legitimate extreme observations are NOT removed.
    """
    columns = columns or [c for c in INDICATORS + [TARGET] if c in df]
    rows = []
    for c in columns:
        s = df[c].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
        med = s.median()
        mad = (s - med).abs().median() or 1e-9
        rz = 0.6745 * (df[c] - med) / mad
        mask = df[c].notna() & ((df[c] < lo) | (df[c] > hi) | (rz.abs() > z_thresh))
        for idx in df.index[mask]:
            rows.append(
                {
                    "country": df.at[idx, "country"],
                    "year": df.at[idx, "year"] if "year" in df else None,
                    "variable": c,
                    "value": round(float(df.at[idx, c]), 4),
                    "iqr_low": round(float(lo), 3),
                    "iqr_high": round(float(hi), 3),
                    "robust_z": round(float(rz[idx]), 2),
                    "flag": "beyond IQR fence" + (" & |robust z|>%.1f" % z_thresh if abs(rz[idx]) > z_thresh else ""),
                }
            )
    return pd.DataFrame(rows)


def outlier_summary(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    flags = outlier_analysis(df, columns)
    if flags.empty:
        return pd.DataFrame(columns=["variable", "n_flagged", "pct_flagged", "countries"])
    total = len(df)
    return (
        flags.groupby("variable")
        .agg(n_flagged=("value", "size"), countries=("country", lambda s: ", ".join(sorted(set(s))[:8])))
        .assign(pct_flagged=lambda d: (100 * d["n_flagged"] / total).round(2))
        .reset_index()
        .sort_values("n_flagged", ascending=False)
    )


def numeric_matrix(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return df[columns].to_numpy(dtype=float)
