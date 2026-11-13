"""Happiness prediction: input specification, validation, prediction with a
split-conformal interval, extrapolation checks and nearest observed profiles.

The model estimates the *country-level average life evaluation* that is
typical of historical country-years with similar indicators. It cannot
determine the happiness of any individual.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import DEFINITIONS, LABELS, VALID_RANGES

STEP = {
    "log_gdp_per_capita": 0.01, "social_support": 0.005, "healthy_life_expectancy": 0.1, "freedom": 0.005,
    "generosity": 0.005, "perceptions_of_corruption": 0.005, "positive_affect": 0.005, "negative_affect": 0.005,
    "year": 1.0, "gdp_growth_1y": 0.005,
}
HARD_LIMITS = {**VALID_RANGES, "gdp_growth_1y": (-0.5, 0.5)}


@dataclass
class InputSpec:
    feature: str
    label: str
    hard_min: float
    hard_max: float
    train_min: float
    train_max: float
    default: float
    step: float
    help: str
    categories: list[str] = field(default_factory=list)

    @property
    def is_categorical(self) -> bool:
        return bool(self.categories)


def input_specs(feature_ranges: dict, features: list[str]) -> dict[str, InputSpec]:
    """Build UI/validation specs from the training-data ranges stored with the model.

    Hard limits = definitional range intersected with a margin around the
    observed training range (inputs far outside anything observed are refused).
    """
    specs = {}
    for f in features:
        r = feature_ranges.get(f, {})
        if "categories" in r:
            specs[f] = InputSpec(f, LABELS.get(f, f), 0, 0, 0, 0, 0, 0, DEFINITIONS.get(f, ""), categories=r["categories"])
            continue
        lo_def, hi_def = HARD_LIMITS.get(f, (-np.inf, np.inf))
        span = r["max"] - r["min"]
        margin = 0.25 * span
        hard_min = max(lo_def, r["min"] - margin)
        hard_max = min(hi_def, r["max"] + margin)
        if f == "year":
            hard_min, hard_max = r["min"], r["max"] + 5
        specs[f] = InputSpec(
            f, LABELS.get(f, f), float(hard_min), float(hard_max), float(r["min"]), float(r["max"]),
            float(r["median"]), STEP.get(f, 0.01), DEFINITIONS.get(f, ""),
        )
    return specs


def validate_inputs(values: dict, specs: dict[str, InputSpec]) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors block prediction; warnings flag extrapolation."""
    errors, warns = [], []
    for f, s in specs.items():
        v = values.get(f)
        if s.is_categorical:
            if v not in s.categories:
                errors.append(f"{s.label}: '{v}' is not a known category.")
            continue
        if v is None or (isinstance(v, float) and np.isnan(v)):
            errors.append(f"{s.label}: a value is required.")
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            errors.append(f"{s.label}: must be numeric.")
            continue
        if v < s.hard_min or v > s.hard_max:
            errors.append(f"{s.label} = {v:g} is outside the accepted range [{s.hard_min:.3g}, {s.hard_max:.3g}].")
        elif v < s.train_min or v > s.train_max:
            warns.append(f"{s.label} = {v:g} lies outside the range observed in training data [{s.train_min:.3g}, {s.train_max:.3g}] — the model is extrapolating.")
    return errors, warns


def predict_one(pipe, values: dict, features: list[str], conformal_q: float | None = None) -> dict:
    row = pd.DataFrame([{f: values.get(f) for f in features}])
    pred = float(pipe.predict(row)[0])
    out = {"prediction": pred, "clipped": False}
    if not 0 <= pred <= 10:
        out["clipped"] = True
        pred = float(np.clip(pred, 0, 10))
        out["prediction"] = pred
    if conformal_q is not None:
        out["lower"] = max(0.0, pred - conformal_q)
        out["upper"] = min(10.0, pred + conformal_q)
        out["interval_halfwidth"] = conformal_q
    return out


def nearest_profiles(panel: pd.DataFrame, values: dict, features: list[str], k: int = 5) -> pd.DataFrame:
    """Most similar observed country-years (standardised Euclidean distance on numeric features)."""
    num = [f for f in features if f in panel and pd.api.types.is_numeric_dtype(panel[f]) and f != "year"]
    d = panel.dropna(subset=num + ["life_ladder"]).copy()
    mu, sd = d[num].mean(), d[num].std().replace(0, 1)
    q = np.array([(float(values[f]) - mu[f]) / sd[f] for f in num])
    Z = ((d[num] - mu) / sd).to_numpy()
    d["distance"] = np.sqrt(((Z - q) ** 2).sum(axis=1))
    return d.nsmallest(k, "distance")[["country", "year", "region", "life_ladder", "distance"] + num]


def mahalanobis_extrapolation(panel: pd.DataFrame, values: dict, features: list[str]) -> dict:
    """How unusual is the *combination* of inputs? Compares the Mahalanobis distance
    of the input to the distribution of distances of the observed data."""
    num = [f for f in features if f in panel and pd.api.types.is_numeric_dtype(panel[f]) and f != "year"]
    d = panel[num].dropna().to_numpy()
    mu = d.mean(axis=0)
    cov = np.cov(d, rowvar=False)
    inv = np.linalg.pinv(cov)
    x = np.array([float(values[f]) for f in num]) - mu
    md = float(np.sqrt(x @ inv @ x))
    ref = np.sqrt(np.einsum("ij,jk,ik->i", d - mu, inv, d - mu))
    pct = float((ref < md).mean() * 100)
    return {"distance": md, "percentile": pct, "unusual": pct > 99}
