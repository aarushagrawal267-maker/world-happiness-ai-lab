"""What-If Happiness Simulator (model-based hypothetical scenarios).

A scenario changes one or more indicator values and asks the trained model
what life evaluation it would associate with the modified profile. This is a
statement about patterns in historical cross-country data — **not** a causal
prediction of what would happen if a country actually changed a policy or an
indicator. Changes that move inputs outside the training range are flagged.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from .config import SHORT_LABELS
from .prediction import InputSpec, predict_one


@dataclass
class Scenario:
    name: str
    base_label: str
    base_values: dict
    modified_values: dict
    base_prediction: float
    modified_prediction: float
    model_name: str
    warnings: list[str] = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))

    @property
    def delta(self) -> float:
        return self.modified_prediction - self.base_prediction

    def changes(self) -> pd.DataFrame:
        rows = []
        for f, v in self.modified_values.items():
            b = self.base_values.get(f)
            if isinstance(v, (int, float)) and isinstance(b, (int, float)) and not math.isclose(v, b, abs_tol=1e-9):
                rows.append({"indicator": SHORT_LABELS.get(f, f), "feature": f, "original": b, "modified": v, "change": v - b})
            elif not isinstance(v, (int, float)) and v != b:
                rows.append({"indicator": SHORT_LABELS.get(f, f), "feature": f, "original": b, "modified": v, "change": np.nan})
        return pd.DataFrame(rows)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["delta"] = self.delta
        return d


def gdp_percent_to_log_delta(pct: float) -> float:
    """A +10 % change in GDP per capita corresponds to log(1.10) ≈ +0.0953 in log GDP."""
    return math.log1p(pct / 100.0)


def log_delta_to_gdp_percent(d: float) -> float:
    return (math.exp(d) - 1) * 100.0


def extrapolation_warnings(values: dict, specs: dict[str, InputSpec]) -> list[str]:
    out = []
    for f, s in specs.items():
        if s.is_categorical:
            continue
        v = values.get(f)
        if v is None:
            continue
        if v < s.train_min or v > s.train_max:
            out.append(f"{SHORT_LABELS.get(f, f)} = {v:.3g} is outside the historical range [{s.train_min:.3g}, {s.train_max:.3g}]; the model is extrapolating and the result is unreliable.")
    return out


def clamp_to_hard_limits(values: dict, specs: dict[str, InputSpec]) -> dict:
    out = dict(values)
    for f, s in specs.items():
        if not s.is_categorical and f in out and out[f] is not None:
            out[f] = float(np.clip(out[f], s.hard_min, s.hard_max))
    return out


def simulate(pipe, features: list[str], base_values: dict, modified_values: dict, specs: dict[str, InputSpec],
             name: str = "Scenario", base_label: str = "Baseline", model_name: str = "") -> Scenario:
    modified_values = clamp_to_hard_limits(modified_values, specs)
    b = predict_one(pipe, base_values, features)["prediction"]
    m = predict_one(pipe, modified_values, features)["prediction"]
    return Scenario(name, base_label, dict(base_values), dict(modified_values), b, m, model_name, extrapolation_warnings(modified_values, specs))


def one_at_a_time(pipe, features: list[str], base_values: dict, specs: dict[str, InputSpec], feature: str, n: int = 40) -> pd.DataFrame:
    """Prediction as one indicator sweeps its observed range, others fixed (ceteris paribus curve)."""
    s = specs[feature]
    grid = np.linspace(s.train_min, s.train_max, n)
    rows = pd.DataFrame([{**base_values, feature: g} for g in grid])[features]
    return pd.DataFrame({"value": grid, "prediction": pipe.predict(rows)})


def tornado(pipe, features: list[str], base_values: dict, specs: dict[str, InputSpec], feature_ranges: dict, sd_shift: float = 1.0) -> pd.DataFrame:
    """Change in prediction when each indicator alone moves ±``sd_shift`` standard deviations
    (bounded by the observed range)."""
    base = predict_one(pipe, base_values, features)["prediction"]
    rows = []
    for f in features:
        s = specs.get(f)
        if s is None or s.is_categorical or f == "year" or f not in feature_ranges:
            continue
        sd = feature_ranges[f]["std"]
        lo = max(s.train_min, base_values[f] - sd_shift * sd)
        hi = min(s.train_max, base_values[f] + sd_shift * sd)
        p_lo = predict_one(pipe, {**base_values, f: lo}, features)["prediction"]
        p_hi = predict_one(pipe, {**base_values, f: hi}, features)["prediction"]
        rows.append({"feature": f, "indicator": SHORT_LABELS.get(f, f), "low_value": lo, "high_value": hi,
                     "pred_low": p_lo - base, "pred_high": p_hi - base, "range": abs(p_hi - p_lo)})
    return pd.DataFrame(rows).sort_values("range")


def scenarios_table(scenarios: list[Scenario]) -> pd.DataFrame:
    rows = []
    for s in scenarios:
        ch = s.changes()
        rows.append({
            "scenario": s.name, "baseline": s.base_label, "model": s.model_name,
            "baseline_prediction": s.base_prediction, "scenario_prediction": s.modified_prediction, "difference": s.delta,
            "changes": "; ".join(f"{r.indicator}: {r.original:.3g}→{r.modified:.3g}" if isinstance(r.original, (int, float)) else f"{r.indicator}: {r.original}→{r.modified}" for r in ch.itertuples()) or "none",
            "extrapolation": "yes" if s.warnings else "no", "created": s.created,
        })
    return pd.DataFrame(rows)
