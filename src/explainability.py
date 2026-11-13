"""Explainable AI: permutation importance, model-native importance, SHAP
(with graceful fallback), partial dependence and individual explanations.

Feature importance describes what a *particular model* relies on to make
predictions. It is model-dependent, can be shared among correlated features,
and does not establish that a factor causes higher or lower life evaluations.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.pipeline import Pipeline

from .config import RANDOM_STATE

try:
    import shap  # type: ignore

    SHAP_AVAILABLE = True
except Exception:  # pragma: no cover
    shap = None
    SHAP_AVAILABLE = False


def permutation_importance_df(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, n_repeats: int = 10) -> pd.DataFrame:
    """Increase in RMSE when each *original* feature is shuffled."""
    r = permutation_importance(pipe, X, y, n_repeats=n_repeats, random_state=RANDOM_STATE, scoring="neg_root_mean_squared_error", n_jobs=1)
    df = pd.DataFrame({"feature": X.columns, "importance_mean": r.importances_mean, "importance_std": r.importances_std})
    return df.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def native_importance(pipe: Pipeline) -> pd.DataFrame | None:
    """Impurity importance for trees or |standardised coefficient| for linear models."""
    model = pipe.named_steps["model"]
    try:
        names = list(pipe.named_steps["preprocess"].get_feature_names_out())
    except Exception:
        return None
    if hasattr(model, "feature_importances_"):
        vals, kind = model.feature_importances_, "Impurity-based importance (training data)"
    elif hasattr(model, "coef_"):
        vals, kind = np.abs(np.ravel(model.coef_)), "|Coefficient| on standardised features"
    else:
        return None
    df = pd.DataFrame({"feature": names[: len(vals)], "importance": vals, "kind": kind})
    if hasattr(model, "coef_"):
        df["signed_coef"] = np.ravel(model.coef_)[: len(df)]
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def _transform(pipe: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    pre = pipe.named_steps["preprocess"]
    Xt = pre.transform(X)
    try:
        names = list(pre.get_feature_names_out())
    except Exception:
        names = [f"f{i}" for i in range(Xt.shape[1])]
    return np.asarray(Xt, dtype=float), names


def shap_explanation(pipe: Pipeline, X_background: pd.DataFrame, X_explain: pd.DataFrame, max_background: int = 100) -> dict | None:
    """SHAP values in the transformed feature space.

    Tree models use the exact TreeExplainer; linear models LinearExplainer;
    anything else a small KernelExplainer. Returns ``None`` if SHAP is unavailable
    or fails (callers fall back to permutation importance).
    """
    if not SHAP_AVAILABLE:
        return None
    model = pipe.named_steps["model"]
    try:
        bg_t, names = _transform(pipe, X_background.sample(min(len(X_background), max_background), random_state=RANDOM_STATE))
        ex_t, _ = _transform(pipe, X_explain)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if hasattr(model, "feature_importances_"):
                explainer = shap.TreeExplainer(model, data=bg_t, feature_perturbation="interventional")
                values = explainer.shap_values(ex_t, check_additivity=False)
                method = "TreeExplainer (interventional)"
            elif hasattr(model, "coef_"):
                explainer = shap.LinearExplainer(model, bg_t)
                values = explainer.shap_values(ex_t)
                method = "LinearExplainer"
            else:
                explainer = shap.KernelExplainer(model.predict, shap.kmeans(bg_t, 20))
                values = explainer.shap_values(ex_t[:100], nsamples=200)
                ex_t = ex_t[:100]
                method = "KernelExplainer"
        base = explainer.expected_value
        base = float(np.ravel(base)[0])
        return {"values": np.asarray(values), "base_value": base, "data": ex_t, "feature_names": names, "method": method}
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(f"SHAP failed: {exc}")
        return None


def shap_importance(expl: dict) -> pd.DataFrame:
    return (
        pd.DataFrame({"feature": expl["feature_names"], "mean_abs_shap": np.abs(expl["values"]).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def pdp(pipe: Pipeline, X: pd.DataFrame, feature: str, grid_resolution: int = 30, ice: bool = False, ice_samples: int = 60) -> dict:
    """Partial dependence (average prediction as ``feature`` varies), optional ICE curves."""
    kind = "both" if ice else "average"
    Xs = X.dropna(subset=[feature])
    if ice and len(Xs) > ice_samples:
        Xs = Xs.sample(ice_samples, random_state=RANDOM_STATE)
    r = partial_dependence(pipe, Xs, [feature], grid_resolution=grid_resolution, kind=kind, percentiles=(0.02, 0.98))
    grid = r["grid_values"][0] if "grid_values" in r else r["values"][0]
    out = {"grid": np.asarray(grid), "average": np.asarray(r["average"][0])}
    if ice:
        out["individual"] = np.asarray(r["individual"][0])
    return out


def explain_instance(pipe: Pipeline, X_background: pd.DataFrame, row: pd.DataFrame) -> dict:
    """Prediction + per-feature contributions for one observation.

    Uses SHAP when available; otherwise a model-agnostic fallback that replaces
    each feature by its background median and records the prediction change.
    """
    pred = float(pipe.predict(row)[0])
    expl = shap_explanation(pipe, X_background, row)
    if expl is not None:
        contrib = pd.DataFrame({"feature": expl["feature_names"], "contribution": expl["values"][0], "value": expl["data"][0]})
        return {"prediction": pred, "base_value": expl["base_value"], "contributions": contrib, "method": "SHAP " + expl["method"]}
    med = X_background.median(numeric_only=True)
    rows = []
    for f in row.columns:
        if f not in med:
            continue
        alt = row.copy()
        alt[f] = med[f]
        rows.append({"feature": f, "contribution": pred - float(pipe.predict(alt)[0]), "value": row[f].iloc[0]})
    base = float(pipe.predict(pd.DataFrame([med]))[0]) if set(row.columns) <= set(med.index) else np.nan
    return {"prediction": pred, "base_value": base, "contributions": pd.DataFrame(rows), "method": "Median-replacement (fallback)"}


def pretty_feature(name: str) -> str:
    from .config import SHORT_LABELS

    if name in SHORT_LABELS:
        return SHORT_LABELS[name]
    if name.startswith("region_"):
        return "Region: " + name[len("region_"):]
    return name
