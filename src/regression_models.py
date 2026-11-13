"""Regression model registry, pipelines and hyper-parameter search spaces."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from .config import RANDOM_STATE
from .preprocessing import build_preprocessor

try:  # XGBoost is optional
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except Exception:  # pragma: no cover
    XGBRegressor = None
    XGBOOST_AVAILABLE = False


@dataclass
class ModelSpec:
    key: str
    name: str
    family: str
    factory: Callable[..., Any]
    defaults: dict[str, Any] = field(default_factory=dict)
    search_space: dict[str, Any] = field(default_factory=dict)
    scaler: str = "standard"
    description: str = ""
    tunable: bool = True
    ui_params: dict[str, dict] = field(default_factory=dict)  # for the training page


MODEL_SPECS: dict[str, ModelSpec] = {
    "linear": ModelSpec(
        "linear", "Linear Regression", "Linear", LinearRegression, {}, {}, "standard",
        "Ordinary least squares baseline: one additive slope per feature.", tunable=False,
    ),
    "ridge": ModelSpec(
        "ridge", "Ridge Regression", "Linear (L2)", Ridge, {"alpha": 1.0},
        {"alpha": loguniform(1e-3, 1e3)}, "standard",
        "Linear model with an L2 penalty that shrinks correlated coefficients.",
        ui_params={"alpha": {"type": "log", "min": 0.001, "max": 1000.0, "default": 1.0}},
    ),
    "lasso": ModelSpec(
        "lasso", "Lasso Regression", "Linear (L1)", Lasso, {"alpha": 0.01, "max_iter": 20000},
        {"alpha": loguniform(1e-4, 1.0)}, "standard",
        "Linear model with an L1 penalty that can set weak coefficients exactly to zero (feature selection).",
        ui_params={"alpha": {"type": "log", "min": 0.0001, "max": 1.0, "default": 0.01}},
    ),
    "decision_tree": ModelSpec(
        "decision_tree", "Decision Tree", "Tree", DecisionTreeRegressor,
        {"max_depth": 6, "min_samples_leaf": 5, "random_state": RANDOM_STATE},
        {"max_depth": [3, 4, 5, 6, 8, 10, 12, None], "min_samples_split": randint(2, 30), "min_samples_leaf": randint(1, 25)},
        "none", "Single tree of if/then splits; captures non-linearity and interactions, prone to overfitting.",
        ui_params={
            "max_depth": {"type": "int", "min": 2, "max": 20, "default": 6},
            "min_samples_leaf": {"type": "int", "min": 1, "max": 50, "default": 5},
        },
    ),
    "random_forest": ModelSpec(
        "random_forest", "Random Forest", "Bagged trees", RandomForestRegressor,
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2, "max_features": 0.6, "n_jobs": -1, "random_state": RANDOM_STATE},
        {
            "n_estimators": [150, 250, 400],
            "max_depth": [None, 6, 10, 14, 20],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", 0.4, 0.6, 0.8, 1.0],
        },
        "none", "Average of many decorrelated trees grown on bootstrap samples.",
        ui_params={
            "n_estimators": {"type": "int", "min": 50, "max": 600, "default": 300, "step": 50},
            "max_depth": {"type": "int_or_none", "min": 2, "max": 30, "default": 0},
            "min_samples_split": {"type": "int", "min": 2, "max": 30, "default": 2},
            "min_samples_leaf": {"type": "int", "min": 1, "max": 20, "default": 2},
            "max_features": {"type": "choice", "options": ["sqrt", 0.4, 0.6, 0.8, 1.0], "default": 0.6},
        },
    ),
    "gradient_boosting": ModelSpec(
        "gradient_boosting", "Gradient Boosting", "Boosted trees", GradientBoostingRegressor,
        {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3, "subsample": 0.8, "random_state": RANDOM_STATE},
        {
            "n_estimators": [100, 200, 300, 500],
            "learning_rate": loguniform(0.01, 0.2),
            "max_depth": [2, 3, 4, 5],
            "subsample": uniform(0.6, 0.4),
            "min_samples_leaf": [1, 3, 5, 10],
        },
        "none", "Trees added sequentially, each fitting the remaining errors of the ensemble.",
        ui_params={
            "n_estimators": {"type": "int", "min": 50, "max": 800, "default": 300, "step": 50},
            "learning_rate": {"type": "float", "min": 0.005, "max": 0.3, "default": 0.05},
            "max_depth": {"type": "int", "min": 1, "max": 8, "default": 3},
            "subsample": {"type": "float", "min": 0.5, "max": 1.0, "default": 0.8},
        },
    ),
}

if XGBOOST_AVAILABLE:
    MODEL_SPECS["xgboost"] = ModelSpec(
        "xgboost", "XGBoost", "Boosted trees (regularised)", XGBRegressor,
        {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 4, "subsample": 0.8, "colsample_bytree": 0.8,
         "reg_lambda": 1.0, "random_state": RANDOM_STATE, "n_jobs": 2, "tree_method": "hist", "verbosity": 0},
        {
            "n_estimators": [150, 300, 500],
            "learning_rate": loguniform(0.01, 0.2),
            "max_depth": [2, 3, 4, 5, 6],
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.5, 0.5),
            "reg_lambda": loguniform(0.1, 10),
        },
        "none", "Regularised gradient boosting with second-order optimisation (CPU 'hist' algorithm).",
        ui_params={
            "n_estimators": {"type": "int", "min": 50, "max": 1000, "default": 400, "step": 50},
            "learning_rate": {"type": "float", "min": 0.005, "max": 0.3, "default": 0.05},
            "max_depth": {"type": "int", "min": 1, "max": 10, "default": 4},
            "subsample": {"type": "float", "min": 0.5, "max": 1.0, "default": 0.8},
            "colsample_bytree": {"type": "float", "min": 0.3, "max": 1.0, "default": 0.8},
        },
    )

MODEL_KEYS = list(MODEL_SPECS)
TREE_MODELS = {"decision_tree", "random_forest", "gradient_boosting", "xgboost"}
LINEAR_MODELS = {"linear", "ridge", "lasso"}


def make_pipeline(
    key: str,
    features: list[str],
    params: dict[str, Any] | None = None,
    add_composite: bool = False,
    imputer: str = "median",
    scaler: str | None = None,
) -> Pipeline:
    spec = MODEL_SPECS[key]
    kwargs = {**spec.defaults, **(params or {})}
    model = spec.factory(**kwargs)
    pre = build_preprocessor(features, scaler=scaler or spec.scaler, imputer=imputer, add_composite=add_composite)
    return Pipeline([("preprocess", pre), ("model", model)])


def search_space(key: str) -> dict[str, Any]:
    return {f"model__{k}": v for k, v in MODEL_SPECS[key].search_space.items()}


def transformed_feature_names(pipe: Pipeline) -> list[str]:
    try:
        return list(pipe.named_steps["preprocess"].get_feature_names_out())
    except Exception:
        return []


def clean_params(params: dict[str, Any]) -> dict[str, Any]:
    """Strip ``model__`` prefixes and convert numpy scalars for JSON."""
    out = {}
    for k, v in params.items():
        k = k.replace("model__", "")
        if isinstance(v, np.generic):
            v = v.item()
        out[k] = v
    return out
