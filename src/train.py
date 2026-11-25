"""Reproducible training and validation experiments.

Three evaluation questions are answered separately:

1. **random**    — random train / validation / test split (K-fold CV for tuning)
2. **group**     — country-grouped split (GroupKFold): generalisation to unseen countries
3. **temporal**  — forward-in-time split (train ≤ 2016, validate 2017–18, test ≥ 2019)

For every model and every question: hyper-parameters are tuned with
RandomizedSearchCV on the training portion only, then the model is scored on
validation and test sets. Additional experiments compare cross-validation
strategies, feature sets and imputation strategies. The best model (chosen on
country-grouped validation RMSE, not on R² alone) is refitted on all data and
saved for prediction, with a split-conformal prediction interval.

Usage::

    python -m src.train                # full run (tuning, all experiments)
    python -m src.train --quick        # fewer tuning iterations
    python -m src.train --models ridge random_forest --no-tune
"""
from __future__ import annotations

import argparse
import math
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)

from .config import CFG, FEATURE_SETS, RANDOM_STATE, REPORTS_DIR, TARGET
from .data_loader import load_bundle
from .evaluate import ForwardYearSplit, describe_validation, regression_metrics
from .feature_engineering import add_time_features
from .model_manager import save_model, save_results
from .preprocessing import assert_no_leakage, compare_imputation_strategies
from .regression_models import MODEL_KEYS, MODEL_SPECS, clean_params, make_pipeline, search_space

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ProgressCB = Callable[[float, str], None]
META_COLS = ["country", "year", "region", "iso3"]
STRATEGIES = ["random", "group", "temporal"]


def _noop(_f: float, _m: str) -> None:
    pass


# --------------------------------------------------------------------------- data
def build_modeling_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Annual panel + backward-looking features; rows without a target are dropped
    (targets are never imputed)."""
    frame = add_time_features(panel)
    frame = frame.dropna(subset=[TARGET]).reset_index(drop=True)
    return frame


def feature_list(feature_set: str) -> list[str]:
    feats = list(FEATURE_SETS[feature_set])
    assert_no_leakage(feats)
    return feats


@dataclass
class SplitData:
    strategy: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    cv: object
    cv_groups_col: str | None

    def cv_groups(self, frame: pd.DataFrame):
        return frame[self.cv_groups_col].to_numpy() if self.cv_groups_col else None

    def describe(self) -> dict:
        d = {
            "description": describe_validation(self.strategy),
            "n_train": len(self.train), "n_val": len(self.val), "n_test": len(self.test),
            "countries_train": int(self.train["country"].nunique()),
            "countries_val": int(self.val["country"].nunique()),
            "countries_test": int(self.test["country"].nunique()),
            "years_train": [int(self.train["year"].min()), int(self.train["year"].max())],
            "years_val": [int(self.val["year"].min()), int(self.val["year"].max())],
            "years_test": [int(self.test["year"].min()), int(self.test["year"].max())],
            "country_overlap_train_test": int(len(set(self.train["country"]) & set(self.test["country"]))),
        }
        return d


def make_splits(frame: pd.DataFrame, strategy: str, cv_folds: int | None = None) -> SplitData:
    cv_folds = cv_folds or int(CFG["modeling"]["cv_folds"])
    test_size = float(CFG["modeling"]["test_size"])
    val_size = float(CFG["modeling"]["validation_size"])
    if strategy == "random":
        trval, test = train_test_split(frame, test_size=test_size, random_state=RANDOM_STATE)
        train, val = train_test_split(trval, test_size=val_size / (1 - test_size), random_state=RANDOM_STATE)
        cv, gcol = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE), None
    elif strategy == "group":
        g1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
        tr_idx, te_idx = next(g1.split(frame, groups=frame["country"]))
        trval, test = frame.iloc[tr_idx], frame.iloc[te_idx]
        g2 = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=RANDOM_STATE)
        a, b = next(g2.split(trval, groups=trval["country"]))
        train, val = trval.iloc[a], trval.iloc[b]
        cv, gcol = GroupKFold(n_splits=cv_folds), "country"
    elif strategy == "temporal":
        t_end = int(CFG["modeling"]["time_split"]["train_end"])
        v_end = int(CFG["modeling"]["time_split"]["val_end"])
        train = frame[frame["year"] <= t_end]
        val = frame[(frame["year"] > t_end) & (frame["year"] <= v_end)]
        test = frame[frame["year"] > v_end]
        cv, gcol = ForwardYearSplit(n_splits=4), "year"
    else:
        raise ValueError(strategy)
    return SplitData(strategy, train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True), cv, gcol)


# --------------------------------------------------------------------------- fitting
def fit_and_evaluate(
    key: str,
    features: list[str],
    split: SplitData,
    params: dict | None = None,
    tune: bool = True,
    n_iter: int = 20,
) -> tuple[object, dict]:
    spec = MODEL_SPECS[key]
    t0 = time.perf_counter()
    Xtr, ytr = split.train[features], split.train[TARGET]
    groups = split.cv_groups(split.train)
    pipe = make_pipeline(key, features, params)
    tuned_params: dict = {}
    cv_rmse = cv_rmse_std = cv_r2 = np.nan
    if tune and spec.tunable and spec.search_space:
        if "n_jobs" in spec.defaults:  # avoid nested parallelism during search
            pipe.set_params(model__n_jobs=1)
        search = RandomizedSearchCV(
            pipe, search_space(key), n_iter=n_iter, cv=split.cv, scoring="neg_root_mean_squared_error",
            random_state=RANDOM_STATE, n_jobs=-1, refit=True, error_score="raise",
        )
        search.fit(Xtr, ytr, groups=groups)
        pipe = search.best_estimator_
        if "n_jobs" in spec.defaults:
            pipe.set_params(model__n_jobs=spec.defaults["n_jobs"])
        tuned_params = clean_params(search.best_params_)
        i = search.best_index_
        cv_rmse = -search.cv_results_["mean_test_score"][i]
        cv_rmse_std = search.cv_results_["std_test_score"][i]
        cvr = cross_validate(make_pipeline(key, features, {**(params or {}), **tuned_params}), Xtr, ytr, groups=groups, cv=split.cv, scoring="r2", n_jobs=-1)
        cv_r2 = float(np.mean(cvr["test_score"]))
    else:
        cvr = cross_validate(pipe, Xtr, ytr, groups=groups, cv=split.cv, scoring=("neg_root_mean_squared_error", "r2"), n_jobs=-1)
        cv_rmse = float(-np.mean(cvr["test_neg_root_mean_squared_error"]))
        cv_rmse_std = float(np.std(cvr["test_neg_root_mean_squared_error"]))
        cv_r2 = float(np.mean(cvr["test_r2"]))
        pipe.fit(Xtr, ytr)
    p = len(features)
    res = {
        "key": key,
        "name": spec.name,
        "tuned": bool(tuned_params),
        "params": {**clean_params(spec.defaults), **(params or {}), **tuned_params},
        "cv_rmse_mean": float(cv_rmse),
        "cv_rmse_std": float(cv_rmse_std),
        "cv_r2_mean": float(cv_r2),
        "train": regression_metrics(ytr, pipe.predict(Xtr), p),
        "val": regression_metrics(split.val[TARGET], pipe.predict(split.val[features]), p),
        "test": regression_metrics(split.test[TARGET], pipe.predict(split.test[features]), p),
        "fit_seconds": round(time.perf_counter() - t0, 2),
    }
    return pipe, res


def cv_strategy_comparison(frame: pd.DataFrame, features: list[str], models: list[str], params: dict[str, dict], folds: int = 5) -> pd.DataFrame:
    """Same model, same data, three CV schemes → shows how optimistic random K-fold is."""
    X, y = frame[features], frame[TARGET]
    schemes = {
        "random": (KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE), None),
        "group": (GroupKFold(n_splits=folds), frame["country"].to_numpy()),
        "temporal": (ForwardYearSplit(n_splits=folds), frame["year"].to_numpy()),
    }
    rows = []
    for key in models:
        for sname, (cv, groups) in schemes.items():
            pipe = make_pipeline(key, features, params.get(key))
            r = cross_validate(pipe, X, y, groups=groups, cv=cv, scoring=("neg_root_mean_squared_error", "r2", "neg_mean_absolute_error"), n_jobs=-1)
            rmse = -r["test_neg_root_mean_squared_error"]
            rows.append({"model": MODEL_SPECS[key].name, "key": key, "strategy": sname, "rmse_mean": rmse.mean(), "rmse_std": rmse.std(),
                         "r2_mean": r["test_r2"].mean(), "r2_std": r["test_r2"].std(), "mae_mean": -r["test_neg_mean_absolute_error"].mean(),
                         "fold_rmse": rmse.round(4).tolist()})
    return pd.DataFrame(rows)


def feature_set_comparison(frame: pd.DataFrame, models: list[str], params: dict[str, dict], folds: int = 5) -> pd.DataFrame:
    """Before/after feature engineering: grouped and forward-in-time CV per feature set."""
    rows = []
    for fs in FEATURE_SETS:
        feats = feature_list(fs)
        X, y = frame[feats], frame[TARGET]
        for key in models:
            for sname, cv, groups in (
                ("group", GroupKFold(n_splits=folds), frame["country"].to_numpy()),
                ("temporal", ForwardYearSplit(n_splits=folds), frame["year"].to_numpy()),
            ):
                r = cross_validate(make_pipeline(key, feats, params.get(key)), X, y, groups=groups, cv=cv,
                                   scoring=("neg_root_mean_squared_error", "r2"), n_jobs=-1)
                rows.append({"feature_set": fs, "n_features": len(feats), "model": MODEL_SPECS[key].name, "key": key, "strategy": sname,
                             "rmse_mean": float(-r["test_neg_root_mean_squared_error"].mean()), "r2_mean": float(r["test_r2"].mean())})
    return pd.DataFrame(rows)


def conformal_quantile(abs_residuals: np.ndarray, coverage: float) -> float:
    """Split-conformal quantile with finite-sample correction."""
    n = len(abs_residuals)
    k = min(n, math.ceil((n + 1) * coverage))
    return float(np.sort(abs_residuals)[k - 1])


def conformal_calibration(frame: pd.DataFrame, features: list[str], key: str, params: dict, coverage: float, split: SplitData) -> dict:
    """Two complementary interval estimates.

    * ``q_cv``: quantile of out-of-fold |residuals| from country-grouped 5-fold CV
      on all data (CV+/jackknife-style), used for production predictions.
    * Validation: split-conformal on the group split — calibrate on the
      validation countries, measure empirical coverage on unseen test countries.
    """
    X, y, g = frame[features], frame[TARGET], frame["country"]
    oof = np.zeros(len(frame))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        m = make_pipeline(key, features, params).fit(X.iloc[tr], y.iloc[tr])
        oof[te] = m.predict(X.iloc[te])
    q_cv = conformal_quantile(np.abs(y.to_numpy() - oof), coverage)
    m = make_pipeline(key, features, params).fit(split.train[features], split.train[TARGET])
    cal_res = np.abs(split.val[TARGET].to_numpy() - m.predict(split.val[features]))
    q_split = conformal_quantile(cal_res, coverage)
    test_res = np.abs(split.test[TARGET].to_numpy() - m.predict(split.test[features]))
    return {
        "method": "Split-conformal (country-grouped)",
        "coverage_target": coverage,
        "q_cv": q_cv,
        "q_split": q_split,
        "empirical_test_coverage_split": float(np.mean(test_res <= q_split)),
        "empirical_oof_coverage_cv": float(np.mean(np.abs(y.to_numpy() - oof) <= q_cv)),
        "n_calibration": int(len(cal_res)),
        "n_test": int(len(test_res)),
    }


DIFFICULTY_PARAMS = {"n_estimators": 200, "min_samples_leaf": 10, "max_features": 0.6, "n_jobs": -1}
DIFFICULTY_FLOOR = 0.10


def extremes_report(y: np.ndarray, pred: np.ndarray) -> list[dict]:
    """Bias / error of out-of-fold predictions by *actual* score band (shows shrinkage at the extremes)."""
    rows = []
    for name, lo, hi in (("< 4 (very low)", 0, 4), ("4–5", 4, 5), ("5–6", 5, 6), ("6–7", 6, 7), ("≥ 7 (very high)", 7, 10.01)):
        m = (y >= lo) & (y < hi)
        if m.sum():
            e = y[m] - pred[m]
            rows.append({"actual_band": name, "n": int(m.sum()), "bias_actual_minus_pred": float(e.mean()),
                         "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean()))})
    return rows


def _cross_conformal_scale(X: pd.DataFrame, y: np.ndarray, g: np.ndarray, key: str, features: list[str], params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Country-grouped 5-fold out-of-fold predictions, |residuals| and out-of-fold difficulty scale."""
    oof = np.zeros(len(X))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        oof[te] = make_pipeline(key, features, params).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    absr = np.abs(y - oof)
    s_oof = np.zeros(len(X))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        s_oof[te] = make_pipeline("random_forest", features, DIFFICULTY_PARAMS).fit(X.iloc[tr], absr[tr]).predict(X.iloc[te])
    return oof, absr, np.maximum(s_oof, DIFFICULTY_FLOOR)


def adaptive_interval(frame: pd.DataFrame, features: list[str], key: str, params: dict, coverage: float) -> tuple[dict, object]:
    """Normalised (locally adaptive) split-conformal interval.

    1. Country-grouped 5-fold out-of-fold predictions of the main model → residuals r.
    2. A 'difficulty' model (random forest) learns |r| from the features; its own
       predictions are also made out-of-fold (cross-conformal).
    3. Score = |r| / difficulty; q = finite-sample 90% quantile of the scores.
    Prediction interval = ŷ ± q · difficulty(x): wider for profiles where the model
    has historically erred more (e.g. crisis-like or unusual combinations).

    ``coverage_all`` / ``fixed_coverage_all`` are measured on the same out-of-fold
    scores that set ``q``, so they equal the target by construction; the per-band
    coverages are diagnostics on those same rows. See
    :func:`adaptive_interval_holdout` for an independent check on unseen countries.
    """
    X, y, g = frame[features], frame[TARGET].to_numpy(), frame["country"].to_numpy()
    oof, absr, s_oof = _cross_conformal_scale(X, y, g, key, features, params)
    q = conformal_quantile(absr / s_oof, coverage)
    half = q * s_oof
    covered = absr <= half
    q_fixed = conformal_quantile(absr, coverage)
    fixed_cov = absr <= q_fixed
    difficulty = make_pipeline("random_forest", features, DIFFICULTY_PARAMS).fit(X, absr)
    lo, hi = y < 4, y >= 7
    info = {
        "method": "Normalised split-conformal (country-grouped cross-conformal, random-forest difficulty model)",
        "coverage_target": coverage, "q_norm": q, "floor": DIFFICULTY_FLOOR,
        "coverage_all": float(covered.mean()), "coverage_very_low": float(covered[lo].mean()), "coverage_very_high": float(covered[hi].mean()),
        "mean_halfwidth": float(half.mean()), "min_halfwidth": float(half.min()), "max_halfwidth": float(half.max()),
        "fixed_q": q_fixed, "fixed_coverage_all": float(fixed_cov.mean()), "fixed_coverage_very_low": float(fixed_cov[lo].mean()),
        "fixed_coverage_very_high": float(fixed_cov[hi].mean()),
        "oof_rmse": float(np.sqrt(np.mean((y - oof) ** 2))),
        "extremes": extremes_report(y, oof),
    }
    return info, difficulty


def adaptive_interval_holdout(split: SplitData, features: list[str], key: str, params: dict, coverage: float) -> dict:
    """Independent check of the adaptive (and fixed-width) interval on unseen countries.

    The whole interval procedure (out-of-fold residuals, difficulty model, quantile) is
    run on the train + validation countries of the country-grouped split only, then
    coverage is measured on the test countries, which play no part in calibration.
    """
    cal = pd.concat([split.train, split.val], ignore_index=True)
    X, y, g = cal[features], cal[TARGET].to_numpy(), cal["country"].to_numpy()
    _, absr, s_oof = _cross_conformal_scale(X, y, g, key, features, params)
    q = conformal_quantile(absr / s_oof, coverage)
    q_fixed = conformal_quantile(absr, coverage)
    main = make_pipeline(key, features, params).fit(X, y)
    difficulty = make_pipeline("random_forest", features, DIFFICULTY_PARAMS).fit(X, absr)
    Xt, yt = split.test[features], split.test[TARGET].to_numpy()
    r = np.abs(yt - main.predict(Xt))
    half = q * np.maximum(difficulty.predict(Xt), DIFFICULTY_FLOOR)
    covered, fixed_cov = r <= half, r <= q_fixed
    lo, hi = yt < 4, yt >= 7
    rate = lambda c, m: float(c[m].mean()) if m.any() else float("nan")  # noqa: E731
    return {
        "method": "Interval procedure fitted on train + validation countries of the grouped split; coverage measured on unseen test countries",
        "n_calibration": int(len(cal)), "n_test": int(len(yt)), "countries_test": int(split.test["country"].nunique()),
        "n_test_very_low": int(lo.sum()), "countries_test_very_low": int(split.test.loc[lo, "country"].nunique()), "n_test_very_high": int(hi.sum()),
        "coverage_all": float(covered.mean()), "coverage_very_low": rate(covered, lo), "coverage_very_high": rate(covered, hi),
        "mean_halfwidth": float(half.mean()),
        "fixed_q": q_fixed, "fixed_coverage_all": float(fixed_cov.mean()), "fixed_coverage_very_low": rate(fixed_cov, lo),
        "fixed_coverage_very_high": rate(fixed_cov, hi),
    }


def baseline_metrics(split: SplitData) -> dict:
    """Naive reference predictors (no indicators used), fitted on the training portion only.

    * ``train_mean``: every row gets the training-set mean Life Ladder.
    * ``region_mean``: every row gets its WHR region's training-set mean (global mean if the region is unseen).
    """
    mu = float(split.train[TARGET].mean())
    region_mu = split.train.groupby("region")[TARGET].mean()

    def predict(d: pd.DataFrame, by_region: bool) -> np.ndarray:
        return d["region"].map(region_mu).fillna(mu).to_numpy(dtype=float) if by_region else np.full(len(d), mu)

    out = {}
    for name, by_region in (("train_mean", False), ("region_mean", True)):
        out[name] = {part: regression_metrics(d[TARGET], predict(d, by_region)) for part, d in (("val", split.val), ("test", split.test))}
    return out


def feature_ranges(frame: pd.DataFrame, features: list[str]) -> dict:
    out = {}
    for f in features:
        if f in frame and pd.api.types.is_numeric_dtype(frame[f]):
            s = frame[f].dropna()
            out[f] = {"min": float(s.min()), "max": float(s.max()), "p01": float(s.quantile(0.01)), "p99": float(s.quantile(0.99)),
                      "median": float(s.median()), "mean": float(s.mean()), "std": float(s.std())}
        elif f in frame:
            out[f] = {"categories": sorted(frame[f].dropna().astype(str).unique().tolist())}
    return out


# --------------------------------------------------------------------------- orchestration
def run_training(
    models: list[str] | None = None,
    feature_set: str | None = None,
    tune: bool = True,
    n_iter: int | None = None,
    strategies: list[str] | None = None,
    extra_experiments: bool = True,
    save: bool = True,
    progress: ProgressCB = _noop,
    panel: pd.DataFrame | None = None,
) -> dict:
    models = [m for m in (models or MODEL_KEYS) if m in MODEL_SPECS]
    feature_set = feature_set or CFG["modeling"]["default_feature_set"]
    n_iter = n_iter or int(CFG["modeling"]["tuning_iterations"])
    strategies = strategies or STRATEGIES
    features = feature_list(feature_set)
    if panel is None:
        panel = load_bundle().panel
    frame = build_modeling_frame(panel)
    t_start = time.perf_counter()

    results: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_set": feature_set,
        "features": features,
        "target": TARGET,
        "n_rows": int(len(frame)),
        "n_countries": int(frame["country"].nunique()),
        "years": [int(frame["year"].min()), int(frame["year"].max())],
        "tuning": {"enabled": tune, "n_iter": n_iter, "method": "RandomizedSearchCV (scoring = RMSE)"},
        "experiments": {},
    }
    total_steps = len(strategies) * len(models) + (4 if extra_experiments else 2)
    step = 0
    prediction_frames = []
    fitted: dict[tuple[str, str], object] = {}
    splits: dict[str, SplitData] = {}

    for strategy in strategies:
        split = make_splits(frame, strategy)
        splits[strategy] = split
        exp = {**split.describe(), "models": {}}
        for key in models:
            progress(step / total_steps, f"[{strategy}] training {MODEL_SPECS[key].name}…")
            pipe, res = fit_and_evaluate(key, features, split, tune=tune, n_iter=n_iter)
            exp["models"][key] = res
            fitted[(strategy, key)] = pipe
            pf = split.test[[c for c in META_COLS if c in split.test]].copy()
            pf["actual"] = split.test[TARGET].to_numpy()
            pf["predicted"] = pipe.predict(split.test[features])
            pf["model"] = key
            pf["strategy"] = strategy
            prediction_frames.append(pf)
            step += 1
        exp["baselines"] = baseline_metrics(split)
        results["experiments"][strategy] = exp

    # ---- best model: lowest validation RMSE on unseen countries (group), tie-break MAE
    sel_strategy = "group" if "group" in results["experiments"] else strategies[0]
    ranked = sorted(results["experiments"][sel_strategy]["models"].values(), key=lambda r: (round(r["val"]["rmse"], 3), r["val"]["mae"]))
    best = ranked[0]
    results["best_model"] = {
        "key": best["key"],
        "name": best["name"],
        "params": best["params"],
        "selection_strategy": sel_strategy,
        "selection_rule": "Lowest validation RMSE on held-out countries (country-grouped split); MAE as tie-breaker. R² is reported but not used alone.",
        "val": best["val"],
        "test": best["test"],
        "ranking": [{"key": r["key"], "name": r["name"], "val_rmse": r["val"]["rmse"], "val_mae": r["val"]["mae"], "val_r2": r["val"]["r2"]} for r in ranked],
    }

    group_params = {k: results["experiments"][sel_strategy]["models"][k]["params"] for k in models}
    group_params = {k: {p: v for p, v in d.items() if p not in ("n_jobs",)} for k, d in group_params.items()}

    if extra_experiments:
        progress(step / total_steps, "Comparing cross-validation strategies…")
        results["cv_comparison"] = cv_strategy_comparison(frame, features, models, group_params).to_dict(orient="records")
        step += 1
        progress(step / total_steps, "Comparing feature sets (before / after feature engineering)…")
        fs_models = [m for m in ("ridge", "random_forest", "xgboost" if "xgboost" in models else "gradient_boosting") if m in models]
        results["feature_set_comparison"] = feature_set_comparison(frame, fs_models, group_params).to_dict(orient="records")
        step += 1
        progress(step / total_steps, "Comparing imputation strategies…")
        results["imputation_comparison"] = compare_imputation_strategies(frame, features).to_dict(orient="records")
    step += 1

    progress(step / total_steps, "Calibrating prediction intervals & refitting production models…")
    coverage = float(CFG["modeling"]["conformal_coverage"])
    results["conformal"] = conformal_calibration(frame, features, best["key"], group_params[best["key"]], coverage, splits.get("group") or make_splits(frame, "group"))
    results["feature_ranges"] = feature_ranges(frame, features)
    bk, bparams = best["key"], group_params[best["key"]]
    progress(min(step / total_steps, 0.97), "Fitting adaptive intervals (core model)…")
    adaptive, difficulty = adaptive_interval(frame, features, bk, bparams, coverage)
    if "group" in splits:
        adaptive["holdout"] = adaptive_interval_holdout(splits["group"], features, bk, bparams, coverage)
    results["adaptive_interval"] = adaptive
    # Optional companion model with emotional-wellbeing inputs (positive / negative affect)
    affect_feats = feature_list("core_plus_affect") if "core_plus_affect" in FEATURE_SETS else None
    affect_pack = None
    if affect_feats and feature_set != "core_plus_affect":
        progress(min(step / total_steps, 0.98), "Training companion model with emotional-wellbeing inputs…")
        a_info, a_diff = adaptive_interval(frame, affect_feats, bk, bparams, coverage)
        results["affect_model"] = {"key": bk, "name": MODEL_SPECS[bk].name, "features": affect_feats, "params": bparams, **a_info}
        affect_pack = (affect_feats, a_info, a_diff)
    results["training_seconds"] = round(time.perf_counter() - t_start, 1)

    if save:
        preds = pd.concat(prediction_frames, ignore_index=True)
        preds.to_csv(REPORTS_DIR / "test_predictions.csv", index=False)
        X_all, y_all = frame[features], frame[TARGET]
        for key in models:
            prod = make_pipeline(key, features, group_params[key]).fit(X_all, y_all)
            meta = {
                "kind": "production", "model_key": key, "model_name": MODEL_SPECS[key].name, "features": features,
                "feature_set": feature_set, "params": group_params[key], "trained_on_rows": int(len(frame)),
                "group_val_metrics": results["experiments"][sel_strategy]["models"][key]["val"],
                "group_test_metrics": results["experiments"][sel_strategy]["models"][key]["test"],
                "feature_ranges": results["feature_ranges"],
            }
            save_model(prod, f"production_{key}", meta)
            if key == best["key"]:
                save_model(prod, "production_best", {**meta, "conformal": results["conformal"], "adaptive": adaptive, "is_best": True})
                save_model(difficulty, "production_best_difficulty", {"kind": "difficulty", "model_key": "random_forest", "features": features,
                                                                       "model_name": "Difficulty model (|residual|)", "feature_set": feature_set})
        if affect_pack:
            af, ai, ad = affect_pack
            prod_a = make_pipeline(bk, af, bparams).fit(frame[af], frame[TARGET])
            save_model(prod_a, "production_best_affect", {
                "kind": "production", "model_key": bk, "model_name": MODEL_SPECS[bk].name + " + emotional wellbeing", "features": af,
                "feature_set": "core_plus_affect", "params": bparams, "trained_on_rows": int(len(frame)), "adaptive": ai,
                "feature_ranges": feature_ranges(frame, af), "conformal": {"q_cv": ai["fixed_q"]},
            })
            save_model(ad, "production_best_affect_difficulty", {"kind": "difficulty", "model_key": "random_forest", "features": af,
                                                                  "model_name": "Difficulty model (|residual|)", "feature_set": "core_plus_affect"})
        # evaluation models (fitted on training portion) for diagnostics / explainability
        for strategy in strategies:
            for key in models:
                save_model(fitted[(strategy, key)], f"eval_{strategy}_{key}", {"kind": "evaluation", "strategy": strategy, "model_key": key,
                                                                               "model_name": MODEL_SPECS[key].name, "features": features, "feature_set": feature_set})
        save_results(results)
    progress(1.0, "Training complete.")
    return results


def train_single(key: str, params: dict, feature_set: str, strategy: str, panel: pd.DataFrame, tune: bool = False, n_iter: int = 10, save_as: str | None = None) -> dict:
    """Train one model interactively (used by the Machine Learning Training page)."""
    features = feature_list(feature_set)
    frame = build_modeling_frame(panel)
    split = make_splits(frame, strategy)
    pipe, res = fit_and_evaluate(key, features, split, params=params, tune=tune, n_iter=n_iter)
    res["strategy"] = strategy
    res["feature_set"] = feature_set
    res["features"] = features
    res["split"] = split.describe()
    pf = split.test[[c for c in META_COLS if c in split.test]].copy()
    pf["actual"] = split.test[TARGET].to_numpy()
    pf["predicted"] = pipe.predict(split.test[features])
    res["_predictions"] = pf
    res["_pipeline"] = pipe
    if save_as:
        prod = make_pipeline(key, features, {k: v for k, v in res["params"].items() if k != "n_jobs"}).fit(frame[features], frame[TARGET])
        save_model(prod, save_as, {"kind": "custom", "model_key": key, "model_name": MODEL_SPECS[key].name, "features": features,
                                   "feature_set": feature_set, "params": res["params"], "strategy": strategy, "val_metrics": res["val"],
                                   "test_metrics": res["test"], "feature_ranges": feature_ranges(frame, features)})
    return res


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="Train World Happiness AI Lab models")
    ap.add_argument("--models", nargs="*", default=None, help=f"subset of {MODEL_KEYS}")
    ap.add_argument("--feature-set", default=None, choices=list(FEATURE_SETS))
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--quick", action="store_true", help="8 tuning iterations")
    ap.add_argument("--iterations", type=int, default=None)
    args = ap.parse_args()
    n_iter = 8 if args.quick else args.iterations

    def cb(f: float, m: str) -> None:
        print(f"[{f:5.0%}] {m}", flush=True)

    res = run_training(models=args.models, feature_set=args.feature_set, tune=not args.no_tune, n_iter=n_iter, progress=cb)
    print("\n=== Test RMSE by validation question ===")
    for s, exp in res["experiments"].items():
        print(f"\n{s.upper()} (train {exp['n_train']}, val {exp['n_val']}, test {exp['n_test']})")
        for k, r in exp["models"].items():
            print(f"  {r['name']:20s} val RMSE {r['val']['rmse']:.3f}  test RMSE {r['test']['rmse']:.3f}  test R² {r['test']['r2']:.3f}")
    b = res["best_model"]
    print(f"\nBest model: {b['name']} (group val RMSE {b['val']['rmse']:.3f}, test RMSE {b['test']['rmse']:.3f})")
    cf = res["conformal"]
    print(f"Split-conformal 90% half-width ±{cf['q_split']:.3f} (validation countries): coverage {cf['empirical_test_coverage_split']:.1%} on unseen test countries; "
          f"production half-width ±{cf['q_cv']:.3f}")
    ai = res["adaptive_interval"]
    print(f"Adaptive 90% interval (cross-conformal, all data): coverage {ai['coverage_all']:.1%} by construction (very low scores {ai['coverage_very_low']:.0%} vs fixed "
          f"{ai['fixed_coverage_very_low']:.0%}); half-width {ai['min_halfwidth']:.2f}–{ai['max_halfwidth']:.2f}")
    if "holdout" in ai:
        h = ai["holdout"]
        print(f"Adaptive interval held-out check ({h['countries_test']} unseen countries): coverage {h['coverage_all']:.1%}; very low scores "
              f"{h['coverage_very_low']:.0%} vs fixed {h['fixed_coverage_very_low']:.0%} (n = {h['n_test_very_low']} rows)")
    for s, exp in res["experiments"].items():
        bl = exp.get("baselines", {})
        if bl:
            print(f"Naive baselines, {s} test RMSE: training mean {bl['train_mean']['test']['rmse']:.3f}, region mean {bl['region_mean']['test']['rmse']:.3f}")
    if "affect_model" in res:
        print(f"Companion model with emotional wellbeing: grouped OOF RMSE {res['affect_model']['oof_rmse']:.3f} vs core {ai['oof_rmse']:.3f}")
    print(f"Total time {res['training_seconds']} s")


if __name__ == "__main__":  # pragma: no cover
    main()
