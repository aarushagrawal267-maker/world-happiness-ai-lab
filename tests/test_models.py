import numpy as np
import pytest

from src.config import INDICATORS, TARGET
from src.evaluate import ForwardYearSplit, adjusted_r2, error_by_group, error_by_score_range, regression_metrics, residual_frame
from src.regression_models import MODEL_KEYS, MODEL_SPECS, XGBOOST_AVAILABLE, make_pipeline, search_space
from src.statistical_analysis import correlation_tests, group_comparison, ols_regression, country_level
from src.train import conformal_quantile, fit_and_evaluate, make_splits


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_every_model_trains_and_predicts(key, frame):
    small = {"random_forest": {"n_estimators": 30, "n_jobs": 1}, "gradient_boosting": {"n_estimators": 50}, "xgboost": {"n_estimators": 50}}.get(key, {})
    d = frame.sample(600, random_state=1)
    pipe = make_pipeline(key, INDICATORS, small).fit(d[INDICATORS], d[TARGET])
    pred = pipe.predict(frame[INDICATORS].iloc[:50])
    assert pred.shape == (50,) and np.isfinite(pred).all()
    assert pred.min() > -1 and pred.max() < 11


def test_xgboost_registered_when_available():
    assert ("xgboost" in MODEL_SPECS) == XGBOOST_AVAILABLE


def test_search_spaces_prefixed():
    for k in ("ridge", "lasso", "random_forest", "gradient_boosting"):
        assert all(p.startswith("model__") for p in search_space(k))
    assert set(MODEL_SPECS["random_forest"].search_space) >= {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features"}
    assert set(MODEL_SPECS["gradient_boosting"].search_space) >= {"n_estimators", "learning_rate", "max_depth", "subsample"}


def test_group_split_has_no_country_overlap(frame):
    s = make_splits(frame, "group")
    assert not set(s.train["country"]) & set(s.test["country"])
    assert not set(s.train["country"]) & set(s.val["country"])
    assert len(s.train) + len(s.val) + len(s.test) == len(frame)


def test_temporal_split_is_forward_in_time(frame):
    s = make_splits(frame, "temporal")
    assert s.train["year"].max() < s.val["year"].min() <= s.val["year"].max() < s.test["year"].min()


def test_forward_year_split_never_trains_on_future(frame):
    years = frame["year"].to_numpy()
    for tr, va in ForwardYearSplit(n_splits=4).split(frame, groups=years):
        assert years[tr].max() < years[va].min()


def test_fit_and_evaluate_with_tuning(frame):
    split = make_splits(frame, "group")
    pipe, res = fit_and_evaluate("ridge", INDICATORS, split, tune=True, n_iter=4)
    assert res["tuned"] and "alpha" in res["params"]
    for part in ("train", "val", "test"):
        assert {"r2", "mae", "mse", "rmse", "adj_r2"} <= set(res[part])
        assert 0 < res[part]["rmse"] < 1.5
    assert np.isfinite(res["cv_rmse_mean"])


def test_random_split_is_more_optimistic_than_group(frame):
    from src.train import cv_strategy_comparison

    r = cv_strategy_comparison(frame, INDICATORS, ["random_forest"], {"random_forest": {"n_estimators": 60}}, folds=4)
    rm = r.set_index("strategy")["rmse_mean"]
    assert rm["random"] < rm["group"]


def test_metrics():
    y, p = np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0, 3.0, 5.0])
    m = regression_metrics(y, p, n_features=1)
    assert m["mae"] == 0.25 and m["mse"] == 0.25 and m["rmse"] == 0.5 and m["max_error"] == 1.0
    assert m["bias"] == 0.25
    assert m["adj_r2"] == pytest.approx(adjusted_r2(m["r2"], 4, 1))


def test_error_breakdowns(frame, ridge_pipeline):
    res = residual_frame(frame[["country", "year", "region"]], frame[TARGET], ridge_pipeline.predict(frame[INDICATORS]))
    by_r = error_by_group(res, "region")
    assert by_r["n"].sum() == len(frame)
    by_s = error_by_score_range(res)
    assert len(by_s) >= 4


def test_conformal_quantile():
    r = np.arange(1, 101, dtype=float)
    assert conformal_quantile(r, 0.9) == 91.0


def test_model_persistence_roundtrip(tmp_path, monkeypatch, ridge_pipeline, frame):
    from src import model_manager as mm

    monkeypatch.setattr(mm, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(mm, "REGISTRY_FILE", tmp_path / "registry.json")
    mm.save_model(ridge_pipeline, "unit_test_model", {"features": INDICATORS})
    loaded = mm.load_model("unit_test_model")
    np.testing.assert_allclose(loaded.predict(frame[INDICATORS].iloc[:5]), ridge_pipeline.predict(frame[INDICATORS].iloc[:5]))
    assert "unit_test_model" in mm.list_models()
    mm.delete_model("unit_test_model")
    assert mm.load_model("unit_test_model") is None


# ---------------------------------------------------------------- statistics
def test_correlation_tests_match_scipy(panel):
    from scipy import stats

    ct = correlation_tests(panel, TARGET, ["log_gdp_per_capita"]).iloc[0]
    d = panel[[TARGET, "log_gdp_per_capita"]].dropna()
    r, p = stats.pearsonr(d["log_gdp_per_capita"], d[TARGET])
    assert ct["pearson_r"] == pytest.approx(r) and ct["n"] == len(d)
    assert ct["pearson_ci_low"] < r < ct["pearson_ci_high"]


def test_ols_matches_sklearn(panel):
    from sklearn.linear_model import LinearRegression

    r = ols_regression(panel, TARGET, INDICATORS)
    d = panel[[TARGET] + INDICATORS].dropna()
    lr = LinearRegression().fit(d[INDICATORS], d[TARGET])
    assert r["r2"] == pytest.approx(lr.score(d[INDICATORS], d[TARGET]))
    assert r["adj_r2"] < r["r2"] and r["n"] == len(d)
    assert {"cluster_std_err", "hc3_std_err", "ci_low", "ci_high", "p_value"} <= set(r["table"].columns)
    assert len(r["diagnostics"]["vif"]) == len(INDICATORS)


def test_group_comparison(panel):
    cl = country_level(panel, TARGET, "region", (2019, 2021))
    assert not cl.duplicated("country").any()
    g = group_comparison(cl, TARGET, "region")
    assert 0 <= g["anova_p"] <= 1 and 0 <= g["kruskal_p"] <= 1
    assert 0 < g["eta_squared"] < 1
    assert "mann_whitney_p_holm" in g["pairwise"] and (g["pairwise"]["mann_whitney_p_holm"] >= g["pairwise"]["mann_whitney_p"]).all()
