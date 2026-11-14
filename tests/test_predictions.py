import numpy as np
import plotly.graph_objects as go
import pytest

from src import visualization as viz
from src.config import INDICATORS, TARGET
from src.eda import correlation_matrix, dataset_overview, ranking_overview, region_summary
from src.explainability import explain_instance, native_importance, pdp, permutation_importance_df, shap_explanation
from src.insight_engine import generate_insights
from src.prediction import input_specs, mahalanobis_extrapolation, nearest_profiles, predict_one, validate_inputs
from src.train import feature_ranges


@pytest.fixture(scope="module")
def specs(frame):
    return input_specs(feature_ranges(frame, INDICATORS), INDICATORS)


def test_input_specs_grounded_in_data(specs, frame):
    s = specs["social_support"]
    assert s.hard_min >= 0 and s.hard_max <= 1
    assert s.train_min == pytest.approx(frame["social_support"].min())
    assert s.train_min <= s.default <= s.train_max


def test_validation_rejects_nonsense(specs):
    vals = {f: s.default for f, s in specs.items()}
    assert validate_inputs(vals, specs) == ([], [])
    bad = {**vals, "social_support": 1.7}
    errs, _ = validate_inputs(bad, specs)
    assert errs
    errs, _ = validate_inputs({**vals, "freedom": None}, specs)
    assert errs
    ext = {**vals, "healthy_life_expectancy": specs["healthy_life_expectancy"].train_max + 1}
    errs, warns = validate_inputs(ext, specs)
    assert not errs and warns


def test_prediction_numeric_in_range(ridge_pipeline, forest_pipeline, specs):
    vals = {f: s.default for f, s in specs.items()}
    for pipe in (ridge_pipeline, forest_pipeline):
        out = predict_one(pipe, vals, INDICATORS, conformal_q=0.9)
        assert isinstance(out["prediction"], float) and 0 <= out["prediction"] <= 10
        assert out["lower"] <= out["prediction"] <= out["upper"]
        assert 0 <= out["lower"] and out["upper"] <= 10


def test_prediction_monotone_in_income_for_linear(ridge_pipeline, specs):
    vals = {f: s.default for f, s in specs.items()}
    lo = predict_one(ridge_pipeline, {**vals, "log_gdp_per_capita": 8}, INDICATORS)["prediction"]
    hi = predict_one(ridge_pipeline, {**vals, "log_gdp_per_capita": 11}, INDICATORS)["prediction"]
    assert hi > lo


def test_nearest_and_extrapolation(panel, specs):
    vals = {f: s.default for f, s in specs.items()}
    nn = nearest_profiles(panel, vals, INDICATORS, 5)
    assert len(nn) == 5 and nn["distance"].is_monotonic_increasing
    m = mahalanobis_extrapolation(panel, vals, INDICATORS)
    assert 0 <= m["percentile"] <= 100 and not m["unusual"]
    weird = {**vals, "log_gdp_per_capita": 11.5, "healthy_life_expectancy": 45, "social_support": 0.4}
    assert mahalanobis_extrapolation(panel, weird, INDICATORS)["percentile"] > 99


def test_explainability(forest_pipeline, ridge_pipeline, frame):
    X, y = frame[INDICATORS].iloc[:300], frame[TARGET].iloc[:300]
    pi = permutation_importance_df(forest_pipeline, X, y, n_repeats=2)
    assert set(pi["feature"]) == set(INDICATORS)
    assert native_importance(forest_pipeline)["importance"].sum() == pytest.approx(1.0)
    assert "signed_coef" in native_importance(ridge_pipeline)
    e = shap_explanation(forest_pipeline, frame[INDICATORS], X.iloc[:20])
    assert e is not None and e["values"].shape == (20, len(INDICATORS))
    recon = e["values"].sum(axis=1) + e["base_value"]
    np.testing.assert_allclose(recon, forest_pipeline.predict(X.iloc[:20]), atol=0.05)
    d = pdp(forest_pipeline, frame[INDICATORS], "social_support", grid_resolution=10, ice=True)
    assert len(d["grid"]) == len(d["average"]) and d["individual"].shape[1] == len(d["grid"])
    ex = explain_instance(ridge_pipeline, frame[INDICATORS], X.iloc[[0]])
    assert abs(ex["contributions"]["contribution"].sum() + ex["base_value"] - ex["prediction"]) < 0.05


def test_map_data_preparation(bundle):
    y = int(bundle.rankings["year"].max())
    d = bundle.rankings[bundle.rankings["year"] == y]
    fig = viz.choropleth(d, "ladder_score", hover_cols=["region"])
    assert isinstance(fig, go.Figure)
    locs = fig.data[0].locations
    assert len(locs) == d["iso3"].notna().sum() and all(isinstance(l, str) and len(l) == 3 for l in locs)
    assert len(set(locs)) == len(locs)


def test_visualizations_receive_valid_inputs(panel, bundle, frame, ridge_pipeline):
    res = frame[["country", "year", "region"]].assign(actual=frame[TARGET], predicted=ridge_pipeline.predict(frame[INDICATORS]))
    res["residual"] = res["actual"] - res["predicted"]
    figs = [
        viz.histogram(panel[TARGET]), viz.density(panel, TARGET), viz.box_by(panel, TARGET, "region"), viz.year_violin(panel, TARGET),
        viz.scatter_trend(panel, "log_gdp_per_capita", TARGET, highlight=["Western Europe"], lowess=True),
        viz.heatmap(correlation_matrix(panel)), viz.missing_heatmap(panel, INDICATORS), viz.region_bars(region_summary(panel, TARGET)),
        viz.trend_lines(panel[panel["country"].isin(["Finland", "India"])], TARGET), viz.actual_vs_predicted(res), viz.residual_hist(res),
        viz.residual_vs_predicted(res), viz.gauge(5.5, 4.6, 6.4), viz.scenario_compare(["a", "b"], [5.0, 6.0], 5.5),
    ]
    assert all(isinstance(f, go.Figure) and len(f.data) > 0 for f in figs)
    cats = viz.color_map([f"c{i}" for i in range(12)])
    assert len(set(list(cats.values())[:8])) == 8 and set(list(cats.values())[8:]) == {viz.OTHER_COLOR}


def test_overview_and_insights_from_real_data(bundle):
    ov = dataset_overview(bundle.panel)
    assert ov["observations"] == len(bundle.panel) and ov["max_score"] == bundle.panel[TARGET].max()
    ro = ranking_overview(bundle.rankings)
    assert ro["edition"] == ro["year"] + 1
    from src.feature_engineering import engineer_panel

    ins = generate_insights(bundle.panel, engineer_panel(bundle.panel), bundle.rankings, bundle.panel_source, bundle.rankings_source)
    assert len(ins) >= 8
    assert all(i.source and i.metric and i.value for i in ins)
    top = bundle.rankings[bundle.rankings["year"] == ro["year"]].nlargest(1, "ladder_score")["country"].iloc[0]
    assert top in ins[0].text  # ranking insight reflects the actual data


def test_adaptive_interval_calibrated_and_wider_for_hard_cases(frame):
    from src.train import adaptive_interval

    small = frame.sample(900, random_state=3)
    info, diff = adaptive_interval(small, INDICATORS, "ridge", {}, 0.9)
    assert 0.85 <= info["coverage_all"] <= 0.95
    assert info["max_halfwidth"] > info["min_halfwidth"] > 0
    assert {r["actual_band"] for r in info["extremes"]} >= {"< 4 (very low)", "≥ 7 (very high)"}
    s = diff.predict(frame[INDICATORS].iloc[:20])
    assert np.isfinite(s).all() and (s >= 0).all()


def test_adaptive_prediction_and_extreme_warning(ridge_pipeline, frame, specs):
    from src.prediction import extreme_warning
    from src.train import adaptive_interval

    info, diff = adaptive_interval(frame.sample(700, random_state=4), INDICATORS, "ridge", {}, 0.9)
    vals = {f: s.default for f, s in specs.items()}
    out = predict_one(ridge_pipeline, vals, INDICATORS, difficulty=diff, adaptive=info)
    assert out["lower"] < out["prediction"] < out["upper"]
    assert extreme_warning(3.5, info) and extreme_warning(7.2, info) and extreme_warning(5.5, info) is None
