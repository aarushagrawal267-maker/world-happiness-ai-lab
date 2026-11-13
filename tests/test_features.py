import numpy as np

from src.config import FEATURE_SETS, INDICATORS, LEAKAGE_COLUMNS, TARGET
from src.feature_engineering import (
    FEATURE_DOCS,
    CompositeIndexTransformer,
    add_categories,
    add_gdp_gap,
    add_time_features,
    composite_index,
    engineer_panel,
)


def test_change_features_only_for_consecutive_years(toy_panel):
    out = add_time_features(toy_panel).set_index(["country", "year"])
    assert np.isnan(out.loc[("A", 2010), "gdp_growth_1y"])
    assert out.loc[("A", 2011), "gdp_growth_1y"] == np.float64(0.1).round(10) or abs(out.loc[("A", 2011), "gdp_growth_1y"] - 0.1) < 1e-9
    assert np.isnan(out.loc[("A", 2014), "gdp_growth_1y"])  # 2013 missing → no gap-bridging
    assert np.isnan(out.loc[("C", 2012), "ladder_yoy_change"])
    assert abs(out.loc[("C", 2013), "ladder_yoy_change"] - 0.1) < 1e-9


def test_time_features_use_past_only(panel):
    out = add_time_features(panel)
    fin = out[out["country"] == "Finland"].sort_values("year")
    manual = fin["log_gdp_per_capita"].diff()
    consecutive = fin["year"].diff() == 1
    np.testing.assert_allclose(fin.loc[consecutive, "gdp_growth_1y"], manual[consecutive])


def test_categories_use_reference_thresholds(panel):
    ref = panel[panel["year"] <= 2015]
    out = add_categories(panel, reference=ref)
    q = ref["log_gdp_per_capita"].quantile([0.25, 0.5, 0.75])
    top = out[out["gdp_category"] == "Q4 (highest)"]["log_gdp_per_capita"]
    assert (top > q.loc[0.75]).all()
    assert set(out["happiness_category"].dropna().unique()) <= {"Low (<4.5)", "Lower-middle (4.5–5.5)", "Upper-middle (5.5–6.5)", "High (≥6.5)"}


def test_target_derived_features_are_marked_as_leakage():
    for f in ("ladder_yoy_change", "happiness_category"):
        assert f in LEAKAGE_COLUMNS
        assert "NO" in FEATURE_DOCS[f]["model_input"]
    for fs in FEATURE_SETS.values():
        assert not set(fs) & LEAKAGE_COLUMNS
        assert TARGET not in fs


def test_engineered_features_documented():
    for f in FEATURE_SETS["engineered"]:
        assert f in INDICATORS or f in FEATURE_DOCS


def test_composite_transformer_train_only(frame):
    X = frame[INDICATORS].fillna(frame[INDICATORS].median())
    t = CompositeIndexTransformer(columns=INDICATORS).fit(X.iloc[:500])
    np.testing.assert_allclose(t.mean_, X.iloc[:500].mean().to_numpy())
    out = t.transform(X)
    assert out.shape == (len(X), len(INDICATORS) + 1)
    assert list(t.get_feature_names_out(INDICATORS))[-1] == "composite_index"
    assert np.corrcoef(out[:, -1], frame[TARGET])[0, 1] > 0.6


def test_composite_index_direction(panel):
    ci = composite_index(panel)
    assert np.corrcoef(ci.fillna(0), panel["perceptions_of_corruption"].fillna(panel["perceptions_of_corruption"].mean()))[0, 1] < 0


def test_engineer_panel_and_gdp_gap(panel):
    out = engineer_panel(panel)
    for c in ("gdp_growth_1y", "ladder_yoy_change", "gdp_category", "happiness_category", "ladder_vs_gdp_residual"):
        assert c in out
    assert len(out) == len(panel)
    assert abs(add_gdp_gap(panel)["ladder_vs_gdp_residual"].mean()) < 0.05
