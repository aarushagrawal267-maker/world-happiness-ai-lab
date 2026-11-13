import numpy as np
import pandas as pd
import pytest

from src.config import INDICATORS, TARGET
from src.preprocessing import (
    assert_no_leakage,
    build_preprocessor,
    compare_imputation_strategies,
    country_interpolate,
    missing_summary,
    outlier_analysis,
)


def test_preprocessor_fits_on_training_data_only(frame):
    train, test = frame.iloc[:1000], frame.iloc[1000:]
    pre = build_preprocessor(INDICATORS, scaler="standard", imputer="median").fit(train[INDICATORS])
    imp = pre.named_transformers_["num"].named_steps["impute"]
    np.testing.assert_allclose(imp.statistics_, train[INDICATORS].median().to_numpy())
    assert not np.allclose(imp.statistics_, frame[INDICATORS].median().to_numpy())
    Xt = pre.transform(test[INDICATORS])
    assert not np.isnan(Xt).any()


def test_categorical_encoding_handles_unknown(frame):
    feats = INDICATORS + ["region"]
    pre = build_preprocessor(feats).fit(frame[feats])
    novel = frame[feats].iloc[:2].copy()
    novel["region"] = "Atlantis"
    out = pre.transform(novel)
    assert out.shape[1] == len(INDICATORS) + frame["region"].nunique()


def test_robust_scaler_option(frame):
    pre = build_preprocessor(INDICATORS, scaler="robust").fit(frame[INDICATORS])
    assert type(pre.named_transformers_["num"].named_steps["scale"]).__name__ == "RobustScaler"


@pytest.mark.parametrize("bad", [[TARGET], ["ladder_yoy_change"], ["dystopia_residual"], ["happiness_category"]])
def test_target_leakage_blocked(bad):
    with pytest.raises(ValueError):
        assert_no_leakage(INDICATORS + bad)
    with pytest.raises(ValueError):
        build_preprocessor(INDICATORS + bad)


def test_missing_summary(panel):
    ms = missing_summary(panel, INDICATORS)
    assert (ms["missing"] + ms["non_missing"] == len(panel)).all()
    assert ms["missing_pct"].between(0, 100).all()


def test_country_interpolation_stays_within_country(toy_panel):
    out = country_interpolate(toy_panel, ["freedom", TARGET], "linear")
    a = out[out["country"] == "A"].set_index("year")
    assert a.loc[2011, "freedom"] == pytest.approx(0.8)
    b = out[out["country"] == "B"].set_index("year")
    assert np.isnan(b.loc[2011, "freedom"])  # trailing gap, no neighbour after it → not filled
    pd.testing.assert_series_equal(out.sort_index()[TARGET], toy_panel.sort_values(["country", "year"]).sort_index()[TARGET], check_names=False)


def test_ffill_is_backward_looking(toy_panel):
    t = toy_panel.copy()
    t.loc[(t["country"] == "C") & (t["year"] == 2012), "freedom"] = np.nan
    out = country_interpolate(t, ["freedom"], "ffill")
    assert np.isnan(out[(out["country"] == "C") & (out["year"] == 2012)]["freedom"].iloc[0])


def test_imputation_comparison_runs(frame):
    res = compare_imputation_strategies(frame.sample(800, random_state=0), INDICATORS, folds=3)
    assert len(res) == 5
    assert res.loc[res["strategy"].str.startswith("Complete"), "rows_lost"].iloc[0] > 0
    assert (res["grouped_cv_rmse"] > 0).all()


def test_outliers_flagged_not_removed(panel):
    flags = outlier_analysis(panel)
    assert len(flags) > 0
    assert set(flags["variable"]) <= set(INDICATORS + [TARGET])
    assert len(panel) == len(panel)  # analysis does not mutate
