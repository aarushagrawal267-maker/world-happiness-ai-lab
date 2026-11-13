import math

import pytest

from src.config import INDICATORS
from src.prediction import input_specs
from src.scenario_simulator import (
    clamp_to_hard_limits,
    extrapolation_warnings,
    gdp_percent_to_log_delta,
    log_delta_to_gdp_percent,
    one_at_a_time,
    scenarios_table,
    simulate,
    tornado,
)
from src.train import feature_ranges


@pytest.fixture(scope="module")
def setup(frame):
    ranges = feature_ranges(frame, INDICATORS)
    specs = input_specs(ranges, INDICATORS)
    base = frame[frame["country"] == "India"].sort_values("year").iloc[-1][INDICATORS].fillna(frame[INDICATORS].median()).to_dict()
    return ranges, specs, base


def test_gdp_percent_conversion():
    assert gdp_percent_to_log_delta(10) == pytest.approx(math.log(1.1))
    assert log_delta_to_gdp_percent(gdp_percent_to_log_delta(37.5)) == pytest.approx(37.5)


def test_no_change_means_no_difference(ridge_pipeline, setup):
    _, specs, base = setup
    s = simulate(ridge_pipeline, INDICATORS, base, dict(base), specs)
    assert s.delta == pytest.approx(0.0) and s.changes().empty and not s.warnings


def test_scenario_direction_linear(ridge_pipeline, setup):
    _, specs, base = setup
    mod = {**base, "social_support": base["social_support"] + 0.1}
    s = simulate(ridge_pipeline, INDICATORS, base, mod, specs, name="more support")
    assert s.delta > 0
    ch = s.changes()
    assert list(ch["feature"]) == ["social_support"] and ch["change"].iloc[0] == pytest.approx(0.1)


def test_extrapolation_flagged_and_clamped(forest_pipeline, setup):
    _, specs, base = setup
    mod = {**base, "healthy_life_expectancy": 200.0}
    clamped = clamp_to_hard_limits(mod, specs)
    assert clamped["healthy_life_expectancy"] == specs["healthy_life_expectancy"].hard_max
    s = simulate(forest_pipeline, INDICATORS, base, mod, specs)
    assert s.warnings and 0 <= s.modified_prediction <= 10
    assert extrapolation_warnings(base, specs) == []


def test_sensitivity_tools(forest_pipeline, setup):
    ranges, specs, base = setup
    curve = one_at_a_time(forest_pipeline, INDICATORS, base, specs, "log_gdp_per_capita", n=15)
    assert len(curve) == 15 and curve["value"].is_monotonic_increasing
    t = tornado(forest_pipeline, INDICATORS, base, specs, ranges)
    assert set(t["feature"]) == set(INDICATORS) and (t["range"] >= 0).all()


def test_multiple_scenarios_table(ridge_pipeline, setup):
    _, specs, base = setup
    sc = [simulate(ridge_pipeline, INDICATORS, base, {**base, "freedom": base["freedom"] + d}, specs, name=f"s{i}") for i, d in enumerate([0.05, -0.05])]
    tbl = scenarios_table(sc)
    assert list(tbl["scenario"]) == ["s0", "s1"]
    assert tbl["difference"].iloc[0] > 0 > tbl["difference"].iloc[1]
