import numpy as np
import pandas as pd
import pytest

from src.config import ALL_NUMERIC, INDICATORS, TARGET, VALID_RANGES
from src.data_acquisition import SOURCES, verify_panel_against_official
from src.data_cleaning import (
    CleaningLog,
    clean_frame,
    deduplicate_country_year,
    invalidate_out_of_range,
    parse_edition_from_filename,
    parse_year,
    reclassify_contribution_columns,
    standardize_columns,
)
from src.data_loader import ingest_upload
from src.data_validation import check_no_leakage, detect_kind, validate_dataset
from src.geography import standardize_country_name


# ---------------------------------------------------------------- loading
def test_bundle_loads_real_data(bundle):
    assert len(bundle.panel) > 2000
    assert bundle.panel["country"].nunique() > 150
    assert bundle.panel["year"].min() == 2005
    assert not bundle.rankings.empty and bundle.rankings["year"].max() >= 2024
    assert len(bundle.editions) >= 5


def test_every_file_detected_and_valid(bundle):
    kinds = {f.name: f.kind for f in bundle.files}
    assert "annual_panel" in kinds.values()
    assert "ranking_panel" in kinds.values()
    assert all(k != "unknown" for k in kinds.values()), kinds
    assert all(f.report.is_valid for f in bundle.files)


def test_panel_schema_and_ranges(panel):
    for col in ["country", "iso3", "region", "year", TARGET] + ALL_NUMERIC:
        assert col in panel.columns
    for col, (lo, hi) in VALID_RANGES.items():
        if col in panel and col != "year":
            v = panel[col].dropna()
            assert v.between(lo, hi).all(), col
    assert panel[TARGET].between(0, 10).all()
    assert panel[TARGET].notna().all()


def test_no_duplicate_country_years(bundle):
    assert not bundle.panel.duplicated(["country", "year"]).any()
    assert not bundle.rankings.duplicated(["country", "year"]).any()


def test_geography_attached(panel, bundle):
    assert panel["region"].notna().mean() == 1.0
    assert panel["iso3"].notna().mean() > 0.98
    assert bundle.rankings["region"].notna().all()


def test_panel_matches_official_whr2023():
    r = verify_panel_against_official()
    assert r["status"] == "passed"
    assert r["pearson_r"] > 0.999


def test_sources_have_provenance():
    for s in SOURCES:
        assert s.url.startswith("https://") and s.publisher and s.methodology_note
        assert s.path.exists(), f"missing {s.local_path}"


# ---------------------------------------------------------------- standardisation
@pytest.mark.parametrize("raw,expected", [
    ("Turkey", "Türkiye"), ("Turkiye", "Türkiye"), ("Türkiye", "Türkiye"),
    ("Czech Republic", "Czechia"), ("DR Congo", "Congo (Kinshasa)"), ("Congo", "Congo (Brazzaville)"),
    ("Hong Kong S.A.R., China", "Hong Kong S.A.R. of China"), ("Botswana*", "Botswana"),
    ("  Finland ", "Finland"), ("Côte d’Ivoire", "Ivory Coast"), ("Swaziland", "Eswatini"),
    ("Palestinian Territories", "State of Palestine"), ("xx", None), (None, None), (np.nan, None),
])
def test_country_standardisation(raw, expected):
    assert standardize_country_name(raw) == expected


def test_column_aliases_cover_editions():
    df = pd.DataFrame(columns=["Country name", "Ladder score", "Logged GDP per capita", "Explained by: Social support", "Dystopia (1.83) + residual", "upperwhisker"])
    std, mapping = standardize_columns(df)
    assert list(std.columns) == ["country", "ladder_score", "log_gdp_per_capita", "exp_social_support", "dystopia_residual", "upper_whisker"]


def test_kaggle_contribution_columns_reclassified():
    df = pd.DataFrame({"Country or region": ["X", "Y"], "Score": [7.0, 5.0], "GDP per capita": [1.3, 0.8],
                       "Social support": [1.5, 1.1], "Healthy life expectancy": [0.9, 0.6]})
    std, _ = standardize_columns(df)
    std = reclassify_contribution_columns(std)
    assert "exp_gdp" in std and "log_gdp_per_capita" not in std
    assert "exp_social_support" in std and "exp_health" in std


@pytest.mark.parametrize("v,expected", [(2019, 2019), ("2019", 2019), (2019.0, 2019), ("2017-2019", 2019), ("WHR2020", 2020)])
def test_year_parsing(v, expected):
    assert parse_year(v) == expected


def test_year_parsing_invalid():
    assert np.isnan(parse_year("n/a"))


@pytest.mark.parametrize("name,ed", [("WHR20_Data_Figure_2.1.xls", 2020), ("2015.csv", 2015), ("WHR2023.csv", 2023), ("WHR26_Data_Figure_2.1.xlsx", 2026)])
def test_edition_from_filename(name, ed):
    assert parse_edition_from_filename(name) == ed


# ---------------------------------------------------------------- cleaning & validation
def test_out_of_range_set_to_missing_not_clipped():
    df = pd.DataFrame({"country": ["A", "B"], "year": [2010, 2010], "social_support": [1.4, 0.8], "life_ladder": [5, 11]})
    log = CleaningLog()
    out = invalidate_out_of_range(df, log)
    assert np.isnan(out.loc[0, "social_support"]) and out.loc[1, "social_support"] == 0.8
    assert np.isnan(out.loc[1, "life_ladder"])
    assert log.invalid_values == {"social_support": 1, "life_ladder": 1}


def test_deduplication_exact_and_conflicting():
    df = pd.DataFrame({"country": ["A", "A", "B", "B"], "year": [2010, 2010, 2011, 2011], "x": [1.0, 1.0, np.nan, 2.0]})
    log = CleaningLog()
    out = deduplicate_country_year(df, log=log)
    assert len(out) == 2
    assert log.exact_duplicates_removed == 1 and log.conflicting_duplicates == 1
    assert out.loc[out["country"] == "B", "x"].iloc[0] == 2.0  # most complete row kept


def test_validation_flags_duplicates_and_ranges():
    df = pd.DataFrame({"country": ["A", "A"], "year": [2010, 2010], "life_ladder": [5.0, 12.0], "log_gdp_per_capita": [9, 9],
                       "social_support": [0.8, 0.8], "freedom": [0.7, 0.7]})
    rep = validate_dataset(df)
    checks = {i.check for i in rep.issues}
    assert "duplicates" in checks and "range" in checks and not rep.is_valid


def test_detect_kind():
    panel_like = pd.DataFrame({"country": ["A", "A"], "year": [2010, 2011], "life_ladder": [5, 6], "log_gdp_per_capita": [9, 9],
                               "social_support": [0.8, 0.8], "freedom": [0.7, 0.7]})
    assert detect_kind(panel_like) == "annual_panel"
    assert detect_kind(pd.DataFrame({"foo": [1]})) == "unknown"


def test_missing_values_reported(panel):
    rep = validate_dataset(panel, "panel")
    assert rep.missing_counts.get("perceptions_of_corruption", 0) > 0
    assert TARGET not in rep.missing_counts


def test_leakage_checker():
    assert check_no_leakage(["log_gdp_per_capita", "life_ladder", "dystopia_residual", "ladder_yoy_change"]) == ["life_ladder", "dystopia_residual", "ladder_yoy_change"]
    assert check_no_leakage(INDICATORS) == []


def test_upload_csv_detected_and_validated(tmp_path):
    csv = "Country name,year,Life Ladder,Log GDP per capita,Social support,Freedom to make life choices\nFinland,2020,7.8,10.8,0.95,0.95\nFinland,2021,7.7,10.8,0.96,0.94\nTurkey,2021,4.7,10.2,0.8,0.6\n"
    lf = ingest_upload(csv.encode(), "my_panel.csv", save=False)
    assert lf.kind == "annual_panel" and lf.report.is_valid
    assert "Türkiye" in set(lf.frame["country"])
    with pytest.raises(ValueError):
        ingest_upload(b"x", "bad.txt", save=False)


def test_clean_frame_drops_invalid_country_rows():
    df = pd.DataFrame({"country": ["Finland", "xx", None], "year": ["2019", "2019", "2019"], "life_ladder": ["7.8", "5", "5"]})
    log = CleaningLog()
    out = clean_frame(df, log)
    assert list(out["country"]) == ["Finland"] and log.dropped_invalid_country == 2
    assert out["year"].dtype.kind == "i" and out["life_ladder"].dtype.kind == "f"
