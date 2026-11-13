"""Shared fixtures. Real WHR data is used wherever possible; the few small
synthetic frames below are clearly-labelled *test fixtures* for unit tests of
edge cases (they are never shown in the app or used for analysis)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import INDICATORS, TARGET  # noqa: E402
from src.data_loader import load_bundle  # noqa: E402
from src.train import build_modeling_frame  # noqa: E402


@pytest.fixture(scope="session")
def bundle():
    return load_bundle(use_cache=False)


@pytest.fixture(scope="session")
def panel(bundle):
    return bundle.panel


@pytest.fixture(scope="session")
def frame(panel):
    return build_modeling_frame(panel)


@pytest.fixture(scope="session")
def ridge_pipeline(frame):
    from src.regression_models import make_pipeline

    return make_pipeline("ridge", INDICATORS).fit(frame[INDICATORS], frame[TARGET])


@pytest.fixture(scope="session")
def forest_pipeline(frame):
    from src.regression_models import make_pipeline

    return make_pipeline("random_forest", INDICATORS, {"n_estimators": 40, "n_jobs": 1}).fit(frame[INDICATORS], frame[TARGET])


@pytest.fixture()
def toy_panel():
    """Synthetic fixture: 3 countries with gaps, used only for edge-case unit tests."""
    rows = []
    for c, years in {"A": [2010, 2011, 2012, 2014], "B": [2010, 2011], "C": [2012, 2013, 2014]}.items():
        for i, y in enumerate(years):
            rows.append({"country": c, "year": y, "region": "Western Europe", TARGET: 5 + i * 0.1,
                         "log_gdp_per_capita": 9 + 0.1 * i, "social_support": 0.8, "healthy_life_expectancy": 65.0,
                         "freedom": 0.8 if i != 1 else np.nan, "generosity": 0.0, "perceptions_of_corruption": 0.7})
    return pd.DataFrame(rows)
