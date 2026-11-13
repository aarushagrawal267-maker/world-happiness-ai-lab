"""Central configuration: paths, feature definitions, labels and constants.

All other modules import from here so that column names, feature lists and
directories are defined exactly once.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Load ``config.yaml`` (cached)."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


CFG = load_config()

# --------------------------------------------------------------------------- paths
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = ROOT_DIR / CFG["data"]["raw_dir"]
PROCESSED_DIR = ROOT_DIR / CFG["data"]["processed_dir"]
REFERENCE_DIR = ROOT_DIR / CFG["data"]["reference_dir"]
UPLOADS_DIR = ROOT_DIR / CFG["data"]["uploads_dir"]
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
SOURCES_MANIFEST = RAW_DIR / "sources_manifest.json"

for _d in (RAW_DIR, PROCESSED_DIR, REFERENCE_DIR, UPLOADS_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE: int = int(CFG["project"]["random_state"])

# --------------------------------------------------------------------------- columns
TARGET = CFG["modeling"]["target"]  # "life_ladder"

INDICATORS = [
    "log_gdp_per_capita",
    "social_support",
    "healthy_life_expectancy",
    "freedom",
    "generosity",
    "perceptions_of_corruption",
]
AFFECT = ["positive_affect", "negative_affect"]
ALL_NUMERIC = INDICATORS + AFFECT

FEATURE_SETS: dict[str, list[str]] = CFG["modeling"]["feature_sets"]
DEFAULT_FEATURE_SET: str = CFG["modeling"]["default_feature_set"]
CATEGORICAL_FEATURES = ["region"]

# Columns that must NEVER be used as model inputs (target or target-derived).
LEAKAGE_COLUMNS = {
    TARGET,
    "ladder_score",
    "rank",
    "lower_whisker",
    "upper_whisker",
    "dystopia_residual",
    "ladder_yoy_change",
    "happiness_category",
    "ladder_rank_in_year",
}

LABELS: dict[str, str] = {
    "life_ladder": "Life Ladder (life evaluation, 0–10)",
    "ladder_score": "Ladder score (3-yr avg, 0–10)",
    "log_gdp_per_capita": "Log GDP per capita (PPP)",
    "social_support": "Social support",
    "healthy_life_expectancy": "Healthy life expectancy (years)",
    "freedom": "Freedom to make life choices",
    "generosity": "Generosity (residual)",
    "perceptions_of_corruption": "Perceptions of corruption",
    "positive_affect": "Positive affect",
    "negative_affect": "Negative affect",
    "year": "Year",
    "gdp_growth_1y": "GDP growth vs previous year (Δ log GDP)",
    "region": "Region",
    "rank": "Rank",
    "composite_index": "Composite socio-economic index",
}

SHORT_LABELS: dict[str, str] = {
    "life_ladder": "Life Ladder",
    "ladder_score": "Ladder score",
    "log_gdp_per_capita": "Log GDP pc",
    "social_support": "Social support",
    "healthy_life_expectancy": "Healthy life exp.",
    "freedom": "Freedom",
    "generosity": "Generosity",
    "perceptions_of_corruption": "Corruption (perceived)",
    "positive_affect": "Positive affect",
    "negative_affect": "Negative affect",
    "year": "Year",
    "gdp_growth_1y": "GDP growth (1y)",
    "region": "Region",
}

DEFINITIONS: dict[str, str] = {
    "life_ladder": (
        "National average response to the Cantril ladder question (Gallup World Poll): "
        "'imagine a ladder with steps numbered from 0 at the bottom to 10 at the top… on which "
        "step do you personally feel you stand at this time?' A reported life evaluation, not an "
        "objective measure of every individual's happiness."
    ),
    "log_gdp_per_capita": "Natural log of GDP per capita in purchasing-power-parity terms (World Bank WDI, constant international dollars).",
    "social_support": "Share answering yes to 'If you were in trouble, do you have relatives or friends you can count on to help you?' (0–1).",
    "healthy_life_expectancy": "Healthy life expectancy at birth in years (WHO Global Health Observatory based, interpolated by WHR).",
    "freedom": "Share satisfied with their 'freedom to choose what you do with your life' (0–1).",
    "generosity": "Residual of regressing national donation rate on log GDP per capita — positive means more giving than income predicts.",
    "perceptions_of_corruption": "Average of perceived corruption in government and in business (0–1, higher = more corruption perceived).",
    "positive_affect": "Average of yesterday's laughter, enjoyment and doing something interesting (0–1).",
    "negative_affect": "Average of yesterday's worry, sadness and anger (0–1).",
}

# Valid value ranges, grounded in the variable definitions (hard validation limits)
VALID_RANGES: dict[str, tuple[float, float]] = {
    "life_ladder": (0.0, 10.0),
    "ladder_score": (0.0, 10.0),
    "log_gdp_per_capita": (5.0, 13.0),
    "social_support": (0.0, 1.0),
    "healthy_life_expectancy": (20.0, 90.0),
    "freedom": (0.0, 1.0),
    "generosity": (-1.0, 1.0),
    "perceptions_of_corruption": (0.0, 1.0),
    "positive_affect": (0.0, 1.0),
    "negative_affect": (0.0, 1.0),
    "year": (2005, 2030),
}

REGIONS = [
    "Western Europe",
    "North America and ANZ",
    "Central and Eastern Europe",
    "Commonwealth of Independent States",
    "East Asia",
    "Southeast Asia",
    "South Asia",
    "Latin America and Caribbean",
    "Middle East and North Africa",
    "Sub-Saharan Africa",
]

# --------------------------------------------------------------------------- design tokens
@dataclass(frozen=True)
class Palette:
    bg: str = "#070B1A"
    surface: str = "#0E1430"
    surface_2: str = "#141B3D"
    border: str = "rgba(140,160,255,0.16)"
    text: str = "#E8ECFF"
    muted: str = "#8E98C8"
    primary: str = "#7C9CFF"
    accent: str = "#4FE3C1"
    warm: str = "#FFB86B"
    danger: str = "#FF6B8B"
    violet: str = "#B18CFF"


PALETTE = Palette()

# Categorical series colours — validated (scripts/validate_palette.js, dark mode,
# surface #0E1430): lightness band, chroma, adjacent CVD ΔE ≥ 8.4, normal-vision ΔE ≥ 19.3,
# contrast ≥ 3:1. Assigned in fixed order, never cycled; >8 series fold into "Other".
SERIES_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
OTHER_COLOR = "#6B7194"

# Sequential (magnitude): single blue hue. On the dark surface low values recede
# (dark) and high values glow (light).
SEQUENTIAL_SCALE = [
    [0.00, "#0d366b"], [0.17, "#184f95"], [0.33, "#256abf"], [0.50, "#3987e5"],
    [0.67, "#6da7ec"], [0.83, "#9ec5f4"], [1.00, "#cde2fb"],
]
HAPPINESS_SCALE = SEQUENTIAL_SCALE
# Diverging (polarity): red ↔ neutral gray ↔ blue
DIVERGING_SCALE = [[0.0, "#e66767"], [0.25, "#a5504f"], [0.5, "#383835"], [0.75, "#2a6bb8"], [1.0, "#3987e5"]]
STATUS = {"good": "#0ca30c", "warning": "#c98500", "critical": "#e66767"}
