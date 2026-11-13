"""Data acquisition: source registry, downloading, provenance manifest and
cross-source verification.

Every file the platform uses is described in ``SOURCES`` with its publisher,
edition, survey period, URL and methodology notes. Downloads are recorded in
``data/raw/sources_manifest.json`` together with a SHA-256 checksum so the
origin of every observation can be traced.

If a download fails (offline laptop, blocked network) the platform keeps
working with files already in ``data/raw`` and the app shows manual-download
instructions (``MANUAL_INSTRUCTIONS``). No observation is ever synthesised.

Run ``python -m src.data_acquisition`` to (re-)download all sources.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import RAW_DIR, SOURCES_MANIFEST


@dataclass(frozen=True)
class Source:
    key: str
    title: str
    publisher: str
    url: str
    local_path: str  # relative to data/raw
    edition: int | None
    survey_years: str
    content: str
    methodology_note: str
    role: str  # which part of the platform uses it

    @property
    def path(self) -> Path:
        return RAW_DIR / self.local_path


WHR_FILES = "https://files.worldhappiness.report/"

_OFFICIAL_NOTE = (
    "Official 'Data for Figure 2.1' file. Ladder scores are 3-year averages of national "
    "Cantril-ladder responses from the Gallup World Poll. 'Explained by' columns are the WHR "
    "regression contributions of each factor relative to the hypothetical 'Dystopia' benchmark; "
    "their scale and coefficients differ between editions, so they are not compared across editions."
)

SOURCES: list[Source] = [
    Source(
        "whr2026_fig21", "World Happiness Report 2026 — Data for Figure 2.1 (2011–2025)",
        "Wellbeing Research Centre, University of Oxford / Gallup / UN SDSN",
        WHR_FILES + "WHR26_Data_Figure_2.1.xlsx", "whr_official/WHR26_Data_Figure_2.1.xlsx", 2026,
        "Ranking years 2011–2025 (each a 3-year window ending that year)",
        "Rank, 3-year average life evaluation, 95% CI whiskers, six explained-by contributions",
        _OFFICIAL_NOTE + " This file stacks the historical ranking tables, so it is the longest official ranking series.",
        "Rankings, world map, historical trends, insights",
    ),
    Source(
        "whr2025_fig21", "World Happiness Report 2025 — Data for Figure 2.1 (2011–2024)",
        "Wellbeing Research Centre, University of Oxford / Gallup / UN SDSN",
        WHR_FILES + "WHR25_Data_Figure_2.1v3.xlsx", "whr_official/WHR25_Data_Figure_2.1v3.xlsx", 2025,
        "Ranking years 2011–2024", "Same structure as WHR 2026 file", _OFFICIAL_NOTE,
        "Fallback ranking series; edition archive",
    ),
    Source(
        "whr2024_fig21", "World Happiness Report 2024 — Data for Figure 2.1",
        "UN SDSN / Wellbeing Research Centre", WHR_FILES + "WHR24_Data_Figure_2.1.xls",
        "whr_official/WHR24_Data_Figure_2.1.xls", 2024, "2021–2023",
        "Ladder score, whiskers, explained-by contributions (no raw indicators)", _OFFICIAL_NOTE, "Edition archive",
    ),
    Source(
        "whr2023_fig21", "World Happiness Report 2023 — Data for Figure 2.1",
        "UN SDSN", WHR_FILES + "WHR23_Data_Figure_2.1.xls", "whr_official/WHR23_Data_Figure_2.1.xls", 2023,
        "2020–2022", "Ladder score, SE, whiskers, raw indicators (3-yr means), contributions", _OFFICIAL_NOTE,
        "Edition archive; panel verification",
    ),
    Source(
        "whr2022_fig21", "World Happiness Report 2022 — Data for Figure 2.1",
        "UN SDSN", WHR_FILES + "WHR22_Data_Figure_2.1.xls", "whr_official/WHR22_Data_Figure_2.1.xls", 2022,
        "2019–2021", "Rank, happiness score, whiskers, contributions (* marks small samples)", _OFFICIAL_NOTE,
        "Edition archive",
    ),
    Source(
        "whr2021_fig21", "World Happiness Report 2021 — Data for Figure 2.1",
        "UN SDSN", WHR_FILES + "WHR21_Data_Figure_2.1.xls", "whr_official/WHR21_Data_Figure_2.1.xls", 2021,
        "2018–2020", "Regional indicator, ladder score, raw indicators, contributions", _OFFICIAL_NOTE,
        "Edition archive; regional classification",
    ),
    Source(
        "whr2020_fig21", "World Happiness Report 2020 — Data for Figure 2.1",
        "UN SDSN", WHR_FILES + "WHR20_Data_Figure_2.1.xls", "whr_official/WHR20_Data_Figure_2.1.xls", 2020,
        "2017–2019", "Regional indicator, ladder score, raw indicators, contributions", _OFFICIAL_NOTE,
        "Edition archive; regional classification",
    ),
    Source(
        "whr2023_panel", "World Happiness Report 2023 — DataForTable2.1 annual panel (2005–2022)",
        "UN SDSN / Gallup World Poll (WHR 2023 statistical appendix data); obtained via the public "
        "Hugging Face mirror 'Slapping/Happiness' because the original S3 link now returns HTTP 403",
        "https://huggingface.co/datasets/Slapping/Happiness/resolve/main/WHR-1.csv",
        "whr_panel/WHR2023_DataForTable2.1_panel.csv", 2023, "Annual observations 2005–2022",
        "Country, ISO-3, year, Life Ladder, log GDP pc, social support, healthy life expectancy, freedom, "
        "generosity, perceptions of corruption, positive affect, negative affect",
        "Annual national means (single survey year, not 3-year averages). Because it is a third-party "
        "mirror it is verified on every acquisition: its 2020–2022 means must reproduce the official "
        "WHR 2023 Figure 2.1 ladder scores (see verification block in the manifest).",
        "Primary dataset for EDA, statistics, machine learning, clustering and prediction",
    ),
] + [
    Source(
        f"kaggle_{y}", f"World Happiness Report {y} (Kaggle unsdsn/world-happiness)",
        "UN SDSN via Kaggle (CC0); mirrored on Hugging Face 'nateraw/world-happiness'",
        f"https://huggingface.co/datasets/nateraw/world-happiness/resolve/main/{y}.csv",
        f"kaggle_unsdsn/{y}.csv", y, f"{y - 3}–{y - 1}",
        "Rank, score and explained-by contributions" + (" plus region" if y in (2015, 2016) else ""),
        "Indicator columns in these files (e.g. 'Economy (GDP per Capita)', 'Family') are explained-by "
        "contributions on edition-specific scales, not raw indicators.",
        "Edition archive (legacy editions)",
    )
    for y in (2015, 2016, 2017, 2018, 2019)
]

SOURCES_BY_KEY = {s.key: s for s in SOURCES}

MANUAL_INSTRUCTIONS = """
**Manual dataset setup** (use this if automatic download is blocked):

1. Official WHR files — open <https://worldhappiness.report/data-sharing/> and download
   *Data for Figure 2.1* for each edition you want. Save them in `data/raw/whr_official/`
   keeping the original file names (e.g. `WHR26_Data_Figure_2.1.xlsx`).
2. Annual panel — download the *World Happiness Report* country-year panel
   (`DataForTable2.1.xls` from the WHR statistical appendix, or the Kaggle dataset
   *"World Happiness Report up to 2023"*). Save it in `data/raw/whr_panel/` (CSV or Excel).
3. Kaggle legacy editions — <https://www.kaggle.com/datasets/unsdsn/world-happiness>
   (`2015.csv` … `2019.csv`) into `data/raw/kaggle_unsdsn/`.
4. Or use **Dataset Explorer → Upload** in the app: CSV/Excel files are validated,
   standardised and stored in `data/raw/uploads/`.

Restart the app (or press *Reload data*) after adding files.
"""


# --------------------------------------------------------------------------- manifest helpers
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    if SOURCES_MANIFEST.exists():
        try:
            return json.loads(SOURCES_MANIFEST.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save_manifest(manifest: dict) -> None:
    SOURCES_MANIFEST.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def _record(source: Source, status: str, extra: dict | None = None) -> dict:
    rec = asdict(source)
    rec["status"] = status
    if source.path.exists():
        rec["sha256"] = sha256(source.path)
        rec["bytes"] = source.path.stat().st_size
    rec.update(extra or {})
    return rec


def download_source(source: Source, force: bool = False, timeout: int = 60) -> dict:
    """Download one source; returns its manifest record (never raises on network errors)."""
    if source.path.exists() and not force:
        return _record(source, "present")
    source.path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import requests

        resp = requests.get(source.url, timeout=timeout, headers={"User-Agent": "world-happiness-ai-lab/1.0"})
        resp.raise_for_status()
        content = resp.content
        if content[:15].lower().startswith((b"<!doctype html", b"<html")):
            raise ValueError("server returned an HTML page instead of a data file")
        tmp = source.path.with_suffix(source.path.suffix + ".part")
        tmp.write_bytes(content)
        tmp.replace(source.path)
        return _record(source, "downloaded", {"downloaded_at": datetime.now(timezone.utc).isoformat()})
    except Exception as exc:  # network blocked, 403, offline …
        return _record(source, "failed" if not source.path.exists() else "present (download failed)", {"error": str(exc)})


def verify_panel_against_official() -> dict:
    """Cross-check the annual panel against the official WHR 2023 Figure 2.1 file.

    WHR 2023 ranks countries on the mean Life Ladder of survey years 2020–2022;
    reproducing those scores from the panel confirms the mirror's integrity.
    """
    import numpy as np
    import pandas as pd

    from .data_cleaning import clean_frame, standardize_columns

    panel_src, off_src = SOURCES_BY_KEY["whr2023_panel"], SOURCES_BY_KEY["whr2023_fig21"]
    if not (panel_src.path.exists() and off_src.path.exists()):
        return {"status": "skipped", "reason": "panel or official WHR 2023 file missing"}
    panel = clean_frame(standardize_columns(pd.read_csv(panel_src.path, encoding="utf-8-sig"))[0])
    panel = panel.rename(columns={"ladder_score": "life_ladder"})
    from .data_loader import read_table

    off = clean_frame(standardize_columns(read_table(off_src.path))[0])
    means = panel[panel["year"].between(2020, 2022)].groupby("country")["life_ladder"].mean()
    j = off.set_index("country")[["ladder_score"]].join(means, how="inner").dropna()
    diff = (j["ladder_score"] - j["life_ladder"]).abs()
    result = {
        "status": "passed" if (len(j) >= 0.9 * len(off) and diff.median() < 0.01) else "failed",
        "official_countries": int(len(off)),
        "matched_countries": int(len(j)),
        "median_abs_diff": round(float(diff.median()), 5),
        "max_abs_diff": round(float(diff.max()), 5),
        "share_within_0_01": round(float((diff < 0.01).mean()), 4),
        "pearson_r": round(float(np.corrcoef(j["ladder_score"], j["life_ladder"])[0, 1]), 6),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    return result


def acquire_all(force: bool = False) -> dict:
    """Download every registered source and refresh the provenance manifest."""
    manifest = load_manifest()
    records = {s.key: download_source(s, force=force) for s in SOURCES}
    manifest["sources"] = records
    manifest["panel_verification"] = verify_panel_against_official()
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    save_manifest(manifest)
    return manifest


def refresh_manifest_offline() -> dict:
    """Record checksums/status of local files without touching the network."""
    manifest = load_manifest()
    manifest["sources"] = {s.key: _record(s, "present" if s.path.exists() else "missing") for s in SOURCES}
    manifest["panel_verification"] = verify_panel_against_official()
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    save_manifest(manifest)
    return manifest


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Download World Happiness Report datasets")
    ap.add_argument("--force", action="store_true", help="re-download even if files exist")
    ap.add_argument("--offline", action="store_true", help="only checksum local files")
    args = ap.parse_args()
    m = refresh_manifest_offline() if args.offline else acquire_all(force=args.force)
    for k, r in m["sources"].items():
        print(f"{k:16s} {r['status']:28s} {r.get('error', '')}")
    print("Panel verification:", m["panel_verification"])
