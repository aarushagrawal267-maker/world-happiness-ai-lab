"""Dataset discovery and loading.

* Recursively scans ``data/raw`` for CSV / Excel files (any WHR edition).
* Standardises headers, reclassifies contribution columns, cleans values,
  standardises country names and detects each file's structure.
* Assembles a ``DataBundle``:

  - ``panel``     annual country-year panel (primary modelling dataset)
  - ``rankings``  official multi-year ranking table (latest WHR edition)
  - ``editions``  single-edition tables keyed by edition year
  - ``reports``   validation reports, ``logs`` cleaning logs, ``files`` inventory

Every row keeps a ``source`` column naming the file it came from.
"""
from __future__ import annotations

import io
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import RAW_DIR, UPLOADS_DIR
from .data_cleaning import (
    CleaningLog,
    clean_frame,
    deduplicate_country_year,
    parse_edition_from_filename,
    reclassify_contribution_columns,
    standardize_columns,
)
from .data_validation import ValidationReport, detect_kind, validate_dataset
from .geography import REFERENCE_FILE, attach_geography, build_country_reference

SUPPORTED_EXT = {".csv", ".xls", ".xlsx", ".xlsm"}


@dataclass
class LoadedFile:
    path: Path
    kind: str
    edition: int | None
    frame: pd.DataFrame
    column_mapping: dict[str, str]
    log: CleaningLog
    report: ValidationReport

    @property
    def name(self) -> str:
        return self.path.name


@dataclass
class DataBundle:
    panel: pd.DataFrame
    rankings: pd.DataFrame
    editions: dict[int, pd.DataFrame]
    files: list[LoadedFile] = field(default_factory=list)
    panel_source: str = ""
    rankings_source: str = ""

    @property
    def reports(self) -> dict[str, ValidationReport]:
        return {f.name: f.report for f in self.files}

    @property
    def latest_ranking_year(self) -> int | None:
        return int(self.rankings["year"].max()) if not self.rankings.empty else None

    def inventory(self) -> pd.DataFrame:
        rows = []
        for f in self.files:
            rows.append(
                {
                    "file": str(f.path.relative_to(RAW_DIR)) if f.path.is_relative_to(RAW_DIR) else f.name,
                    "kind": f.kind,
                    "edition": f.edition,
                    "rows": f.report.n_rows,
                    "countries": f.report.n_countries,
                    "years": f"{min(f.report.years)}–{max(f.report.years)}" if f.report.years else "",
                    "valid": f.report.is_valid,
                    "used_for": "primary panel" if f.name == self.panel_source else ("rankings" if f.name == self.rankings_source else ("edition archive" if f.kind.startswith("edition") else "available")),
                }
            )
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- reading
def _read_excel(source, name: str) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            book = pd.read_excel(source, sheet_name=None, engine="calamine")
        except Exception:
            if hasattr(source, "seek"):
                source.seek(0)
            book = pd.read_excel(source, sheet_name=None)
    # choose the sheet that looks like a country table with the most rows
    best, best_score = None, -1
    for _, sheet in book.items():
        std, _ = standardize_columns(sheet)
        score = len(sheet) * (2 if "country" in std.columns else 1)
        if score > best_score:
            best, best_score = sheet, score
    return best if best is not None else pd.DataFrame()


def read_table(source: Path | bytes, name: str | None = None) -> pd.DataFrame:
    """Read CSV or Excel from a path or raw bytes."""
    if isinstance(source, (bytes, bytearray)):
        name = name or "upload.csv"
        buf = io.BytesIO(source)
        if name.lower().endswith((".xls", ".xlsx", ".xlsm")):
            return _read_excel(buf, name)
        for enc in ("utf-8-sig", "latin-1"):
            try:
                buf.seek(0)
                return pd.read_csv(buf, encoding=enc, sep=None, engine="python")
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode CSV file")
    path = Path(source)
    if path.suffix.lower() in {".xls", ".xlsx", ".xlsm"}:
        return _read_excel(path, path.name)
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path}")


def process_frame(raw: pd.DataFrame, name: str) -> tuple[pd.DataFrame, str, dict, CleaningLog]:
    """Standardise + clean one raw table and detect its kind."""
    log = CleaningLog()
    std, mapping = standardize_columns(raw)
    std = reclassify_contribution_columns(std)
    std = clean_frame(std, log)
    kind = detect_kind(std)
    if kind == "annual_panel" and "ladder_score" in std and "life_ladder" not in std:
        std = std.rename(columns={"ladder_score": "life_ladder"})
    keys = ("country", "year") if "year" in std else ("country",)
    std = deduplicate_country_year(std, keys, log)
    std["source"] = name
    return std, kind, mapping, log


def load_file(path: Path) -> LoadedFile:
    raw = read_table(path)
    frame, kind, mapping, log = process_frame(raw, path.name)
    edition = parse_edition_from_filename(path.name) if kind.startswith("edition") else None
    if kind.startswith("edition") and edition:
        frame["edition"] = edition
        frame["year"] = edition - 1  # final survey year of the edition's 3-year window
    if kind == "ranking_panel":
        frame["edition"] = frame["year"] + 1
    report = validate_dataset(frame, name=path.name, kind=kind)
    return LoadedFile(path, kind, edition, frame, mapping, log, report)


def discover_files(root: Path = RAW_DIR) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT and not p.name.startswith(("~$", "."))
    )


# --------------------------------------------------------------------------- bundle
_CACHE: dict[tuple, DataBundle] = {}


def _cache_key(files: list[Path], panel_file: str | None) -> tuple:
    return tuple((str(p), p.stat().st_mtime_ns) for p in files) + (panel_file,)


def load_bundle(panel_file: str | None = None, root: Path = RAW_DIR, use_cache: bool = True) -> DataBundle:
    """Load every dataset under ``root``.

    ``panel_file`` selects which annual panel to use when several exist
    (e.g. an uploaded one); default = the largest panel outside ``uploads``.
    """
    paths = discover_files(root)
    key = _cache_key(paths, panel_file)
    if use_cache and key in _CACHE:
        return _CACHE[key]

    loaded: list[LoadedFile] = []
    for p in paths:
        try:
            loaded.append(load_file(p))
        except Exception as exc:  # unreadable file: record but keep going
            log = CleaningLog(notes=[f"Failed to read: {exc}"])
            rep = validate_dataset(pd.DataFrame({"country": []}), name=p.name, kind="unknown")
            loaded.append(LoadedFile(p, "unknown", None, pd.DataFrame(), {}, log, rep))

    panels = [f for f in loaded if f.kind == "annual_panel" and f.report.is_valid]
    if panel_file:
        chosen = [f for f in panels if f.name == panel_file]
        panels = chosen or panels
    panels.sort(key=lambda f: (UPLOADS_DIR in f.path.parents, -len(f.frame)))
    panel_lf = panels[0] if panels else None

    rankings = [f for f in loaded if f.kind == "ranking_panel"]
    rankings.sort(key=lambda f: (-(f.frame["year"].max() if len(f.frame) else 0), -len(f.frame)))
    rank_lf = rankings[0] if rankings else None

    edition_files = [f for f in loaded if f.kind in ("edition_raw", "edition_contrib") and f.edition]

    # Build / refresh the country reference (ISO-3 + regions) when needed
    if not REFERENCE_FILE.exists():
        names = sorted({c for f in loaded if "country" in f.frame for c in f.frame["country"].dropna().unique()})
        build_country_reference(
            panel=panel_lf.frame if panel_lf else None,
            region_sources=[f.frame for f in edition_files if "region" in f.frame],
            names=names,
        )

    panel = attach_geography(panel_lf.frame) if panel_lf else pd.DataFrame()
    if not panel.empty:
        panel = panel.sort_values(["country", "year"]).reset_index(drop=True)
    ranking_df = attach_geography(rank_lf.frame) if rank_lf else pd.DataFrame()
    if not ranking_df.empty:
        ranking_df = ranking_df.sort_values(["year", "rank"]).reset_index(drop=True)

    editions: dict[int, pd.DataFrame] = {}
    for f in sorted(edition_files, key=lambda f: f.edition):
        editions[f.edition] = attach_geography(f.frame)

    bundle = DataBundle(
        panel=panel,
        rankings=ranking_df,
        editions=editions,
        files=loaded,
        panel_source=panel_lf.name if panel_lf else "",
        rankings_source=rank_lf.name if rank_lf else "",
    )
    _CACHE.clear()
    _CACHE[key] = bundle
    return bundle


def ingest_upload(content: bytes, filename: str, save: bool = True) -> LoadedFile:
    """Validate and (optionally) store a user-uploaded CSV/Excel file."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXT:
        raise ValueError(f"Unsupported file type '{suffix}'. Use CSV or Excel.")
    safe_name = "".join(ch for ch in Path(filename).name if ch.isalnum() or ch in "._- ") or "upload.csv"
    raw = read_table(content, safe_name)
    frame, kind, mapping, log = process_frame(raw, safe_name)
    edition = parse_edition_from_filename(safe_name) if kind.startswith("edition") else None
    if kind.startswith("edition") and edition:
        frame["edition"] = edition
        frame["year"] = edition - 1
    frame = attach_geography(frame) if "country" in frame else frame
    report = validate_dataset(frame, name=safe_name, kind=kind)
    path = UPLOADS_DIR / safe_name
    if save and report.is_valid:
        path.write_bytes(content)
        _CACHE.clear()
    return LoadedFile(path, kind, edition, frame, mapping, log, report)
