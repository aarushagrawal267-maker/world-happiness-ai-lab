"""Model persistence: save/load pipelines with joblib plus a JSON registry of
metadata (features, parameters, metrics, library versions, training ranges).

Pickled scikit-learn models are tied to library versions. ``load_model``
returns ``None`` (instead of crashing) if a file cannot be loaded, so the app
can retrain automatically on a different machine.
"""
from __future__ import annotations

import json
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from .config import MODELS_DIR, REPORTS_DIR

REGISTRY_FILE = MODELS_DIR / "registry.json"
RESULTS_FILE = REPORTS_DIR / "experiment_results.json"


def library_versions() -> dict[str, str]:
    import numpy
    import pandas
    import sklearn

    v = {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__, "scikit-learn": sklearn.__version__}
    try:
        import xgboost

        v["xgboost"] = xgboost.__version__
    except Exception:
        pass
    return v


def _json_default(o: Any):
    import numpy as np
    import pandas as pd

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    if isinstance(o, Path):
        return str(o)
    return str(o)


def read_registry() -> dict:
    if REGISTRY_FILE.exists():
        try:
            return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def write_registry(reg: dict) -> None:
    REGISTRY_FILE.write_text(json.dumps(reg, indent=2, default=_json_default), encoding="utf-8")


def model_path(name: str) -> Path:
    safe = "".join(ch for ch in name if ch.isalnum() or ch in "_-")
    return MODELS_DIR / f"{safe}.joblib"


def save_model(pipeline, name: str, metadata: dict | None = None) -> Path:
    path = model_path(name)
    joblib.dump(pipeline, path, compress=3)
    reg = read_registry()
    reg[name] = {
        **(metadata or {}),
        "file": path.name,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "versions": library_versions(),
        "size_kb": round(path.stat().st_size / 1024, 1),
    }
    write_registry(reg)
    return path


def load_model(name: str):
    path = model_path(name)
    if not path.exists():
        return None
    reg = read_registry().get(name, {})
    saved_sklearn = reg.get("versions", {}).get("scikit-learn")
    if saved_sklearn and saved_sklearn != library_versions()["scikit-learn"]:
        return None  # incompatible pickle → caller retrains
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return joblib.load(path)
    except Exception:
        return None


def list_models() -> dict:
    reg = read_registry()
    return {k: v for k, v in reg.items() if model_path(k).exists()}


def delete_model(name: str) -> None:
    path = model_path(name)
    if path.exists():
        path.unlink()
    reg = read_registry()
    reg.pop(name, None)
    write_registry(reg)


def save_results(results: dict, path: Path = RESULTS_FILE) -> None:
    path.write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")


def load_results(path: Path = RESULTS_FILE) -> dict | None:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


def artifacts_compatible() -> bool:
    """True if a production model exists and was trained with this sklearn version."""
    reg = read_registry()
    prod = reg.get("production_best")
    if not prod or not model_path("production_best").exists():
        return False
    return prod.get("versions", {}).get("scikit-learn") == library_versions()["scikit-learn"]
