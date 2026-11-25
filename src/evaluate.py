"""Model evaluation: regression metrics, residual analysis, error breakdowns
and validation splitters (random, country-grouped, forward-in-time)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, r2_score

METRIC_INFO = {
    "r2": ("R²", "Share of variance in the target explained by the model (1 = perfect, 0 = no better than predicting the mean, can be negative)."),
    "adj_r2": ("Adjusted R²", "R² penalised for the number of predictors p: 1 − (1 − R²)(n − 1)/(n − p − 1)."),
    "mae": ("MAE", "Mean absolute error in ladder points: the typical size of a miss."),
    "mse": ("MSE", "Mean squared error: averages squared misses, so large errors weigh more."),
    "rmse": ("RMSE", "Root mean squared error, in ladder points; sensitive to large errors."),
    "max_error": ("Max error", "Largest single absolute miss."),
    "bias": ("Bias", "Mean of (predicted − actual); positive = systematic over-prediction."),
}


def adjusted_r2(r2: float, n: int, p: int) -> float:
    if n - p - 1 <= 0:
        return float("nan")
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def regression_metrics(y_true, y_pred, n_features: int | None = None) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    out = {
        "r2": float(r2),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "max_error": float(max_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "n": int(len(y_true)),
    }
    if n_features is not None:
        out["adj_r2"] = float(adjusted_r2(r2, len(y_true), n_features))
    return out


def cluster_bootstrap_rmse(y_true, preds: dict[str, np.ndarray], groups, n_boot: int = 2000, seed: int = 0) -> dict:
    """95% percentile intervals for test RMSE, resampling whole countries (clusters) with replacement.

    ``preds`` maps a model name to its predictions on the same rows. Returns, per model, the
    RMSE interval, and for every pair the interval of the RMSE difference (first − second).
    Country rows are dependent, so resampling rows would understate the uncertainty.
    """
    y = np.asarray(y_true, float)
    codes, uniq = pd.factorize(pd.Series(groups))
    k = len(uniq)
    sq = {m: np.bincount(codes, weights=(y - np.asarray(p, float)) ** 2, minlength=k) for m, p in preds.items()}
    cnt = np.bincount(codes, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    w = np.stack([np.bincount(rng.integers(0, k, k), minlength=k) for _ in range(n_boot)]).astype(float)
    boot = {m: np.sqrt((w @ s) / (w @ cnt)) for m, s in sq.items()}
    ci = lambda a: [float(v) for v in np.percentile(a, [2.5, 97.5])]  # noqa: E731
    names = list(preds)
    return {
        "n_groups": int(k), "n_boot": n_boot,
        "rmse_ci": {m: ci(b) for m, b in boot.items()},
        "diff_ci": {f"{a} - {b}": ci(boot[a] - boot[b]) for i, a in enumerate(names) for b in names[i + 1:]},
    }


def residual_frame(meta: pd.DataFrame, y_true, y_pred) -> pd.DataFrame:
    out = meta.copy().reset_index(drop=True)
    out["actual"] = np.asarray(y_true, float)
    out["predicted"] = np.asarray(y_pred, float)
    out["residual"] = out["actual"] - out["predicted"]
    out["abs_error"] = out["residual"].abs()
    return out


SCORE_BANDS = [0, 4.0, 5.0, 6.0, 7.0, 10.01]
SCORE_BAND_LABELS = ["< 4", "4–5", "5–6", "6–7", "≥ 7"]


def error_by_group(res: pd.DataFrame, group: str) -> pd.DataFrame:
    return (
        res.groupby(group, observed=True)
        .agg(n=("abs_error", "size"), mae=("abs_error", "mean"), rmse=("residual", lambda r: float(np.sqrt(np.mean(r ** 2)))), bias=("residual", lambda r: float(-np.mean(r))))
        .reset_index()
        .sort_values("mae", ascending=False)
    )


def error_by_score_range(res: pd.DataFrame) -> pd.DataFrame:
    r = res.copy()
    r["score_range"] = pd.cut(r["actual"], bins=SCORE_BANDS, labels=SCORE_BAND_LABELS, right=False)
    return error_by_group(r, "score_range").sort_values("score_range")


# --------------------------------------------------------------------------- splitters
class ForwardYearSplit:
    """Expanding-window, forward-in-time cross-validation on calendar years.

    For ``n_splits`` = k, the last k distinct years (or year blocks) are used as
    successive validation folds; each fold trains only on strictly earlier
    years. ``groups`` must be the year of each row.
    """

    def __init__(self, n_splits: int = 4, block: int = 1):
        self.n_splits = n_splits
        self.block = block

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        years = np.asarray(groups)
        uniq = np.sort(np.unique(years))
        needed = self.n_splits * self.block
        if len(uniq) <= needed:
            raise ValueError("Not enough distinct years for forward splits")
        val_years = uniq[-needed:]
        for i in range(self.n_splits):
            block = val_years[i * self.block:(i + 1) * self.block]
            train_idx = np.where(years < block.min())[0]
            val_idx = np.where(np.isin(years, block))[0]
            yield train_idx, val_idx


def describe_validation(strategy: str) -> str:
    return {
        "random": "Random split / K-fold: rows are shuffled, so the same country appears in training and test (different years). Answers: 'how well can the model fill in a country-year it has partly seen?' — typically the most optimistic.",
        "group": "Country-grouped split / GroupKFold: every country is entirely in training *or* test. Answers: 'how well does the model estimate happiness for a country it has never seen?'",
        "temporal": "Forward-in-time split: train on earlier years, test on later years. Answers: 'how well do patterns learned from the past carry over to future observations?'",
    }[strategy]
