"""Statistical analysis: correlation tests, multiple linear regression with
assumption diagnostics, and group comparisons with multiple-comparison
correction and effect sizes.

Country-year observations are **not independent** (the same country appears
in many years). Regression therefore also reports country-clustered standard
errors, and group comparisons are run on one observation per country (country
means over a chosen period) to avoid pseudo-replication.

All results describe statistical *associations*; none establishes causation.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from .config import INDICATORS, TARGET


# --------------------------------------------------------------------------- correlations
def _fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 3 or abs(r) >= 1:
        return (np.nan, np.nan)
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    zc = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se))


def strength_label(r: float) -> str:
    a = abs(r)
    if a >= 0.7:
        return "strong"
    if a >= 0.5:
        return "moderate–strong"
    if a >= 0.3:
        return "moderate"
    if a >= 0.1:
        return "weak"
    return "negligible"


def correlation_tests(df: pd.DataFrame, target: str = TARGET, features: list[str] | None = None) -> pd.DataFrame:
    """Pearson and Spearman correlation of each feature with the target
    (pairwise-complete observations), with p-values, n and Holm-adjusted p."""
    features = features or [c for c in INDICATORS if c in df]
    rows = []
    for f in features:
        d = df[[target, f]].dropna()
        n = len(d)
        if n < 5:
            continue
        pr, pp = stats.pearsonr(d[f], d[target])
        sr, sp = stats.spearmanr(d[f], d[target])
        lo, hi = _fisher_ci(pr, n)
        rows.append({"feature": f, "n": n, "pearson_r": pr, "pearson_p": pp, "pearson_ci_low": lo, "pearson_ci_high": hi,
                     "spearman_rho": sr, "spearman_p": sp, "strength": strength_label(pr)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["pearson_p_holm"] = multipletests(out["pearson_p"], method="holm")[1]
        out["spearman_p_holm"] = multipletests(out["spearman_p"], method="holm")[1]
        out = out.sort_values("pearson_r", key=np.abs, ascending=False).reset_index(drop=True)
    return out


def pairwise_correlation_pvalues(df: pd.DataFrame, columns: list[str], method: str = "pearson") -> pd.DataFrame:
    fn = stats.pearsonr if method == "pearson" else stats.spearmanr
    p = pd.DataFrame(np.ones((len(columns), len(columns))), index=columns, columns=columns)
    for a, b in combinations(columns, 2):
        d = df[[a, b]].dropna()
        if len(d) > 3:
            p.loc[a, b] = p.loc[b, a] = fn(d[a], d[b])[1]
    return p


# --------------------------------------------------------------------------- regression
def ols_regression(df: pd.DataFrame, target: str = TARGET, features: list[str] | None = None, standardize: bool = False) -> dict:
    """Multiple linear regression (complete cases) with classical, HC3 and
    country-clustered standard errors plus assumption diagnostics."""
    features = features or [c for c in INDICATORS if c in df]
    cols = [target] + features + (["country"] if "country" in df else [])
    d = df[cols].dropna().reset_index(drop=True)
    X = d[features].astype(float)
    y = d[target].astype(float)
    if standardize:
        X = (X - X.mean()) / X.std(ddof=0)
        y = (y - y.mean()) / y.std(ddof=0)
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc).fit()
    robust = model.get_robustcov_results(cov_type="HC3")
    clustered = None
    if "country" in d:
        groups = pd.factorize(d["country"])[0]
        clustered = sm.OLS(y, Xc).fit(cov_type="cluster", cov_kwds={"groups": groups})
    ci = model.conf_int()
    table = pd.DataFrame(
        {
            "coef": model.params,
            "std_err": model.bse,
            "t": model.tvalues,
            "p_value": model.pvalues,
            "ci_low": ci[0],
            "ci_high": ci[1],
            "hc3_std_err": robust.bse,
            "hc3_p_value": robust.pvalues,
        }
    )
    if clustered is not None:
        cci = clustered.conf_int()
        table["cluster_std_err"] = clustered.bse
        table["cluster_p_value"] = clustered.pvalues
        table["cluster_ci_low"] = cci[0]
        table["cluster_ci_high"] = cci[1]
    table.index.name = "term"
    return {
        "table": table.reset_index(),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "f_stat": float(model.fvalue),
        "f_pvalue": float(model.f_pvalue),
        "aic": float(model.aic),
        "bic": float(model.bic),
        "n": int(model.nobs),
        "n_countries": int(d["country"].nunique()) if "country" in d else None,
        "features": features,
        "diagnostics": assumption_checks(model, X),
        "fitted": model.fittedvalues.to_numpy(),
        "residuals": model.resid.to_numpy(),
        "summary_text": model.summary().as_text(),
        "standardized": standardize,
    }


def assumption_checks(model, X: pd.DataFrame) -> dict:
    resid = model.resid
    bp_lm, bp_p, _, _ = het_breuschpagan(resid, model.model.exog)
    jb, jb_p, skew, kurt = jarque_bera(resid)
    vif = pd.DataFrame(
        {
            "feature": X.columns,
            "vif": [variance_inflation_factor(sm.add_constant(X).values, i + 1) for i in range(X.shape[1])],
        }
    )
    try:
        reset = linear_reset(model, power=2, use_f=True)
        reset_p = float(reset.pvalue)
    except Exception:
        reset_p = np.nan
    return {
        "breusch_pagan_p": float(bp_p),
        "jarque_bera_p": float(jb_p),
        "residual_skew": float(skew),
        "residual_kurtosis": float(kurt),
        "durbin_watson": float(durbin_watson(resid)),
        "reset_p": reset_p,
        "vif": vif,
        "interpretation": _interpret_diagnostics(bp_p, jb_p, float(durbin_watson(resid)), vif["vif"].max(), reset_p),
    }


def _interpret_diagnostics(bp_p, jb_p, dw, max_vif, reset_p) -> list[str]:
    msgs = []
    msgs.append(
        "Breusch–Pagan p = %.3g: %s" % (bp_p, "evidence of heteroscedasticity → rely on HC3 / clustered SEs." if bp_p < 0.05 else "no strong evidence of heteroscedasticity.")
    )
    msgs.append(
        "Jarque–Bera p = %.3g: %s" % (jb_p, "residuals deviate from normality (with large n, inference on coefficients remains approximately valid)." if jb_p < 0.05 else "residuals consistent with normality.")
    )
    msgs.append(
        "Durbin–Watson = %.2f: %s" % (dw, "positive autocorrelation, expected because rows are sorted country-year → clustered SEs account for within-country dependence." if dw < 1.5 else "no strong first-order autocorrelation in row order.")
    )
    msgs.append(
        "Max VIF = %.2f: %s" % (max_vif, "notable multicollinearity (GDP, health and social support move together) → individual coefficients are less precise." if max_vif > 5 else "multicollinearity is moderate.")
    )
    if not np.isnan(reset_p):
        msgs.append("Ramsey RESET p = %.3g: %s" % (reset_p, "some evidence of non-linearity / omitted terms." if reset_p < 0.05 else "no strong evidence of functional-form misspecification."))
    return msgs


# --------------------------------------------------------------------------- group comparisons
def country_level(df: pd.DataFrame, value: str, group: str, years: tuple[int, int] | None = None) -> pd.DataFrame:
    """One row per country: mean of ``value`` over ``years`` (avoids pseudo-replication)."""
    d = df if years is None else df[df["year"].between(*years)]
    return d.dropna(subset=[value, group]).groupby(["country", group], observed=True)[value].mean().reset_index()


def group_comparison(df: pd.DataFrame, value: str, group: str, alpha: float = 0.05) -> dict:
    """Omnibus tests + post-hoc pairwise comparisons with Holm correction.

    Returns parametric (one-way ANOVA, Welch ANOVA) and non-parametric
    (Kruskal–Wallis) results, assumption checks and effect sizes.
    """
    d = df.dropna(subset=[value, group])
    groups = {str(k): g[value].to_numpy() for k, g in d.groupby(group, observed=True) if len(g) >= 2}
    samples = list(groups.values())
    k, n = len(samples), sum(len(s) for s in samples)
    f, p_anova = stats.f_oneway(*samples)
    h, p_kw = stats.kruskal(*samples)
    lev_stat, p_lev = stats.levene(*samples, center="median")
    # Welch ANOVA
    try:
        from statsmodels.stats.oneway import anova_oneway

        w = anova_oneway(samples, use_var="unequal")
        welch_f, welch_p = float(w.statistic), float(w.pvalue)
    except Exception:
        welch_f, welch_p = np.nan, np.nan
    grand = d[value].mean()
    ss_between = sum(len(s) * (s.mean() - grand) ** 2 for s in samples)
    ss_total = ((d[value] - grand) ** 2).sum()
    ms_within = (ss_total - ss_between) / (n - k)
    eta2 = ss_between / ss_total
    omega2 = (ss_between - (k - 1) * ms_within) / (ss_total + ms_within)
    eps2 = h / ((n ** 2 - 1) / (n + 1))
    normal_p = {g: (float(stats.shapiro(s)[1]) if 3 <= len(s) <= 5000 else np.nan) for g, s in groups.items()}

    pairs = []
    for (a, sa), (b, sb) in combinations(groups.items(), 2):
        u, pu = stats.mannwhitneyu(sa, sb, alternative="two-sided")
        t, pt = stats.ttest_ind(sa, sb, equal_var=False)
        pooled = np.sqrt(((len(sa) - 1) * sa.var(ddof=1) + (len(sb) - 1) * sb.var(ddof=1)) / (len(sa) + len(sb) - 2))
        d_cohen = (sa.mean() - sb.mean()) / pooled if pooled > 0 else np.nan
        rbc = 1 - 2 * u / (len(sa) * len(sb))
        pairs.append({"group_a": a, "group_b": b, "n_a": len(sa), "n_b": len(sb), "mean_diff": sa.mean() - sb.mean(),
                      "welch_t_p": pt, "mann_whitney_p": pu, "cohens_d": d_cohen, "rank_biserial": rbc})
    pw = pd.DataFrame(pairs)
    if not pw.empty:
        pw["welch_t_p_holm"] = multipletests(pw["welch_t_p"], method="holm")[1]
        pw["mann_whitney_p_holm"] = multipletests(pw["mann_whitney_p"], method="holm")[1]
        pw["significant_holm"] = pw["mann_whitney_p_holm"] < alpha
        pw["practical_size"] = pw["cohens_d"].abs().map(lambda x: "large" if x >= 0.8 else "medium" if x >= 0.5 else "small" if x >= 0.2 else "negligible")
        pw = pw.sort_values("mean_diff", key=np.abs, ascending=False)

    summary = d.groupby(group, observed=True)[value].agg(["count", "mean", "median", "std"]).sort_values("mean", ascending=False).reset_index()
    use_nonparam = (p_lev < alpha) or any((p < alpha) for p in normal_p.values() if not np.isnan(p))
    return {
        "k_groups": k,
        "n": n,
        "anova_f": float(f), "anova_p": float(p_anova),
        "welch_f": welch_f, "welch_p": welch_p,
        "kruskal_h": float(h), "kruskal_p": float(p_kw),
        "levene_p": float(p_lev),
        "shapiro_p": normal_p,
        "eta_squared": float(eta2), "omega_squared": float(omega2), "epsilon_squared": float(eps2),
        "recommended_test": "Kruskal–Wallis + Holm-corrected Mann–Whitney (assumptions questionable)" if use_nonparam else "One-way ANOVA (assumptions reasonable)",
        "summary": summary,
        "pairwise": pw,
        "practical_interpretation": _effect_text(eta2),
    }


def _effect_text(eta2: float) -> str:
    size = "large" if eta2 >= 0.14 else "medium" if eta2 >= 0.06 else "small" if eta2 >= 0.01 else "negligible"
    return f"η² = {eta2:.2f}: group membership accounts for {eta2:.0%} of the variance in country scores — a {size} effect by Cohen's benchmarks."
