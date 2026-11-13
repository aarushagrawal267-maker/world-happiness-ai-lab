"""Automated insights engine.

Every insight is computed at run time from the loaded data and the saved
experiment results — nothing is hard-coded. Each carries its source, the
reporting period and the metric it is based on.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import INDICATORS, SHORT_LABELS, TARGET
from .eda import country_period_change, trend_slopes
from .statistical_analysis import correlation_tests, country_level, group_comparison


@dataclass
class Insight:
    category: str
    title: str
    text: str
    metric: str
    value: str
    source: str
    period: str
    icon: str = "◆"
    tone: str = "neutral"  # positive | negative | neutral


def _fmt(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}"


def generate_insights(panel: pd.DataFrame, panel_fe: pd.DataFrame, rankings: pd.DataFrame, panel_source: str, rankings_source: str,
                      results: dict | None = None, clustering=None, importance: pd.DataFrame | None = None) -> list[Insight]:
    ins: list[Insight] = []

    # ---------------- rankings (official, latest edition)
    if not rankings.empty:
        y = int(rankings["year"].max())
        r = rankings[rankings["year"] == y].sort_values("ladder_score", ascending=False)
        top3 = r.head(3)
        ins.append(Insight(
            "Rankings", "Highest reported life evaluations",
            f"In the WHR {y + 1} ranking (survey years {y - 2}–{y}), the highest 3-year average ladder scores are: "
            + ", ".join(f"{t.country} ({_fmt(t.ladder_score, 3)})" for t in top3.itertuples()) + f" — out of {len(r)} countries ranked.",
            "Ladder score (3-yr avg)", _fmt(top3.iloc[0]["ladder_score"], 3), rankings_source, f"{y - 2}–{y}", "▲", "positive"))
        bot3 = r.tail(3).iloc[::-1]
        ins.append(Insight(
            "Rankings", "Lowest reported life evaluations",
            "The lowest scores in the same ranking are: " + ", ".join(f"{t.country} ({_fmt(t.ladder_score, 3)})" for t in bot3.itertuples())
            + f". The gap between first and last place is {_fmt(r['ladder_score'].max() - r['ladder_score'].min())} points on the 0–10 ladder.",
            "Ladder score (3-yr avg)", _fmt(bot3.iloc[0]["ladder_score"], 3), rankings_source, f"{y - 2}–{y}", "▼", "negative"))

        # whisker overlap → ranking uncertainty
        if {"lower_whisker", "upper_whisker"} <= set(r.columns):
            first = r.iloc[0]
            overlapping = r[(r["upper_whisker"] >= first["lower_whisker"]) & (r["country"] != first["country"])]
            ins.append(Insight(
                "Rankings", "Ranking uncertainty at the top",
                (f"{len(overlapping)} other countries have 95% confidence intervals that overlap with {first['country']}'s, "
                 "so their relative ranks are uncertain (overlapping intervals do not by themselves prove a difference is zero)." if len(overlapping) else
                 f"No other country's 95% confidence interval overlaps with {first['country']}'s ({first['lower_whisker']:.3f}–{first['upper_whisker']:.3f}), "
                 f"so its first place is statistically distinguishable; further down the table most neighbouring ranks have overlapping intervals "
                 f"({int(((r['upper_whisker'].shift(-1) >= r['lower_whisker'])).sum())} of {len(r) - 1} adjacent pairs)."),
                "Countries with overlapping CI", str(len(overlapping)), rankings_source, f"{y - 2}–{y}", "≈"))

        # long-run change between first and last ranking year
        y0 = int(rankings["year"].min())
        ch = country_period_change(rankings, y0, y)
        if len(ch) > 5:
            up, down = ch.iloc[0], ch.iloc[-1]
            ins.append(Insight(
                "Change over time", "Largest long-run improvement",
                f"Among {len(ch)} countries ranked in both WHR {y0 + 1} and WHR {y + 1}, {up['country']} improved the most: "
                f"{_fmt(up['ladder_score_start'])} → {_fmt(up['ladder_score_end'])} (+{_fmt(up['change'])}), moving {int(up['rank_change']):+d} places.",
                "Δ ladder score", f"+{_fmt(up['change'])}", rankings_source, f"{y0}→{y}", "↗", "positive"))
            ins.append(Insight(
                "Change over time", "Largest long-run decline",
                f"{down['country']} declined the most: {_fmt(down['ladder_score_start'])} → {_fmt(down['ladder_score_end'])} ({_fmt(down['change'])}), "
                f"moving {int(down['rank_change']):+d} places.",
                "Δ ladder score", _fmt(down["change"]), rankings_source, f"{y0}→{y}", "↘", "negative"))

    # ---------------- annual panel
    if not panel_fe.empty and "ladder_yoy_change" in panel_fe:
        yoy = panel_fe.dropna(subset=["ladder_yoy_change"])
        best, worst = yoy.loc[yoy["ladder_yoy_change"].idxmax()], yoy.loc[yoy["ladder_yoy_change"].idxmin()]
        ins.append(Insight(
            "Change over time", "Largest single-year rise",
            f"{best['country']} recorded the largest year-over-year increase in the annual panel: +{_fmt(best['ladder_yoy_change'])} points "
            f"from {int(best['year']) - 1} to {int(best['year'])}. Single-year national survey means are noisy, so such jumps should be read with caution.",
            "Δ Life Ladder (1 yr)", f"+{_fmt(best['ladder_yoy_change'])}", panel_source, f"{int(best['year']) - 1}→{int(best['year'])}", "⇡", "positive"))
        ins.append(Insight(
            "Change over time", "Largest single-year fall",
            f"{worst['country']} recorded the largest year-over-year fall: {_fmt(worst['ladder_yoy_change'])} points from {int(worst['year']) - 1} to {int(worst['year'])}.",
            "Δ Life Ladder (1 yr)", _fmt(worst["ladder_yoy_change"]), panel_source, f"{int(worst['year']) - 1}→{int(worst['year'])}", "⇣", "negative"))
        slopes = trend_slopes(panel)
        if not slopes.empty:
            s_up, s_dn = slopes.iloc[0], slopes.iloc[-1]
            ins.append(Insight(
                "Change over time", "Steadiest upward and downward trends",
                f"Fitting a linear trend to each country with ≥ 8 annual observations, {s_up['country']} shows the steepest rise "
                f"({s_up['slope_per_year']:+.3f} points/year, {s_up['first_year']}–{s_up['last_year']}) and {s_dn['country']} the steepest decline "
                f"({s_dn['slope_per_year']:+.3f} points/year, {s_dn['first_year']}–{s_dn['last_year']}).",
                "Trend slope (pts / yr)", f"{s_up['slope_per_year']:+.3f}", panel_source, f"{int(panel['year'].min())}–{int(panel['year'].max())}", "∿"))

    if not panel.empty:
        ct = correlation_tests(panel, TARGET, [c for c in INDICATORS if c in panel])
        if not ct.empty:
            s = ct.iloc[0]
            ins.append(Insight(
                "Relationships", "Strongest observed correlation",
                f"Across {int(s['n'])} country-years, {SHORT_LABELS.get(s['feature'], s['feature'])} has the strongest linear association with Life Ladder "
                f"(Pearson r = {s['pearson_r']:.2f}, Spearman ρ = {s['spearman_rho']:.2f}, p {'< 0.001' if s['pearson_p'] < 0.001 else '= ' + format(s['pearson_p'], '.3f')}). "
                "This is an association, not evidence of causation.",
                "Pearson r", f"{s['pearson_r']:.2f}", panel_source, f"{int(panel['year'].min())}–{int(panel['year'].max())}", "↔"))
            w = ct.iloc[-1]
            ins.append(Insight(
                "Relationships", "Weakest observed correlation",
                f"{SHORT_LABELS.get(w['feature'], w['feature'])} shows the weakest association (r = {w['pearson_r']:.2f}, n = {int(w['n'])}).",
                "Pearson r", f"{w['pearson_r']:.2f}", panel_source, f"{int(panel['year'].min())}–{int(panel['year'].max())}", "↔"))
            neg = ct[ct["pearson_r"] < 0]
            if not neg.empty:
                n0 = neg.iloc[0]
                ins.append(Insight(
                    "Relationships", "Negative association",
                    f"Higher {SHORT_LABELS.get(n0['feature'], n0['feature']).lower()} goes with lower life evaluations (r = {n0['pearson_r']:.2f}).",
                    "Pearson r", f"{n0['pearson_r']:.2f}", panel_source, f"{int(panel['year'].min())}–{int(panel['year'].max())}", "↔", "negative"))

        # regional differences on country means of the last 3 panel years
        y1 = int(panel["year"].max())
        cl = country_level(panel, TARGET, "region", (y1 - 2, y1))
        if cl["region"].nunique() >= 3:
            gc = group_comparison(cl, TARGET, "region")
            hi, lo = gc["summary"].iloc[0], gc["summary"].iloc[-1]
            ins.append(Insight(
                "Regions", "Regional differences",
                f"Using one average per country ({y1 - 2}–{y1}), {hi['region']} has the highest mean Life Ladder ({_fmt(hi['mean'])}) and {lo['region']} the lowest "
                f"({_fmt(lo['mean'])}). Kruskal–Wallis p {'< 0.001' if gc['kruskal_p'] < 0.001 else '= ' + format(gc['kruskal_p'], '.3f')}; η² = {gc['eta_squared']:.2f} "
                f"(region accounts for about {gc['eta_squared']:.0%} of between-country variance).",
                "Regional gap (pts)", _fmt(hi["mean"] - lo["mean"]), panel_source, f"{y1 - 2}–{y1}", "◍"))

        if "ladder_vs_gdp_residual" in panel_fe:
            yr = panel_fe[panel_fe["year"] == y1].dropna(subset=["ladder_vs_gdp_residual"])
            if len(yr) > 10:
                a, b = yr.nlargest(1, "ladder_vs_gdp_residual").iloc[0], yr.nsmallest(1, "ladder_vs_gdp_residual").iloc[0]
                ins.append(Insight(
                    "Relationships", "Life evaluations vs income",
                    f"In {y1}, relative to a simple GDP-only line fitted to all country-years, {a['country']} reports life evaluations {_fmt(a['ladder_vs_gdp_residual'])} points higher "
                    f"than its income alone would suggest, while {b['country']} reports {_fmt(-b['ladder_vs_gdp_residual'])} points lower. Other factors matter beyond income.",
                    "Residual vs GDP line", f"+{_fmt(a['ladder_vs_gdp_residual'])}", panel_source, str(y1), "$"))

        miss = panel[INDICATORS].isna().mean().sort_values(ascending=False)
        ins.append(Insight(
            "Data quality", "Most incomplete indicator",
            f"{SHORT_LABELS.get(miss.index[0], miss.index[0])} is missing in {miss.iloc[0]:.1%} of country-years; missing inputs are imputed inside the model pipeline "
            "(training-fold statistics only) and target scores are never imputed.",
            "Missing share", f"{miss.iloc[0]:.1%}", panel_source, f"{int(panel['year'].min())}–{int(panel['year'].max())}", "◌"))

    # ---------------- models
    if results:
        b = results.get("best_model", {})
        if b:
            ins.append(Insight(
                "Machine learning", "Best predictive model",
                f"{b['name']} achieved the lowest validation RMSE on countries never seen in training ({b['val']['rmse']:.3f} points); "
                f"on a further held-out set of unseen countries its test RMSE is {b['test']['rmse']:.3f} (R² = {b['test']['r2']:.2f}).",
                "Test RMSE (unseen countries)", f"{b['test']['rmse']:.3f}", "outputs/reports/experiment_results.json", results.get("generated_at", "")[:10], "◎", "positive"))
        cvc = results.get("cv_comparison")
        if cvc:
            df = pd.DataFrame(cvc)
            key = b.get("key")
            sub = df[df["key"] == key].set_index("strategy") if key else pd.DataFrame()
            if {"random", "group"} <= set(sub.index):
                gap = (sub.loc["group", "rmse_mean"] / sub.loc["random", "rmse_mean"] - 1) * 100
                ins.append(Insight(
                    "Machine learning", "Random splits are optimistic",
                    f"For {b['name']}, random K-fold CV gives RMSE {sub.loc['random', 'rmse_mean']:.3f}, but country-grouped CV gives {sub.loc['group', 'rmse_mean']:.3f} "
                    f"({gap:+.0f}%). Random splits let the model see other years of the same country, overstating accuracy for new countries.",
                    "Optimism gap", f"{gap:+.0f}%", "outputs/reports/experiment_results.json", results.get("generated_at", "")[:10], "⚖"))
    if importance is not None and not importance.empty:
        top = importance.iloc[0]
        ins.append(Insight(
            "Machine learning", "Most important predictor",
            f"Shuffling {SHORT_LABELS.get(top['feature'], top['feature'])} increases the best model's RMSE by {top['importance_mean']:.3f} points on unseen countries — the largest "
            "permutation importance. Importance reflects the model's reliance, shared among correlated features, not causal effect.",
            "Δ RMSE when permuted", f"+{top['importance_mean']:.3f}", "Permutation importance (group test set)", "", "★"))

    if clustering is not None:
        cs = clustering.cluster_summary
        parts = [f"{r['name']} ({int(r['n_countries'])} countries, mean ladder {r[TARGET]:.2f}; e.g. {r['example_countries']})" for _, r in cs.iterrows()]
        ins.append(Insight(
            "Clusters", f"{clustering.k} country profiles",
            "K-means on the six indicators (Life Ladder not used for clustering) finds: " + " | ".join(parts) + f". Silhouette = {clustering.silhouette:.2f}, "
            "indicating moderately separated, overlapping groups rather than sharp categories.",
            "Silhouette score", f"{clustering.silhouette:.2f}", panel_source, ", ".join(map(str, clustering.info["years"])), "⬡"))
    return ins


def insights_frame(ins: list[Insight]) -> pd.DataFrame:
    return pd.DataFrame([i.__dict__ for i in ins])
