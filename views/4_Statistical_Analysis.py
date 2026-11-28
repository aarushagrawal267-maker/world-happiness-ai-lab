import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from src import ui
from src import visualization as viz
from src.config import INDICATORS, LABELS, SHORT_LABELS, TARGET
from src.feature_engineering import add_categories
from src.statistical_analysis import correlation_tests, country_level, group_comparison, ols_regression

ui.page_header(
    "Statistical analysis",
    "Formal tests of association: correlation tests with confidence intervals, multiple linear regression with assumption diagnostics "
    "and country-clustered standard errors, and group comparisons with multiple-comparison correction and effect sizes.",
    "Analyze · Inference",
)
ui.causation_note("Statistical significance means an association is unlikely to be zero in this data — not that it is large, important or causal.")

panel = ui.bundle().panel
t1, t2, t3 = st.tabs(["A · Correlation tests", "B · Multiple regression", "C · Group comparisons"])

with t1:
    yr = st.slider("Years included", int(panel["year"].min()), int(panel["year"].max()), (int(panel["year"].min()), int(panel["year"].max())), key="ct_years")
    feats = [c for c in INDICATORS + ["positive_affect", "negative_affect"] if c in panel]
    ct = correlation_tests(panel[panel["year"].between(*yr)], TARGET, feats)
    for pc in ("pearson_p", "pearson_p_holm", "spearman_p", "spearman_p_holm"):
        ct[pc] = ct[pc].map(lambda p: "< 1e-300" if p < 1e-300 else f"{p:.2e}")
    show = ct.assign(feature=ct["feature"].map(LABELS))[["feature", "n", "pearson_r", "pearson_ci_low", "pearson_ci_high", "pearson_p", "pearson_p_holm", "spearman_rho", "spearman_p", "spearman_p_holm", "strength"]]
    st.dataframe(show, hide_index=True, column_config={
        "pearson_r": st.column_config.ProgressColumn("Pearson r", min_value=-1, max_value=1, format="%.3f"),
        "spearman_rho": st.column_config.NumberColumn("Spearman ρ", format="%.3f"),
                "pearson_ci_low": st.column_config.NumberColumn("CI low", format="%.3f"), "pearson_ci_high": st.column_config.NumberColumn("CI high", format="%.3f"),
    })
    ui.callout("Tests use pairwise-complete country-years. Because each country contributes many years, observations are not independent and p-values are "
               "<b>too small</b>; the confidence intervals and the clustered regression in tab B are the more honest guides. Holm correction controls the family-wise error across the 8 tests.", "warn")
    ycs = st.select_slider("Cross-section check: correlation in one year only", options=sorted(panel["year"].unique()), value=int(panel["year"].max()) - 3)
    cs = correlation_tests(panel[panel["year"] == ycs], TARGET, feats)
    cs["label"] = cs["feature"].map(SHORT_LABELS)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cs["label"], y=cs["pearson_r"], name=f"Pearson r ({ycs})", marker=dict(color=viz.SERIES_COLORS[0], cornerradius=4)))
    fig.add_trace(go.Bar(x=cs["label"], y=cs["spearman_rho"], name=f"Spearman ρ ({ycs})", marker=dict(color=viz.SERIES_COLORS[1], cornerradius=4)))
    fig.update_layout(barmode="group", title=f"One observation per country ({ycs}, n ≈ {int(cs['n'].median())})", height=380)
    ui.chart(fig, key="cs")

with t2:
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        feats = st.multiselect("Explanatory variables", [c for c in INDICATORS + ["positive_affect", "negative_affect"] if c in panel], default=INDICATORS, format_func=lambda c: LABELS[c])
    with c2:
        yr2 = st.slider("Years", int(panel["year"].min()), int(panel["year"].max()), (int(panel["year"].min()), int(panel["year"].max())), key="reg_years")
    with c3:
        stdz = st.toggle("Standardised coefficients (β)", value=False, help="Variables z-scored: coefficients become SD-change in ladder per SD-change in the predictor.")
    if not feats:
        st.info("Select at least one variable.")
    else:
        r = ols_regression(panel[panel["year"].between(*yr2)], TARGET, feats, standardize=stdz)
        ui.kpi_cards([
            {"label": "R²", "value": f"{r['r2']:.3f}", "sub": "share of variance explained"},
            {"label": "Adjusted R²", "value": f"{r['adj_r2']:.3f}", "sub": f"{len(feats)} predictors"},
            {"label": "F-statistic", "value": f"{r['f_stat']:.0f}", "sub": f"p = {r['f_pvalue']:.1e}"},
            {"label": "Observations", "value": f"{r['n']:,}", "sub": f"{r['n_countries']} countries (complete cases)"},
            {"label": "AIC / BIC", "value": f"{r['aic']:.0f}", "sub": f"BIC {r['bic']:.0f}"},
        ])
        tbl = r["table"].copy()
        tbl["term"] = tbl["term"].map(lambda t: LABELS.get(t, t))
        st.dataframe(ui.fmt_table(tbl, 4), hide_index=True)
        # forest plot with clustered CIs
        tt = r["table"][r["table"]["term"] != "const"].copy()
        lo_col, hi_col = ("cluster_ci_low", "cluster_ci_high") if "cluster_ci_low" in tt else ("ci_low", "ci_high")
        tt["label"] = tt["term"].map(SHORT_LABELS)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tt["coef"], y=tt["label"], mode="markers", marker=dict(size=11, color=viz.SERIES_COLORS[0], line=dict(color=viz.SURFACE, width=2)),
                                 error_x=dict(type="data", symmetric=False, array=tt[hi_col] - tt["coef"], arrayminus=tt["coef"] - tt[lo_col], color=viz.MUTED, thickness=1.5, width=5),
                                 name="coefficient (95% CI, country-clustered)", hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
        fig.add_vline(x=0, line=dict(color=viz.MUTED, width=1, dash="dash"))
        fig.update_layout(title="Coefficient plot — 95% CIs from country-clustered standard errors", height=380)
        fig.update_xaxes(title="Standardised coefficient (β)" if stdz else "Coefficient (ladder points per unit)")
        ui.chart(fig, key="forest")
        ui.callout("Three sets of standard errors are reported: <b>classical</b> (assume independent, equal-variance errors), <b>HC3</b> (robust to heteroscedasticity) and "
                   "<b>country-clustered</b> (robust to correlation among a country's repeated observations — the most defensible here). "
                   "Each coefficient is the association holding the other included variables fixed; with correlated predictors these partial associations can differ "
                   "sharply from simple correlations.", "info")
        ui.section("Assumption checks", "", "linearity · homoscedasticity · normality · independence · multicollinearity")
        dg = r["diagnostics"]
        c1, c2 = st.columns([1, 1.3])
        with c1:
            for msg in dg["interpretation"]:
                st.markdown(f"- {msg}")
            v = dg["vif"].copy()
            v["feature"] = v["feature"].map(LABELS)
            st.dataframe(ui.fmt_table(v, 2), hide_index=True)
        with c2:
            res = pd.DataFrame({"predicted": r["fitted"], "residual": r["residuals"], "actual": r["fitted"] + r["residuals"]})
            res["country"], res["year"] = "obs", 0
            ui.chart(viz.residual_vs_predicted(res, "Residuals vs fitted (linearity / constant variance)"), key="olsresid")
        osm, osr = stats.probplot(r["residuals"], dist="norm", fit=False)
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=osm, y=np.sort(r["residuals"]), mode="markers", marker=dict(size=5, color=viz.SERIES_COLORS[2]), name="residual quantiles"))
        lim = [float(osm.min()), float(osm.max())]
        sd = float(np.std(r["residuals"]))
        fig.add_trace(go.Scatter(x=lim, y=[lim[0] * sd, lim[1] * sd], mode="lines", line=dict(color=viz.MUTED, dash="dash"), name="normal reference"))
        fig.update_layout(title="Normal Q–Q plot of residuals", height=360)
        fig.update_xaxes(title="Theoretical quantile")
        fig.update_yaxes(title="Residual")
        ui.chart(fig, key="qq")
        with st.expander("Full statsmodels summary"):
            st.code(r["summary_text"])

with t3:
    pfe = add_categories(panel)
    groupings = {"Region": "region", "GDP quartile": "gdp_category", "Social support quartile": "social_support_category", "Freedom quartile": "freedom_category", "Health quartile": "health_category"}
    c1, c2, c3 = st.columns(3)
    with c1:
        g_label = st.selectbox("Compare groups by", list(groupings))
    with c2:
        value = st.selectbox("Outcome", [TARGET] + INDICATORS, format_func=lambda c: LABELS[c])
    with c3:
        yr3 = st.slider("Period (one mean per country)", int(panel["year"].min()), int(panel["year"].max()), (int(panel["year"].max()) - 2, int(panel["year"].max())), key="grp_years")
    gcol = groupings[g_label]
    cl = country_level(pfe, value, gcol, yr3)
    if cl[gcol].nunique() < 2:
        st.warning("Not enough groups for this selection.")
    else:
        gc = group_comparison(cl, value, gcol)
        ui.kpi_cards([
            {"label": "One-way ANOVA", "value": f"F = {gc['anova_f']:.1f}", "sub": f"p = {gc['anova_p']:.2e}"},
            {"label": "Welch ANOVA", "value": f"F = {gc['welch_f']:.1f}", "sub": f"p = {gc['welch_p']:.2e} (unequal variances)"},
            {"label": "Kruskal–Wallis", "value": f"H = {gc['kruskal_h']:.1f}", "sub": f"p = {gc['kruskal_p']:.2e} (rank-based)"},
            {"label": "Effect size η²", "value": f"{gc['eta_squared']:.2f}", "sub": f"ω² = {gc['omega_squared']:.2f} · ε² = {gc['epsilon_squared']:.2f}"},
            {"label": "Countries", "value": str(gc["n"]), "sub": f"{gc['k_groups']} groups"},
        ])
        ui.callout(f"<b>Recommended test:</b> {gc['recommended_test']} — Levene p = {gc['levene_p']:.3g}. <b>Practical significance:</b> {gc['practical_interpretation']}", "info")
        ui.chart(viz.box_by(cl, value, gcol, f"{LABELS[value]} by {g_label.lower()} — country means {yr3[0]}–{yr3[1]}"), key="grpbox")
        pw = gc["pairwise"]
        if not pw.empty:
            sig = int(pw["significant_holm"].sum())
            st.markdown(f"**Post-hoc pairwise comparisons** — {sig} of {len(pw)} pairs differ after Holm correction (Mann–Whitney). "
                        "Cohen's d gives the practical size of each difference.")
            st.dataframe(ui.fmt_table(pw, 4), hide_index=True, height=380, column_config={"significant_holm": st.column_config.CheckboxColumn("sig. (Holm)")})
        ui.card("Statistical vs practical significance", "A p-value answers 'could this difference plausibly be zero?' and shrinks with sample size. Effect sizes "
                "(η², Cohen's d) answer 'how big is it?'. A difference can be statistically significant yet small in ladder points, or large but uncertain with few countries.")
ui.footer()
