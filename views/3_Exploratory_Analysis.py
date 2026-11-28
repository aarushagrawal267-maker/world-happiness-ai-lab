import pandas as pd
import streamlit as st

from src import ui
from src import visualization as viz
from src.config import ALL_NUMERIC, DEFINITIONS, INDICATORS, LABELS, REGIONS, SHORT_LABELS, TARGET
from src.eda import (
    best_reporting_year,
    correlation_matrix,
    coverage_by_year,
    dataset_overview,
    distribution_stats,
    gdp_growth_vs_happiness,
    region_summary,
    top_bottom,
)
from src.preprocessing import missing_summary, outlier_analysis, outlier_summary
from src.statistical_analysis import pairwise_correlation_pvalues

ui.page_header(
    "Exploratory data analysis",
    "Distributions, rankings, economic and social relationships, correlations and missing data — computed live from the annual "
    "WHR country-year panel and the official ranking tables.",
    "Explore · EDA",
)

b = ui.bundle()
panel, rk = b.panel, b.rankings
pfe = ui.panel_fe()
ov = dataset_overview(panel)
tabs = st.tabs(["A · Overview", "B · Distribution", "C · Countries", "D · Economic", "E · Social", "F · Correlation", "G · Missing data & outliers"])

# ------------------------------------------------------------------ A
with tabs[0]:
    ui.kpi_cards([
        {"label": "Observations", "value": f"{ov['observations']:,}", "sub": "country-years"},
        {"label": "Countries", "value": str(ov["countries"])},
        {"label": "Years", "value": str(ov["years"]), "sub": f"{ov['year_min']}–{ov['year_max']}"},
        {"label": "Features", "value": str(ov["features"]), "sub": "explanatory indicators"},
        {"label": "Mean Life Ladder", "value": ui.score(ov['mean_score']), "sub": f"median {ov['median_score']:.2f} · SD {ov['std_score']:.2f}"},
        {"label": "Highest", "value": ui.score(ov['max_score']), "sub": f"{ov['max_country']} {ov['max_year']}"},
        {"label": "Lowest", "value": ui.score(ov['min_score']), "sub": f"{ov['min_country']} {ov['min_year']}"},
        {"label": "Missing cells", "value": f"{ov['missing_cells']:,}", "sub": f"{ov['missing_pct']:.1f}% of indicator cells"},
        {"label": "Duplicates", "value": str(ov["duplicate_country_years"]), "sub": "country-year keys"},
    ])
    cov = coverage_by_year(panel)
    c1, c2 = st.columns(2)
    with c1:
        import plotly.graph_objects as go

        fig = go.Figure(go.Bar(x=cov["year"], y=cov["countries"], marker=dict(color=viz.SERIES_COLORS[0], cornerradius=4), hovertemplate="%{x}: %{y} countries<extra></extra>"))
        fig.update_layout(title="Countries surveyed per year", height=360)
        ui.chart(fig, key="cov")
    with c2:
        fig = go.Figure(go.Scatter(x=cov["year"], y=cov["mean_score"], mode="lines+markers", line=dict(color=viz.SERIES_COLORS[2], width=2), marker=dict(size=8),
                                   hovertemplate="%{x}: mean %{y:.3f}<extra></extra>"))
        fig.update_layout(title="Unweighted mean Life Ladder of surveyed countries", height=360)
        fig.update_yaxes(title="Life Ladder")
        ui.chart(fig, key="cov2")
    ui.callout("Coverage changes over time (27 countries in 2005, ~140 in the 2010s, fewer during the pandemic years). Year-to-year changes in the mean partly reflect "
               "<b>which countries were surveyed</b>, not only changes within countries.", "warn")
    with st.expander("Variable definitions"):
        for k, v in DEFINITIONS.items():
            st.markdown(f"**{LABELS.get(k, k)}** — {v}")

# ------------------------------------------------------------------ B
with tabs[1]:
    ds = distribution_stats(panel[TARGET])
    ui.stat_row([(k, f"{v:.3f}" if isinstance(v, float) else str(v)) for k, v in ds.items()])
    c1, c2 = st.columns(2)
    with c1:
        ui.chart(viz.histogram(panel[TARGET], "Life Ladder distribution (all country-years)", xlabel=LABELS[TARGET]), key="hist")
    with c2:
        ui.chart(viz.density(pfe, TARGET, title="Density estimate (Gaussian KDE)"), key="dens")
    ui.chart(viz.box_by(panel, TARGET, "region", "Life Ladder by region (box = IQR, whiskers = 1.5 × IQR)"), key="box")
    c1, c2 = st.columns(2)
    with c1:
        y = int(rk["year"].max())
        r = rk[rk["year"] == y].sort_values("rank")
        fig = go.Figure(go.Scatter(x=r["rank"], y=r["ladder_score"], mode="markers", marker=dict(size=7, color=viz.SERIES_COLORS[0]),
                                   error_y=dict(type="data", symmetric=False, array=r["upper_whisker"] - r["ladder_score"], arrayminus=r["ladder_score"] - r["lower_whisker"], color="rgba(142,152,200,0.35)", thickness=1, width=0),
                                   text=r["country"], hovertemplate="#%{x} %{text}<br>%{y:.3f}<extra></extra>"))
        fig.update_layout(title=f"Ranking distribution · WHR {y + 1} (score vs rank, 95% CI)", height=400)
        fig.update_xaxes(title="Rank")
        fig.update_yaxes(title="Ladder score")
        ui.chart(fig, key="rankdist")
    with c2:
        ui.chart(viz.year_violin(panel, TARGET, "Year-wise distribution (violin, mean line)"), key="violin")

# ------------------------------------------------------------------ C
with tabs[2]:
    mode = st.segmented_control("Reporting basis", ["Official ranking (3-yr average)", "Annual panel (single year)"], default="Official ranking (3-yr average)")
    if mode == "Annual panel (single year)":
        default_y = best_reporting_year(panel)
        yrs = sorted(panel["year"].unique())[::-1]
        y = st.selectbox("Survey year", yrs, index=yrs.index(default_y))
        d, score = panel[panel["year"] == y], TARGET
        ui.callout(f"Default year <b>{default_y}</b> = the latest panel year in which at least 80% of the maximum number of countries were surveyed "
                   f"({d['country'].nunique()} countries in {y}). Single-year means are noisier than the official 3-year averages.", "info")
    else:
        yrs = sorted(rk["year"].unique())[::-1]
        y = st.selectbox("Ranking year", yrs, format_func=lambda v: f"{v} (WHR {v + 1}, survey {v - 2}–{v})")
        d, score = rk[rk["year"] == y], "ladder_score"
        ui.callout(f"The WHR {y + 1} ranking uses 3-year averages of survey years {y - 2}–{y}; {d['country'].nunique()} countries are ranked.", "info")
    top, bot = top_bottom(d, score, 10)
    c1, c2 = st.columns(2)
    ci = ("lower_whisker", "upper_whisker") if "lower_whisker" in d else None
    with c1:
        ui.chart(viz.ranking_bars(top, score, title="Top 10", ci=ci, color=viz.SERIES_COLORS[2]), key="top10")
    with c2:
        ui.chart(viz.ranking_bars(bot, score, title="Bottom 10", ci=ci, color=viz.SERIES_COLORS[7]), key="bot10")
    rs = region_summary(d, score)
    ui.chart(viz.region_bars(rs, title="Average score by region"), key="regbars")
    ui.section("Country trends", "", "official ranking series")
    pick = st.multiselect("Countries", sorted(rk["country"].unique()), default=[c for c in ["Finland", "India", "Brazil", "United States"] if c in set(rk["country"])], max_selections=8)
    if pick:
        ui.chart(viz.trend_lines(rk[rk["country"].isin(pick)], "ladder_score", band=("lower_whisker", "upper_whisker"), title="Official ladder score with 95% CI"), key="ctrend")
    st.page_link("views/6_Historical_Trends.py", label="More in Historical Trends", icon=":material/timeline:")

# ------------------------------------------------------------------ D
with tabs[3]:
    hl = st.multiselect("Highlight regions (others shown in grey)", REGIONS, default=["Western Europe", "Sub-Saharan Africa", "Latin America and Caribbean"], max_selections=8, key="hl_econ")
    ui.chart(viz.scatter_trend(panel, "log_gdp_per_capita", TARGET, highlight=hl, lowess=True, title="GDP per capita (log) vs Life Ladder — all country-years"), key="gdp")
    from src.statistical_analysis import correlation_tests

    ct = correlation_tests(panel, TARGET, ["log_gdp_per_capita"]).iloc[0]
    ui.callout(f"Pearson r = <b>{ct['pearson_r']:.3f}</b> (95% CI {ct['pearson_ci_low']:.3f}–{ct['pearson_ci_high']:.3f}), Spearman ρ = {ct['spearman_rho']:.3f}, n = {ct['n']}. "
               "Because GDP is on a log scale, a fixed step on the x-axis is a fixed <i>percentage</i> difference in income: the roughly linear pattern means each doubling "
               "of income is associated with a similar increase in life evaluations.", "info")
    c1, c2 = st.columns(2)
    with c1:
        g = gdp_growth_vs_happiness(pfe)
        ui.chart(viz.scatter_trend(g, "gdp_growth_1y", "ladder_yoy_change", color=None, title="Economic growth vs change in Life Ladder (same country, consecutive years)"), key="growth")
        if len(g) > 10:
            gc = correlation_tests(g.rename(columns={"ladder_yoy_change": "y"}), "y", ["gdp_growth_1y"]).iloc[0]
            st.caption(f"Within-country annual changes: r = {gc['pearson_r']:.3f}, p = {gc['pearson_p']:.3g}, n = {gc['n']}. Short-run growth is a much weaker correlate than income level; "
                       "annual changes are dominated by survey noise.")
    with c2:
        rm = panel.groupby("region")[[TARGET, "log_gdp_per_capita"]].mean().reset_index()
        rm["country"] = rm["region"]
        fig = viz.scatter_trend(rm, "log_gdp_per_capita", TARGET, color=None, title="Regional means: income vs Life Ladder", hover="region")
        for _, r in rm.iterrows():
            fig.add_annotation(x=r["log_gdp_per_capita"], y=r[TARGET], text=r["region"], showarrow=False, yshift=12, font=dict(size=10, color=viz.MUTED))
        ui.chart(fig, key="regincome")
    ui.causation_note("Richer countries also tend to have better health, institutions and social support, so income's association partly reflects these related factors.")

# ------------------------------------------------------------------ E
with tabs[4]:
    soc = ["social_support", "freedom", "generosity", "perceptions_of_corruption", "healthy_life_expectancy", "positive_affect", "negative_affect"]
    c1, c2 = st.columns([1, 2])
    with c1:
        f = st.radio("Indicator", soc, format_func=lambda x: LABELS[x])
        st.markdown(f"<div class='card'><p>{DEFINITIONS.get(f, '')}</p></div>", unsafe_allow_html=True)
        ct = correlation_tests(panel, TARGET, [f]).iloc[0]
        ui.stat_row([("Pearson r", f"{ct['pearson_r']:.3f}"), ("Spearman ρ", f"{ct['spearman_rho']:.3f}"), ("n", str(ct["n"])), ("Strength", ct["strength"])])
    with c2:
        hl2 = st.multiselect("Highlight regions", REGIONS, default=["Western Europe", "Sub-Saharan Africa", "South Asia"], max_selections=8, key="hl_soc")
        ui.chart(viz.scatter_trend(panel, f, TARGET, highlight=hl2, lowess=True, title=f"{LABELS[f]} vs Life Ladder"), key="socscatter")
    allc = correlation_tests(panel, TARGET, [c for c in INDICATORS + ["positive_affect", "negative_affect"] if c in panel])
    allc["label"] = allc["feature"].map(SHORT_LABELS)
    fig = go.Figure(go.Bar(x=allc["pearson_r"], y=allc["label"], orientation="h", marker=dict(color=[viz.SERIES_COLORS[0] if v > 0 else viz.SERIES_COLORS[7] for v in allc["pearson_r"]], cornerradius=4),
                           error_x=dict(type="data", symmetric=False, array=allc["pearson_ci_high"] - allc["pearson_r"], arrayminus=allc["pearson_r"] - allc["pearson_ci_low"], color=viz.MUTED),
                           hovertemplate="%{y}: r = %{x:.3f}<extra></extra>"))
    fig.update_layout(title="Correlation with Life Ladder (Pearson r, 95% CI)", height=380)
    fig.update_yaxes(autorange="reversed")
    fig.add_vline(x=0, line=dict(color=viz.MUTED, width=1))
    ui.chart(fig, key="corrbars")
    ui.callout("Positive and negative affect are themselves measures of emotional wellbeing (yesterday's emotions). They are reported for context; they are "
               "<b>excluded from the default prediction model</b> because they are closer to outcomes than to explanatory conditions.", "info")

# ------------------------------------------------------------------ F
with tabs[5]:
    cols = [TARGET] + [c for c in ALL_NUMERIC if c in panel]
    c1, c2 = st.columns(2)
    with c1:
        pm = correlation_matrix(panel, cols, "pearson")
        ui.chart(viz.heatmap(pm, "Pearson correlation (linear)", pvals=pairwise_correlation_pvalues(panel, cols, "pearson")), key="pearson")
    with c2:
        sm = correlation_matrix(panel, cols, "spearman")
        ui.chart(viz.heatmap(sm, "Spearman rank correlation (monotonic)", pvals=pairwise_correlation_pvalues(panel, cols, "spearman")), key="spearman")
    st.caption("Pairwise-complete observations. ⁿˢ marks coefficients not significant at p < 0.05. Blue = positive, red = negative.")
    ui.card("Pearson vs Spearman", "Pearson measures the strength of a <b>linear</b> relationship on the original scale and is sensitive to outliers. Spearman correlates "
            "<b>ranks</b>, capturing any monotonic relationship and being robust to outliers. Similar values (as here) suggest the relationships are close to linear.")
    ui.card("Correlation is not causation", "A strong correlation can arise because (1) X influences Y, (2) Y influences X (happier societies may become more productive or trusting), "
            "(3) a third factor drives both (institutions, history, culture), or (4) measurement artefacts (e.g. culturally different response styles). Observational "
            "country-level data cannot separate these explanations; the platform therefore reports associations only.")

# ------------------------------------------------------------------ G
with tabs[6]:
    cols = [TARGET] + [c for c in ALL_NUMERIC if c in panel]
    ms = missing_summary(panel, cols).reset_index().rename(columns={"index": "variable"})
    c1, c2 = st.columns([1, 1.6])
    with c1:
        st.dataframe(ms, hide_index=True, height=360)
    with c2:
        ui.chart(viz.missing_heatmap(panel, cols[1:], "year", "Missing share by year"), key="missyear")
    ui.chart(viz.missing_matrix(panel, cols[1:], "Missingness pattern (each column = one country-year)"), key="missmatrix")
    ui.section("Imputation options", "", "evaluated, not applied blindly")
    res = ui.results()
    if res and res.get("imputation_comparison"):
        st.dataframe(ui.fmt_table(pd.DataFrame(res["imputation_comparison"]), 4), hide_index=True)
    st.markdown(
        "- **Complete cases** drops every row with any missing indicator — loses countries systematically (e.g. those without corruption questions).\n"
        "- **Median / mean** imputation inside the scikit-learn pipeline: statistics are learned from training folds only (leakage-safe). The platform default is **median** (robust to skew).\n"
        "- **Within-country interpolation** fills gaps between a country's own observed years; it uses later years, so it is avoided for forward-in-time evaluation.\n"
        "- **Past-only carry-forward** uses only earlier observations of the same country.\n"
        "- **Target scores are never imputed** — rows without a Life Ladder value are excluded from modelling."
    )
    ui.section("Outlier analysis", "", "flagged, investigated — not deleted")
    osum = outlier_summary(panel)
    st.dataframe(osum, hide_index=True)
    with st.expander("All flagged observations"):
        st.dataframe(outlier_analysis(panel), hide_index=True, height=360)
    notes = [n for f in b.files if f.name == b.panel_source for n in f.log.notes]
    if notes:
        ui.callout("<b>Values removed as definitionally implausible during cleaning:</b><br/>" + "<br/>".join(ui.esc(n) for n in notes), "warn")
    ui.callout("Extreme values such as Afghanistan's very low recent scores or the high GDP of Gulf states and Luxembourg are genuine observations and are kept. "
               "Tree models are insensitive to them; linear models use them as they are, and robust standard errors are reported in the statistical analysis.", "info")
ui.footer()
