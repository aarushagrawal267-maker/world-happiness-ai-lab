import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src import visualization as viz
from src.config import INDICATORS, LABELS, REGIONS, TARGET
from src.eda import country_period_change, trend_slopes, yoy_changes

ui.page_header(
    "Historical trends",
    "How reported life evaluations have moved since 2005: global and regional trends, the biggest improvers and decliners, rank movements and year-over-year changes.",
    "Explore · Change over time",
)
b = ui.bundle()
panel, rk = b.panel, b.rankings
pfe = ui.panel_fe()

ui.section("Global & regional trends", "01")
c1, c2 = st.columns([1, 2.4])
with c1:
    ind = st.selectbox("Indicator", [TARGET] + INDICATORS, format_func=lambda c: LABELS[c])
    y0, y1 = st.slider("Window", int(panel["year"].min()), int(panel["year"].max()), (2011, int(panel["year"].max())))
    balanced = st.toggle("Balanced panel only", value=True, help="Only countries observed in every year of the window, so composition changes don't drive the trend.")
    regs = st.multiselect("Regions (max 8)", REGIONS, default=["Western Europe", "Latin America and Caribbean", "Sub-Saharan Africa", "South Asia", "Central and Eastern Europe"], max_selections=8)
w = panel[panel["year"].between(y0, y1)].dropna(subset=[ind])
if balanced:
    n_years = y1 - y0 + 1
    keep = w.groupby("country")["year"].nunique()
    keep = keep[keep == n_years].index
    w = w[w["country"].isin(keep)]
with c2:
    if w.empty:
        st.warning("No country is observed in every year of this window — widen it or switch off the balanced panel.")
    else:
        g = w.groupby("year")[ind].agg(["mean", "count"]).reset_index()
        rg = w[w["region"].isin(regs)].groupby(["region", "year"])[ind].mean().reset_index()
        fig = go.Figure()
        cmap = viz.color_map(regs)
        for r in regs:
            s = rg[rg["region"] == r]
            fig.add_trace(go.Scatter(x=s["year"], y=s[ind], mode="lines", name=r, line=dict(color=cmap[r], width=2), hovertemplate=f"{r} %{{x}}: %{{y:.3f}}<extra></extra>"))
        fig.add_trace(go.Scatter(x=g["year"], y=g["mean"], mode="lines+markers", name="All countries in sample", line=dict(color=viz.TEXT, width=3, dash="dot"), marker=dict(size=8),
                                 customdata=g["count"], hovertemplate="All %{x}: %{y:.3f} (n=%{customdata})<extra></extra>"))
        fig.update_layout(title=f"Mean {LABELS[ind]} · {'balanced panel of ' + str(w['country'].nunique()) + ' countries' if balanced else 'all surveyed countries'}", height=460, hovermode="x unified")
        fig.update_yaxes(title=LABELS[ind])
        ui.chart(fig, key="globaltrend")

ui.section("Improvers & decliners", "02", "official ranking scores (3-year averages)")
ry = sorted(rk["year"].unique())
c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    s0 = st.selectbox("From ranking year", ry, index=0, format_func=lambda v: f"{v} (WHR {v + 1})")
with c2:
    s1 = st.selectbox("To ranking year", ry[::-1], index=0, format_func=lambda v: f"{v} (WHR {v + 1})")
with c3:
    topn = st.slider("Show top/bottom", 5, 25, 12)
ch = country_period_change(rk, s0, s1)
if ch.empty or s0 >= s1:
    st.warning("Choose a 'from' year earlier than the 'to' year.")
else:
    d = pd.concat([ch.head(topn), ch.tail(topn)]).drop_duplicates("country").sort_values("change")
    fig = go.Figure(go.Bar(x=d["change"], y=d["country"], orientation="h", marker=dict(color=[viz.SERIES_COLORS[0] if v > 0 else viz.SERIES_COLORS[7] for v in d["change"]], cornerradius=4),
                           customdata=d[["ladder_score_start", "ladder_score_end", "rank_change"]].to_numpy(),
                           hovertemplate="<b>%{y}</b><br>%{customdata[0]:.2f} → %{customdata[1]:.2f} (%{x:+.2f})<br>rank change %{customdata[2]:+d}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=viz.MUTED, width=1))
    fig.update_layout(title=f"Change in ladder score, WHR {s0 + 1} → WHR {s1 + 1} ({len(ch)} countries in both)", height=max(420, 22 * len(d) + 100))
    fig.update_xaxes(title="Δ ladder score")
    c1, c2 = st.columns([1, 1.3])
    with c1:
        ui.chart(fig, key="chg")
    with c2:
        m = ch.merge(rk[rk["year"] == s1][["country", "iso3"]], on="country")
        ui.chart(viz.choropleth(m, "change", diverging=True, hover_cols=["region"], height=520, title="Map of change (blue = improved, red = declined)"), key="chgmap")
    ui.callout("Changes smaller than about the width of the confidence intervals (≈ ±0.1–0.2) are within sampling uncertainty. Countries ranked in only one of the two years are excluded.", "info")

ui.section("Rank movements", "03")
pick = st.multiselect("Countries", sorted(rk["country"].unique()), default=[c for c in ["Costa Rica", "Mexico", "United States", "Lithuania", "Afghanistan", "Lebanon"] if c in set(rk["country"])], max_selections=8)
if pick:
    rs = rk[rk["country"].isin(pick)]
    fig = viz.trend_lines(rs, "rank", order=pick, title="Official rank over time (1 = highest)")
    fig.update_yaxes(autorange="reversed", title="Rank")
    ui.chart(fig, key="bump")

ui.section("Year-over-year changes", "04", "annual panel, consecutive observed years only")
yy = yoy_changes(pfe)
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Largest single-year rises**")
    st.dataframe(ui.fmt_table(yy.sort_values("ladder_yoy_change", ascending=False).head(12)), hide_index=True)
with c2:
    st.markdown("**Largest single-year falls**")
    st.dataframe(ui.fmt_table(yy.head(12)), hide_index=True)
ui.chart(viz.histogram(yy["ladder_yoy_change"], "Distribution of year-over-year changes", xlabel="Δ Life Ladder (1 year)", color=viz.SERIES_COLORS[6]), key="yoyhist")

ui.section("Long-run trend slopes", "05", "linear trend per country with ≥ 8 annual observations")
ts = trend_slopes(panel)
c1, c2 = st.columns(2)
with c1:
    ui.chart(viz.ranking_bars(ts.head(12), "slope_per_year", title="Steepest upward trends", color=viz.SERIES_COLORS[0], fmt="+.3f"), key="up")
with c2:
    dn = ts.tail(12).copy()
    dn["decline"] = -dn["slope_per_year"]
    ui.chart(viz.ranking_bars(dn, "decline", title="Steepest downward trends (points lost per year)", color=viz.SERIES_COLORS[7], fmt=".3f"), key="dn")
ui.footer()
