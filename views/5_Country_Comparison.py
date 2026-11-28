import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src import ui
from src import visualization as viz
from src.config import INDICATORS, LABELS, SHORT_LABELS, TARGET
from src.country_comparison import (
    COMPARE_FEATURES,
    availability,
    common_years,
    comparison_table,
    percentile_profile,
    radar_axis_labels,
    trend_series,
)

ui.page_header(
    "Country comparison",
    "Compare two to six countries side by side: percentile radar profiles, indicator-by-indicator bars, historical trends and data availability.",
    "Explore · Compare nations",
)

b = ui.bundle()
panel, rk = b.panel, b.rankings
countries = sorted(panel["country"].unique())
default = [c for c in ["Finland", "United States", "India", "Brazil"] if c in countries]
sel = st.multiselect("Countries (2–6)", countries, default=st.session_state.get("cmp_sel", default), max_selections=6)
st.session_state["cmp_sel"] = sel
if len(sel) < 2:
    st.info("Select at least two countries.")
    st.stop()

cy = common_years(panel, sel)
yrs = sorted(panel["year"].unique())[::-1]
c1, c2 = st.columns([1, 3])
with c1:
    year = st.selectbox("Comparison year (annual panel)", yrs, index=yrs.index(max(cy)) if cy else 0, help="Defaults to the latest year in which all selected countries were surveyed.")
with c2:
    if not cy:
        ui.callout("The selected countries were never all surveyed in the same year; some values below will be unavailable.", "warn")
    elif year not in cy:
        ui.callout(f"Not every selected country has an observation in {year}. Common years: {', '.join(map(str, cy[-8:]))}.", "warn")
    else:
        ui.callout(f"All selected countries were surveyed in {year}. Values are single-year national means from the annual panel.", "ok")

tbl = comparison_table(panel, sel, year)
cmap = viz.color_map(sel)

# --------------------------------------------------------- side-by-side cards
cols = st.columns(len(sel))
for col, (_, r) in zip(cols, tbl.iterrows()):
    with col:
        last = rk[(rk["country"] == r["country"])].sort_values("year").tail(1)
        rank_txt = f"Rank {int(last['rank'].iloc[0])} · WHR {int(last['year'].iloc[0]) + 1}" if len(last) else "not ranked"
        rows = "".join(
            f"<tr><td style='color:var(--muted);padding:2px 8px 2px 0;font-size:.8rem'>{SHORT_LABELS[f]}</td><td style='text-align:right;color:var(--text);font-size:.85rem'>{'n/a' if np.isnan(r[f]) else f'{r[f]:.3f}'}</td></tr>"
            for f in INDICATORS
        )
        st.markdown(
            f"<div class='card' style='border-top:2px solid {cmap[r['country']]}'><h4>{ui.esc(r['country'])}</h4><div style='font-size:.74rem;color:var(--muted)'>{rank_txt}</div>"
            f"<div class='big-number' style='font-size:2.4rem;margin:8px 0'>{'n/a' if np.isnan(r[TARGET]) else ui.score(r[TARGET])}</div>"
            f"<div style='font-size:.72rem;color:var(--muted);margin-bottom:6px'>Life Ladder {year}</div><table style='width:100%'>{rows}</table></div>",
            unsafe_allow_html=True,
        )

c1, c2 = st.columns([1.1, 1], gap="large")
with c1:
    prof = percentile_profile(panel, sel, year)
    ui.chart(viz.radar(prof, COMPARE_FEATURES, radar_axis_labels(), f"Percentile profile among countries surveyed in {year}"), key="radar")
    st.caption("Each axis = percentile rank among all countries observed that year (100 = highest). The corruption axis is reversed (outward = lower perceived corruption). Gaps = no data.")
with c2:
    feats = COMPARE_FEATURES
    fig = make_subplots(rows=4, cols=2, subplot_titles=[SHORT_LABELS[f] for f in feats], vertical_spacing=0.09, horizontal_spacing=0.12)
    for i, f in enumerate(feats):
        rr, cc = i // 2 + 1, i % 2 + 1
        fig.add_trace(go.Bar(x=tbl[f], y=tbl["country"], orientation="h", marker=dict(color=[cmap[c] for c in tbl["country"]], cornerradius=3),
                             hovertemplate="%{y}: %{x:.3f}<extra>" + SHORT_LABELS[f] + "</extra>", showlegend=False), row=rr, col=cc)
    fig.update_annotations(font=dict(size=11, color=viz.MUTED))
    fig.update_yaxes(tickfont=dict(size=10))
    fig.update_layout(height=620, title=f"Indicator values ({year}) — each panel on its own scale")
    ui.chart(fig, key="smallmult")

ui.section("Historical comparison", "", "annual panel and official ranking series")
f = st.selectbox("Indicator", [TARGET] + INDICATORS + ["positive_affect", "negative_affect"], format_func=lambda c: LABELS[c])
ts = trend_series(panel, sel, f)
ui.chart(viz.trend_lines(ts, f, order=sel, title=f"{LABELS[f]} — annual observations"), key="trend_p")
rs = rk[rk["country"].isin(sel)]
if not rs.empty:
    ui.chart(viz.trend_lines(rs, "ladder_score", order=[c for c in sel if c in set(rs["country"])], band=("lower_whisker", "upper_whisker"),
                             title="Official ranking score (3-year averages, 95% CI bands)"), key="trend_r")
    rank_tbl = rs.pivot_table(index="year", columns="country", values="rank").sort_index(ascending=False)
    with st.expander("Rank history table"):
        st.dataframe(rank_tbl.astype("Int64"))

ui.section("Data availability & comparability", "")
st.dataframe(availability(panel, rk, sel), hide_index=True)
ui.callout("Annual panel values (single survey year) and official ranking scores (3-year averages) are different quantities and should not be mixed. "
           "Across WHR editions, small methodological revisions (e.g. updated GDP series or HLE sources) mean historical values can differ slightly between editions; "
           "within this platform each series comes from a single, internally consistent file.", "info")
ui.footer()
