import pandas as pd
import streamlit as st

from src import ui
from src import visualization as viz
from src.config import INDICATORS, LABELS, REGIONS, SHORT_LABELS, TARGET
from src.geography import load_country_reference

ui.page_header(
    "Global Happiness Explorer",
    "Explore every indicator on an interactive world map. Hover for statistics, click countries to inspect and compare them, "
    "filter by region, and see exactly which countries have no data.",
    "Explore · Interactive world map",
)

b = ui.bundle()
panel, rk = b.panel, b.rankings

INDICATOR_OPTIONS = {
    "Life evaluation — official ranking (3-yr avg)": ("rankings", "ladder_score"),
    "Happiness rank — official ranking": ("rankings", "rank"),
    "Life Ladder — annual panel": ("panel", TARGET),
    **{f"{LABELS[f]} — annual panel": ("panel", f) for f in INDICATORS},
    "Positive affect — annual panel": ("panel", "positive_affect"),
    "Negative affect — annual panel": ("panel", "negative_affect"),
    "Country cluster (k-means profile)": ("cluster", "cluster_name"),
}

c1, c2, c3 = st.columns([2.2, 1, 2])
with c1:
    ind_label = st.selectbox("Indicator", list(INDICATOR_OPTIONS), index=0)
source, col = INDICATOR_OPTIONS[ind_label]
with c2:
    if source == "rankings":
        years = sorted(rk["year"].unique())
        year = st.selectbox("Ranking year", years[::-1], index=0, format_func=lambda y: f"{y} (WHR {y + 1})")
    elif source == "panel":
        years = sorted(panel["year"].unique())
        year = st.selectbox("Survey year", years[::-1], index=0)
    else:
        year = None
        st.selectbox("Period", ["2020–2022 profile"], disabled=True)
with c3:
    regions = st.multiselect("Filter regions", REGIONS, default=[], placeholder="All regions")

if source == "rankings":
    d = rk[rk["year"] == year].copy()
    period_txt = f"WHR {year + 1} ranking · survey years {year - 2}–{year} (3-year averages)"
    src_txt = b.rankings_source
elif source == "panel":
    d = panel[panel["year"] == year].copy()
    period_txt = f"Annual observations for {year}"
    src_txt = b.panel_source
else:
    cl = ui.clustering()
    d = cl.profiles.copy()
    period_txt = f"Country profiles = mean of {', '.join(map(str, cl.info['years']))} (k = {cl.k})"
    src_txt = b.panel_source
if regions:
    d = d[d["region"].isin(regions)]

ref = load_country_reference()
universe = ref if not regions else ref[ref["region"].isin(regions)]
have = set(d.dropna(subset=[col])["country"])
missing = sorted(set(universe["country"]) - have)

ui.stat_row([
    ("Countries with data", f"{len(have)}"),
    ("No data for this view", f"{len(missing)}"),
    ("Mean", f"{d[col].mean():.3f}" if col not in ("cluster_name",) else "—"),
    ("Min → max", f"{d[col].min():.2f} → {d[col].max():.2f}" if col not in ("cluster_name",) else f"{d[col].nunique()} clusters"),
    ("Source", f"<span style='font-size:.8rem'>{src_txt}</span>"),
])

if "rank" in d:
    d["rank_txt"] = d["rank"].astype("Int64").astype(str)
else:
    d["rank_txt"] = "—"
hover_cols = ["region", "rank_txt"] if source == "rankings" else ["region"]
if source == "panel":
    d["ladder_txt"] = d[TARGET].map(lambda v: f"{v:.2f}" if pd.notna(v) else "n/a")
    hover_cols = ["region", "ladder_txt"]

if col == "cluster_name":
    fig = viz.choropleth(d, col, categorical=True, hover_cols=["region"], height=600)
else:
    fig = viz.choropleth(d.dropna(subset=[col]), col, hover_cols=hover_cols, height=600, reverse=(col == "rank" or col in ("perceptions_of_corruption", "negative_affect")))
event = ui.chart(fig, key="explorer_map", on_select="rerun", selection_mode="points")
st.caption(f"{period_txt}. Countries in flat grey have no observation for this indicator and year — no values are imputed on the map. "
           + ("Colour scale reversed so that darker = lower rank number." if col == "rank" else "")
           + (" Colour scale reversed: lighter = lower values." if col in ("perceptions_of_corruption", "negative_affect") else ""))

# ---------------------------------------------------------------- selection
clicked = []
try:
    pts = event.selection["points"] if event and event.selection else []
    iso_to_country = dict(zip(d["iso3"], d["country"]))
    for p in pts:
        loc = p.get("location")
        if loc in iso_to_country:
            clicked.append(iso_to_country[loc])
except Exception:
    pass
all_countries = sorted(set(panel["country"]) | set(rk["country"]))
if "explorer_ms" not in st.session_state:
    st.session_state["explorer_ms"] = [c for c in ("Finland", "Costa Rica", "India") if c in all_countries]
if clicked and tuple(clicked) != st.session_state.get("explorer_last_click"):
    st.session_state["explorer_last_click"] = tuple(clicked)
    merged = list(dict.fromkeys(st.session_state["explorer_ms"] + [c for c in clicked if c in all_countries]))
    st.session_state["explorer_ms"] = merged[-6:]
sel = st.multiselect("Selected countries (click the map or choose here)", all_countries, max_selections=6, key="explorer_ms")

if sel:
    ui.section("Country details", "◉", "latest available observations — missing values are shown as n/a, never estimated")
    cols = st.columns(min(len(sel), 3))
    for i, c in enumerate(sel):
        with cols[i % len(cols)]:
            r_hist = rk[rk["country"] == c].sort_values("year")
            p_hist = panel[panel["country"] == c].sort_values("year")
            last_r = r_hist.iloc[-1] if len(r_hist) else None
            last_p = p_hist.iloc[-1] if len(p_hist) else None
            rows = []
            if last_p is not None:
                for f in INDICATORS:
                    v = last_p[f]
                    rows.append(f"<tr><td style='color:var(--muted);padding:2px 10px 2px 0'>{SHORT_LABELS[f]}</td><td style='text-align:right;color:var(--text)'>{'n/a' if pd.isna(v) else f'{v:.3f}'}</td></tr>")
            region = (last_r["region"] if last_r is not None else last_p["region"]) if (last_r is not None or last_p is not None) else "—"
            head = (f"<div class='big-number' style='font-size:2.6rem'>{last_r['ladder_score']:.2f}</div><div style='color:var(--muted);font-size:.8rem'>Rank {int(last_r['rank'])} · WHR {int(last_r['year']) + 1}</div>"
                    if last_r is not None else "<div style='color:var(--muted)'>Not in official rankings</div>")
            st.markdown(
                f"<div class='card glow'><div style='display:flex;justify-content:space-between;align-items:flex-start'><div><h4>{ui.esc(c)}</h4>"
                f"<div style='font-size:.78rem;color:var(--muted)'>{ui.esc(region)}</div></div><div style='text-align:right'>{head}</div></div>"
                f"<div style='margin-top:10px;font-size:.78rem;color:var(--muted)'>Annual panel {int(last_p['year']) if last_p is not None else ''} · Life Ladder "
                f"{'n/a' if last_p is None else f'{last_p[TARGET]:.2f}'}</div><table style='width:100%;font-size:.84rem;margin-top:6px'>{''.join(rows)}</table></div>",
                unsafe_allow_html=True,
            )
            if len(r_hist) > 1:
                ui.chart(viz.sparkline(r_hist["ladder_score"].tolist(), height=70), key=f"spark_{c}")
                st.caption(f"Official ladder score {int(r_hist['year'].min())}–{int(r_hist['year'].max())}")
    if len(sel) >= 2:
        ui.section("Quick comparison", "⇄")
        y_cmp = int(rk["year"].max())
        cmp = rk[(rk["country"].isin(sel)) & (rk["year"] == y_cmp)][["country", "rank", "ladder_score", "lower_whisker", "upper_whisker", "region"]]
        not_ranked = sorted(set(sel) - set(cmp["country"]))
        ui.chart(viz.ranking_bars(cmp, "ladder_score", ci=("lower_whisker", "upper_whisker"), title=f"Official ladder score with 95% CI · WHR {y_cmp + 1}"), key="cmp_bars")
        if not_ranked:
            ui.callout(f"Not ranked in WHR {y_cmp + 1}: {', '.join(not_ranked)}.", "warn")
        st.page_link("views/5_Country_Comparison.py", label="Deep comparison (radar, trends, statistics)", icon=":material/compare_arrows:")

with st.expander(f"Countries without data in this view ({len(missing)})"):
    if missing:
        st.write(", ".join(missing))
        st.caption("These countries appear elsewhere in the WHR data but have no observation for the selected indicator/year (survey not fielded, indicator not collected, or excluded by WHR for sample-size reasons).")
    else:
        st.write("Every country in the reference list has data for this view.")
    no_iso = sorted(set(d["country"]) - set(d.dropna(subset=["iso3"])["country"]))
    if no_iso:
        st.caption(f"Have data but cannot be drawn (no separate map geometry): {', '.join(no_iso)}.")

with st.expander("Data table for this view"):
    show = [c for c in ["country", "region", "year", "rank", col, "lower_whisker", "upper_whisker"] if c in d.columns]
    tbl = d[show].sort_values(col, ascending=col in ("rank",) or col == "cluster_name")
    st.dataframe(ui.fmt_table(tbl), hide_index=True, height=380)
    st.download_button("Download CSV", tbl.to_csv(index=False).encode(), file_name=f"whr_map_{col}_{year or 'profile'}.csv", mime="text/csv")
ui.footer()
