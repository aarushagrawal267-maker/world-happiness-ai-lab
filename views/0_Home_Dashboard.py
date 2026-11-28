import streamlit as st

from src import ui
from src import visualization as viz
from src.eda import dataset_overview, ranking_overview

b = ui.bundle()
res = ui.results()
panel, rk = b.panel, b.rankings
ov = dataset_overview(panel)
ro = ranking_overview(rk) if not rk.empty else None

ui.hero(
    "World Happiness AI Lab",
    "A research platform for understanding how economic, social and institutional conditions relate to reported life "
    "evaluations across countries — with rigorous statistics, validated machine learning, explainable AI and model-based "
    "what-if scenarios. Every number on these pages is computed live from World Happiness Report data.",
    eyebrow="Understanding and predicting global happiness",
    chips=[
        f"<b>{ov['observations']:,}</b> country-years",
        f"<b>{ov['countries']}</b> countries",
        f"<b>{ov['year_min']}–{ov['year_max']}</b> annual panel",
        f"rankings to <b>WHR {ro['edition']}</b>" if ro else "",
        f"<b>{len(b.files)}</b> source files",
    ],
)

if ro:
    best = res["best_model"] if res else None
    ui.kpi_cards([
        {"label": "Latest ranking", "value": f"WHR {ro['edition']}", "sub": f"survey years {ro['year'] - 2}–{ro['year']}"},
        {"label": "Countries ranked", "value": f"{ro['countries']}", "sub": f"panel covers {ov['countries']} countries"},
        {"label": "Average score", "value": ui.score(ro['mean_score']), "sub": "unweighted mean of countries"},
        {"label": "Highest score", "value": ui.score(ro['max_score'], ".3f"), "sub": ro["max_country"]},
        {"label": "Lowest score", "value": ui.score(ro['min_score'], ".3f"), "sub": ro["min_country"]},
        {"label": "Best model", "value": best["name"] if best else "—", "sub": f"RMSE {best['test']['rmse']:.3f} on unseen countries" if best else "train in ML Training Lab"},
    ])

c1, c2 = st.columns([1.65, 1], gap="large")
with c1:
    ui.section("The world, ranked", "01", f"official 3-year average life evaluations · WHR {ro['edition'] if ro else ''}")
    if ro:
        latest = rk[rk["year"] == ro["year"]].copy()
        latest["rank_txt"] = latest["rank"].astype("Int64").astype(str)
        ui.chart(viz.choropleth(latest, "ladder_score", hover_cols=["rank_txt", "region"], height=470), key="home_map")
        st.page_link("views/1_Global_Happiness_Explorer.py", label="Open the full interactive explorer", icon=":material/arrow_forward:")
with c2:
    ui.section("Top 10", "02", "highest reported life evaluations")
    if ro:
        top = latest.nsmallest(10, "rank")
        fig = viz.ranking_bars(top, "ladder_score", ci=("lower_whisker", "upper_whisker"), height=470)
        # thin ruled bars on the full 0–10 ladder scale
        fig.update_traces(width=0.28, marker_cornerradius=2)
        fig.update_xaxes(range=[0, 10], dtick=2, title="Ladder score / 10")
        fig.update_layout(margin=dict(r=40))
        ui.chart(fig, key="home_top")

if res:
    ui.section("Model performance at a glance", "03", "same data, three different questions — accuracy depends on what you ask")
    best = res["best_model"]
    cols = st.columns(3)
    labels = {"random": ("Random split", "familiar countries, other years"), "group": ("Unseen countries", "country-grouped hold-out"),
              "temporal": ("Future years", "train ≤ 2016 · test ≥ 2019")}
    for col, (s, (t, d)) in zip(cols, labels.items()):
        m = res["experiments"][s]["models"][best["key"]]["test"]
        with col:
            st.markdown(
                f"<div class='card'><div class='stat'><div class='l'>{t}</div>"
                f"<div class='v' style='font-size:1.875rem'>{m['rmse']:.3f}<small class='scale'>RMSE</small></div>"
                f"<div style='font-size:.8125rem;margin-top:4px;font-variant-numeric:tabular-nums'>MAE {m['mae']:.3f} · R² {m['r2']:.2f} · n = {m['n']}</div></div>"
                f"<div class='meta'>{d} · {best['name']}</div></div>",
                unsafe_allow_html=True,
            )
    ui.callout(f"The prediction model is <b>{best['name']}</b>, selected by validation RMSE on held-out countries (not R² alone). "
               f"Its 90% prediction interval is ±{res['conformal']['q_cv']:.2f} ladder points, with {res['conformal']['empirical_test_coverage_split']:.0%} empirical coverage on unseen test countries.", "ok")

ui.section("Launch a tool", "04")
tiles = [
    ("views/1_Global_Happiness_Explorer.py", "Global explorer", "Interactive world map of every indicator, year and region."),
    ("views/12_Happiness_Prediction.py", "Happiness prediction", "Estimate a life-evaluation score from six indicators with an interval."),
    ("views/13_What_If_Simulator.py", "What-if simulator", "Model-based hypothetical scenarios with saved comparisons."),
    ("views/5_Country_Comparison.py", "Compare countries", "Radar profiles, trends and side-by-side statistics."),
    ("views/11_Explainable_AI.py", "Explainable AI", "SHAP, permutation importance and partial dependence."),
    ("views/14_Automated_Insights.py", "Automated insights", "Findings generated live from the data, with sources."),
]
cols = st.columns(3)
for i, (page, t, d) in enumerate(tiles):
    with cols[i % 3]:
        st.markdown(f"<div class='feature-tile'><div class='t'>{t}</div><div class='d'>{d}</div></div>", unsafe_allow_html=True)
        st.page_link(page, label=f"Open {t}", icon=":material/arrow_forward:")

ui.section("About the data", "05")
a, bcol = st.columns(2, gap="large")
with a:
    ui.card(
        "Annual country-year panel (primary dataset)",
        f"{ov['observations']:,} observations of {ov['countries']} countries, {ov['year_min']}–{ov['year_max']}, with Life Ladder and "
        f"{ov['features']} explanatory indicators. Mean {ov['mean_score']:.2f}; highest {ov['max_score']:.2f} ({ov['max_country']} {ov['max_year']}); "
        f"lowest {ov['min_score']:.2f} ({ov['min_country']} {ov['min_year']}). {ov['missing_pct']:.1f}% of indicator cells are missing; "
        f"{ov['duplicate_country_years']} duplicate country-years.",
        meta=f"source: {b.panel_source}",
    )
with bcol:
    ui.card(
        "Official rankings and edition archive",
        f"Official WHR ranking tables for {rk['year'].nunique()} ranking years ({int(rk['year'].min())}–{int(rk['year'].max())}) plus "
        f"{len(b.editions)} single-edition files (WHR {min(b.editions)}–{max(b.editions)}), each standardised and validated. "
        "Ladder scores in rankings are 3-year averages, so they differ from single-year panel values.",
        meta=f"source: {b.rankings_source} + edition files",
    )
ui.callout("Happiness scores are <b>reported life evaluations</b> (the Cantril ladder, 0–10) averaged nationally — not an objective measurement of how happy every individual is.", "info")
ui.footer()
