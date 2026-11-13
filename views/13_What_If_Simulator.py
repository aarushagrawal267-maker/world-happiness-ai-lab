import math

import pandas as pd
import streamlit as st

from src import ui
from src import visualization as viz
from src.config import INDICATORS, LABELS, SHORT_LABELS
from src.prediction import input_specs
from src.scenario_simulator import (
    gdp_percent_to_log_delta,
    log_delta_to_gdp_percent,
    one_at_a_time,
    scenarios_table,
    simulate,
    tornado,
)

ui.page_header(
    "What-If Happiness Simulator",
    "Adjust a country's indicators and see how the trained model's estimate changes. Save scenarios and compare them side by side.",
    "Simulate · Model-based hypothetical scenarios",
)
st.markdown(
    "<div class='callout warn'><span class='ci'>⚗</span><div><b>Model-based hypothetical scenario simulator.</b> Results show how the model's estimate responds "
    "when inputs change, based on patterns across countries in historical data. Moving a slider is <b>not</b> a real-world causal intervention: it does not "
    "predict what would happen if a country changed a policy, and it guarantees nothing. Changes that leave the range of observed data are flagged as extrapolation.</div></div>",
    unsafe_allow_html=True,
)

res = ui.ensure_trained()
reg = ui.registry()
panel = ui.bundle().panel
meta = reg["production_best"]
pipe = ui.model("production_best")
feats = meta["features"]
ranges = meta["feature_ranges"]
specs = input_specs(ranges, feats)
model_name = meta["model_name"]

# --------------------------------------------------- baseline
c1, c2, c3 = st.columns([1.6, 1, 1.4])
latest = panel.sort_values("year").groupby("country").tail(1)
with c1:
    base_country = st.selectbox("Baseline country", ["Global median profile"] + sorted(latest["country"]), index=1 + sorted(latest["country"]).index("India") if "India" in set(latest["country"]) else 0)
if base_country == "Global median profile":
    base = {f: ranges[f]["median"] for f in feats if "median" in ranges[f]}
    base_label, filled = "Global median", []
    with c2:
        st.selectbox("Year", ["—"], disabled=True)
else:
    rows = panel[panel["country"] == base_country].sort_values("year")
    with c2:
        by = st.selectbox("Year", rows["year"].tolist()[::-1])
    r = rows[rows["year"] == by].iloc[0]
    base, filled = {}, []
    for f in feats:
        v = r.get(f)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            base[f] = ranges[f]["median"]
            filled.append(SHORT_LABELS.get(f, f))
        else:
            base[f] = float(v)
    base_label = f"{base_country} {by}"
with c3:
    bench = st.selectbox("Benchmark preset (optional)", ["None"] + sorted(latest["country"]), help="Copy selected indicators from another country's latest observation.")
if filled:
    ui.callout(f"Missing baseline indicators replaced by the training median: {', '.join(filled)}.", "warn")

# reset sliders when baseline changes
sig = (base_label,)
if st.session_state.get("wi_sig") != sig:
    st.session_state["wi_sig"] = sig
    for f in INDICATORS:
        st.session_state.pop(f"wi_{f}", None)

if bench != "None":
    br = latest[latest["country"] == bench].iloc[0]
    bfeat = st.multiselect(f"Indicators to copy from {bench} ({int(br['year'])})", [f for f in feats if pd.notna(br.get(f))], format_func=lambda f: SHORT_LABELS.get(f, f))
    if st.button(f"Apply {bench}'s values"):
        for f in bfeat:
            delta = float(br[f]) - base[f]
            st.session_state[f"wi_{f}"] = round(log_delta_to_gdp_percent(delta), 1) if f == "log_gdp_per_capita" else round(delta, 3)
        st.rerun()

# --------------------------------------------------- sliders
ui.section("Adjust indicators", "01", "changes relative to the baseline")
mod = dict(base)
cols = st.columns(3)
slider_cfg = {
    "log_gdp_per_capita": ("GDP per capita change (%)", -60.0, 150.0, 1.0),
    "social_support": ("Social support change (share)", -0.30, 0.30, 0.005),
    "healthy_life_expectancy": ("Healthy life expectancy change (years)", -12.0, 12.0, 0.5),
    "freedom": ("Freedom change (share)", -0.30, 0.30, 0.005),
    "generosity": ("Generosity change", -0.30, 0.30, 0.005),
    "perceptions_of_corruption": ("Perceived corruption change (share)", -0.40, 0.40, 0.005),
}
for i, f in enumerate([f for f in feats if f in slider_cfg]):
    label, lo, hi, step = slider_cfg[f]
    with cols[i % 3]:
        k = f"wi_{f}"
        st.session_state[k] = float(min(max(st.session_state.get(k, 0.0), lo), hi))
        v = st.slider(label, lo, hi, step=step, key=k)
        if f == "log_gdp_per_capita":
            mod[f] = base[f] + gdp_percent_to_log_delta(v)
            st.caption(f"${math.exp(base[f]):,.0f} → **${math.exp(mod[f]):,.0f}** (log {base[f]:.2f} → {mod[f]:.2f})")
        else:
            mod[f] = base[f] + v
            st.caption(f"{base[f]:.3f} → **{mod[f]:.3f}** · observed range {specs[f].train_min:.2f}–{specs[f].train_max:.2f}")
for f in feats:  # non-slider features (e.g. year / region in engineered models) stay at baseline
    mod.setdefault(f, base[f])

sc = simulate(pipe, feats, base, mod, specs, name="current", base_label=base_label, model_name=model_name)

# --------------------------------------------------- results
ui.section("Model response", "02")
c1, c2 = st.columns([1, 1.3], gap="large")
with c1:
    ui.chart(viz.gauge(sc.modified_prediction, title="Scenario estimate (Life Ladder)", reference=sc.base_prediction if abs(sc.delta) >= 0.005 else None), key="wi_gauge")
with c2:
    ui.kpi_cards([
        {"label": "Original prediction", "value": f"{sc.base_prediction:.2f}", "sub": base_label, "icon": "○"},
        {"label": "Scenario prediction", "value": f"{sc.modified_prediction:.2f}", "sub": "modified inputs", "icon": "●"},
        {"label": "Difference", "value": f"{(0.0 if abs(sc.delta) < 0.005 else sc.delta):+.2f}", "sub": "ladder points", "icon": "Δ",
         "accent": "linear-gradient(90deg,#4FE3C1,#9BE57A)" if sc.delta >= 0 else "linear-gradient(90deg,#FF6B8B,#FFB86B)"},
    ])
    rmse = res["best_model"]["test"]["rmse"]
    if abs(sc.delta) < 0.1:
        ui.callout(f"A change of {(0.0 if abs(sc.delta) < 0.005 else sc.delta):+.2f} is very small relative to the model's typical error for unseen countries (RMSE {rmse:.2f}).", "info")
    for w in sc.warnings:
        ui.callout(w, "danger")
    ch = sc.changes()
    if not ch.empty:
        st.dataframe(ui.fmt_table(ch.drop(columns=["feature"])), hide_index=True)
    else:
        st.caption("Move a slider to create a scenario.")
    st.caption(f"Model: {model_name} (production). Tree ensembles respond in steps and flatten outside the observed range; linear models extrapolate linearly.")

ui.section("Sensitivity", "03", "one indicator at a time, everything else held at the scenario values")
c1, c2 = st.columns([1, 1.2])
with c1:
    tor = tornado(pipe, feats, mod, specs, ranges)
    ui.chart(viz.tornado_chart(tor, "Effect of a ±1 SD change in each indicator alone"), key="tornado")
with c2:
    f_sw = st.selectbox("Sweep indicator", [f for f in feats if f in slider_cfg], format_func=lambda f: LABELS[f])
    curve = one_at_a_time(pipe, feats, mod, specs, f_sw)
    ui.chart(viz.sweep_plot(curve, f_sw, base[f_sw], mod[f_sw], f"Model response across the observed range of {SHORT_LABELS[f_sw]}"), key="sweep")

# --------------------------------------------------- scenarios
ui.section("Saved scenarios", "04", "kept for this browser session")
saved = st.session_state.setdefault("scenarios", [])
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    sname = st.text_input("Scenario name", value=f"Scenario {len(saved) + 1}")
with c2:
    st.write("")
    st.write("")
    if st.button("💾 Save scenario", type="primary"):
        s2 = simulate(pipe, feats, base, mod, specs, name=sname, base_label=base_label, model_name=model_name)
        saved.append(s2)
        st.rerun()
with c3:
    st.write("")
    st.write("")
    if st.button("Clear all") and saved:
        st.session_state["scenarios"] = []
        st.rerun()
if saved:
    tbl = scenarios_table(saved)
    st.dataframe(ui.fmt_table(tbl), hide_index=True)
    same_base = [s for s in saved if s.base_label == saved[-1].base_label]
    ui.chart(viz.scenario_compare([s.name for s in same_base], [s.modified_prediction for s in same_base], same_base[-1].base_prediction,
                                  f"Scenarios for baseline {same_base[-1].base_label}"), key="sc_cmp")
    st.download_button("Download scenarios (CSV)", tbl.to_csv(index=False).encode(), file_name="what_if_scenarios.csv", mime="text/csv")
else:
    st.caption("No saved scenarios yet. Adjust the sliders and press *Save scenario* to build a comparison.")
ui.footer()
