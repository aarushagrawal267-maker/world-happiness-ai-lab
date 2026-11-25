import math

import streamlit as st

from src import ui
from src import visualization as viz
from src.config import TARGET
from src.explainability import explain_instance
from src.prediction import extreme_warning, input_specs, mahalanobis_extrapolation, nearest_profiles, predict_one, validate_inputs
from src.train import build_modeling_frame

ui.page_header(
    "Happiness Prediction",
    "Enter a country profile and the model estimates the national average life evaluation typical of historical country-years with similar indicators — "
    "with a calibrated 90% prediction interval and an explanation.",
    "Simulate · Predict",
)
res = ui.ensure_trained()
reg = ui.registry()
b = ui.bundle()
panel = b.panel
# Modelling frame = panel + backward-looking engineered features (e.g. gdp_growth_1y) that some saved models use.
frame = build_modeling_frame(panel)

lead = [k for k in ("production_best", "production_best_affect") if k in reg]
choices = lead + sorted(k for k, v in reg.items() if k.startswith(("production_", "custom_")) and k not in lead and not k.endswith("_difficulty"))
c1, c2 = st.columns([1.2, 2])
with c1:
    mname = st.selectbox("Model", choices, format_func=lambda n: f"★ {reg[n]['model_name']} — 6 core indicators (recommended)" if n == "production_best"
                         else (f"★ {reg[n]['model_name']} — 8 inputs (better for crisis profiles)" if n == "production_best_affect" else f"{reg[n]['model_name']} · {n}"))
meta = reg[mname]
pipe = ui.require_model(mname)
feats = meta["features"]
specs = input_specs(meta["feature_ranges"], feats)
adaptive = meta.get("adaptive")
difficulty = ui.model(mname + "_difficulty") if adaptive else None
q = res["conformal"]["q_cv"] if mname == "production_best" else None
# Independent coverage check on unseen countries (computed for the 6-indicator best model only).
holdout = (res.get("adaptive_interval") or {}).get("holdout") if mname == "production_best" else None
interval_ok = bool(adaptive and difficulty is not None) or q is not None
if mname == "production_best_affect":
    ui.callout("This model also uses <b>positive and negative affect</b> (yesterday's emotions). They are themselves wellbeing measures, so this model is closer to "
               "'explaining wellbeing with wellbeing' — but it captures crisis states (conflict, collapse) noticeably better: "
               f"unseen-country RMSE {adaptive['oof_rmse']:.3f} vs {reg['production_best']['adaptive']['oof_rmse']:.3f} for the 6-indicator model.", "info")
with c2:
    st.markdown(f"<div style='padding-top:28px'>{ui.tags([(meta['model_name'], 'blue'), ('features: ' + meta['feature_set'], ''), ('trained on ' + str(meta.get('trained_on_rows', '—')) + ' country-years', ''), ('adaptive 90% interval' if adaptive else ('90% interval' if q else 'no calibrated interval'), 'green' if interval_ok else 'orange')])}</div>", unsafe_allow_html=True)

# --------------------------------------------------- prefill
with st.expander("Start from a real country profile (optional)", expanded=False):
    cc1, cc2, cc3 = st.columns([2, 1, 1])
    with cc1:
        pc = st.selectbox("Country", sorted(panel["country"].unique()), index=sorted(panel["country"].unique()).index("Brazil") if "Brazil" in set(panel["country"]) else 0)
    rows = frame[frame["country"] == pc].sort_values("year")
    with cc2:
        py = st.selectbox("Year", rows["year"].tolist()[::-1])
    with cc3:
        st.write("")
        st.write("")
        if st.button("Load values"):
            r = rows[rows["year"] == py].iloc[0]
            for f, s in specs.items():
                if s.is_categorical:
                    st.session_state[f"pred_{f}"] = r.get(f, s.categories[0])
                else:
                    v = r.get(f)
                    st.session_state[f"pred_{f}"] = float(s.default if v is None or (isinstance(v, float) and math.isnan(v)) else v)
            st.session_state["pred_loaded"] = f"{pc} {py}"
            st.rerun()
if st.session_state.get("pred_loaded"):
    st.caption(f"Form pre-filled from {st.session_state['pred_loaded']} (missing indicators replaced by the training median).")

# --------------------------------------------------- form
with st.form("predict"):
    st.markdown("**Country indicators**")
    values = {}
    cols = st.columns(3)
    for i, (f, s) in enumerate(specs.items()):
        with cols[i % 3]:
            if s.is_categorical:
                values[f] = st.selectbox(s.label, s.categories, key=f"pred_{f}")
            else:
                if f"pred_{f}" not in st.session_state:
                    st.session_state[f"pred_{f}"] = round(s.default, 4)
                values[f] = st.number_input(s.label, min_value=s.hard_min, max_value=s.hard_max, step=s.step, format="%.3f" if s.step < 0.1 else "%.1f",
                                            key=f"pred_{f}", help=f"{s.help}\n\nObserved range: {s.train_min:.3g} – {s.train_max:.3g}")
                if f == "log_gdp_per_capita":
                    st.caption(f"≈ ${math.exp(values[f]):,.0f} per person (PPP, constant international $)")
                else:
                    st.caption(f"observed {s.train_min:.3g} – {s.train_max:.3g} · median {s.default:.3g}")
    go_btn = st.form_submit_button("Estimate life evaluation", type="primary")

if go_btn or st.session_state.get("pred_last"):
    if go_btn:
        st.session_state["pred_last"] = values
    values = st.session_state["pred_last"]
    errors, warns = validate_inputs(values, specs)
    if errors:
        for e in errors:
            ui.callout(e, "danger")
        st.stop()
    out = predict_one(pipe, values, feats, q, difficulty=difficulty, adaptive=adaptive)
    maha = mahalanobis_extrapolation(panel, values, feats)
    c1, c2 = st.columns([1, 1.25], gap="large")
    with c1:
        ui.chart(viz.gauge(out["prediction"], out.get("lower"), out.get("upper"), "Estimated Life Ladder (0–10)"), key="gauge")
        if "lower" in out:
            st.markdown(f"<div style='text-align:center;margin-top:-10px'><span class='tag blue'>90% prediction interval: {out['lower']:.2f} – {out['upper']:.2f}</span></div>", unsafe_allow_html=True)
            if adaptive:
                st.caption(f"Adaptive conformal interval: ±{out['interval_halfwidth']:.2f} for this profile (range across countries "
                           f"±{adaptive['min_halfwidth']:.2f} to ±{adaptive['max_halfwidth']:.2f}). It widens where the model has historically erred more. "
                           + (f"Held-out check on {holdout['countries_test']} unseen countries: {holdout['coverage_all']:.0%} coverage overall, "
                              f"{holdout['coverage_very_low']:.0%} for actual scores below 4 (only {holdout['n_test_very_low']} rows), "
                              "so treat it as a guide, not a guarantee." if holdout else
                              f"Cross-validated coverage: {adaptive['coverage_all']:.0%} overall (calibrated to 90% by construction), "
                              f"{adaptive['coverage_very_low']:.0%} for very low scores."))
            else:
                st.caption(f"Split-conformal interval (±{q:.2f}) calibrated on countries unseen during training.")
        else:
            st.caption("Calibrated intervals are computed for the best production model only.")
    with c2:
        ui.stat_row([("Model", meta["model_name"]), ("Point estimate", f"{out['prediction']:.3f}"),
                     ("More unusual than", f"{maha['percentile']:.0f}% of data"), ("Global mean", f"{panel[TARGET].mean():.2f}")])
        ew = extreme_warning(out["prediction"], adaptive)
        if ew:
            ui.callout(ew, "warn", "⇅")
        for w in warns:
            ui.callout(w, "warn")
        if maha["unusual"]:
            ui.callout(f"This <b>combination</b> of values is more unusual than {maha['percentile']:.1f}% of observed country-years (Mahalanobis distance {maha['distance']:.1f}). "
                       "Even if each value is plausible alone, the model has seen few profiles like it.", "warn")
        elif not warns:
            ui.callout("All inputs lie inside the training range and the profile resembles observed country-years.", "ok")
        st.markdown("**Most similar observed country-years**")
        nn = nearest_profiles(panel, values, feats, 6)
        st.dataframe(ui.fmt_table(nn[["country", "year", "region", "life_ladder", "distance"]]), hide_index=True)

    ui.section("Why this estimate?", "")
    c1, c2 = st.columns([1.3, 1])
    import pandas as pd

    row = pd.DataFrame([{f: values[f] for f in feats}])
    ex = explain_instance(pipe, frame[feats].dropna(), row)
    with c1:
        ui.chart(viz.waterfall(ex["contributions"], ex["base_value"], ex["prediction"], f"Contribution of each input · {ex['method']}"), key="pwf")
    with c2:
        if mname.startswith("production_") and reg.get(f"eval_group_{meta['model_key']}", {}).get("features") == feats:
            pi = ui.permutation_importance(f"eval_group_{meta['model_key']}", "group")
            ui.chart(viz.importance_bars(pi, "importance_mean", "importance_std", title="Global feature importance (permutation)", xtitle="Δ RMSE"), key="ppi")
    ui.section("Limitations", "")
    ui.card("What this estimate is — and is not",
            "It is the model's estimate of the <b>national average</b> Cantril-ladder answer that historical country-years with similar indicators reported. "
            "It cannot tell how happy any individual is, does not account for culture, conflict, inequality or events not captured by the inputs, and reflects "
            "associations — not the effect of changing any indicator. Typical error for unseen countries is about "
            f"±{res['best_model']['test']['rmse']:.2f} ladder points (RMSE).")
ui.footer()
