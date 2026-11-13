import pandas as pd
import streamlit as st

from src import ui
from src import visualization as viz
from src.config import LABELS, SHORT_LABELS, TARGET
from src.explainability import SHAP_AVAILABLE, explain_instance, native_importance, pdp, shap_explanation, shap_importance
from src.regression_models import MODEL_SPECS
from src.train import build_modeling_frame, make_splits

ui.page_header(
    "Explainable AI",
    "Which inputs does the model rely on, how do predictions change as an input varies, and why did the model predict what it did for a given country?",
    "Machine learning · Explain",
)
ui.callout("<b>Feature importance is model-dependent and is not causal.</b> It shows what a particular fitted model uses to reproduce patterns in historical data. "
           "Correlated indicators (income, health, social support) share and trade importance between them, and a different model can rank them differently.", "warn", "⚖")

res = ui.ensure_trained()
reg = ui.registry()
b = ui.bundle()
keys = list(res["experiments"]["group"]["models"])
k = st.selectbox("Model to explain", keys, index=keys.index(res["best_model"]["key"]), format_func=lambda x: MODEL_SPECS[x].name + (" ★ best" if x == res["best_model"]["key"] else ""))
eval_name = f"eval_group_{k}"
pipe = ui.model(eval_name)
feats = reg[eval_name]["features"]
frame = build_modeling_frame(b.panel)
split = make_splits(frame, "group")
Xtr, Xte, yte = split.train[feats], split.test[feats], split.test[TARGET]
st.caption(f"Explanations use the model fitted on the training countries of the unseen-countries split; importance is measured on {len(Xte)} test rows from {split.test['country'].nunique()} countries it never saw.")

t1, t2, t3, t4 = st.tabs(["Global importance", "SHAP", "Partial dependence", "Explain a country"])

with t1:
    pi = ui.permutation_importance(eval_name, "group")
    c1, c2 = st.columns(2)
    with c1:
        ui.chart(viz.importance_bars(pi, "importance_mean", "importance_std", title="Permutation importance (unseen-country test set)", xtitle="Increase in RMSE when shuffled"), key="pi")
        st.caption("Each feature is randomly shuffled 10 times; the bar is the average loss of accuracy. Model-agnostic and measured on unseen data.")
    with c2:
        ni = native_importance(pipe)
        if ni is not None:
            ui.chart(viz.importance_bars(ni, "importance", title=ni["kind"].iloc[0], xtitle="importance", color=viz.SERIES_COLORS[2]), key="ni")
            if "signed_coef" in ni:
                st.dataframe(ni[["feature", "signed_coef"]].assign(feature=lambda d: d["feature"].map(lambda f: SHORT_LABELS.get(f, f))), hide_index=True)
            else:
                st.caption("Impurity importance is computed on training data and is biased toward high-cardinality continuous features; prefer permutation importance.")

with t2:
    if not SHAP_AVAILABLE:
        ui.callout("SHAP is not installed — permutation importance and partial dependence remain available as fallbacks.", "warn")
    else:
        n = st.slider("Rows explained", 100, len(Xte), min(300, len(Xte)), step=50)
        expl = shap_explanation(pipe, Xtr, Xte.sample(n, random_state=0) if n < len(Xte) else Xte)
        if expl is None:
            ui.callout("SHAP could not explain this model; showing permutation importance instead (tab 1).", "warn")
        else:
            c1, c2 = st.columns([1.4, 1])
            with c1:
                ui.chart(viz.shap_beeswarm(expl, f"SHAP summary · {expl['method']}"), key="bees")
            with c2:
                ui.chart(viz.importance_bars(shap_importance(expl), "mean_abs_shap", title="Mean |SHAP value|", xtitle="avg impact on prediction (points)", color=viz.SERIES_COLORS[4]), key="sbar")
            st.caption(f"Each dot is one test country-year. Position = how much that feature moved the prediction away from the average prediction "
                       f"({expl['base_value']:.2f}); colour = the feature's value (blue high, red low). SHAP values sum exactly to the prediction.")

with t3:
    num_feats = [f for f in feats if f != "region"]
    c1, c2 = st.columns([1, 3])
    with c1:
        f = st.radio("Feature", num_feats, format_func=lambda x: LABELS.get(x, x))
        ice = st.toggle("Show ICE curves", value=True, help="Individual conditional expectation: one line per sampled row")
    with c2:
        out = pdp(pipe, frame[feats], f, ice=ice)
        ui.chart(viz.pdp_plot(out["grid"], out["average"], f, out.get("individual"), f"Partial dependence of predicted Life Ladder on {LABELS.get(f, f)}", rug=frame[f]), key="pdp")
    st.caption("Partial dependence = the model's average prediction when this feature is set to each value for every row, other features left as observed. "
               "Grid limited to the 2nd–98th percentile to avoid extrapolation. Flat regions mean the model does not use the feature there; with correlated features, "
               "some combinations behind the curve rarely occur in reality.")

with t4:
    use_prod = st.toggle("Use production model (trained on all countries)", value=False,
                         help="Off: the evaluation model, which never saw test countries — honest errors for test countries. On: the model used by the prediction tools.")
    pmodel = ui.model(f"production_{k}") if use_prod else pipe
    test_c = set(split.test["country"])
    opts = sorted(frame["country"].unique())
    default = next((c for c in ["Costa Rica", "Mexico", "Japan", "Finland"] if c in opts), opts[0])
    c1, c2 = st.columns(2)
    with c1:
        country = st.selectbox("Country", opts, index=opts.index(default), format_func=lambda c: c + ("  · unseen in training" if (c in test_c and not use_prod) else ""))
    rows = frame[frame["country"] == country].sort_values("year")
    with c2:
        year = st.selectbox("Year", rows["year"].tolist()[::-1])
    row = rows[rows["year"] == year]
    ex = explain_instance(pmodel, Xtr, row[feats])
    actual = float(row[TARGET].iloc[0])
    err = actual - ex["prediction"]
    in_train = (country not in test_c and country not in set(split.val["country"])) or use_prod
    ui.kpi_cards([
        {"label": "Actual Life Ladder", "value": f"{actual:.2f}", "sub": f"{country} {year}", "icon": "●"},
        {"label": "Predicted", "value": f"{ex['prediction']:.2f}", "sub": MODEL_SPECS[k].name, "icon": "◎"},
        {"label": "Prediction error", "value": f"{err:+.2f}", "sub": "actual − predicted", "icon": "±",
         "accent": "linear-gradient(90deg,#4FE3C1,#9BE57A)" if abs(err) < 0.5 else "linear-gradient(90deg,#FF6B8B,#FFB86B)"},
        {"label": "Country seen in training?", "value": "yes" if in_train else "no", "sub": "errors are optimistic if yes" if in_train else "honest out-of-sample error", "icon": "?"},
    ])
    c1, c2 = st.columns([1.3, 1])
    with c1:
        ui.chart(viz.waterfall(ex["contributions"], ex["base_value"], ex["prediction"], f"Why the model predicts {ex['prediction']:.2f} · {ex['method']}"), key="wf")
    with c2:
        cdf = ex["contributions"].copy()
        cdf["feature"] = cdf["feature"].map(lambda x: SHORT_LABELS.get(x, x))
        up = cdf[cdf["contribution"] > 0].sort_values("contribution", ascending=False)
        dn = cdf[cdf["contribution"] < 0].sort_values("contribution")
        st.markdown("**Factors associated with a higher prediction**")
        st.markdown("\n".join(f"- {r.feature} = {r.value:.3g} → **{r.contribution:+.2f}**" for r in up.itertuples()) or "—")
        st.markdown("**Factors associated with a lower prediction**")
        st.markdown("\n".join(f"- {r.feature} = {r.value:.3g} → **{r.contribution:+.2f}**" for r in dn.itertuples()) or "—")
        st.caption("Contributions are relative to the average prediction over the training data; they describe the model's arithmetic, not real-world effects.")
    hist = rows.copy()
    hist["predicted"] = pmodel.predict(hist[feats])
    long = pd.concat([hist[["year"]].assign(series="Actual", v=hist[TARGET]), hist[["year"]].assign(series="Predicted", v=hist["predicted"])])
    long["country"] = long["series"]
    fig = viz.trend_lines(long.rename(columns={"v": "Life Ladder"}), "Life Ladder", group="series", order=["Actual", "Predicted"], title=f"{country}: actual vs predicted over time")
    ui.chart(fig, key="cty_trend")
ui.footer()
