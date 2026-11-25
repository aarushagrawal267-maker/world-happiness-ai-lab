import pandas as pd
import streamlit as st

from src import ui
from src import visualization as viz
from src.evaluate import describe_validation
from src.regression_models import MODEL_KEYS, MODEL_SPECS

ui.page_header(
    "Model Comparison",
    "Seven regression models, three validation questions, identical data and preprocessing. Models are compared on several metrics and "
    "validation schemes — never on R² alone.",
    "Machine learning · Compare",
)
res = ui.ensure_trained()
exps = res["experiments"]
SNAME = {"random": "Random split", "group": "Unseen countries", "temporal": "Future years"}

rows = []
for s, e in exps.items():
    for k, m in e["models"].items():
        rows.append({"strategy": SNAME[s], "skey": s, "key": k, "model": m["name"], "cv_rmse": m["cv_rmse_mean"], "cv_rmse_std": m["cv_rmse_std"],
                     "val_rmse": m["val"]["rmse"], "test_rmse": m["test"]["rmse"], "test_mae": m["test"]["mae"], "test_r2": m["test"]["r2"],
                     "test_adj_r2": m["test"].get("adj_r2"), "train_rmse": m["train"]["rmse"], "tuned": m["tuned"], "seconds": m["fit_seconds"]})
df = pd.DataFrame(rows)
order = [MODEL_SPECS[k].name for k in MODEL_KEYS if k in set(df["key"])]

b = res["best_model"]
ui.callout(f"<b>Selected model: {b['name']}.</b> {b['selection_rule']} Validation RMSE {b['val']['rmse']:.3f} · test RMSE {b['test']['rmse']:.3f} · "
           f"test MAE {b['test']['mae']:.3f} · test R² {b['test']['r2']:.3f}.", "ok", "◎")

ui.section("Leaderboard", "01")
s = st.segmented_control("Validation question", list(SNAME), default="group", format_func=lambda x: SNAME[x])
st.caption(describe_validation(s))
e = exps[s]
st.markdown(f"<span class='tag'>train {e['n_train']} rows · {e['countries_train']} countries · {e['years_train'][0]}–{e['years_train'][1]}</span>"
            f"<span class='tag'>validation {e['n_val']} · {e['countries_val']} countries · {e['years_val'][0]}–{e['years_val'][1]}</span>"
            f"<span class='tag'>test {e['n_test']} · {e['countries_test']} countries · {e['years_test'][0]}–{e['years_test'][1]}</span>"
            f"<span class='tag {'green' if e['country_overlap_train_test'] == 0 else 'orange'}'>countries in both train & test: {e['country_overlap_train_test']}</span>", unsafe_allow_html=True)
lb = df[df["skey"] == s].sort_values("val_rmse").drop(columns=["skey", "key", "strategy"])
st.dataframe(ui.fmt_table(lb, 4), hide_index=True, column_config={
    "val_rmse": st.column_config.ProgressColumn("val RMSE", min_value=0, max_value=float(lb["val_rmse"].max()) * 1.1, format="%.4f"),
    "test_r2": st.column_config.ProgressColumn("test R²", min_value=0, max_value=1, format="%.3f"),
})
bl = e.get("baselines")
if bl:
    st.caption(f"Naive baselines on the same test set (no indicators used, fitted on the training rows): training mean → RMSE {bl['train_mean']['test']['rmse']:.3f} "
               f"(R² {bl['train_mean']['test']['r2']:.2f}); region mean → RMSE {bl['region_mean']['test']['rmse']:.3f} (R² {bl['region_mean']['test']['r2']:.2f}).")

ui.section("Test performance across validation questions", "02")
c1, c2 = st.columns(2)
with c1:
    ui.chart(viz.grouped_bars(df, "model", "test_rmse", "strategy", order=order, group_order=list(SNAME.values()), title="Test RMSE (lower is better)", ytitle="RMSE"), key="rmse")
with c2:
    ui.chart(viz.grouped_bars(df, "model", "test_mae", "strategy", order=order, group_order=list(SNAME.values()), title="Test MAE (lower is better)", ytitle="MAE"), key="mae")
ui.chart(viz.grouped_bars(df, "model", "test_r2", "strategy", order=order, group_order=list(SNAME.values()), title="Test R² (higher is better)", ytitle="R²"), key="r2")
ui.callout("Test sets differ between questions (different countries/years with different spreads of scores), so R² is not comparable across questions — "
           "a test set with wider score variance can produce a higher R² even with larger errors. RMSE/MAE in ladder points are the primary comparison.", "info")

ui.section("Cross-validation strategy comparison", "03", "same model and data, three CV schemes on the full panel")
cv = pd.DataFrame(res["cv_comparison"])
cv["strategy_name"] = cv["strategy"].map(SNAME)
ui.chart(viz.grouped_bars(cv, "model", "rmse_mean", "strategy_name", error="rmse_std", order=order, group_order=list(SNAME.values()),
                          title="5-fold CV RMSE (mean ± SD across folds)", ytitle="RMSE"), key="cv")
piv = cv.pivot(index="model", columns="strategy", values="rmse_mean")
piv["optimism_vs_group_%"] = (piv["group"] / piv["random"] - 1) * 100
piv["optimism_vs_temporal_%"] = (piv["temporal"] / piv["random"] - 1) * 100
st.dataframe(ui.fmt_table(piv.reset_index(), 3), hide_index=True)
mean_opt = piv["optimism_vs_group_%"].mean()
ui.callout(f"Random K-fold CV underestimates the error for <b>new countries</b> by about {mean_opt:.0f}% on average: rows of the same country in other years leak "
           "country-specific information (culture, measurement quirks) into training. Use the grouped result when the question is 'what would the model say for a country it has never seen?', "
           "and the forward-in-time result for 'how well do past patterns hold in later years?'.", "warn")

ui.section("Before vs after feature engineering", "04")
fs = pd.DataFrame(res["feature_set_comparison"])
fs["label"] = fs["model"] + " · " + fs["strategy"].map(SNAME)
ui.chart(viz.grouped_bars(fs, "label", "rmse_mean", "feature_set", group_order=["core", "engineered", "core_plus_affect"], title="CV RMSE by feature set", ytitle="RMSE"), key="fs")
piv2 = fs.pivot_table(index=["model", "strategy"], columns="feature_set", values="rmse_mean")
piv2["engineered_gain_%"] = (1 - piv2["engineered"] / piv2["core"]) * 100
st.dataframe(ui.fmt_table(piv2.reset_index(), 4), hide_index=True)
ui.callout("<b>engineered</b> adds year, region (one-hot) and backward-looking GDP growth to the six core indicators. <b>core_plus_affect</b> adds positive/negative affect, "
           "which improves accuracy but uses emotional-wellbeing measures that are close to the outcome itself, so it is not used as the default prediction model.", "info")

ui.section("Over-fitting check", "05", "train vs test error on the unseen-countries split")
g = df[df["skey"] == "group"].melt(id_vars="model", value_vars=["train_rmse", "val_rmse", "test_rmse"], var_name="set", value_name="rmse")
ui.chart(viz.grouped_bars(g, "model", "rmse", "set", order=order, title="RMSE on train / validation / test", ytitle="RMSE"), key="ovf")
st.caption("Flexible models (deep trees, forests) fit training data almost perfectly; what matters is their error on unseen data.")

with st.expander("Tuned hyper-parameters (unseen-countries split)"):
    for k, m in exps["group"]["models"].items():
        st.markdown(f"**{m['name']}** {'(tuned)' if m['tuned'] else '(defaults)'}")
        st.json({p: v for p, v in m["params"].items() if p not in ("random_state", "n_jobs", "verbosity")}, expanded=False)
ui.footer()
