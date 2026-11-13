import pandas as pd
import streamlit as st

from src import ui
from src import visualization as viz
from src.config import REPORTS_DIR
from src.evaluate import METRIC_INFO, error_by_group, error_by_score_range, regression_metrics
from src.regression_models import MODEL_SPECS

ui.page_header(
    "Model Evaluation",
    "Diagnostics on held-out test data: accuracy metrics, actual vs predicted, residual behaviour, and where the model errs — by region and by score range.",
    "Machine learning · Evaluate",
)
res = ui.ensure_trained()
preds = pd.read_csv(REPORTS_DIR / "test_predictions.csv")
SNAME = {"group": "Unseen countries", "temporal": "Future years", "random": "Random split"}
c1, c2 = st.columns(2)
with c1:
    s = st.selectbox("Validation question", list(SNAME), format_func=lambda x: SNAME[x])
with c2:
    keys = list(res["experiments"][s]["models"])
    k = st.selectbox("Model", keys, index=keys.index(res["best_model"]["key"]) if res["best_model"]["key"] in keys else 0, format_func=lambda x: MODEL_SPECS[x].name)

r = preds[(preds["strategy"] == s) & (preds["model"] == k)].copy()
r["residual"] = r["actual"] - r["predicted"]
r["abs_error"] = r["residual"].abs()
m = regression_metrics(r["actual"], r["predicted"], len(res["features"]))
ui.kpi_cards([
    {"label": "R²", "value": f"{m['r2']:.3f}", "icon": "◔"},
    {"label": "Adjusted R²", "value": f"{m['adj_r2']:.3f}", "sub": f"p = {len(res['features'])} features", "icon": "◑"},
    {"label": "MAE", "value": f"{m['mae']:.3f}", "sub": "ladder points", "icon": "|e|"},
    {"label": "MSE", "value": f"{m['mse']:.3f}", "icon": "e²"},
    {"label": "RMSE", "value": f"{m['rmse']:.3f}", "sub": "ladder points", "icon": "√", "accent": "linear-gradient(90deg,#4FE3C1,#7C9CFF)"},
    {"label": "Bias", "value": f"{m['bias']:+.3f}", "sub": "mean(pred − actual)", "icon": "±"},
    {"label": "Max error", "value": f"{m['max_error']:.2f}", "sub": f"n = {m['n']} test rows", "icon": "⇕"},
])

c1, c2 = st.columns(2)
with c1:
    ui.chart(viz.actual_vs_predicted(r, "Actual vs predicted (test set)"), key="avp")
with c2:
    ui.chart(viz.residual_vs_predicted(r, "Residuals vs predicted"), key="rvp")
c1, c2 = st.columns(2)
with c1:
    ui.chart(viz.residual_hist(r, "Residual distribution"), key="rh")
with c2:
    ui.chart(viz.error_bars(error_by_score_range(r), "score_range", "Error by actual score range (MAE)"), key="esr")
ui.chart(viz.error_bars(error_by_group(r.dropna(subset=["region"]), "region"), "region", "Error by region (MAE)"), key="ereg")
ui.callout("Typical pattern: errors are largest at the extremes of the score range — the model pulls very high and very low scores towards the middle "
           "(regression to the mean) — and in regions whose life evaluations differ from what their indicators suggest (e.g. Latin America tends to report higher "
           "life evaluations than predicted). Positive residual = actual above prediction.", "info")

ui.section("Largest misses", "")
c1, c2 = st.columns(2)
cols = ["country", "year", "region", "actual", "predicted", "residual"]
with c1:
    st.markdown("**Happier than predicted** (actual ≫ predicted)")
    st.dataframe(ui.fmt_table(r.nlargest(10, "residual")[cols]), hide_index=True)
with c2:
    st.markdown("**Less happy than predicted** (actual ≪ predicted)")
    st.dataframe(ui.fmt_table(r.nsmallest(10, "residual")[cols]), hide_index=True)

if s == "group" and k == res["best_model"]["key"]:
    cf = res["conformal"]
    ui.section("Prediction-interval calibration", "", cf["method"])
    ui.kpi_cards([
        {"label": "Target coverage", "value": f"{cf['coverage_target']:.0%}", "icon": "◎"},
        {"label": "Interval half-width", "value": f"±{cf['q_split']:.2f}", "sub": f"calibrated on {cf['n_calibration']} validation rows", "icon": "↔"},
        {"label": "Empirical test coverage", "value": f"{cf['empirical_test_coverage_split']:.1%}", "sub": f"{cf['n_test']} rows of unseen countries", "icon": "✓",
         "accent": "linear-gradient(90deg,#4FE3C1,#9BE57A)"},
        {"label": "Production half-width", "value": f"±{cf['q_cv']:.2f}", "sub": f"grouped 5-fold OOF, coverage {cf['empirical_oof_coverage_cv']:.1%}", "icon": "⟷"},
    ])
    st.caption("Split-conformal intervals: the 90th percentile (finite-sample corrected) of absolute errors on data from countries the model never saw. "
               "They guarantee average coverage under exchangeability, not coverage for every individual country.")

with st.expander("What each metric means"):
    for key, (n, d) in METRIC_INFO.items():
        st.markdown(f"- **{n}** — {d}")
    st.markdown("A model should not be chosen on R² alone: R² depends on the variance of the test set, while MAE/RMSE are in the units that matter (ladder points), "
                "and the validation design determines which question the number answers.")
ui.footer()
