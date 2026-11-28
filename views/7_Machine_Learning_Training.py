import math
import time

import pandas as pd
import streamlit as st

from src import ui
from src import visualization as viz
from src.config import FEATURE_SETS, LABELS
from src.evaluate import METRIC_INFO, describe_validation
from src.feature_engineering import FEATURE_DOCS
from src.model_manager import delete_model, list_models
from src.regression_models import MODEL_SPECS, XGBOOST_AVAILABLE
from src.train import run_training, train_single

ui.page_header(
    "Machine learning training lab",
    "Train and tune regression models that estimate a country's Life Ladder from its indicators. Choose the model, hyper-parameters, feature set and the "
    "validation question; watch progress; compare runs; save models for the prediction tools.",
    "Machine learning · Train",
)
res = ui.ensure_trained()
b = ui.bundle()

tab_single, tab_full, tab_saved, tab_fe = st.tabs(["Train a model", "Full experiment pipeline", "Saved models", "Feature sets"])

with tab_single:
    c1, c2, c3 = st.columns(3)
    with c1:
        key = st.selectbox("Model", list(MODEL_SPECS), format_func=lambda k: MODEL_SPECS[k].name)
        st.caption(MODEL_SPECS[key].description)
    with c2:
        fs = st.selectbox("Feature set", list(FEATURE_SETS), format_func=lambda f: f"{f} ({len(FEATURE_SETS[f])} features)")
        st.caption(", ".join(LABELS.get(f, f) for f in FEATURE_SETS[fs]))
    with c3:
        strategy = st.selectbox("Validation question", ["group", "temporal", "random"], format_func=lambda s: {"group": "Unseen countries (grouped)", "temporal": "Future years (time-aware)", "random": "Random split"}[s])
        st.caption(describe_validation(strategy))
    spec = MODEL_SPECS[key]
    tune = st.toggle("Automatic hyper-parameter search (RandomizedSearchCV)", value=False, disabled=not spec.tunable,
                     help="Searches the model's parameter space with cross-validation on the training portion only.")
    params = {}
    if spec.ui_params and not tune:
        st.markdown("**Hyper-parameters**")
        pcols = st.columns(min(len(spec.ui_params), 5))
        for i, (p, cfg) in enumerate(spec.ui_params.items()):
            with pcols[i % len(pcols)]:
                if cfg["type"] == "int":
                    params[p] = st.slider(p, cfg["min"], cfg["max"], cfg["default"], step=cfg.get("step", 1), key=f"{key}_{p}")
                elif cfg["type"] == "int_or_none":
                    v = st.slider(f"{p} (0 = unlimited)", 0, cfg["max"], cfg["default"], key=f"{key}_{p}")
                    params[p] = None if v == 0 else v
                elif cfg["type"] == "float":
                    params[p] = st.slider(p, float(cfg["min"]), float(cfg["max"]), float(cfg["default"]), key=f"{key}_{p}")
                elif cfg["type"] == "log":
                    e = st.slider(f"log10({p})", math.log10(cfg["min"]), math.log10(cfg["max"]), math.log10(cfg["default"]), step=0.1, key=f"{key}_{p}")
                    params[p] = float(10 ** e)
                    st.caption(f"{p} = {params[p]:.4g}")
                elif cfg["type"] == "choice":
                    params[p] = st.selectbox(p, cfg["options"], index=cfg["options"].index(cfg["default"]), key=f"{key}_{p}")
    n_iter = st.slider("Search iterations", 5, 40, 12) if tune else 0
    save_name = st.text_input("Save as (optional, letters/numbers/_)", value="", placeholder="e.g. my_forest")
    if st.button("Train model", type="primary"):
        with st.status(f"Training {spec.name}…", expanded=True) as status:
            st.write(f"Splitting data for the **{strategy}** question…")
            t0 = time.perf_counter()
            st.write("Fitting pipeline (imputation → scaling → model) on training data" + (f" with {n_iter}-iteration randomized search…" if tune else "…"))
            out = train_single(key, params, fs, strategy, b.panel, tune=tune, n_iter=max(n_iter, 1), save_as=("custom_" + "".join(ch for ch in save_name if ch.isalnum() or ch == "_")) if save_name else None)
            st.write("Scoring validation and test sets…")
            status.update(label=f"{spec.name} trained in {time.perf_counter() - t0:.1f} s", state="complete")
        hist = st.session_state.setdefault("train_history", [])
        hist.append({"model": spec.name, "feature_set": fs, "strategy": strategy, "tuned": out["tuned"], "cv_rmse": out["cv_rmse_mean"],
                     "val_rmse": out["val"]["rmse"], "test_rmse": out["test"]["rmse"], "test_mae": out["test"]["mae"], "test_r2": out["test"]["r2"],
                     "train_rmse": out["train"]["rmse"], "seconds": out["fit_seconds"], "params": str({k: v for k, v in out["params"].items() if k not in ("random_state", "n_jobs", "verbosity")})})
        st.session_state["last_train"] = out
        if save_name:
            ui.get_model.clear()
            st.success(f"Saved as custom_{save_name} — available in the prediction tools.")
    out = st.session_state.get("last_train")
    if out:
        ui.section("Result", "", f"{out['name']} · {out['feature_set']} · {out['strategy']} split")
        ui.kpi_cards([
            {"label": "Train RMSE", "value": f"{out['train']['rmse']:.3f}", "sub": f"R² {out['train']['r2']:.3f}"},
            {"label": "CV RMSE", "value": f"{out['cv_rmse_mean']:.3f}", "sub": f"± {out['cv_rmse_std']:.3f} across folds"},
            {"label": "Validation RMSE", "value": f"{out['val']['rmse']:.3f}", "sub": f"MAE {out['val']['mae']:.3f} · R² {out['val']['r2']:.3f}"},
            {"label": "Test RMSE", "value": f"{out['test']['rmse']:.3f}", "sub": f"MAE {out['test']['mae']:.3f} · R² {out['test']['r2']:.3f} · adj {out['test'].get('adj_r2', float('nan')):.3f}"},
        ])
        gap = out["val"]["rmse"] - out["train"]["rmse"]
        if gap > 0.15:
            ui.callout(f"Training error is much lower than validation error (gap {gap:.2f}) — a sign of <b>overfitting</b>. Try shallower trees, more regularisation or larger leaves.", "warn")
        c1, c2 = st.columns(2)
        pr = out["_predictions"].copy()
        pr["residual"] = pr["actual"] - pr["predicted"]
        with c1:
            ui.chart(viz.actual_vs_predicted(pr, "Test set: actual vs predicted"), key="t_avp")
        with c2:
            ui.chart(viz.residual_hist(pr, "Test residuals"), key="t_res")
        with st.expander("Parameters used"):
            st.json({k: v for k, v in out["params"].items()})
    if st.session_state.get("train_history"):
        ui.section("Runs this session", "")
        h = pd.DataFrame(st.session_state["train_history"])
        st.dataframe(ui.fmt_table(h), hide_index=True)
        if len(h) > 1:
            h["run"] = [f"#{i + 1} {m}" for i, m in enumerate(h["model"])]
            long = h.melt(id_vars="run", value_vars=["train_rmse", "val_rmse", "test_rmse"], var_name="set", value_name="rmse")
            ui.chart(viz.grouped_bars(long, "run", "rmse", "set", title="RMSE by run", ytitle="RMSE"), key="runs")
        if st.button("Clear history"):
            st.session_state["train_history"] = []
            st.rerun()

with tab_full:
    if res:
        st.markdown(f"Last full run: **{res['generated_at'][:19].replace('T', ' ')} UTC** · feature set `{res['feature_set']}` · "
                    f"{res['n_rows']:,} rows · tuning {'on (' + str(res['tuning']['n_iter']) + ' iterations)' if res['tuning']['enabled'] else 'off'} · {res.get('training_seconds', '?')} s")
    st.markdown(
        "The full pipeline trains **all models for all three validation questions**, tunes them with RandomizedSearchCV, compares cross-validation strategies, "
        "feature sets and imputation strategies, calibrates conformal prediction intervals, refits production models on all data and saves everything."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        models = st.multiselect("Models", list(MODEL_SPECS), default=list(MODEL_SPECS), format_func=lambda k: MODEL_SPECS[k].name)
    with c2:
        fs_full = st.selectbox("Feature set ", list(FEATURE_SETS), index=list(FEATURE_SETS).index("core"))
    with c3:
        iters = st.slider("Tuning iterations per model", 4, 40, 15)
    if not XGBOOST_AVAILABLE:
        ui.callout("XGBoost is not installed in this environment; it is skipped automatically.", "warn")
    if st.button("Run full pipeline", type="primary", disabled=not models):
        bar = st.progress(0.0, text="Starting…")
        t0 = time.perf_counter()
        r = run_training(models=models, feature_set=fs_full, n_iter=iters, progress=lambda f, m: bar.progress(min(f, 1.0), text=m), panel=b.panel)
        ui.get_model.clear()
        st.cache_data.clear()
        st.success(f"Done in {time.perf_counter() - t0:.0f} s. Best model: {r['best_model']['name']} (group validation RMSE {r['best_model']['val']['rmse']:.3f}).")
    st.markdown("Command-line equivalent:")
    st.code("python -m src.train            # full run\npython -m src.train --quick    # faster search", language="bash")

with tab_saved:
    ms = list_models()
    if ms:
        df = pd.DataFrame([{"name": k, "kind": v.get("kind"), "model": v.get("model_name"), "feature_set": v.get("feature_set"), "size_kb": v.get("size_kb"),
                            "saved_at": v.get("saved_at", "")[:19], "sklearn": v.get("versions", {}).get("scikit-learn")} for k, v in ms.items()])
        st.dataframe(df.sort_values(["kind", "name"]), hide_index=True, height=420)
        custom = [k for k, v in ms.items() if v.get("kind") == "custom"]
        if custom:
            dname = st.selectbox("Delete a custom model", custom)
            if st.button("Delete"):
                delete_model(dname)
                ui.get_model.clear()
                st.rerun()
    st.caption("production_* = refitted on all data with the hyper-parameters tuned on the country-grouped split (used by Prediction and What-If). "
               "eval_* = fitted on the training portion of each split (used for evaluation and explainability). custom_* = saved from this page.")

with tab_fe:
    for fs_name, feats in FEATURE_SETS.items():
        ui.card(f"{fs_name} · {len(feats)} features", ", ".join(LABELS.get(f, f) for f in feats))
    st.markdown("**Engineered variables**")
    st.dataframe(pd.DataFrame(FEATURE_DOCS).T.reset_index().rename(columns={"index": "variable"}), hide_index=True)
    st.markdown("**Metrics**")
    for k, (n, d) in METRIC_INFO.items():
        st.markdown(f"- **{n}** — {d}")
ui.footer()
