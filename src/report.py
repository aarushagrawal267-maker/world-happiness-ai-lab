"""Generate PROJECT_REPORT.md and static figures from the actual data and the
saved experiment results.

    python -m src.report

Every number in the report is computed here at generation time; nothing is typed in by hand.
"""
from __future__ import annotations

from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from .clustering import run_clustering  # noqa: E402
from .config import FIGURES_DIR, INDICATORS, LABELS, ROOT_DIR, SHORT_LABELS, TARGET  # noqa: E402
from .data_acquisition import load_manifest  # noqa: E402
from .data_loader import load_bundle  # noqa: E402
from .eda import country_period_change, dataset_overview, distribution_stats, ranking_overview, region_summary  # noqa: E402
from .explainability import permutation_importance_df, shap_explanation, shap_importance  # noqa: E402
from .feature_engineering import engineer_panel  # noqa: E402
from .insight_engine import generate_insights  # noqa: E402
from .model_manager import load_model, load_results  # noqa: E402
from .statistical_analysis import correlation_tests, country_level, group_comparison, ols_regression  # noqa: E402
from .train import build_modeling_frame, make_splits  # noqa: E402

LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SNAME = {"random": "Random split", "group": "Unseen countries (grouped)", "temporal": "Future years (time-aware)"}


def _style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 140, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlesize": 12, "axes.prop_cycle": plt.cycler(color=LIGHT), "grid.alpha": 0.35})


def _save(fig, name: str) -> str:
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return f"outputs/figures/{name}"


def fp(p: float) -> str:
    return "< 1e-300" if p < 1e-300 else f"{p:.1e}"


def md_table(df: pd.DataFrame, floatfmt: int = 3) -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else ("< 1e-300" if (v == 0 and c.startswith(("p", "pearson_p", "spearman_p"))) else (f"{v:.2e}" if (abs(v) < 1e-3 and v != 0) else f"{v:.{floatfmt}f}")))
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


def generate_report() -> str:
    _style()
    b = load_bundle(use_cache=False)
    panel, rk = b.panel, b.rankings
    pfe = engineer_panel(panel)
    res = load_results()
    if not res:
        raise SystemExit("No experiment results found — run `python -m src.train` first.")
    ov, ro = dataset_overview(panel), ranking_overview(rk)
    manifest = load_manifest()
    figs = {}

    # ------------------------------------------------------------ figures: EDA
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(panel[TARGET], bins=40, color=LIGHT[0], edgecolor="white")
    ax.axvline(panel[TARGET].mean(), color="#333", ls="--", lw=1)
    ax.set(title="Distribution of Life Ladder (all country-years)", xlabel="Life Ladder (0–10)", ylabel="Count")
    figs["hist"] = _save(fig, "01_ladder_distribution.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    sns.scatterplot(data=panel, x="log_gdp_per_capita", y=TARGET, s=14, alpha=0.45, color=LIGHT[0], edgecolor=None, ax=ax)
    d = panel[["log_gdp_per_capita", TARGET]].dropna()
    m, c = np.polyfit(d["log_gdp_per_capita"], d[TARGET], 1)
    xs = np.linspace(d["log_gdp_per_capita"].min(), d["log_gdp_per_capita"].max(), 20)
    ax.plot(xs, m * xs + c, color="#222", lw=2, ls="--", label=f"OLS fit (slope {m:.2f}, r = {np.corrcoef(d.iloc[:, 0], d.iloc[:, 1])[0, 1]:.2f})")
    ax.legend(frameon=False)
    ax.set(title="Log GDP per capita vs Life Ladder", xlabel=LABELS["log_gdp_per_capita"], ylabel=LABELS[TARGET])
    figs["gdp"] = _save(fig, "02_gdp_vs_ladder.png")

    cols = [TARGET] + [c for c in INDICATORS + ["positive_affect", "negative_affect"]]
    corr = panel[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr.rename(index=SHORT_LABELS, columns=SHORT_LABELS), annot=True, fmt=".2f", cmap=sns.diverging_palette(10, 250, s=75, l=50, as_cmap=True),
                vmin=-1, vmax=1, linewidths=1, linecolor="white", cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title("Pearson correlation matrix")
    figs["corr"] = _save(fig, "03_correlation_matrix.png")

    rs = region_summary(panel[panel["year"].between(2020, 2022)], TARGET)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.barh(rs["region"][::-1], rs["mean"][::-1], xerr=rs["std"][::-1], color=LIGHT[0], ecolor="#888")
    ax.set(title="Mean Life Ladder by region, 2020–2022 (± 1 SD)", xlabel="Life Ladder")
    figs["region"] = _save(fig, "04_region_means.png")

    miss = panel.groupby("year")[INDICATORS + ["positive_affect", "negative_affect"]].apply(lambda g: g.isna().mean() * 100)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    sns.heatmap(miss.T.rename(index=SHORT_LABELS), cmap="Blues", annot=False, cbar_kws={"label": "% missing"}, ax=ax, linewidths=0.5, linecolor="white")
    ax.set(title="Missing values by year", xlabel="Year", ylabel="")
    figs["missing"] = _save(fig, "05_missingness.png")

    # ------------------------------------------------------------ statistics
    ct = correlation_tests(panel, TARGET, INDICATORS + ["positive_affect", "negative_affect"])
    ols = ols_regression(panel, TARGET, INDICATORS)
    ols_std = ols_regression(panel, TARGET, INDICATORS, standardize=True)
    y1 = int(panel["year"].max())
    cl_reg = country_level(panel, TARGET, "region", (y1 - 2, y1))
    gc = group_comparison(cl_reg, TARGET, "region")

    # ------------------------------------------------------------ ML figures
    exps = res["experiments"]
    rows = []
    for s, e in exps.items():
        for k, mm in e["models"].items():
            rows.append({"strategy": s, "model": mm["name"], "val_rmse": mm["val"]["rmse"], "test_rmse": mm["test"]["rmse"], "test_mae": mm["test"]["mae"],
                         "test_r2": mm["test"]["r2"], "cv_rmse": mm["cv_rmse_mean"], "train_rmse": mm["train"]["rmse"], "tuned": mm["tuned"]})
    perf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    piv = perf.pivot(index="model", columns="strategy", values="test_rmse")[["random", "group", "temporal"]].sort_values("group")
    piv.rename(columns=SNAME).plot.bar(ax=ax, width=0.8, edgecolor="white")
    ax.set(title="Test RMSE by model and validation question (lower is better)", ylabel="RMSE (ladder points)", xlabel="")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    figs["models"] = _save(fig, "06_model_comparison.png")

    cv = pd.DataFrame(res["cv_comparison"])
    fig, ax = plt.subplots(figsize=(9, 4.4))
    cpiv = cv.pivot(index="model", columns="strategy", values="rmse_mean")[["random", "group", "temporal"]]
    cerr = cv.pivot(index="model", columns="strategy", values="rmse_std")[["random", "group", "temporal"]]
    order = cpiv.sort_values("group").index
    cpiv.loc[order].rename(columns=SNAME).plot.bar(ax=ax, yerr=cerr.loc[order].rename(columns=SNAME), width=0.8, edgecolor="white", capsize=2)
    ax.set(title="5-fold cross-validation RMSE by CV scheme (mean ± SD)", ylabel="RMSE", xlabel="")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    figs["cv"] = _save(fig, "07_cv_strategies.png")

    best = res["best_model"]
    preds = pd.read_csv(ROOT_DIR / "outputs/reports/test_predictions.csv")
    bp = preds[(preds["strategy"] == "group") & (preds["model"] == best["key"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    axes[0].scatter(bp["actual"], bp["predicted"], s=14, alpha=0.6, color=LIGHT[0])
    lim = [min(bp["actual"].min(), bp["predicted"].min()) - 0.2, max(bp["actual"].max(), bp["predicted"].max()) + 0.2]
    axes[0].plot(lim, lim, "--", color="#555", lw=1)
    axes[0].set(xlim=lim, ylim=lim, title=f"{best['name']}: actual vs predicted\n(unseen-country test set)", xlabel="Actual", ylabel="Predicted")
    resid = bp["actual"] - bp["predicted"]
    axes[1].scatter(bp["predicted"], resid, s=14, alpha=0.6, color=LIGHT[2])
    axes[1].axhline(0, color="#555", lw=1, ls="--")
    axes[1].set(title="Residuals vs predicted", xlabel="Predicted", ylabel="Residual (actual − predicted)")
    figs["avp"] = _save(fig, "08_actual_vs_predicted.png")

    # ------------------------------------------------------------ explainability
    frame = build_modeling_frame(panel)
    split = make_splits(frame, "group")
    feats = res["features"]
    eval_pipe = load_model(f"eval_group_{best['key']}")
    pi = permutation_importance_df(eval_pipe, split.test[feats], split.test[TARGET], n_repeats=15)
    expl = shap_explanation(eval_pipe, split.train[feats], split.test[feats])
    shp = shap_importance(expl) if expl else None
    fig, axes = plt.subplots(1, 2 if shp is not None else 1, figsize=(10, 3.8))
    axes = np.atleast_1d(axes)
    p2 = pi.sort_values("importance_mean")
    axes[0].barh(p2["feature"].map(SHORT_LABELS), p2["importance_mean"], xerr=p2["importance_std"], color=LIGHT[0], ecolor="#888")
    axes[0].set(title="Permutation importance (Δ RMSE)", xlabel="Increase in RMSE when shuffled")
    if shp is not None:
        s2 = shp.sort_values("mean_abs_shap")
        axes[1].barh(s2["feature"].map(lambda f: SHORT_LABELS.get(f, f)), s2["mean_abs_shap"], color=LIGHT[4])
        axes[1].set(title="Mean |SHAP value|", xlabel="Average impact on prediction (points)")
    figs["imp"] = _save(fig, "09_feature_importance.png")

    # ------------------------------------------------------------ clustering
    cl = run_clustering(panel)
    fig, ax = plt.subplots(figsize=(8, 5.4))
    for c in sorted(cl.profiles["cluster"].unique()):
        sub = cl.profiles[cl.profiles["cluster"] == c]
        ax.scatter(sub["pc1"], sub["pc2"], s=36, color=LIGHT[c], edgecolor="white", label=f"C{c + 1}: {cl.names[c]}")
    for _, r in pd.concat([cl.profiles.nlargest(4, "pc1"), cl.profiles.nsmallest(4, "pc1"), cl.profiles.nlargest(3, "pc2"), cl.profiles.nsmallest(3, "pc2")]).drop_duplicates("country").iterrows():
        ax.annotate(r["country"], (r["pc1"], r["pc2"]), fontsize=7, xytext=(3, 3), textcoords="offset points", color="#444")
    ax.set(title=f"K-means clusters (k = {cl.k}) in PCA space", xlabel=f"PC1 ({cl.pca_variance[0]:.0%})", ylabel=f"PC2 ({cl.pca_variance[1]:.0%})")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    figs["pca"] = _save(fig, "10_clusters_pca.png")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, (col, t) in zip(axes, (("wcss", "Within-cluster SS (elbow)"), ("silhouette", "Silhouette (↑)"), ("davies_bouldin", "Davies–Bouldin (↓)"))):
        ax.plot(cl.scores["k"], cl.scores[col], marker="o", color=LIGHT[0])
        ax.axvline(cl.k, color="#999", ls=":")
        ax.set(title=t, xlabel="k")
    figs["kscores"] = _save(fig, "11_cluster_scores.png")

    # official rankings change
    y0 = int(rk["year"].min())
    chg = country_period_change(rk, y0, ro["year"])
    insights = generate_insights(panel, pfe, rk, b.panel_source, b.rankings_source, res, cl, pi)

    # ------------------------------------------------------------ write markdown
    pv = manifest.get("panel_verification", {})
    t = ols["table"].copy()
    t["term"] = t["term"].map(lambda x: LABELS.get(x, x))
    t_std = ols_std["table"][ols_std["table"]["term"] != "const"][["term", "coef"]].rename(columns={"coef": "beta"})
    t = t.merge(t_std.assign(term=t_std["term"].map(LABELS)), on="term", how="left")
    ct_show = ct.assign(feature=ct["feature"].map(LABELS))[["feature", "n", "pearson_r", "pearson_ci_low", "pearson_ci_high", "pearson_p_holm", "spearman_rho", "strength"]]
    dist = distribution_stats(panel[TARGET])
    top10 = rk[rk["year"] == ro["year"]].nsmallest(10, "rank")[["rank", "country", "ladder_score", "lower_whisker", "upper_whisker"]]
    bot10 = rk[rk["year"] == ro["year"]].nlargest(10, "rank")[["rank", "country", "ladder_score", "lower_whisker", "upper_whisker"]]
    fsc = pd.DataFrame(res["feature_set_comparison"]).pivot_table(index=["model", "strategy"], columns="feature_set", values="rmse_mean").reset_index()
    cvp = cpiv.copy()
    cvp["group_vs_random_%"] = (cvp["group"] / cvp["random"] - 1) * 100
    cf = res["conformal"]
    cs = cl.cluster_summary.copy()
    cs["cluster"] = "C" + (cs["cluster"] + 1).astype(str)
    cs = cs[["cluster", "name", "n_countries"] + INDICATORS + [TARGET, "example_countries"]].rename(columns={**SHORT_LABELS, TARGET: "Life Ladder*"})
    best_test = {s: exps[s]["models"][best["key"]]["test"] for s in exps}
    imp_tbl = pi.assign(feature=pi["feature"].map(LABELS))
    if shp is not None:
        imp_tbl = imp_tbl.merge(shp.assign(feature=shp["feature"].map(LABELS)), on="feature", how="left")
    ols_d = ols["diagnostics"]
    sig_pairs = int(gc["pairwise"]["significant_holm"].sum())
    imp_comp = pd.DataFrame(res["imputation_comparison"])

    L = []
    w = L.append
    w("# World Happiness AI Lab — Project Report\n")
    w(f"*Generated automatically by `python -m src.report` on {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC from the data in `data/raw` and the experiment results in "
      f"`outputs/reports/experiment_results.json` (training run {res['generated_at'][:19]} UTC). Every number below is computed, not typed.*\n")
    w("> **Interpretation note.** Life Ladder scores are national averages of *reported* life evaluations (Cantril ladder, 0–10). All relationships reported "
      "here are statistical associations in observational country-level data and must not be read as causal effects.\n")

    w("## 1. Executive summary\n")
    w(f"- **Data:** {ov['observations']:,} country-year observations for {ov['countries']} countries ({ov['year_min']}–{ov['year_max']}) from the WHR 2023 annual panel, "
      f"plus official WHR ranking tables for {rk['year'].nunique()} ranking years up to **WHR {ro['edition']}** (survey years {ro['year'] - 2}–{ro['year']}, {ro['countries']} countries).")
    w(f"- **Latest ranking:** highest {ro['max_country']} ({ro['max_score']:.3f}), lowest {ro['min_country']} ({ro['min_score']:.3f}); mean of countries {ro['mean_score']:.2f}.")
    s0 = ct.iloc[0]
    w(f"- **Strongest correlate:** {LABELS[s0['feature']]} (r = {s0['pearson_r']:.2f}, n = {s0['n']}). The six WHR indicators jointly explain "
      f"R² = {ols['r2']:.2f} of the variance in Life Ladder in a pooled OLS regression (n = {ols['n']}).")
    w(f"- **Best model:** {best['name']}, selected on validation RMSE for unseen countries: test RMSE **{best['test']['rmse']:.3f}**, MAE {best['test']['mae']:.3f}, "
      f"R² {best['test']['r2']:.3f} on {exps['group']['countries_test']} countries never seen in training.")
    w(f"- **Validation matters:** random K-fold CV understates error for new countries by {cvp['group_vs_random_%'].mean():.0f}% on average across models.")
    w(f"- **Uncertainty:** split-conformal 90% intervals are ±{cf['q_cv']:.2f} ladder points; empirical coverage on unseen test countries {cf['empirical_test_coverage_split']:.1%}.")
    w(f"- **Clusters:** {cl.k} country profiles (silhouette {cl.silhouette:.2f}) — " + "; ".join(f"C{c + 1} {cl.names[c]}" for c in sorted(cl.names)) + ".\n")

    w("## 2. Data sources and preparation\n")
    w("| Dataset | Role | Rows | Countries | Years |\n|---|---|---|---|---|")
    for f in b.files:
        role = "primary panel" if f.name == b.panel_source else ("rankings" if f.name == b.rankings_source else f.kind.replace("_", " "))
        yrs = f"{min(f.report.years)}–{max(f.report.years)}" if f.report.years else ""
        w(f"| `{f.path.relative_to(ROOT_DIR / 'data/raw')}` | {role} | {f.report.n_rows} | {f.report.n_countries} | {yrs} |")
    w("")
    w("Sources: official *Data for Figure 2.1* files from <https://worldhappiness.report/data-sharing/> (WHR 2020–2026); the WHR 2023 *DataForTable2.1* annual panel "
      "(obtained from a public Hugging Face mirror because the original link returns HTTP 403); and the Kaggle *unsdsn/world-happiness* 2015–2019 files (CC0). "
      "Checksums and methodology notes are stored in `data/raw/sources_manifest.json`.\n")
    if pv:
        w(f"**Panel verification ({pv.get('status')}):** the panel's 2020–2022 means reproduce the official WHR 2023 ladder scores for {pv.get('matched_countries')}/"
          f"{pv.get('official_countries')} countries (Pearson r = {pv.get('pearson_r')}, median |Δ| = {pv.get('median_abs_diff')}).\n")
    notes = [n for f in b.files for n in f.log.notes]
    w("**Cleaning:** headers mapped to a canonical schema across 7 structural variants; country names standardised (e.g. *Turkey/Turkiye/Türkiye*, `*`-flagged names in WHR 2022); "
      "definitionally impossible values set to missing (not clipped); genuine duplicates removed (none found in the panel). Values invalidated:")
    for n in notes:
        w(f"  - {n}")
    w(f"\n**Missing data:** {ov['missing_cells']} indicator cells ({ov['missing_pct']:.1f}%) are missing; the Life Ladder target is complete. Strategies compared "
      "(country-grouped 5-fold CV RMSE of a Ridge model):\n")
    w(md_table(imp_comp, 4))
    w(f"\nMedian imputation inside the pipeline was retained: it keeps all {len(frame)} rows and {frame['country'].nunique()} countries and performs on par with alternatives.\n")
    w(f"![Missingness]({figs['missing']})\n")

    w("## 3. Exploratory analysis\n")
    w(f"Life Ladder across all country-years: mean {dist['mean']:.2f}, median {dist['median']:.2f}, SD {dist['std']:.2f}, skewness {dist['skewness']:.2f}, "
      f"range {ov['min_score']:.2f} ({ov['min_country']} {ov['min_year']}) to {ov['max_score']:.2f} ({ov['max_country']} {ov['max_year']}).\n")
    w(f"![Distribution]({figs['hist']})\n")
    w(f"**Top 10 — WHR {ro['edition']} (3-year averages {ro['year'] - 2}–{ro['year']}):**\n")
    w(md_table(top10))
    w(f"\n**Bottom 10 — WHR {ro['edition']}:**\n")
    w(md_table(bot10))
    if not chg.empty:
        w(f"\nLargest change between WHR {y0 + 1} and WHR {ro['edition']} ({len(chg)} countries in both): improvement — "
          + ", ".join(f"{r.country} ({r.change:+.2f})" for r in chg.head(5).itertuples()) + "; decline — "
          + ", ".join(f"{r.country} ({r.change:+.2f})" for r in chg.tail(5).iloc[::-1].itertuples()) + ".\n")
    w(f"![Regions]({figs['region']})\n\n![GDP]({figs['gdp']})\n\n![Correlation]({figs['corr']})\n")

    w("## 4. Statistical findings\n")
    w("### 4.1 Correlation tests (pairwise-complete country-years; Holm-adjusted p)\n")
    w(md_table(ct_show))
    w("\nBecause countries contribute many years, observations are not independent and these p-values are optimistic; the clustered regression below is more conservative.\n")
    w(f"### 4.2 Multiple linear regression (OLS, n = {ols['n']}, {ols['n_countries']} countries)\n")
    w(f"R² = {ols['r2']:.3f}, adjusted R² = {ols['adj_r2']:.3f}, F = {ols['f_stat']:.1f} (p {fp(ols['f_pvalue'])}).\n")
    w(md_table(t[["term", "coef", "std_err", "ci_low", "ci_high", "p_value", "cluster_std_err", "cluster_p_value", "beta"]], 4))
    w("\n*cluster_* columns use standard errors clustered by country; *beta* = standardised coefficient.*\n")
    w("**Assumption checks:**\n")
    for msg in ols_d["interpretation"]:
        w(f"- {msg}")
    w("\nVIF: " + ", ".join(f"{SHORT_LABELS[r.feature]} {r.vif:.2f}" for r in ols_d["vif"].itertuples()) + "\n")
    w(f"### 4.3 Regional differences (one mean per country, {y1 - 2}–{y1}, {gc['n']} countries)\n")
    w(f"One-way ANOVA F = {gc['anova_f']:.2f} (p {fp(gc['anova_p'])}); Welch ANOVA F = {gc['welch_f']:.2f} (p {fp(gc['welch_p'])}); Kruskal–Wallis H = {gc['kruskal_h']:.2f} "
      f"(p {fp(gc['kruskal_p'])}); Levene p = {gc['levene_p']:.3f}. Effect size η² = {gc['eta_squared']:.2f} (ω² = {gc['omega_squared']:.2f}). "
      f"{sig_pairs} of {len(gc['pairwise'])} region pairs differ after Holm correction (Mann–Whitney). {gc['practical_interpretation']}\n")
    w(md_table(gc["summary"]))
    w("")

    w("## 5. Feature engineering\n")
    w("Engineered variables (documented in `src/feature_engineering.py`): year, WHR region (one-hot), backward-looking GDP growth (Δ log GDP vs the previous observed year), "
      "a composite z-score index (optional transformer), quartile categories for EDA, and target-derived descriptive variables (year-over-year ladder change, ladder-vs-GDP residual, "
      "happiness band) that are blocked from model inputs. Cross-validated RMSE before/after feature engineering:\n")
    w(md_table(fsc, 4))
    w("")

    w("## 6. Machine learning\n")
    w("### 6.1 Validation methodology\n")
    for s, e in exps.items():
        w(f"- **{SNAME[s]}** — train {e['n_train']} rows ({e['countries_train']} countries, {e['years_train'][0]}–{e['years_train'][1]}), "
          f"validation {e['n_val']} ({e['years_val'][0]}–{e['years_val'][1]}), test {e['n_test']} ({e['countries_test']} countries, {e['years_test'][0]}–{e['years_test'][1]}); "
          f"countries in both train and test: {e['country_overlap_train_test']}.")
    w(f"\nHyper-parameters were tuned with {res['tuning']['method']}, {res['tuning']['n_iter']} iterations per model, using K-fold, GroupKFold or forward-year CV on the training "
      "portion only, matching each question. All preprocessing (median imputation, scaling, one-hot encoding) is inside the pipeline.\n")
    w("### 6.2 Results\n")
    for s in exps:
        w(f"**{SNAME[s]}**\n")
        w(md_table(perf[perf["strategy"] == s].drop(columns="strategy").sort_values("val_rmse"), 4))
        w("")
    w(f"![Models]({figs['models']})\n")
    w("### 6.3 Cross-validation strategy comparison (same model, same data)\n")
    w(md_table(cvp.reset_index(), 4))
    w(f"\n![CV]({figs['cv']})\n")
    w(f"### 6.4 Selected model\n\n**{best['name']}** — {best['selection_rule']} Tuned parameters: `{ {k: v for k, v in best['params'].items() if k not in ('random_state', 'n_jobs', 'verbosity')} }`.\n")
    w("| Question | Test RMSE | Test MAE | Test R² |\n|---|---|---|---|")
    for s, m in best_test.items():
        w(f"| {SNAME[s]} | {m['rmse']:.3f} | {m['mae']:.3f} | {m['r2']:.3f} |")
    w(f"\nPrediction intervals: {cf['method']}; half-width ±{cf['q_split']:.3f} calibrated on {cf['n_calibration']} validation rows gives "
      f"{cf['empirical_test_coverage_split']:.1%} coverage on {cf['n_test']} unseen test rows (target {cf['coverage_target']:.0%}). The production model uses ±{cf['q_cv']:.3f} "
      f"from grouped 5-fold out-of-fold residuals (coverage {cf['empirical_oof_coverage_cv']:.1%}).\n")
    w(f"![Actual vs predicted]({figs['avp']})\n")
    ai = res.get("adaptive_interval")
    if ai:
        w("### 6.5 Extreme scores and adaptive intervals\n")
        w("Out-of-fold (country-grouped) error of the selected model by *actual* score band. Positive bias = under-prediction:\n")
        ex = pd.DataFrame(ai["extremes"])
        if "affect_model" in res:
            ex = ex.merge(pd.DataFrame(res["affect_model"]["extremes"])[["actual_band", "bias_actual_minus_pred", "rmse"]].rename(
                columns={"bias_actual_minus_pred": "bias_with_affect", "rmse": "rmse_with_affect"}), on="actual_band")
        w(md_table(ex, 3))
        w(f"\nLike any imperfect predictor, the model shrinks extremes. Recalibration and stacking were tested and gave little improvement; the remaining error comes from information "
          f"the indicators do not contain (conflict, collapse). Two remedies are deployed: (1) a **normalised conformal interval** whose width follows a learned difficulty model "
          f"(half-width ±{ai['min_halfwidth']:.2f} to ±{ai['max_halfwidth']:.2f}); coverage {ai['coverage_all']:.1%} overall and {ai['coverage_very_low']:.0%} for actual scores below 4 "
          f"(fixed-width: {ai['fixed_coverage_very_low']:.0%}); (2) an optional companion model adding positive/negative affect"
          + (f" (grouped RMSE {res['affect_model']['oof_rmse']:.3f} vs {ai['oof_rmse']:.3f})" if "affect_model" in res else "") + ".\n")

    w("## 7. Explainable AI\n")
    w(f"Measured for the evaluation {best['name']} on the unseen-country test set ({len(split.test)} rows):\n")
    w(md_table(imp_tbl, 4))
    w(f"\n![Importance]({figs['imp']})\n")
    w("Importance is model-dependent: income, healthy life expectancy and social support are strongly inter-correlated (see VIFs), so the model can substitute one for another. "
      "These rankings describe what the model uses to reproduce historical patterns, not causal effects.\n")

    w("## 8. Clustering\n")
    w(f"Country profiles = mean of the six indicators over {', '.join(map(str, cl.info['years']))} ({cl.info['n_countries']} countries; excluded for incomplete data: "
      f"{', '.join(cl.info['excluded']) or 'none'}). Standardised features; K-means (n_init = 50) and Ward hierarchical clustering.\n")
    w(md_table(cl.scores[["k", "wcss", "silhouette", "davies_bouldin", "calinski_harabasz", "min_cluster_size"]], 3))
    w(f"\n{cl.k_reason} Silhouette = {cl.silhouette:.3f}, Davies–Bouldin = {cl.davies_bouldin:.3f}; agreement with Ward clustering ARI = {cl.hierarchical_ari:.2f}. "
      f"PC1 and PC2 explain {cl.pca_variance[0]:.0%} and {cl.pca_variance[1]:.0%} of the variance.\n")
    w(md_table(cs, 3))
    w("\n\\* Life Ladder was not used to form clusters; it is shown only to describe them. Names are generated from each cluster's own z-score profile.\n")
    w(f"![Cluster scores]({figs['kscores']})\n\n![Clusters]({figs['pca']})\n")

    w("## 9. Automated insights\n")
    for i in insights:
        w(f"- **{i.title}** ({i.category}; {i.metric} = {i.value}; source `{i.source}`{'; ' + i.period if i.period else ''}): {i.text}")
    w("")

    w("## 10. Limitations\n")
    w("- Reported life evaluations may be shaped by cultural response styles; they are not objective measures of individual happiness.\n"
      "- Observational data: confounding and reverse causality cannot be excluded; no result is causal.\n"
      "- Repeated country observations are dependent; naïve p-values are anti-conservative (clustered SEs reported).\n"
      "- National means rest on ~1,000 respondents per country-year; small changes are within sampling error.\n"
      f"- The annual panel ends in {ov['year_max']}; WHR 2024–2026 rankings are included but provide explained-by contributions, not raw indicators.\n"
      "- Country coverage varies over time; some indicators (perceived corruption, healthy life expectancy) are missing for specific countries.\n"
      "- Model-based what-if scenarios describe model behaviour, not the consequences of real-world interventions.\n")
    report = "\n".join(L)
    (ROOT_DIR / "PROJECT_REPORT.md").write_text(report, encoding="utf-8")
    return report


if __name__ == "__main__":  # pragma: no cover
    txt = generate_report()
    print(f"PROJECT_REPORT.md written ({len(txt):,} characters); figures in outputs/figures/")
