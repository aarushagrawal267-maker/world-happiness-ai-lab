"""Build notebooks/exploratory_analysis.ipynb (executed with real outputs).

    python scripts/build_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("# World Happiness AI Lab — Exploratory Analysis Notebook\n\nA reproducible walk-through of the core analysis using the project modules in `src/`. "
       "All numbers are computed from the World Happiness Report files in `data/raw/`.\n\n> Life Ladder = national average of *reported* life evaluations (0–10). "
       "Every relationship below is an association, not a causal effect."),
    code("import sys, pathlib\nROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == 'notebooks' else pathlib.Path.cwd()\nsys.path.insert(0, str(ROOT))\n"
         "import os; os.chdir(ROOT)\nimport warnings; warnings.filterwarnings('ignore')\nimport pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns\n"
         "sns.set_theme(style='whitegrid')\npd.set_option('display.precision', 3)\n"
         "from src.data_loader import load_bundle\nfrom src.config import INDICATORS, TARGET, SHORT_LABELS\nb = load_bundle(use_cache=False)\nb.inventory()"),
    md("## 1. Dataset overview"),
    code("from src.eda import dataset_overview, ranking_overview\npd.Series(dataset_overview(b.panel)), pd.Series(ranking_overview(b.rankings))"),
    code("b.panel.head()"),
    md("## 2. Distribution of Life Ladder"),
    code("fig, ax = plt.subplots(1, 2, figsize=(12, 4))\nsns.histplot(b.panel[TARGET], bins=40, kde=True, ax=ax[0])\nax[0].set_title('All country-years')\n"
         "order = b.panel.groupby('region')[TARGET].median().sort_values().index\nsns.boxplot(data=b.panel, y='region', x=TARGET, order=order, ax=ax[1], color='#2a78d6')\nax[1].set_title('By region'); plt.tight_layout()"),
    md("## 3. Correlations"),
    code("from src.statistical_analysis import correlation_tests\ncorrelation_tests(b.panel, TARGET, INDICATORS + ['positive_affect', 'negative_affect'])"
         "[['feature','n','pearson_r','pearson_ci_low','pearson_ci_high','spearman_rho','strength']]"),
    code("plt.figure(figsize=(8, 6))\nsns.heatmap(b.panel[[TARGET] + INDICATORS].corr().rename(index=SHORT_LABELS, columns=SHORT_LABELS), annot=True, fmt='.2f', cmap='RdBu', vmin=-1, vmax=1)\nplt.title('Pearson correlation');"),
    code("sns.lmplot(data=b.panel, x='log_gdp_per_capita', y=TARGET, scatter_kws={'s': 8, 'alpha': .4}, height=4.5, aspect=1.4, lowess=True, line_kws={'color': '#eb6834'})\nplt.title('Log GDP per capita vs Life Ladder (LOWESS)');"),
    md("## 4. Multiple regression with country-clustered standard errors"),
    code("from src.statistical_analysis import ols_regression\nr = ols_regression(b.panel, TARGET, INDICATORS)\nprint(f\"R² = {r['r2']:.3f}, adj. R² = {r['adj_r2']:.3f}, n = {r['n']}\")\n"
         "r['table'][['term','coef','std_err','cluster_std_err','cluster_p_value','cluster_ci_low','cluster_ci_high']]"),
    code("for m in r['diagnostics']['interpretation']: print('-', m)\nr['diagnostics']['vif']"),
    md("## 5. Missing values and outliers"),
    code("from src.preprocessing import missing_summary, outlier_summary\nmissing_summary(b.panel, INDICATORS + ['positive_affect', 'negative_affect'])"),
    code("outlier_summary(b.panel)"),
    md("## 6. Machine-learning results (from `python -m src.train`)"),
    code("from src.model_manager import load_results\nres = load_results()\nrows = [{'question': s, 'model': m['name'], 'val_rmse': m['val']['rmse'], 'test_rmse': m['test']['rmse'], 'test_r2': m['test']['r2']}\n"
         "        for s, e in res['experiments'].items() for m in e['models'].values()]\npd.DataFrame(rows).pivot(index='model', columns='question', values='test_rmse').round(3)"),
    code("cv = pd.DataFrame(res['cv_comparison']).pivot(index='model', columns='strategy', values='rmse_mean')\ncv['group_vs_random_%'] = (cv['group']/cv['random']-1)*100\ncv.round(3)"),
    code("res['best_model']['name'], res['best_model']['test'], res['conformal']"),
    md("## 7. Clustering"),
    code("from src.clustering import run_clustering\ncl = run_clustering(b.panel)\nprint(cl.k_reason)\ncl.cluster_summary[['cluster','name','n_countries', TARGET, 'example_countries']]"),
    code("fig, ax = plt.subplots(figsize=(8, 5))\nsns.scatterplot(data=cl.profiles, x='pc1', y='pc2', hue='cluster_name', s=50, ax=ax)\nax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.12))\nax.set_title('Clusters in PCA space');"),
    md("## 8. Automated insights"),
    code("from src.insight_engine import generate_insights\nfrom src.feature_engineering import engineer_panel\nfor i in generate_insights(b.panel, engineer_panel(b.panel), b.rankings, b.panel_source, b.rankings_source, res, cl)[:8]:\n    print(f'• {i.title}: {i.text}\\n')"),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "notebooks" / "exploratory_analysis.ipynb"
nbf.write(nb, out)
print("written", out)
