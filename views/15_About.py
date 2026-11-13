import streamlit as st

from src import ui
from src.config import DEFINITIONS, LABELS
from src.data_acquisition import SOURCES

ui.page_header("About the Project", "Purpose, data, methods, design decisions and limitations of the World Happiness AI Lab.", "Project")

ui.section("Purpose", "01")
st.markdown(
    "The World Happiness AI Lab investigates how economic, social, institutional and health conditions relate to **reported life evaluations** across countries. "
    "It combines exploratory analysis, formal statistics, supervised and unsupervised machine learning, explainable AI and model-based scenarios in one place."
)
qs = [
    "Are countries with higher GDP per capita generally happier?", "How does social support relate to happiness?", "Does healthy life expectancy relate to happiness?",
    "How important is freedom to make life choices?", "How does perceived corruption relate to reported happiness?", "How do generosity and trust relate to happiness?",
    "Which countries have improved or declined over time?", "Can machine learning estimate a country's happiness score from its indicators?",
    "Can countries be grouped into meaningful profiles?", "Which variables are most useful for prediction?",
]
pages = ["3_Exploratory_Analysis", "4_Statistical_Analysis", "4_Statistical_Analysis", "11_Explainable_AI", "4_Statistical_Analysis", "3_Exploratory_Analysis",
         "6_Historical_Trends", "8_Model_Comparison", "10_Country_Clustering", "11_Explainable_AI"]
cols = st.columns(2)
for i, (q, p) in enumerate(zip(qs, pages)):
    with cols[i % 2]:
        st.page_link(f"views/{p}.py", label=f"Q{i + 1}. {q}", icon=":material/help:")

ui.section("Data", "02")
for s in SOURCES[:8]:
    st.markdown(f"- **{s.title}** — {s.publisher}. {s.content}. [{s.url}]({s.url})")
st.markdown("- **Kaggle *unsdsn/world-happiness*** (WHR 2015–2019 editions) — mirrored on Hugging Face (CC0).")
with st.expander("Variable definitions"):
    for k, v in DEFINITIONS.items():
        st.markdown(f"**{LABELS.get(k, k)}** — {v}")

ui.section("Methods", "03")
c1, c2 = st.columns(2)
with c1:
    ui.card("Preprocessing", "Standardised headers and country names across editions; impossible values set to missing; genuine duplicates removed; "
            "imputation and scaling inside scikit-learn Pipelines fitted on training folds only; targets never imputed.")
    ui.card("Statistics", "Pearson/Spearman with Fisher CIs and Holm correction; OLS with classical, HC3 and country-clustered SEs; VIF, Breusch–Pagan, Jarque–Bera, "
            "Durbin–Watson and RESET diagnostics; ANOVA, Welch ANOVA and Kruskal–Wallis on country means with Holm-corrected post-hoc tests and effect sizes.")
    ui.card("Clustering", "K-means (k chosen by consensus of silhouette, Davies–Bouldin and Calinski–Harabasz), Ward hierarchical clustering, PCA; names generated from cluster profiles.")
with c2:
    ui.card("Machine learning", "Linear, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting and XGBoost; RandomizedSearchCV tuning; three validation "
            "questions — random, country-grouped (GroupKFold) and forward-in-time; model selected on grouped validation RMSE.")
    ui.card("Uncertainty & explainability", "Split-conformal 90% prediction intervals calibrated on unseen countries; permutation importance, SHAP (Tree/Linear explainers), "
            "partial dependence with ICE curves, per-country waterfall explanations.")
    ui.card("Visual design", "Dark navy research theme; categorical palette validated for colour-vision deficiency (adjacent ΔE ≥ 8 in OKLab), single-hue sequential "
            "and red–grey–blue diverging scales; one axis per chart; tooltips on every chart.")

ui.section("Limitations", "04")
st.markdown(
    "- **Reported, not objective, wellbeing.** The Cantril ladder is a self-reported evaluation; cultural response styles may shift national averages.\n"
    "- **Observational data.** All relationships are associations. Reverse causality and confounding (institutions, history, conflict) cannot be ruled out.\n"
    "- **Non-independent observations.** Country-years repeat the same countries; naïve p-values are too small (clustered SEs mitigate this).\n"
    "- **Sampling uncertainty.** National means come from ~1,000 respondents per year; small year-to-year changes are often noise.\n"
    "- **Coverage.** Country coverage varies by year (27 countries in 2005) and some indicators are missing (e.g. perceived corruption in a few countries).\n"
    "- **Annual panel ends in 2022.** The publicly available WHR 2023 panel is used for modelling; the official rankings extend to WHR 2026 (survey years to 2025) but contain explained-by contributions rather than raw indicators.\n"
    "- **Positive/negative affect** are available but excluded from the default model because they are themselves wellbeing measures.\n"
    "- **What-if scenarios** describe the model's response, not causal effects, and become unreliable outside the observed data range."
)
ui.section("Run it yourself", "05")
st.code("pip install -r requirements.txt\npython -m src.data_acquisition   # optional: refresh/verify datasets\npython -m src.train              # optional: full training run\nstreamlit run app.py", language="bash")
ui.footer()
