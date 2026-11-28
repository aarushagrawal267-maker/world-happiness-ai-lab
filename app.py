"""WORLD HAPPINESS AI LAB — Streamlit entry point.

Run with:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="World Happiness AI Lab",
    page_icon=":material/public:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "World Happiness AI Lab — understanding and predicting global happiness with World Happiness Report data."},
)

from src import ui  # noqa: E402

ui.inject_css()
ui.sidebar_brand()

P = "views/"
pages = {
    "Overview": [
        st.Page(P + "0_Home_Dashboard.py", title="Home dashboard", icon=":material/space_dashboard:", default=True),
        st.Page(P + "14_Automated_Insights.py", title="Automated insights", icon=":material/lightbulb:"),
    ],
    "Explore": [
        st.Page(P + "1_Global_Happiness_Explorer.py", title="Global happiness explorer", icon=":material/public:"),
        st.Page(P + "2_Dataset_Explorer.py", title="Dataset explorer", icon=":material/table_view:"),
        st.Page(P + "3_Exploratory_Analysis.py", title="Exploratory analysis", icon=":material/query_stats:"),
        st.Page(P + "5_Country_Comparison.py", title="Country comparison", icon=":material/compare_arrows:"),
        st.Page(P + "6_Historical_Trends.py", title="Historical trends", icon=":material/timeline:"),
    ],
    "Analyze": [
        st.Page(P + "4_Statistical_Analysis.py", title="Statistical analysis", icon=":material/functions:"),
        st.Page(P + "10_Country_Clustering.py", title="Country clustering", icon=":material/bubble_chart:"),
    ],
    "Machine learning": [
        st.Page(P + "7_Machine_Learning_Training.py", title="ML training lab", icon=":material/model_training:"),
        st.Page(P + "8_Model_Comparison.py", title="Model comparison", icon=":material/leaderboard:"),
        st.Page(P + "9_Model_Evaluation.py", title="Model evaluation", icon=":material/fact_check:"),
        st.Page(P + "11_Explainable_AI.py", title="Explainable AI", icon=":material/account_tree:"),
    ],
    "Simulate": [
        st.Page(P + "12_Happiness_Prediction.py", title="Happiness prediction", icon=":material/online_prediction:"),
        st.Page(P + "13_What_If_Simulator.py", title="What-if simulator", icon=":material/tune:"),
    ],
    "Project": [
        st.Page(P + "15_About.py", title="About the project", icon=":material/info:"),
    ],
}

nav = st.navigation(pages, expanded=True)
ui.sidebar_status(ui.bundle(), ui.results())
nav.run()
