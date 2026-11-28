import streamlit as st

from src import ui
from src.insight_engine import generate_insights, insights_frame

ui.page_header(
    "Automated insights",
    "Findings generated on the fly from the loaded data and the saved model results. Nothing here is hard-coded: change the data and the insights change. "
    "Every card states its source, reporting period and metric.",
    "Overview · Data-driven findings",
)
res = ui.results()
b = ui.bundle()
imp = None
if res:
    try:
        imp = ui.permutation_importance(f"eval_group_{res['best_model']['key']}", "group")
    except Exception:
        imp = None
cl = ui.clustering()
ins = generate_insights(b.panel, ui.panel_fe(), b.rankings, b.panel_source, b.rankings_source, res, cl, imp)

cats = list(dict.fromkeys(i.category for i in ins))
pick = st.pills("Filter by theme", cats, selection_mode="multi", default=cats)
shown = [i for i in ins if i.category in (pick or cats)]
st.markdown(f"<span class='tag blue'>{len(shown)} insights</span><span class='tag'>generated from {b.panel_source} and {b.rankings_source}</span>", unsafe_allow_html=True)
c1, c2 = st.columns(2, gap="medium")
for n, i in enumerate(shown):
    with (c1 if n % 2 == 0 else c2):
        ui.insight_card(i)
if not res:
    ui.callout("Model-based insights appear after models are trained (ML Training Lab).", "warn")
st.download_button("Download insights (CSV)", insights_frame(ins).to_csv(index=False).encode(), file_name="whr_insights.csv", mime="text/csv")
ui.causation_note()
ui.footer()
