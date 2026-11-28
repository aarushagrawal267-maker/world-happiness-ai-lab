import pandas as pd
import streamlit as st

from src import ui
from src import visualization as viz
from src.config import INDICATORS, LABELS, SHORT_LABELS, TARGET

ui.page_header(
    "Country clustering",
    "Unsupervised learning groups countries by similarity in their six socio-economic indicators. K-means and Ward hierarchical clustering, "
    "evaluated with silhouette, Davies–Bouldin and within-cluster sum of squares, visualised in PCA space and on the map.",
    "Analyze · Unsupervised learning",
)
b = ui.bundle()
panel = b.panel
years_all = sorted(panel["year"].unique())
c1, c2, c3 = st.columns([1.4, 1, 1])
with c1:
    yrs = st.multiselect("Profile years (country mean)", years_all, default=[2020, 2021, 2022])
with c2:
    auto = st.toggle("Data-driven k", value=True)
with c3:
    k_manual = st.slider("k (manual)", 2, 8, 4, disabled=auto)
if not yrs:
    st.info("Pick at least one year.")
    st.stop()
cl = ui.clustering(None if auto else k_manual, tuple(sorted(yrs)))
p = cl.profiles

ui.kpi_cards([
    {"label": "Clusters (k)", "value": str(cl.k), "sub": "data-driven" if auto else "manual"},
    {"label": "Countries", "value": str(cl.info["n_countries"]), "sub": f"{len(cl.info['excluded'])} excluded (incomplete)"},
    {"label": "Silhouette", "value": f"{cl.silhouette:.3f}", "sub": "−1…1, higher = better separated"},
    {"label": "Davies–Bouldin", "value": f"{cl.davies_bouldin:.3f}", "sub": "lower = more compact/separate"},
    {"label": "K-means vs Ward", "value": f"{cl.hierarchical_ari:.2f}", "sub": "adjusted Rand agreement"},
    {"label": "PCA: PC1 + PC2", "value": f"{(cl.pca_variance[0] + cl.pca_variance[1]):.0%}", "sub": "variance shown in 2-D"},
])

ui.section("Choosing the number of clusters", "01")
ui.chart(viz.cluster_scores(cl.scores, cl.k), key="scores")
ui.callout(cl.k_reason, "info")
with st.expander("All cluster-quality scores"):
    st.dataframe(ui.fmt_table(cl.scores, 3), hide_index=True)

ui.section("Clusters in PCA space & on the map", "02")
c1, c2 = st.columns([1.15, 1])
with c1:
    ui.chart(viz.pca_scatter(p, cl.pca_variance, "Countries projected on the first two principal components"), key="pca")
with c2:
    m = p.copy()
    m["cluster_label"] = "C" + (m["cluster"] + 1).astype(str) + " · " + m["cluster_name"].str.split(" · ").str[0]
    ui.chart(viz.choropleth(m, "cluster_label", categorical=True, hover_cols=["region"], height=540, title="Cluster membership"), key="clmap")
lo = cl.pca_loadings.copy()
lo.index = [SHORT_LABELS[i] for i in lo.index]
ui.chart(viz.heatmap(lo.T, "PCA loadings (how each indicator contributes to each component)", zmin=-1, zmax=1), key="load")
st.caption(f"PC1 explains {cl.pca_variance[0]:.0%} of the variance and loads on income, health and social support together — a general 'development' axis. "
           f"PC2 ({cl.pca_variance[1]:.0%}) mostly separates generosity and perceived corruption.")

ui.section("What characterises each cluster?", "03", "names are generated from each cluster's own profile")
ui.chart(viz.cluster_profile_heatmap(cl.cluster_z, cl.names, "Cluster means in standard deviations from the global mean"), key="zheat")
cs = cl.cluster_summary.copy()
cs["cluster"] = "C" + (cs["cluster"] + 1).astype(str)
cs = cs.rename(columns={**{f: SHORT_LABELS[f] for f in INDICATORS}, TARGET: "Life Ladder (not used for clustering)"})
st.dataframe(ui.fmt_table(cs, 3), hide_index=True)
pl = p.copy()
pl["cluster_label"] = "C" + (pl["cluster"] + 1).astype(str)
ui.chart(viz.box_by(pl, TARGET, "cluster_label", "Life Ladder by cluster (descriptive — Life Ladder was not an input)", order=sorted(pl["cluster_label"].unique())), key="clbox")
for c in sorted(p["cluster"].unique()):
    members = p[p["cluster"] == c].sort_values(TARGET, ascending=False)
    with st.expander(f"C{c + 1} · {cl.names[c]} — {len(members)} countries"):
        st.write(", ".join(members["country"]))

ui.section("Hierarchical clustering (Ward linkage)", "04")
ui.chart(viz.dendrogram(cl.linkage_matrix, p["country"].tolist(), cl.k, "Dendrogram of countries"), key="dendro")
ct = pd.crosstab(p["cluster"].map(lambda c: f"K-means C{c + 1}"), p["hier_cluster"].map(lambda c: f"Ward {c + 1}"))
st.markdown("**Agreement between K-means and Ward clusters**")
st.dataframe(ct)

ui.section("Notes", "")
if cl.info["excluded"]:
    ui.callout(f"Excluded because indicators are missing in the window and in their own history: {', '.join(cl.info['excluded'])}.", "warn")
if cl.info["filled_from_own_history"]:
    st.caption("Gaps filled from the same country's most recent earlier observation: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in cl.info["filled_from_own_history"].items()))
ui.callout("Clusters summarise similarity in six chosen indicators; with a silhouette around 0.3 they overlap substantially. They are a descriptive lens, "
           "<b>not objective categories of national happiness</b>, and would change with other indicators, years or scaling choices.", "warn")
ui.footer()
_ = LABELS
