import pandas as pd
import streamlit as st

from src import ui
from src.config import ALL_NUMERIC, TARGET
from src.data_acquisition import MANUAL_INSTRUCTIONS, SOURCES, acquire_all, load_manifest
from src.data_loader import ingest_upload
from src.data_validation import DATASET_KINDS

ui.page_header(
    "Dataset explorer",
    "Every file the platform uses: where it came from, how it was standardised, what validation found, and the data itself. "
    "Upload your own WHR edition (CSV or Excel) and it will be detected, validated and standardised automatically.",
    "Explore · Data acquisition & provenance",
)

b = ui.bundle()
manifest = load_manifest()
tab_inv, tab_browse, tab_val, tab_up, tab_src = st.tabs(["Inventory", "Browse data", "Validation", "Upload", "Sources & methodology"])

with tab_inv:
    inv = b.inventory()
    ui.kpi_cards([
        {"label": "Files detected", "value": str(len(inv)), "sub": "CSV / Excel under data/raw"},
        {"label": "Valid files", "value": str(int(inv["valid"].sum())), "sub": "passed all error-level checks"},
        {"label": "Editions", "value": f"{len(b.editions) + (b.rankings['year'].nunique() if not b.rankings.empty else 0)}", "sub": "edition tables + ranking years"},
        {"label": "Panel rows", "value": f"{len(b.panel):,}", "sub": b.panel_source},
    ])
    st.dataframe(inv, hide_index=True, height=480, column_config={"valid": st.column_config.CheckboxColumn("valid")})
    st.caption("Structure is detected from the columns, not the file name: " + "; ".join(f"**{k}** = {v}" for k, v in DATASET_KINDS.items() if k != "unknown"))
    pv = manifest.get("panel_verification", {})
    if pv:
        kind = "ok" if pv.get("status") == "passed" else "danger"
        ui.callout(
            f"<b>Panel integrity check: {pv.get('status', '?').upper()}.</b> The annual panel is a third-party mirror of the WHR 2023 DataForTable2.1 file, so it is verified against the "
            f"official WHR 2023 Figure 2.1 file: its 2020–2022 means reproduce the official ladder scores for {pv.get('matched_countries')}/{pv.get('official_countries')} countries "
            f"(Pearson r = {pv.get('pearson_r')}, median |difference| = {pv.get('median_abs_diff')}, {pv.get('share_within_0_01', 0):.0%} within 0.01).",
            kind,
        )

with tab_browse:
    options = {"Annual panel (primary)": b.panel, "Official rankings (multi-year)": b.rankings}
    options.update({f"Edition WHR {e}": d for e, d in b.editions.items()})
    name = st.selectbox("Dataset", list(options))
    df = options[name]
    f1, f2, f3 = st.columns(3)
    with f1:
        countries = st.multiselect("Countries", sorted(df["country"].unique()), placeholder="All")
    with f2:
        regs = st.multiselect("Regions", sorted(df["region"].dropna().unique()) if "region" in df else [], placeholder="All")
    with f3:
        if "year" in df and df["year"].nunique() > 1:
            yr = st.slider("Years", int(df["year"].min()), int(df["year"].max()), (int(df["year"].min()), int(df["year"].max())))
        else:
            yr = None
    v = df
    if countries:
        v = v[v["country"].isin(countries)]
    if regs:
        v = v[v["region"].isin(regs)]
    if yr:
        v = v[v["year"].between(*yr)]
    st.markdown(f"<span class='tag blue'>{len(v):,} rows</span><span class='tag'>{v['country'].nunique()} countries</span><span class='tag'>{v.shape[1]} columns</span>", unsafe_allow_html=True)
    st.dataframe(ui.fmt_table(v), hide_index=True, height=460)
    st.download_button("Download filtered CSV", v.to_csv(index=False).encode(), file_name="whr_filtered.csv", mime="text/csv")
    num = [c for c in v.select_dtypes("number").columns if c not in ("year", "edition")]
    with st.expander("Descriptive statistics"):
        st.dataframe(ui.fmt_table(v[num].describe().T), height=360)

with tab_val:
    fname = st.selectbox("File", [f.name for f in b.files])
    lf = next(f for f in b.files if f.name == fname)
    rep = lf.report
    ui.kpi_cards([
        {"label": "Detected structure", "value": rep.kind.replace("_", " "), "sub": DATASET_KINDS.get(rep.kind, "")},
        {"label": "Rows · columns", "value": f"{rep.n_rows} · {rep.n_columns}"},
        {"label": "Countries", "value": str(rep.n_countries)},
        {"label": "Duplicate keys", "value": str(rep.duplicate_country_year), "tone": "positive" if rep.duplicate_country_year == 0 else "negative"},
    ])
    for i in rep.issues:
        ui.callout(f"<b>{i.check}</b> — {i.message}", {"error": "danger", "warning": "warn", "info": "ok"}[i.level])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Column mapping (original → standard)**")
        st.dataframe(pd.DataFrame({"original": list(lf.column_mapping), "standard": list(lf.column_mapping.values())}), hide_index=True, height=360)
    with c2:
        st.markdown("**Missing values**")
        if rep.missing_counts:
            st.dataframe(pd.DataFrame({"missing": rep.missing_counts, "pct": rep.missing_pct}).sort_values("missing", ascending=False), height=360)
        else:
            st.write("No missing values.")
    log = lf.log.as_dict()
    st.markdown("**Cleaning log**")
    st.json(log, expanded=False)

with tab_up:
    st.markdown("Upload a WHR dataset (any edition, CSV or Excel). Column names are matched to the standard schema automatically; "
                "the file is validated before it is stored in `data/raw/uploads/`.")
    up = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
    if up is not None:
        try:
            lf = ingest_upload(up.getvalue(), up.name, save=False)
            rep = lf.report
            ui.callout(f"Detected <b>{rep.kind}</b>: {rep.n_rows} rows, {rep.n_countries} countries, years {rep.years[:1]}…{rep.years[-1:]}.", "ok" if rep.is_valid else "danger")
            for i in rep.issues:
                ui.callout(f"<b>{i.check}</b> — {i.message}", {"error": "danger", "warning": "warn", "info": "ok"}[i.level])
            st.dataframe(ui.fmt_table(lf.frame.head(50)), hide_index=True)
            if rep.is_valid and st.button("Save upload to data/raw/uploads", type="primary"):
                saved = ingest_upload(up.getvalue(), up.name, save=True)
                st.session_state["data_version"] = st.session_state.get("data_version", 0) + 1
                st.success(f"Saved {saved.path.name}. It is now part of the inventory.")
                if saved.kind == "annual_panel":
                    st.session_state["panel_file_candidate"] = saved.path.name
        except Exception as exc:
            st.error(f"Could not read this file: {exc}")
    panels = [f.name for f in b.files if f.kind == "annual_panel" and f.report.is_valid]
    if len(panels) > 1:
        choice = st.selectbox("Annual panel used by the platform", panels, index=panels.index(b.panel_source) if b.panel_source in panels else 0)
        if choice != b.panel_source and st.button("Switch panel"):
            st.session_state["panel_file"] = choice
            st.rerun()
        ui.callout("Switching the panel changes EDA, statistics and clustering immediately. Retrain models in the ML Training Lab so predictions use the new data.", "warn")

with tab_src:
    rows = []
    srcs = manifest.get("sources", {})
    for s in SOURCES:
        rec = srcs.get(s.key, {})
        rows.append({"key": s.key, "title": s.title, "edition": s.edition, "survey years": s.survey_years, "status": rec.get("status", "unknown"),
                     "sha256": (rec.get("sha256") or "")[:12], "url": s.url, "used for": s.role})
    st.dataframe(pd.DataFrame(rows), hide_index=True, height=440, column_config={"url": st.column_config.LinkColumn("url")})
    sel = st.selectbox("Methodology record", [s.key for s in SOURCES], format_func=lambda k: next(s.title for s in SOURCES if s.key == k))
    s = next(s for s in SOURCES if s.key == sel)
    ui.card(s.title, f"<b>Publisher:</b> {ui.esc(s.publisher)}<br/><b>Content:</b> {ui.esc(s.content)}<br/><b>Methodology:</b> {ui.esc(s.methodology_note)}",
            meta=f"local file: data/raw/{s.local_path} · role: {s.role}")
    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("Check / download missing sources", help="Downloads only files that are not yet present"):
            with st.spinner("Contacting sources…"):
                m = acquire_all(force=False)
            failed = [k for k, r in m["sources"].items() if r["status"].startswith("failed")]
            st.success("All sources present.") if not failed else st.warning(f"Failed: {', '.join(failed)} — see manual instructions.")
            st.session_state["data_version"] = st.session_state.get("data_version", 0) + 1
    st.markdown(MANUAL_INSTRUCTIONS)
ui.footer()
_ = (ALL_NUMERIC, TARGET)
