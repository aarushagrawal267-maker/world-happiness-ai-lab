"""Streamlit UI kit: design system (CSS), reusable components and cached
accessors for data, models and analysis results.

Kept separate from the analytical modules so that ``src`` stays importable
(and testable) without Streamlit running.
"""
from __future__ import annotations

import html
from typing import Iterable

import pandas as pd
import streamlit as st

from .config import CFG, PALETTE

# --------------------------------------------------------------------------- CSS
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500&family=JetBrains+Mono:wght@400;500&display=swap');

/* Graphite ramp, one daylight-blue accent. Chart colours live in visualization.py. */
:root {
  --bg: #141415; --sidebar: #0f0f10; --surface: #19191b; --surface-2: #1e1e20; --surface-3: #232325;
  --border: rgba(237,237,237,0.08); --border-strong: rgba(237,237,237,0.14);
  --text: #ededed; --text-2: #a1a1a6; --muted: #85858b;
  --primary: #6da7ec; --primary-ink: #0f0f10;
  --positive: #4cc38a; --negative: #e5484d; --caution: #d4a24c;
  --radius: 12px; --radius-sm: 8px;
  --ease: cubic-bezier(0.22, 1, 0.36, 1);
  --font: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
  --display: 'Newsreader', Georgia, 'Times New Roman', serif;
  --mono: 'JetBrains Mono', ui-monospace, Consolas, monospace;
}

html, body, [class*="css"], .stApp { font-family: var(--font); }
.stApp { background: var(--bg); }
.block-container { padding-top: 32px; padding-bottom: 64px; max-width: 1440px; }
h1, h2, h3, h4 { font-family: var(--font) !important; font-weight: 500 !important; letter-spacing: 0; color: var(--text); }
h2 { font-size: 1.375rem !important; margin-top: 16px !important; }
h3 { font-size: 1.125rem !important; }
p, li, label, .stMarkdown { color: var(--text-2); }
b, strong { font-weight: 500; color: var(--text); }
a { color: var(--primary) !important; }
code { font-family: var(--mono) !important; background: var(--surface-3) !important; color: var(--text) !important; border-radius: 4px; }
hr { border-color: var(--border) !important; }
.scale { font-family: var(--mono); font-size: 0.45em; font-weight: 400; color: var(--muted); margin-left: 4px; letter-spacing: 0; white-space: nowrap; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { background: var(--sidebar); border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a { border-radius: var(--radius-sm); margin: 2px 8px; transition: background-color .15s var(--ease), color .15s var(--ease); }
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover { background: var(--surface-2); }
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] { background: var(--surface-3); box-shadow: inset 0 1px 0 rgba(255,255,255,0.06); }
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] span { color: var(--text) !important; }
[data-testid="stSidebarHeader"] img, img[data-testid="stLogo"] { height: 2.5rem !important; max-width: 100%; }
section[data-testid="stSidebar"] [data-testid="stNavSectionHeader"] { font-family: var(--font); font-size: .75rem !important; font-weight: 500; color: var(--muted) !important; text-transform: none; letter-spacing: 0; }
.side-status { margin: 8px; padding: 12px; border-radius: var(--radius-sm); background: var(--surface); border: 1px solid var(--border); font-size: .8125rem; line-height: 1.6; color: var(--text-2); font-variant-numeric: tabular-nums; }
.dot { display:inline-block; width:6px; height:6px; border-radius:50%; margin-right:8px; vertical-align:middle; background: var(--muted); }
.dot.ok { background: var(--positive); } .dot.warn { background: var(--caution); } .dot.bad { background: var(--negative); }

/* ---------- hero ---------- */
.hero { border-radius: 16px; padding: 40px; margin-bottom: 24px; background: var(--surface); border: 1px solid var(--border); }
.eyebrow { font-size: .875rem; color: var(--text-2); }
.hero h1 { font-family: var(--display) !important; font-size: clamp(2.25rem, 4.4vw, 3.5rem) !important; line-height: 1.06; letter-spacing: -0.02em; margin: 8px 0 16px 0 !important; font-weight: 400 !important; color: var(--text); }
.hero p.lead { font-size: 1.0625rem; line-height: 1.65; max-width: 46rem; color: var(--text-2); margin: 0; }
.hero .chips { margin-top: 24px; display:flex; gap: 8px 24px; flex-wrap: wrap; font-family: var(--mono); font-size: .8125rem; color: var(--muted); font-variant-numeric: tabular-nums; }
.chip b { color: var(--text); font-weight: 500; }
.chip:empty { display: none; }

/* ---------- page header ---------- */
.page-head { margin: 0 0 24px 0; padding-bottom: 20px; border-bottom: 1px solid var(--border); }
.page-head .kicker { font-size: .875rem; color: var(--muted); }
.page-head h1 { font-family: var(--display) !important; font-size: clamp(1.875rem, 3vw, 2.375rem) !important; line-height: 1.15; margin: 4px 0 8px 0 !important; font-weight: 400 !important; color: var(--text); letter-spacing: -0.01em; }
.page-head p { margin: 0; color: var(--text-2); max-width: 900px; font-size: 1rem; }

/* ---------- cards ---------- */
.kpi-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(136px, 1fr)); gap: 12px; margin: 8px 0 24px 0; }
.kpi { padding: 20px; border-radius: var(--radius); background: var(--surface); border: 1px solid var(--border); }
.kpi .label { font-size: .8125rem; color: var(--text-2); }
.kpi .value { font-size: 1.875rem; font-weight: 500; color: var(--text); line-height: 1.15; margin-top: 8px; font-variant-numeric: tabular-nums; }
.kpi .value.long { font-size: 1.25rem; padding-top: 8px; }
.kpi .value small { font-family: var(--mono); font-size: .8125rem; color: var(--muted); font-weight: 400; margin-left: 4px; }
.kpi .sub { font-size: .8125rem; color: var(--muted); margin-top: 4px; }

.card { padding: 20px 24px; border-radius: var(--radius); background: var(--surface); border: 1px solid var(--border); margin-bottom: 12px; }
.card h4 { margin: 0 0 8px 0 !important; font-size: 1rem !important; }
.card p { margin: 0; font-size: .9375rem; }
.card .meta { margin-top: 12px; font-size: .75rem; color: var(--muted); }
.card.glow { border-color: var(--border-strong); }

.insight { padding: 20px 24px; border-radius: var(--radius); background: var(--surface); border: 1px solid var(--border); margin-bottom: 12px; }
.insight .cat { font-size: .8125rem; color: var(--muted); display: flex; align-items: center; }
.insight h4 { margin: 4px 0 8px 0 !important; font-size: 1.0625rem !important; }
.insight p { margin: 0; font-size: .9375rem; color: var(--text-2); }
.insight .metric { float: right; text-align: right; margin-left: 16px; }
.insight .metric .v { font-size: 1.5rem; font-weight: 500; color: var(--text); font-variant-numeric: tabular-nums; }
.insight .metric .l { font-size: .75rem; color: var(--muted); }
.insight .src { margin-top: 12px; font-size: .75rem; color: var(--muted); font-family: var(--mono); }

.callout { display:flex; gap: 12px; align-items: baseline; padding: 16px; border-radius: var(--radius); margin: 12px 0 16px 0; font-size: .9375rem; line-height: 1.55;
  background: var(--surface); border: 1px solid var(--border); color: var(--text-2); }
.callout .ci { flex: none; width: 6px; height: 6px; border-radius: 50%; background: var(--muted); transform: translateY(-2px); }
.callout.warn .ci { background: var(--caution); }
.callout.danger .ci { background: var(--negative); }
.callout.ok .ci { background: var(--positive); }
.callout.danger { border-color: rgba(229,72,77,0.35); }

.section { display:flex; align-items: baseline; gap: 12px; flex-wrap: wrap; margin: 40px 0 12px 0; }
.section .num { font-family: var(--mono); font-size: .8125rem; color: var(--muted); }
.section h2 { margin: 0 !important; font-size: 1.25rem !important; }
.section .desc { color: var(--muted); font-size: .875rem; }

.stat-row { display:flex; gap: 24px; flex-wrap: wrap; padding: 4px 0 8px 0; }
.stat { min-width: 120px; }
.stat .l { font-size: .8125rem; color: var(--text-2); }
.stat .v { font-size: 1.375rem; color: var(--text); font-weight: 500; font-variant-numeric: tabular-nums; }
.stat .v .scale, .insight .metric .v .scale { font-size: .8125rem; }

.big-number { font-weight: 500; font-size: 3.5rem; line-height: 1; letter-spacing: -0.02em; color: var(--text); font-variant-numeric: tabular-nums; }

.tag { display:inline-block; font-size: .75rem; padding: 2px 8px; border-radius: 4px; margin: 2px 4px 2px 0; border: 1px solid var(--border); color: var(--text-2); background: var(--surface-2); }
.tag.blue { color: var(--primary); }
.tag.green { color: var(--positive); }
.tag.orange { color: var(--caution); }
.tag.red { color: #f07a7e; }

.feature-tile { display:block; min-height: 120px; padding: 20px; border-radius: var(--radius); background: var(--surface); border:1px solid var(--border); height: 100%; transition: border-color .15s var(--ease); }
.feature-tile:hover { border-color: var(--border-strong); }
.feature-tile .t { color: var(--text); font-weight: 500; font-size: 1rem; }
.feature-tile .d { color: var(--text-2); font-size: .875rem; margin-top: 4px; }

.footer { margin-top: 48px; padding-top: 20px; border-top: 1px solid var(--border); font-size: .75rem; color: var(--muted); }

/* ---------- widgets ---------- */
div[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px; }
div[data-testid="stMetricValue"] { font-weight: 500; font-variant-numeric: tabular-nums; }
div[data-testid="stMetricLabel"] p { color: var(--text-2) !important; font-size: .8125rem !important; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { height: 40px; border-radius: var(--radius-sm) var(--radius-sm) 0 0; padding: 0 16px; color: var(--text-2); font-weight: 400; }
.stTabs [aria-selected="true"] { color: var(--text) !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--primary) !important; }
div[data-testid="stExpander"] { border: 1px solid var(--border) !important; border-radius: var(--radius) !important; background: var(--surface); }
div[data-testid="stExpander"] summary p { color: var(--text) !important; font-weight: 500; }
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button { border-radius: var(--radius-sm); border: 1px solid var(--border-strong); background: var(--surface-3); font-weight: 500;
  transition: background-color .15s var(--ease), border-color .15s var(--ease), transform .12s var(--ease); }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: rgba(237,237,237,0.24); }
.stButton > button:active, .stFormSubmitButton > button:active { transform: scale(0.98); }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind^="primary"], button[data-testid^="stBaseButton-primary"] { background: var(--primary); border: 1px solid var(--primary); color: var(--primary-ink); }
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind^="primary"]:hover, button[data-testid^="stBaseButton-primary"]:hover { background: #82b4ef; border-color: #82b4ef; }
button[data-testid^="stBaseButton-primary"] p, button[data-testid^="stBaseButton-primary"] span, .stButton > button[kind="primary"] p, .stFormSubmitButton > button[kind^="primary"] p { color: var(--primary-ink) !important; font-weight: 500; }
button:focus-visible, a:focus-visible { outline: 2px solid var(--primary) !important; outline-offset: 2px; }
div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, .stNumberInput input { background: var(--surface-3) !important; border-color: var(--border) !important; border-radius: var(--radius-sm) !important; }
div[data-testid="stPlotlyChart"] { border-radius: var(--radius); background: var(--surface); border: 1px solid var(--border); padding: 12px 12px 4px 12px; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: var(--radius); }
div[data-testid="stAlert"] { border-radius: var(--radius); }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
@media (max-width: 640px) {
  .block-container { padding-left: 16px; padding-right: 16px; }
  .hero { padding: 24px 20px; border-radius: 12px; }
  .hero h1 { font-size: 2.25rem !important; }
  .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
  .kpi { padding: 16px; }
  .kpi .value { font-size: 1.5rem; }
  .big-number { font-size: 2.75rem; }
}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape(str(x))


# --------------------------------------------------------------------------- components
def sidebar_brand() -> None:
    from .config import ROOT_DIR

    logo, icon = ROOT_DIR / "assets" / "logo.svg", ROOT_DIR / "assets" / "icon.svg"
    if logo.exists():
        st.logo(str(logo), size="large", icon_image=str(icon) if icon.exists() else None)


def sidebar_status(bundle, results) -> None:
    with st.sidebar:
        n_c = bundle.panel["country"].nunique() if not bundle.panel.empty else 0
        model_txt = f"<span class='dot ok'></span><b>{esc(results['best_model']['name'])}</b> ready" if results else "<span class='dot warn'></span>Models not trained"
        st.markdown(
            f"""<div class="side-status"><span class='dot'></span><b>{len(bundle.panel):,}</b> country-years · <b>{n_c}</b> countries<br/>
            <span class='dot'></span>Rankings to <b>WHR {bundle.latest_ranking_year + 1 if bundle.latest_ranking_year else '—'}</b><br/>{model_txt}</div>""",
            unsafe_allow_html=True,
        )


def hero(title: str, lead: str, eyebrow: str = "", chips: Iterable[str] = ()) -> None:
    chips_html = "".join(f"<span class='chip'>{c}</span>" for c in chips if c)
    st.markdown(
        f"""<div class="hero">{f'<div class="eyebrow">{esc(eyebrow)}</div>' if eyebrow else ''}
        <h1>{esc(title)}</h1><p class="lead">{lead}</p><div class="chips">{chips_html}</div></div>""",
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "", kicker: str = "") -> None:
    st.markdown(
        f"""<div class="page-head"><div class="kicker">{esc(kicker)}</div><h1>{esc(title)}</h1><p>{subtitle}</p></div>""",
        unsafe_allow_html=True,
    )


def section(title: str, num: str = "", desc: str = "") -> None:
    st.markdown(
        f"""<div class="section">{f'<span class="num">{esc(num)}</span>' if num else ''}<h2>{esc(title)}</h2><span class="desc">{desc}</span></div>""",
        unsafe_allow_html=True,
    )


def score(v: float, fmt: str = ".2f") -> str:
    """A Cantril-ladder score with its small mono '/ 10' scale mark (the app's signature)."""
    return f"{v:{fmt}}<small class='scale'>/ 10</small>"


def kpi_cards(items: list[dict]) -> None:
    """items: {label, value, sub?}. Legacy ``icon``/``accent`` keys are ignored: KPIs stay neutral."""
    cells = []
    for it in items:
        cells.append(
            f"""<div class="kpi"><div class="label">{esc(it['label'])}</div>
            <div class="value{' long' if len(str(it['value'])) > 9 and '<' not in str(it['value']) else ''}">{it['value']}</div><div class="sub">{it.get('sub', '')}</div></div>"""
        )
    st.markdown(f"<div class='kpi-grid'>{''.join(cells)}</div>", unsafe_allow_html=True)


def callout(text: str, kind: str = "info", icon: str | None = None) -> None:
    """``kind`` sets a small status dot; ``icon`` is accepted for compatibility and ignored."""
    st.markdown(f"<div class='callout {kind}'><span class='ci'></span><div>{text}</div></div>", unsafe_allow_html=True)


def causation_note(extra: str = "") -> None:
    callout(
        "<b>Correlation ≠ causation.</b> These results describe statistical associations in observational, country-level data. "
        "They do not show that changing a factor would change reported life evaluations. " + extra,
        "warn",
    )


def card(title: str, body: str, meta: str = "", glow: bool = False) -> None:
    st.markdown(f"<div class='card{' glow' if glow else ''}'><h4>{title}</h4><p>{body}</p>{f'<div class=meta>{meta}</div>' if meta else ''}</div>", unsafe_allow_html=True)


_TONE_DOT = {"positive": "ok", "negative": "bad"}


def insight_card(ins) -> None:
    st.markdown(
        f"""<div class="insight {ins.tone}">
        <div class="metric"><div class="v">{esc(ins.value)}</div><div class="l">{esc(ins.metric)}</div></div>
        <div class="cat"><span class="dot {_TONE_DOT.get(ins.tone, '')}"></span>{esc(ins.category)}</div><h4>{esc(ins.title)}</h4><p>{esc(ins.text)}</p>
        <div class="src">source: {esc(ins.source)}{' · period: ' + esc(ins.period) if ins.period else ''}</div></div>""",
        unsafe_allow_html=True,
    )


def tags(items: Iterable[tuple[str, str]]) -> str:
    return "".join(f"<span class='tag {c}'>{esc(t)}</span>" for t, c in items)


def stat_row(items: list[tuple[str, str]]) -> None:
    st.markdown("<div class='stat-row'>" + "".join(f"<div class='stat'><div class='l'>{esc(l)}</div><div class='v'>{v}</div></div>" for l, v in items) + "</div>", unsafe_allow_html=True)


def chart(fig, key: str | None = None, **kw):
    """Render a Plotly figure keeping our template (theme=None)."""
    return st.plotly_chart(fig, theme=None, key=key, config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}, **kw)


def footer(sources: str = "") -> None:
    st.markdown(
        f"<div class='footer'>World Happiness AI Lab · Data: World Happiness Report (Gallup World Poll, UN SDSN, Wellbeing Research Centre). "
        f"{esc(sources)} Life evaluations are self-reported national averages; all modelled relationships are associations, not causal effects.</div>",
        unsafe_allow_html=True,
    )


def fmt_table(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    out = df.copy()
    for c in out.select_dtypes("number").columns:
        out[c] = out[c].round(decimals)
    return out


# --------------------------------------------------------------------------- cached accessors
@st.cache_resource(show_spinner="Loading World Happiness Report datasets…")
def get_bundle(panel_file: str | None = None, _version: int = 0):
    from .data_loader import load_bundle

    return load_bundle(panel_file=panel_file)


def bundle():
    return get_bundle(st.session_state.get("panel_file"), st.session_state.get("data_version", 0))


@st.cache_data(show_spinner=False)
def get_panel_fe(_bundle_id: str, panel: pd.DataFrame) -> pd.DataFrame:
    from .feature_engineering import engineer_panel

    return engineer_panel(panel)


def panel_fe() -> pd.DataFrame:
    b = bundle()
    return get_panel_fe(b.panel_source + str(len(b.panel)), b.panel)


def results():
    from .model_manager import load_results

    return load_results()


@st.cache_resource(show_spinner=False)
def get_model(name: str, _mtime: float = 0.0):
    from .model_manager import load_model

    return load_model(name)


def model(name: str):
    from .model_manager import model_path

    p = model_path(name)
    return get_model(name, p.stat().st_mtime if p.exists() else 0.0)


def registry() -> dict:
    from .model_manager import read_registry

    return read_registry()


def ensure_trained() -> dict | None:
    """Make sure compatible trained artifacts exist; train automatically otherwise."""
    from .model_manager import artifacts_compatible, load_results

    res = load_results()
    if res and artifacts_compatible():
        return res
    st.markdown(
        "<div class='card glow'><h4>Preparing machine-learning models</h4><p>No compatible trained models were found on this machine "
        "(first launch or different library versions). Training all models now with a compact hyper-parameter search — this runs once "
        "and usually takes 1–3 minutes on a laptop CPU. Results are saved to <code>models/</code> and <code>outputs/reports/</code>.</p></div>",
        unsafe_allow_html=True,
    )
    bar = st.progress(0.0, text="Starting…")
    from .train import run_training

    def cb(f: float, m: str) -> None:
        bar.progress(min(max(f, 0.0), 1.0), text=m)

    res = run_training(n_iter=8, progress=cb, panel=bundle().panel)
    get_model.clear()
    st.cache_data.clear()
    bar.empty()
    st.rerun()
    return res


@st.cache_resource(show_spinner="Clustering countries…")
def get_clustering(_key: str, panel: pd.DataFrame, k: int | None, years: tuple[int, ...]):
    from .clustering import run_clustering

    return run_clustering(panel, k=k, years=list(years))


def clustering(k: int | None = None, years: tuple[int, ...] | None = None):
    b = bundle()
    years = years or tuple(CFG["clustering"]["profile_years"])
    return get_clustering(b.panel_source + str(len(b.panel)), b.panel, k, tuple(years))


@st.cache_data(show_spinner="Computing permutation importance…")
def get_permutation_importance(model_name: str, strategy: str, _mtime: float, n_repeats: int = 10) -> pd.DataFrame:
    from .config import TARGET
    from .explainability import permutation_importance_df
    from .train import build_modeling_frame, make_splits

    pipe = model(model_name)
    feats = registry()[model_name]["features"]
    split = make_splits(build_modeling_frame(bundle().panel), strategy)
    return permutation_importance_df(pipe, split.test[feats], split.test[TARGET], n_repeats=n_repeats)


def permutation_importance(model_name: str, strategy: str = "group") -> pd.DataFrame:
    from .model_manager import model_path

    p = model_path(model_name)
    return get_permutation_importance(model_name, strategy, p.stat().st_mtime if p.exists() else 0.0)


def setup_page(title: str, icon: str = "") -> None:
    """Common per-page setup (CSS + sidebar). Page config is set once in app.py."""
    inject_css()
    sidebar_brand()


__all__ = [n for n in dir() if not n.startswith("_")]
_ = PALETTE
