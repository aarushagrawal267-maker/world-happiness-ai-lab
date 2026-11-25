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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --bg: #070B1A; --surface: #0E1430; --surface-2: #141B3D; --surface-3: #1A2350;
  --border: rgba(140,160,255,0.14); --border-strong: rgba(140,160,255,0.28);
  --text: #E8ECFF; --text-2: #B8C0E8; --muted: #8E98C8;
  --primary: #7C9CFF; --accent: #4FE3C1; --warm: #FFB86B; --danger: #FF6B8B; --violet: #B18CFF;
  --radius: 18px;
  --shadow: 0 10px 40px -12px rgba(0,0,0,0.6), 0 0 0 1px var(--border);
  --font: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
  --display: 'Space Grotesk', 'Inter', 'Segoe UI', sans-serif;
  --mono: 'JetBrains Mono', ui-monospace, Consolas, monospace;
}

html, body, [class*="css"], .stApp { font-family: var(--font); }
.stApp {
  background:
    radial-gradient(1200px 600px at 85% -10%, rgba(124,156,255,0.13), transparent 60%),
    radial-gradient(900px 500px at -10% 20%, rgba(79,227,193,0.07), transparent 60%),
    radial-gradient(800px 600px at 50% 120%, rgba(177,140,255,0.08), transparent 60%),
    var(--bg);
  background-attachment: fixed;
}
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1380px; }
h1, h2, h3, h4 { font-family: var(--display) !important; letter-spacing: -0.02em; color: var(--text); }
h2 { font-size: 1.55rem !important; margin-top: 1.2rem !important; }
h3 { font-size: 1.2rem !important; }
p, li, label, .stMarkdown { color: var(--text-2); }
a { color: var(--primary) !important; }
code { font-family: var(--mono) !important; background: rgba(124,156,255,0.12) !important; color: #C9D4FF !important; border-radius: 6px; }
hr { border-color: var(--border) !important; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0A1026 0%, #070B1A 100%);
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] [data-testid="stSidebarNavItems"] a,
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a { border-radius: 10px; margin: 1px 6px; transition: all .15s ease; }
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover { background: rgba(124,156,255,0.10); }
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {
  background: linear-gradient(90deg, rgba(124,156,255,0.22), rgba(124,156,255,0.04));
  box-shadow: inset 2px 0 0 var(--primary);
}
[data-testid="stSidebarHeader"] img, img[data-testid="stLogo"] { height: 3.2rem !important; max-width: 100%; }
section[data-testid="stSidebar"] [data-testid="stNavSectionHeader"] {
  font-family: var(--display); text-transform: uppercase; letter-spacing: .14em; font-size: .68rem !important; color: var(--muted) !important;
}
.brand { display:flex; align-items:center; gap:12px; padding: 4px 6px 14px 6px; }
.brand-mark { width:42px; height:42px; border-radius:12px; display:grid; place-items:center;
  background: conic-gradient(from 200deg, #7C9CFF, #4FE3C1, #FFB86B, #B18CFF, #7C9CFF);
  box-shadow: 0 0 24px rgba(124,156,255,0.45); }
.brand-mark span { width:34px; height:34px; border-radius:9px; background:#0A1026; display:grid; place-items:center; font-size:19px; }
.brand-name { font-family: var(--display); font-weight:700; font-size:1.02rem; color:var(--text); line-height:1.1; letter-spacing:.02em; }
.brand-sub { font-size:.72rem; color:var(--muted); letter-spacing:.08em; text-transform:uppercase; }
.side-status { margin: 8px 4px; padding: 10px 12px; border-radius: 12px; background: rgba(124,156,255,0.06); border: 1px solid var(--border); font-size: .78rem; color: var(--text-2); }
.side-status b { color: var(--text); }
.dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; vertical-align:middle; }

/* ---------- hero ---------- */
.hero { position: relative; overflow: hidden; border-radius: 26px; padding: 44px 44px 38px 44px; margin-bottom: 26px;
  background: linear-gradient(135deg, rgba(20,27,61,0.95), rgba(10,16,38,0.95)); border: 1px solid var(--border-strong); box-shadow: var(--shadow); }
.hero::before { content:""; position:absolute; inset:-40%; background:
  radial-gradient(closest-side, rgba(124,156,255,0.35), transparent 70%) 70% 30% / 50% 50% no-repeat,
  radial-gradient(closest-side, rgba(79,227,193,0.25), transparent 70%) 90% 80% / 40% 40% no-repeat,
  radial-gradient(closest-side, rgba(177,140,255,0.28), transparent 70%) 40% 90% / 45% 45% no-repeat;
  animation: drift 18s ease-in-out infinite alternate; filter: blur(10px); }
.hero::after { content:""; position:absolute; inset:0; background-image: linear-gradient(rgba(140,160,255,0.06) 1px, transparent 1px), linear-gradient(90deg, rgba(140,160,255,0.06) 1px, transparent 1px);
  background-size: 38px 38px; mask-image: linear-gradient(180deg, rgba(0,0,0,.7), transparent 85%); }
@keyframes drift { 0% { transform: translate(0,0) rotate(0deg);} 100% { transform: translate(-6%,4%) rotate(8deg);} }
.hero > * { position: relative; z-index: 1; }
.eyebrow { display:inline-flex; align-items:center; gap:8px; font-size:.72rem; font-weight:600; letter-spacing:.18em; text-transform:uppercase; color: var(--accent);
  padding: 6px 12px; border-radius: 999px; background: rgba(79,227,193,0.08); border: 1px solid rgba(79,227,193,0.25); }
.hero h1 { font-size: clamp(2.1rem, 4.4vw, 3.6rem) !important; line-height: 1.02; margin: 16px 0 12px 0 !important; font-weight: 700;
  background: linear-gradient(92deg, #FFFFFF 10%, #C9D4FF 45%, #7C9CFF 70%, #4FE3C1 100%); -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero p.lead { font-size: 1.08rem; max-width: 820px; color: var(--text-2); margin: 0; }
.hero .chips { margin-top: 20px; display:flex; gap:8px; flex-wrap: wrap; }
.chip { font-size: .76rem; padding: 6px 12px; border-radius: 999px; background: rgba(255,255,255,0.04); border: 1px solid var(--border); color: var(--text-2); }
.chip b { color: var(--text); }

/* ---------- page header ---------- */
.page-head { margin: 0 0 22px 0; padding-bottom: 18px; border-bottom: 1px solid var(--border); }
.page-head .kicker { font-size: .72rem; letter-spacing: .2em; text-transform: uppercase; color: var(--primary); font-weight: 600; }
.page-head h1 { font-size: clamp(1.8rem, 3vw, 2.5rem) !important; margin: 6px 0 6px 0 !important; font-weight: 700; color: var(--text); }
.page-head p { margin: 0; color: var(--muted); max-width: 900px; font-size: 1rem; }

/* ---------- cards ---------- */
.kpi-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(148px, 1fr)); gap: 14px; margin: 6px 0 18px 0; }
.kpi { position:relative; padding: 18px 18px 16px 18px; border-radius: var(--radius); background: linear-gradient(180deg, rgba(20,27,61,0.9), rgba(14,20,48,0.9));
  border: 1px solid var(--border); box-shadow: var(--shadow); overflow:hidden; transition: transform .2s ease, border-color .2s ease; }
.kpi:hover { transform: translateY(-2px); border-color: var(--border-strong); }
.kpi::before { content:""; position:absolute; left:0; top:0; height:3px; width:100%; background: var(--kpi-accent, linear-gradient(90deg,#7C9CFF,#4FE3C1)); opacity:.9; }
.kpi .label { font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
.kpi .value { font-family: var(--display); font-size: 2.0rem; font-weight: 700; color: var(--text); line-height: 1.15; margin-top: 6px; }
.kpi .value.long { font-size: 1.25rem; padding-top: 6px; }
.kpi .value small { font-size: .9rem; color: var(--muted); font-weight: 500; margin-left: 4px; }
.kpi .sub { font-size: .8rem; color: var(--text-2); margin-top: 4px; }
.kpi .icon { position:absolute; right:14px; top:14px; font-size: 1.1rem; opacity: .55; }

.card { padding: 20px 22px; border-radius: var(--radius); background: linear-gradient(180deg, rgba(20,27,61,0.78), rgba(14,20,48,0.78));
  border: 1px solid var(--border); box-shadow: var(--shadow); margin-bottom: 14px; }
.card h4 { margin: 0 0 6px 0 !important; font-size: 1.02rem !important; }
.card p { margin: 0; font-size: .92rem; }
.card .meta { margin-top: 10px; font-size: .74rem; color: var(--muted); letter-spacing: .02em; }
.card.glow { border-color: rgba(124,156,255,0.35); box-shadow: 0 0 0 1px rgba(124,156,255,0.25), 0 20px 50px -20px rgba(124,156,255,0.45); }

.insight { position: relative; padding: 18px 20px 16px 62px; border-radius: var(--radius); background: linear-gradient(180deg, rgba(20,27,61,0.85), rgba(14,20,48,0.85));
  border: 1px solid var(--border); margin-bottom: 12px; }
.insight .ico { position:absolute; left:16px; top:18px; width:34px; height:34px; border-radius:10px; display:grid; place-items:center; font-size: 1rem;
  background: rgba(124,156,255,0.12); color: var(--primary); border: 1px solid rgba(124,156,255,0.25); }
.insight.positive .ico { background: rgba(79,227,193,0.10); color: var(--accent); border-color: rgba(79,227,193,0.3); }
.insight.negative .ico { background: rgba(255,107,139,0.10); color: var(--danger); border-color: rgba(255,107,139,0.3); }
.insight .cat { font-size: .68rem; letter-spacing: .16em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
.insight h4 { margin: 2px 0 6px 0 !important; font-size: 1.05rem !important; }
.insight p { margin: 0; font-size: .92rem; color: var(--text-2); }
.insight .metric { float: right; text-align: right; margin-left: 16px; }
.insight .metric .v { font-family: var(--display); font-size: 1.5rem; font-weight: 700; color: var(--text); }
.insight .metric .l { font-size: .68rem; color: var(--muted); text-transform: uppercase; letter-spacing: .1em; }
.insight .src { margin-top: 10px; font-size: .72rem; color: var(--muted); font-family: var(--mono); }

.callout { display:flex; gap: 12px; padding: 14px 16px; border-radius: 14px; margin: 10px 0 16px 0; font-size: .9rem; line-height: 1.5;
  background: rgba(124,156,255,0.07); border: 1px solid rgba(124,156,255,0.22); color: var(--text-2); }
.callout .ci { font-size: 1.05rem; line-height: 1.3; }
.callout b { color: var(--text); }
.callout.warn { background: rgba(255,184,107,0.07); border-color: rgba(255,184,107,0.28); }
.callout.danger { background: rgba(255,107,139,0.07); border-color: rgba(255,107,139,0.3); }
.callout.ok { background: rgba(79,227,193,0.06); border-color: rgba(79,227,193,0.25); }

.section { display:flex; align-items: baseline; gap: 12px; margin: 30px 0 10px 0; }
.section .num { font-family: var(--mono); font-size: .78rem; color: var(--primary); padding: 3px 8px; border-radius: 8px; background: rgba(124,156,255,0.1); border: 1px solid rgba(124,156,255,0.22); }
.section h2 { margin: 0 !important; font-size: 1.35rem !important; }
.section .desc { color: var(--muted); font-size: .9rem; }

.stat-row { display:flex; gap: 22px; flex-wrap: wrap; padding: 4px 0 8px 0; }
.stat { min-width: 120px; }
.stat .l { font-size: .7rem; color: var(--muted); text-transform: uppercase; letter-spacing: .12em; }
.stat .v { font-family: var(--display); font-size: 1.35rem; color: var(--text); font-weight: 600; }

.big-number { font-family: var(--display); font-weight: 700; font-size: 4.2rem; line-height: 1; letter-spacing: -0.03em;
  background: linear-gradient(92deg, #FFFFFF, #9ec5f4 60%, #4FE3C1); -webkit-background-clip: text; background-clip: text; color: transparent; }

.tag { display:inline-block; font-size: .72rem; padding: 3px 9px; border-radius: 999px; margin: 2px 4px 2px 0; border: 1px solid var(--border); color: var(--text-2); background: rgba(255,255,255,0.03); }
.tag.blue { border-color: rgba(57,135,229,.45); color: #9ec5f4; }
.tag.green { border-color: rgba(79,227,193,.4); color: #8ff0d8; }
.tag.orange { border-color: rgba(255,184,107,.45); color: #FFCF9A; }
.tag.red { border-color: rgba(255,107,139,.45); color: #FFA3B7; }

.feature-tile { display:block; min-height: 150px; padding: 18px; border-radius: 16px; background: rgba(20,27,61,0.6); border:1px solid var(--border); height: 100%; transition: all .2s ease; }
.feature-tile:hover { border-color: var(--border-strong); background: rgba(26,35,80,0.7); transform: translateY(-2px); }
.feature-tile .t { font-family: var(--display); color: var(--text); font-weight: 600; font-size: 1rem; margin-top: 8px; }
.feature-tile .d { color: var(--muted); font-size: .84rem; margin-top: 4px; }
.feature-tile .e { font-size: 1.4rem; }

.footer { margin-top: 48px; padding-top: 18px; border-top: 1px solid var(--border); font-size: .76rem; color: var(--muted); }

/* ---------- widgets ---------- */
div[data-testid="stMetric"] { background: linear-gradient(180deg, rgba(20,27,61,0.85), rgba(14,20,48,0.85)); border: 1px solid var(--border); border-radius: 16px; padding: 14px 16px; }
div[data-testid="stMetricValue"] { font-family: var(--display); }
div[data-testid="stMetricLabel"] p { color: var(--muted) !important; text-transform: uppercase; letter-spacing: .1em; font-size: .72rem !important; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { height: 42px; border-radius: 10px 10px 0 0; padding: 0 16px; color: var(--muted); font-weight: 500; }
.stTabs [aria-selected="true"] { color: var(--text) !important; background: rgba(124,156,255,0.10); }
.stTabs [data-baseweb="tab-highlight"] { background: var(--primary) !important; }
div[data-testid="stExpander"] { border: 1px solid var(--border) !important; border-radius: 14px !important; background: rgba(14,20,48,0.6); }
div[data-testid="stExpander"] summary p { color: var(--text) !important; font-weight: 500; }
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button { border-radius: 12px; border: 1px solid var(--border-strong); font-weight: 600; transition: all .15s ease; }
.stButton > button[kind="primary"] p, .stFormSubmitButton > button[kind="primary"] p, .stButton > button[kind="primary"] span { color: #07102A !important; font-weight: 700; }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] { background: linear-gradient(92deg, #5B7CFF, #7C9CFF 50%, #4FE3C1); border: none; color: #07102A; box-shadow: 0 8px 24px -8px rgba(124,156,255,0.7); }
.stButton > button:hover { transform: translateY(-1px); border-color: var(--primary); }
div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 14px; overflow: hidden; }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, .stNumberInput input { background: rgba(14,20,48,0.85) !important; border-color: var(--border) !important; border-radius: 10px !important; }
div[data-testid="stSlider"] [role="slider"] { box-shadow: 0 0 0 4px rgba(124,156,255,0.25); }
div[data-testid="stPlotlyChart"] { border-radius: var(--radius); background: linear-gradient(180deg, rgba(20,27,61,0.55), rgba(14,20,48,0.55)); border: 1px solid var(--border); padding: 10px 12px 4px 12px; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: var(--radius); }
div[data-testid="stAlert"] { border-radius: 14px; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
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
        model_txt = f"<span class='dot' style='background:#4FE3C1'></span><b>{esc(results['best_model']['name'])}</b> ready" if results else "<span class='dot' style='background:#FFB86B'></span>models not trained"
        st.markdown(
            f"""<div class="side-status"><span class='dot' style='background:#7C9CFF'></span><b>{len(bundle.panel):,}</b> country-years · <b>{n_c}</b> countries<br/>
            <span class='dot' style='background:#B18CFF'></span>Rankings to <b>WHR {bundle.latest_ranking_year + 1 if bundle.latest_ranking_year else '—'}</b><br/>{model_txt}</div>""",
            unsafe_allow_html=True,
        )


def hero(title: str, lead: str, eyebrow: str = "", chips: Iterable[str] = ()) -> None:
    chips_html = "".join(f"<span class='chip'>{c}</span>" for c in chips)
    st.markdown(
        f"""<div class="hero">{f'<span class="eyebrow">◉ {esc(eyebrow)}</span>' if eyebrow else ''}
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


def kpi_cards(items: list[dict]) -> None:
    """items: {label, value, sub?, icon?, accent?}"""
    cells = []
    for it in items:
        accent = f"--kpi-accent:{it['accent']};" if it.get("accent") else ""
        cells.append(
            f"""<div class="kpi" style="{accent}"><div class="icon">{it.get('icon', '')}</div><div class="label">{esc(it['label'])}</div>
            <div class="value{' long' if len(str(it['value'])) > 9 and '<' not in str(it['value']) else ''}">{it['value']}</div><div class="sub">{it.get('sub', '')}</div></div>"""
        )
    st.markdown(f"<div class='kpi-grid'>{''.join(cells)}</div>", unsafe_allow_html=True)


def callout(text: str, kind: str = "info", icon: str | None = None) -> None:
    icons = {"info": "ℹ", "warn": "⚠", "danger": "⛔", "ok": "✓"}
    st.markdown(f"<div class='callout {kind}'><span class='ci'>{icon or icons.get(kind, 'ℹ')}</span><div>{text}</div></div>", unsafe_allow_html=True)


def causation_note(extra: str = "") -> None:
    callout(
        "<b>Correlation ≠ causation.</b> These results describe statistical associations in observational, country-level data. "
        "They do not show that changing a factor would change reported life evaluations. " + extra,
        "warn", "⚖",
    )


def card(title: str, body: str, meta: str = "", glow: bool = False) -> None:
    st.markdown(f"<div class='card{' glow' if glow else ''}'><h4>{title}</h4><p>{body}</p>{f'<div class=meta>{meta}</div>' if meta else ''}</div>", unsafe_allow_html=True)


def insight_card(ins) -> None:
    st.markdown(
        f"""<div class="insight {ins.tone}"><div class="ico">{esc(ins.icon)}</div>
        <div class="metric"><div class="v">{esc(ins.value)}</div><div class="l">{esc(ins.metric)}</div></div>
        <div class="cat">{esc(ins.category)}</div><h4>{esc(ins.title)}</h4><p>{esc(ins.text)}</p>
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
def get_model(name: str, mtime: float = 0.0):
    # ``mtime`` is part of the cache key (no leading underscore, so Streamlit hashes it):
    # a model file rewritten by `python -m src.train` is reloaded without restarting the app.
    from .model_manager import load_model

    return load_model(name)


def model(name: str):
    from .model_manager import model_path

    p = model_path(name)
    return get_model(name, p.stat().st_mtime if p.exists() else 0.0)


def require_model(name: str):
    """Load a saved model or stop the page with a clear, actionable message."""
    m = model(name)
    if m is None:
        from .model_manager import model_path

        st.error(
            f"The saved model **{name}** could not be loaded from `models/{model_path(name).name}` — the file is missing, "
            "corrupt or was saved with a different scikit-learn version. Retrain it with `python -m src.train` "
            "(or on the ML Training Lab page for custom models), then reload this page."
        )
        st.stop()
    return m


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
        "<div class='card glow'><h4>⚙ Preparing machine-learning models</h4><p>No compatible trained models were found on this machine "
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
def get_permutation_importance(model_name: str, strategy: str, mtime: float, n_repeats: int = 10) -> pd.DataFrame:
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


def setup_page(title: str, icon: str = "◐") -> None:
    """Common per-page setup (CSS + sidebar). Page config is set once in app.py."""
    inject_css()
    sidebar_brand()


__all__ = [n for n in dir() if not n.startswith("_")]
_ = PALETTE
