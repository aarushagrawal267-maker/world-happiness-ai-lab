"""Plotly chart builders with a single, consistent visual language.

Design rules (see README → Visual design):
* categorical colours in fixed order, never cycled — at most 8 series, extra
  categories fold into a grey "Other";
* sequential = one hue (blue), diverging = red ↔ neutral grey ↔ blue;
* one y-axis per chart; thin marks, 2px lines, ≥ 8px markers, rounded bar ends;
* text uses text colours, never the series colour; recessive grids; every
  chart has hover tooltips.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from .config import (
    DIVERGING_SCALE,
    LABELS,
    OTHER_COLOR,
    PALETTE,
    SEQUENTIAL_SCALE,
    SERIES_COLORS,
    SHORT_LABELS,
)

FONT = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"
TEXT = PALETTE.text
MUTED = PALETTE.muted
GRID = "rgba(160,175,255,0.07)"
AXIS = "rgba(160,175,255,0.18)"
SURFACE = "#0E1430"

pio.templates["whr"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family=FONT, color=TEXT, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=SERIES_COLORS,
        title=dict(font=dict(size=16, color=TEXT), x=0, xanchor="left", y=0.98, yanchor="top", pad=dict(b=8)),
        xaxis=dict(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS, tickcolor=AXIS, title_font=dict(color=MUTED, size=12), tickfont=dict(color=MUTED, size=11), automargin=True),
        yaxis=dict(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS, tickcolor=AXIS, title_font=dict(color=MUTED, size=12), tickfont=dict(color=MUTED, size=11), automargin=True),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=12), orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
        hoverlabel=dict(bgcolor="#151C42", bordercolor="rgba(140,160,255,0.35)", font=dict(family=FONT, color=TEXT, size=12)),
        margin=dict(l=10, r=10, t=56, b=10),
        coloraxis=dict(colorbar=dict(outlinewidth=0, tickfont=dict(color=MUTED), title_font=dict(color=MUTED))),
        bargap=0.25,
    )
)
pio.templates.default = "whr"


def lab(f: str) -> str:
    return LABELS.get(f, f)


def short(f: str) -> str:
    return SHORT_LABELS.get(f, f)


def color_map(categories: list, highlight: list | None = None) -> dict:
    """Fixed-order colours. Only the first 8 (or the ``highlight`` list) get hues;
    everything else is grey 'Other'."""
    cats = [c for c in categories]
    chosen = list(highlight) if highlight is not None else cats[:8]
    chosen = chosen[:8]
    return {c: (SERIES_COLORS[chosen.index(c)] if c in chosen else OTHER_COLOR) for c in cats}


def _finish(fig: go.Figure, title: str | None = None, height: int = 420, legend: bool = True) -> go.Figure:
    fig.update_layout(height=height, showlegend=legend)
    if title:
        fig.update_layout(title_text=title)
    return fig


def empty_figure(message: str, height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color=MUTED, size=14), x=0.5, y=0.5, xref="paper", yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _finish(fig, height=height, legend=False)


# --------------------------------------------------------------------------- distributions
def histogram(s: pd.Series, title: str = "", nbins: int = 40, xlabel: str = "", color: str = SERIES_COLORS[0]) -> go.Figure:
    s = s.dropna()
    fig = go.Figure(go.Histogram(x=s, nbinsx=nbins, marker=dict(color=color, line=dict(color=SURFACE, width=1)), opacity=0.95,
                                 hovertemplate="%{x}<br>%{y} observations<extra></extra>"))
    fig.add_vline(x=s.mean(), line=dict(color=TEXT, width=1.5, dash="dot"), annotation_text=f"mean {s.mean():.2f}",
                  annotation_font_color=MUTED, annotation_position="top right")
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(title="Count")
    return _finish(fig, title, legend=False)


def density(df: pd.DataFrame, value: str, group: str | None = None, title: str = "", groups: list | None = None) -> go.Figure:
    from scipy.stats import gaussian_kde

    fig = go.Figure()
    d = df.dropna(subset=[value])
    xs = np.linspace(d[value].min() - 0.3, d[value].max() + 0.3, 300)
    if group:
        cats = groups or list(d[group].dropna().unique())[:8]
        cmap = color_map(cats)
        for c in cats:
            v = d.loc[d[group] == c, value]
            if len(v) < 5:
                continue
            ys = gaussian_kde(v)(xs)
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=str(c), line=dict(color=cmap[c], width=2),
                                     hovertemplate=f"{c}<br>%{{x:.2f}}<extra></extra>"))
    else:
        ys = gaussian_kde(d[value])(xs)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="tozeroy", line=dict(color=SERIES_COLORS[0], width=2),
                                 fillcolor="rgba(57,135,229,0.18)", name="density", hovertemplate="%{x:.2f}<extra></extra>"))
    fig.update_xaxes(title=lab(value))
    fig.update_yaxes(title="Density")
    return _finish(fig, title, legend=bool(group))


def box_by(df: pd.DataFrame, value: str, group: str, title: str = "", order: list | None = None, single_color: bool = True) -> go.Figure:
    d = df.dropna(subset=[value, group])
    order = order or d.groupby(group, observed=True)[value].median().sort_values(ascending=False).index.tolist()
    fig = go.Figure()
    for i, g in enumerate(order):
        v = d.loc[d[group] == g, value]
        col = SERIES_COLORS[0] if single_color else color_map(order)[g]
        fig.add_trace(go.Box(y=v, name=str(g), marker=dict(color=col, size=4), line=dict(width=1.5), boxpoints="outliers",
                             fillcolor="rgba(57,135,229,0.22)" if single_color else None,
                             hovertemplate=f"{g}<br>%{{y:.2f}}<extra></extra>"))
    fig.update_yaxes(title=lab(value))
    fig.update_xaxes(tickangle=-30)
    return _finish(fig, title, height=460, legend=False)


def year_violin(df: pd.DataFrame, value: str, title: str = "") -> go.Figure:
    d = df.dropna(subset=[value])
    fig = go.Figure()
    for y in sorted(d["year"].unique()):
        v = d.loc[d["year"] == y, value]
        fig.add_trace(go.Violin(x=[str(y)] * len(v), y=v, name=str(y), line=dict(color=SERIES_COLORS[0], width=1),
                                fillcolor="rgba(57,135,229,0.25)", meanline_visible=True, points=False, spanmode="hard",
                                hoverinfo="y+name"))
    fig.update_yaxes(title=lab(value))
    fig.update_xaxes(title="Year")
    return _finish(fig, title, height=420, legend=False)


# --------------------------------------------------------------------------- rankings & bars
def ranking_bars(df: pd.DataFrame, value: str, label: str = "country", title: str = "", ci: tuple[str, str] | None = None,
                 color: str = SERIES_COLORS[0], height: int | None = None, fmt: str = ".3f") -> go.Figure:
    d = df.sort_values(value)
    err = None
    if ci and set(ci) <= set(d.columns):
        err = dict(type="data", symmetric=False, array=d[ci[1]] - d[value], arrayminus=d[value] - d[ci[0]], color=MUTED, thickness=1.2, width=3)
    fig = go.Figure(go.Bar(
        x=d[value], y=d[label], orientation="h", marker=dict(color=color, cornerradius=4), error_x=err,
        text=d[value].map(lambda v: format(v, fmt)), textposition="outside", textfont=dict(color=MUTED, size=11), cliponaxis=False,
        hovertemplate="%{y}<br>" + short(value) + ": %{x:" + fmt + "}<extra></extra>",
    ))
    fig.update_xaxes(title=lab(value), range=[0, max(d[value].max() * 1.2, 1)])
    fig.update_yaxes(title="")
    return _finish(fig, title, height=height or max(320, 28 * len(d) + 80), legend=False)


def region_bars(summary: pd.DataFrame, value: str = "mean", title: str = "") -> go.Figure:
    d = summary.sort_values(value)
    fig = go.Figure(go.Bar(
        x=d[value], y=d["region"], orientation="h", marker=dict(color=SERIES_COLORS[0], cornerradius=4),
        error_x=dict(type="data", array=d["std"], color=MUTED, thickness=1.2, width=3) if "std" in d else None,
        text=d[value].map("{:.2f}".format), textposition="inside", insidetextanchor="end", textfont=dict(color="#FFFFFF", size=11),
        customdata=np.stack([d["count"]], axis=-1) if "count" in d else None,
        hovertemplate="%{y}<br>mean %{x:.2f}<br>n = %{customdata[0]}<extra></extra>" if "count" in d else "%{y}<br>%{x:.2f}<extra></extra>",
    ))
    fig.update_xaxes(title="Mean score (± 1 SD)")
    return _finish(fig, title, height=420, legend=False)


# --------------------------------------------------------------------------- relationships
def scatter_trend(df: pd.DataFrame, x: str, y: str, color: str | None = "region", highlight: list | None = None, title: str = "",
                  hover: str = "country", trend: bool = True, lowess: bool = False) -> go.Figure:
    d = df.dropna(subset=[x, y]).copy()
    fig = go.Figure()
    if color and color in d:
        cats = sorted(d[color].dropna().unique().tolist())
        cmap = color_map(cats, highlight)
        # draw "Other" first so highlighted series sit on top
        ordered = sorted(cats, key=lambda c: cmap[c] != OTHER_COLOR)
        other_drawn = False
        for c in ordered:
            sub = d[d[color] == c]
            is_other = cmap[c] == OTHER_COLOR
            fig.add_trace(go.Scattergl(
                x=sub[x], y=sub[y], mode="markers", name="Other regions" if is_other else str(c),
                legendgroup="other" if is_other else str(c), showlegend=not (is_other and other_drawn),
                marker=dict(color=cmap[c], size=8, opacity=0.35 if is_other else 0.85, line=dict(color=SURFACE, width=1)),
                text=sub[hover] + (" " + sub["year"].astype(str) if "year" in sub else ""),
                hovertemplate="<b>%{text}</b><br>" + short(x) + ": %{x:.3f}<br>" + short(y) + ": %{y:.3f}<extra>" + str(c) + "</extra>",
            ))
            other_drawn = other_drawn or is_other
    else:
        fig.add_trace(go.Scattergl(x=d[x], y=d[y], mode="markers", marker=dict(color=SERIES_COLORS[0], size=8, opacity=0.7, line=dict(color=SURFACE, width=1)),
                                   text=d[hover] if hover in d else None, name="observations",
                                   hovertemplate="<b>%{text}</b><br>%{x:.3f}, %{y:.3f}<extra></extra>"))
    if trend and len(d) > 3:
        slope, intercept = np.polyfit(d[x], d[y], 1)
        xs = np.linspace(d[x].min(), d[x].max(), 50)
        r = np.corrcoef(d[x], d[y])[0, 1]
        fig.add_trace(go.Scatter(x=xs, y=intercept + slope * xs, mode="lines", name=f"OLS fit (r = {r:.2f})",
                                 line=dict(color=TEXT, width=2, dash="dash"), hoverinfo="skip"))
    if lowess and len(d) > 20:
        from statsmodels.nonparametric.smoothers_lowess import lowess as lw

        sm = lw(d[y], d[x], frac=0.4)
        fig.add_trace(go.Scatter(x=sm[:, 0], y=sm[:, 1], mode="lines", name="LOWESS (non-linear)", line=dict(color=SERIES_COLORS[3], width=2), hoverinfo="skip"))
    fig.update_xaxes(title=lab(x))
    fig.update_yaxes(title=lab(y))
    return _finish(fig, title, height=460)


def heatmap(matrix: pd.DataFrame, title: str = "", zmin: float = -1, zmax: float = 1, fmt: str = ".2f", diverging: bool = True, pvals: pd.DataFrame | None = None) -> go.Figure:
    labels_x = [short(c) for c in matrix.columns]
    labels_y = [short(c) for c in matrix.index]
    text = matrix.map(lambda v: format(v, fmt) if pd.notna(v) else "")
    if pvals is not None:
        text = text + pvals.map(lambda p: "" if p < 0.05 else "ⁿˢ")
    fig = go.Figure(go.Heatmap(
        z=matrix.to_numpy(), x=labels_x, y=labels_y, zmin=zmin, zmax=zmax,
        colorscale=DIVERGING_SCALE if diverging else SEQUENTIAL_SCALE, text=text.to_numpy(), texttemplate="%{text}",
        textfont=dict(size=11, color=TEXT), xgap=2, ygap=2, hovertemplate="%{y} × %{x}<br>%{z:.3f}<extra></extra>",
        colorbar=dict(thickness=10, len=0.8),
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, tickangle=-35)
    return _finish(fig, title, height=520, legend=False)


def missing_heatmap(df: pd.DataFrame, columns: list[str], by: str = "year", title: str = "") -> go.Figure:
    m = df.groupby(by)[columns].apply(lambda g: g.isna().mean() * 100)
    fig = go.Figure(go.Heatmap(
        z=m.T.to_numpy(), x=m.index.astype(str), y=[short(c) for c in columns], colorscale=SEQUENTIAL_SCALE[::-1] if False else SEQUENTIAL_SCALE,
        zmin=0, zmax=max(5, float(np.nanmax(m.to_numpy()))), xgap=2, ygap=2,
        hovertemplate="%{y} · %{x}<br>%{z:.1f}% missing<extra></extra>", colorbar=dict(title="% missing", thickness=10),
    ))
    fig.update_xaxes(showgrid=False, title=by.title())
    fig.update_yaxes(showgrid=False)
    return _finish(fig, title, height=380, legend=False)


def missing_matrix(df: pd.DataFrame, columns: list[str], title: str = "") -> go.Figure:
    """Row-level missingness pattern (sorted by country/year)."""
    d = df.sort_values(["country", "year"])[columns].isna().astype(int)
    fig = go.Figure(go.Heatmap(z=d.to_numpy().T, y=[short(c) for c in columns], colorscale=[[0, "#14204A"], [1, "#e66767"]], showscale=False,
                               hovertemplate="row %{x}<br>%{y}: %{z}<extra>1 = missing</extra>"))
    fig.update_xaxes(showgrid=False, title="Country-year rows (sorted by country, year)", showticklabels=False)
    fig.update_yaxes(showgrid=False)
    return _finish(fig, title, height=300, legend=False)


# --------------------------------------------------------------------------- maps
def choropleth(df: pd.DataFrame, value: str, title: str = "", hover_cols: list[str] | None = None, diverging: bool = False,
               categorical: bool = False, range_color: tuple | None = None, height: int = 560, reverse: bool = False) -> go.Figure:
    d = df.dropna(subset=["iso3"]).copy()
    hover_cols = hover_cols or []
    custom = d[["country"] + hover_cols].to_numpy() if hover_cols else d[["country"]].to_numpy()
    hover = "<b>%{customdata[0]}</b><br>" + short(value) + ": %{z:.3f}"
    for i, h in enumerate(hover_cols, start=1):
        hover += f"<br>{short(h)}: %{{customdata[{i}]}}"
    if categorical:
        cats = sorted(d[value].dropna().unique().tolist(), key=str)
        cmap = color_map(cats)
        fig = go.Figure()
        for c in cats:
            sub = d[d[value] == c]
            fig.add_trace(go.Choropleth(
                locations=sub["iso3"], z=[1] * len(sub), colorscale=[[0, cmap[c]], [1, cmap[c]]], showscale=False, name=str(c),
                customdata=sub[["country"] + hover_cols].to_numpy(), marker_line_color=SURFACE, marker_line_width=0.5,
                hovertemplate="<b>%{customdata[0]}</b><br>" + str(c) + "<extra></extra>", showlegend=True,
            ))
    else:
        scale = DIVERGING_SCALE if diverging else SEQUENTIAL_SCALE
        if reverse:
            scale = [[1 - s, c] for s, c in scale][::-1]
        zmin, zmax = range_color if range_color else (d[value].min(), d[value].max())
        if diverging and not range_color:
            m = np.nanmax(np.abs(d[value]))
            zmin, zmax = -m, m
        fig = go.Figure(go.Choropleth(
            locations=d["iso3"], z=d[value], customdata=custom, colorscale=scale, zmin=zmin, zmax=zmax,
            marker_line_color=SURFACE, marker_line_width=0.5, hovertemplate=hover + "<extra></extra>",
            colorbar=dict(title=dict(text=short(value), side="right"), thickness=10, len=0.75, x=1.0),
        ))
    fig.update_geos(
        projection_type="natural earth", showframe=False, showcoastlines=False, showcountries=True, countrycolor="rgba(160,175,255,0.12)",
        showland=True, landcolor="#1A2147", showocean=True, oceancolor="rgba(0,0,0,0)", bgcolor="rgba(0,0,0,0)", showlakes=False,
        lataxis_range=[-56, 84], lonaxis_range=[-170, 190], projection_scale=1.0,
    )
    fig.update_layout(geo=dict(domain=dict(x=[0, 0.96], y=[0, 1])))
    fig.update_layout(margin=dict(l=0, r=0, t=40 if title else 0, b=0), legend=dict(orientation="h", y=-0.02, x=0.0, yanchor="top"))
    return _finish(fig, title, height=height, legend=categorical)


# --------------------------------------------------------------------------- trends & comparison
def trend_lines(df: pd.DataFrame, value: str, group: str = "country", title: str = "", band: tuple[str, str] | None = None,
                order: list | None = None) -> go.Figure:
    d = df.dropna(subset=[value])
    cats = order or list(dict.fromkeys(d[group]))
    cmap = color_map(cats)
    fig = go.Figure()
    for c in cats:
        sub = d[d[group] == c].sort_values("year")
        if band and set(band) <= set(sub.columns):
            fig.add_trace(go.Scatter(x=pd.concat([sub["year"], sub["year"][::-1]]), y=pd.concat([sub[band[1]], sub[band[0]][::-1]]),
                                     fill="toself", fillcolor=_rgba(cmap[c], 0.14), line=dict(width=0), hoverinfo="skip", showlegend=False, legendgroup=str(c)))
        fig.add_trace(go.Scatter(x=sub["year"], y=sub[value], mode="lines+markers", name=str(c), legendgroup=str(c),
                                 line=dict(color=cmap[c], width=2), marker=dict(size=8, line=dict(color=SURFACE, width=2)),
                                 hovertemplate=f"<b>{c}</b> %{{x}}<br>{short(value)}: %{{y:.3f}}<extra></extra>"))
    # selective direct labels on the last point (≤ 4 series)
    if len(cats) <= 4:
        for c in cats:
            sub = d[d[group] == c].sort_values("year")
            if len(sub):
                fig.add_annotation(x=sub["year"].iloc[-1], y=sub[value].iloc[-1], text=str(c), showarrow=False, xanchor="left", xshift=8,
                                   font=dict(color=MUTED, size=11))
    fig.update_xaxes(title="Year", dtick=2)
    fig.update_yaxes(title=lab(value))
    fig.update_layout(hovermode="x unified")
    return _finish(fig, title, height=440)


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


def radar(profile: pd.DataFrame, features: list[str], axis_labels: list[str], title: str = "") -> go.Figure:
    cats = profile["country"].tolist()
    cmap = color_map(cats)
    fig = go.Figure()
    for _, row in profile.iterrows():
        vals = [row[f] for f in features]
        fig.add_trace(go.Scatterpolar(
            r=vals + vals[:1], theta=axis_labels + axis_labels[:1], name=row["country"], fill="toself",
            fillcolor=_rgba(cmap[row["country"]], 0.12), line=dict(color=cmap[row["country"]], width=2), marker=dict(size=7),
            hovertemplate="<b>" + row["country"] + "</b><br>%{theta}: %{r:.0f}th percentile<extra></extra>", connectgaps=False,
        ))
    fig.update_layout(polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(range=[0, 100], gridcolor=GRID, linecolor=AXIS, tickfont=dict(color=MUTED, size=10), ticksuffix="", angle=90, tickangle=90),
        angularaxis=dict(gridcolor=GRID, linecolor=AXIS, tickfont=dict(color=TEXT, size=11), rotation=90, direction="clockwise"),
        domain=dict(x=[0.16, 0.84], y=[0.08, 0.9]),
    ), margin=dict(l=40, r=40, t=60, b=60), legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center", yanchor="top"))
    return _finish(fig, title, height=540)


def grouped_bars(df: pd.DataFrame, x: str, y: str, group: str, title: str = "", error: str | None = None, order: list | None = None,
                 group_order: list | None = None, ytitle: str = "", fmt: str = ".3f", horizontal: bool = False) -> go.Figure:
    gorder = group_order or list(dict.fromkeys(df[group]))
    cmap = color_map(gorder)
    fig = go.Figure()
    for g in gorder:
        sub = df[df[group] == g]
        if order:
            sub = sub.set_index(x).reindex(order).reset_index()
        kw = dict(name=str(g), marker=dict(color=cmap[g], cornerradius=4, line=dict(color=SURFACE, width=2)),
                  hovertemplate=f"<b>{g}</b><br>%{{{'y' if horizontal else 'x'}}}: %{{{'x' if horizontal else 'y'}:{fmt}}}<extra></extra>")
        err = dict(type="data", array=sub[error], color=MUTED, thickness=1.2, width=3) if error else None
        if horizontal:
            fig.add_trace(go.Bar(y=sub[x], x=sub[y], orientation="h", error_x=err, **kw))
        else:
            fig.add_trace(go.Bar(x=sub[x], y=sub[y], error_y=err, **kw))
    fig.update_layout(barmode="group", bargroupgap=0.08)
    if horizontal:
        fig.update_xaxes(title=ytitle)
    else:
        fig.update_yaxes(title=ytitle)
    return _finish(fig, title, height=440)


# --------------------------------------------------------------------------- model evaluation
def actual_vs_predicted(res: pd.DataFrame, title: str = "", color: str | None = None) -> go.Figure:
    lo, hi = float(min(res["actual"].min(), res["predicted"].min())) - 0.2, float(max(res["actual"].max(), res["predicted"].max())) + 0.2
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="perfect prediction", line=dict(color=MUTED, dash="dash", width=1.5), hoverinfo="skip"))
    fig.add_trace(go.Scattergl(
        x=res["actual"], y=res["predicted"], mode="markers", name="test observations",
        marker=dict(color=SERIES_COLORS[0], size=8, opacity=0.75, line=dict(color=SURFACE, width=1)),
        text=res["country"] + " " + res["year"].astype(str) if "year" in res else res.get("country"),
        hovertemplate="<b>%{text}</b><br>actual %{x:.2f}<br>predicted %{y:.2f}<extra></extra>",
    ))
    fig.update_xaxes(title="Actual Life Ladder", range=[lo, hi])
    fig.update_yaxes(title="Predicted Life Ladder", range=[lo, hi], scaleanchor="x", scaleratio=1)
    return _finish(fig, title, height=460)


def residual_hist(res: pd.DataFrame, title: str = "") -> go.Figure:
    fig = histogram(res["residual"], title, nbins=35, xlabel="Residual (actual − predicted)", color=SERIES_COLORS[2])
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    return fig


def residual_vs_predicted(res: pd.DataFrame, title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dash"))
    fig.add_trace(go.Scattergl(x=res["predicted"], y=res["residual"], mode="markers", marker=dict(color=SERIES_COLORS[2], size=8, opacity=0.7, line=dict(color=SURFACE, width=1)),
                               text=res["country"] + " " + res["year"].astype(str), hovertemplate="<b>%{text}</b><br>pred %{x:.2f}<br>resid %{y:+.2f}<extra></extra>", name="residuals"))
    if len(res) > 20:
        from statsmodels.nonparametric.smoothers_lowess import lowess as lw

        sm = lw(res["residual"], res["predicted"], frac=0.5)
        fig.add_trace(go.Scatter(x=sm[:, 0], y=sm[:, 1], mode="lines", line=dict(color=SERIES_COLORS[3], width=2), name="LOWESS trend", hoverinfo="skip"))
    fig.update_xaxes(title="Predicted Life Ladder")
    fig.update_yaxes(title="Residual")
    return _finish(fig, title, height=420)


def error_bars(tbl: pd.DataFrame, group: str, title: str = "", metric: str = "mae") -> go.Figure:
    d = tbl.copy()
    d[group] = d[group].astype(str)
    fig = go.Figure(go.Bar(
        x=d[metric], y=d[group], orientation="h", marker=dict(color=SERIES_COLORS[1], cornerradius=4),
        text=[f"{m:.2f}  (n={n})" for m, n in zip(d[metric], d["n"])], textposition="outside", textfont=dict(color=MUTED, size=11), cliponaxis=False,
        customdata=np.stack([d["n"], d["bias"]], axis=-1),
        hovertemplate="%{y}<br>" + metric.upper() + " %{x:.3f}<br>n = %{customdata[0]}<br>bias %{customdata[1]:+.3f}<extra></extra>",
    ))
    fig.update_xaxes(title=metric.upper() + " (ladder points)", range=[0, d[metric].max() * 1.35])
    return _finish(fig, title, height=max(300, 34 * len(d) + 90), legend=False)


# --------------------------------------------------------------------------- explainability
def importance_bars(df: pd.DataFrame, value: str, error: str | None = None, label: str = "feature", title: str = "", xtitle: str = "", color: str = SERIES_COLORS[0]) -> go.Figure:
    d = df.sort_values(value).copy()
    d["label"] = d[label].map(lambda f: short(f) if f in SHORT_LABELS else (f.replace("region_", "Region: ") if isinstance(f, str) else f))
    fig = go.Figure(go.Bar(
        x=d[value], y=d["label"], orientation="h", marker=dict(color=color, cornerradius=4),
        error_x=dict(type="data", array=d[error], color=MUTED, thickness=1.2, width=3) if error else None,
        hovertemplate="%{y}<br>%{x:.4f}<extra></extra>",
    ))
    fig.update_xaxes(title=xtitle)
    return _finish(fig, title, height=max(300, 34 * len(d) + 90), legend=False)


def shap_beeswarm(expl: dict, title: str = "", max_display: int = 12) -> go.Figure:
    vals, data, names = expl["values"], expl["data"], expl["feature_names"]
    order = np.argsort(np.abs(vals).mean(axis=0))[::-1][:max_display][::-1]
    fig = go.Figure()
    rng = np.random.default_rng(0)
    for rank, j in enumerate(order):
        v, x = vals[:, j], data[:, j]
        xn = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-12)
        jitter = rng.uniform(-0.28, 0.28, len(v))
        fig.add_trace(go.Scattergl(
            x=v, y=rank + jitter, mode="markers", showlegend=False,
            marker=dict(size=6, color=xn, colorscale=DIVERGING_SCALE, cmin=0, cmax=1, opacity=0.85,
                        colorbar=dict(title="feature value", tickvals=[0, 1], ticktext=["low", "high"], thickness=10, len=0.6) if rank == len(order) - 1 else None,
                        showscale=rank == len(order) - 1),
            customdata=x, hovertemplate=short(names[j]) + "<br>value %{customdata:.3f}<br>SHAP %{x:+.3f}<extra></extra>",
        ))
    fig.update_yaxes(tickvals=list(range(len(order))), ticktext=[short(names[j]).replace("region_", "Region: ") for j in order], showgrid=False)
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    fig.update_xaxes(title="SHAP value (impact on predicted ladder, points)")
    return _finish(fig, title, height=max(360, 40 * len(order) + 100), legend=False)


def waterfall(contrib: pd.DataFrame, base: float, prediction: float, title: str = "") -> go.Figure:
    d = contrib.reindex(contrib["contribution"].abs().sort_values(ascending=False).index)
    labels = [f"{short(f).replace('region_', 'Region: ')} = {v:.3g}" if isinstance(v, (int, float, np.floating)) else short(f) for f, v in zip(d["feature"], d["value"])]
    fig = go.Figure(go.Waterfall(
        orientation="h", measure=["absolute"] + ["relative"] * len(d) + ["total"],
        y=["Average prediction"] + labels + ["Prediction"], x=[base] + d["contribution"].tolist() + [prediction],
        increasing=dict(marker=dict(color=SERIES_COLORS[0])), decreasing=dict(marker=dict(color=SERIES_COLORS[7])),
        totals=dict(marker=dict(color="#6B7194")), connector=dict(line=dict(color=AXIS, width=1)),
        text=[f"{base:.2f}"] + [f"{c:+.2f}" for c in d["contribution"]] + [f"{prediction:.2f}"], textposition="outside",
        textfont=dict(color=MUTED), hovertemplate="%{y}<br>%{x:.3f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Life Ladder (model scale)")
    return _finish(fig, title, height=max(360, 40 * (len(d) + 2) + 80), legend=False)


def pdp_plot(grid, avg, feature: str, individual=None, title: str = "", rug: pd.Series | None = None) -> go.Figure:
    fig = go.Figure()
    if individual is not None:
        for i, row in enumerate(individual[:60]):
            fig.add_trace(go.Scatter(x=grid, y=row, mode="lines", line=dict(color="rgba(142,152,200,0.18)", width=1), hoverinfo="skip", showlegend=i == 0, name="ICE (individual rows)"))
    fig.add_trace(go.Scatter(x=grid, y=avg, mode="lines", line=dict(color=SERIES_COLORS[0], width=3), name="Partial dependence (average)",
                             hovertemplate=short(feature) + " %{x:.3f}<br>avg prediction %{y:.3f}<extra></extra>"))
    if rug is not None:
        q = rug.dropna().quantile(np.linspace(0.05, 0.95, 19))
        fig.add_trace(go.Scatter(x=q, y=[min(np.min(avg), np.min(individual) if individual is not None else np.min(avg))] * len(q), mode="markers",
                                 marker=dict(symbol="line-ns-open", size=12, color=MUTED), name="data deciles", hoverinfo="skip"))
    fig.update_xaxes(title=lab(feature))
    fig.update_yaxes(title="Predicted Life Ladder")
    return _finish(fig, title, height=400)


# --------------------------------------------------------------------------- clustering
def cluster_scores(scores: pd.DataFrame, chosen_k: int) -> go.Figure:
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Elbow: within-cluster SS (↓)", "Silhouette (↑)", "Davies–Bouldin (↓)"), horizontal_spacing=0.08)
    for i, (col, c) in enumerate((("wcss", SERIES_COLORS[0]), ("silhouette", SERIES_COLORS[2]), ("davies_bouldin", SERIES_COLORS[1])), start=1):
        fig.add_trace(go.Scatter(x=scores["k"], y=scores[col], mode="lines+markers", line=dict(color=c, width=2), marker=dict(size=9, line=dict(color=SURFACE, width=2)),
                                 hovertemplate="k=%{x}<br>%{y:.3f}<extra></extra>", showlegend=False), row=1, col=i)
        fig.add_vline(x=chosen_k, line=dict(color=MUTED, dash="dot", width=1), row=1, col=i)
    fig.update_annotations(font=dict(color=MUTED, size=12))
    fig.update_xaxes(dtick=1, title="k")
    return _finish(fig, None, height=330, legend=False)


def pca_scatter(profiles: pd.DataFrame, variance: list[float], title: str = "", label_top: int = 12) -> go.Figure:
    cats = sorted(profiles["cluster"].unique())
    names = profiles.drop_duplicates("cluster").set_index("cluster")["cluster_name"].to_dict()
    cmap = color_map(cats)
    fig = go.Figure()
    for c in cats:
        sub = profiles[profiles["cluster"] == c]
        fig.add_trace(go.Scatter(
            x=sub["pc1"], y=sub["pc2"], mode="markers", name=f"C{c + 1} · {names[c].split(' · ')[0]}",
            marker=dict(color=cmap[c], size=10, line=dict(color=SURFACE, width=2), opacity=0.9),
            text=sub["country"], customdata=sub[["life_ladder"]].to_numpy(),
            hovertemplate="<b>%{text}</b><br>PC1 %{x:.2f} · PC2 %{y:.2f}<br>Life Ladder %{customdata[0]:.2f}<extra>" + f"C{c + 1}: {names[c]}" + "</extra>",
        ))
    # selective labels: extremes on PC1/PC2
    lab_rows = pd.concat([profiles.nlargest(label_top // 4, "pc1"), profiles.nsmallest(label_top // 4, "pc1"),
                          profiles.nlargest(label_top // 4, "pc2"), profiles.nsmallest(label_top // 4, "pc2")]).drop_duplicates("country")
    for _, r in lab_rows.iterrows():
        fig.add_annotation(x=r["pc1"], y=r["pc2"], text=r["country"], showarrow=False, yshift=11, font=dict(color=MUTED, size=10))
    fig.update_xaxes(title=f"PC1 ({variance[0]:.0%} of variance)")
    fig.update_yaxes(title=f"PC2 ({variance[1]:.0%} of variance)")
    fig.update_layout(legend=dict(orientation="h", y=-0.14, x=0, xanchor="left", yanchor="top"))
    return _finish(fig, title, height=560)


def cluster_profile_heatmap(z: pd.DataFrame, names: dict, title: str = "") -> go.Figure:
    m = z.copy()
    m.index = [f"C{i + 1}" for i in m.index]
    m.columns = [short(c) for c in m.columns]
    lim = float(np.nanmax(np.abs(m.to_numpy())))
    fig = go.Figure(go.Heatmap(z=m.to_numpy(), x=m.columns, y=m.index, colorscale=DIVERGING_SCALE, zmin=-lim, zmax=lim,
                               text=m.map(lambda v: f"{v:+.2f}").to_numpy(), texttemplate="%{text}", textfont=dict(color=TEXT, size=12), xgap=2, ygap=2,
                               hovertemplate="%{y} · %{x}<br>%{z:+.2f} SD from global mean<extra></extra>", colorbar=dict(title="z", thickness=10)))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False)
    return _finish(fig, title, height=120 + 60 * len(m), legend=False)


def dendrogram(Z: np.ndarray, labels: list[str], k: int, title: str = "") -> go.Figure:
    from scipy.cluster.hierarchy import dendrogram as scipy_dendro

    cut = Z[-(k - 1), 2] if k > 1 else Z[-1, 2]
    dd = scipy_dendro(Z, labels=labels, no_plot=True, color_threshold=cut, above_threshold_color="#6B7194")
    fig = go.Figure()
    uniq = []
    for c in dd["color_list"]:
        if c not in uniq and c != "#6B7194":
            uniq.append(c)
    cmap = {c: SERIES_COLORS[i % len(SERIES_COLORS)] for i, c in enumerate(uniq)}
    cmap["#6B7194"] = "#6B7194"
    for xs, ys, c in zip(dd["icoord"], dd["dcoord"], dd["color_list"]):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=cmap.get(c, "#6B7194"), width=1.5), hoverinfo="skip", showlegend=False))
    fig.add_hline(y=cut, line=dict(color=MUTED, dash="dot", width=1), annotation_text=f"cut for k = {k}", annotation_font_color=MUTED)
    ticks = [5 + 10 * i for i in range(len(dd["ivl"]))]
    fig.update_xaxes(tickvals=ticks, ticktext=dd["ivl"], tickangle=-90, tickfont=dict(size=8), showgrid=False)
    fig.update_yaxes(title="Ward linkage distance", showgrid=True)
    return _finish(fig, title, height=520, legend=False)


# --------------------------------------------------------------------------- prediction / scenarios
def _z(v):
    return 0.0 if v is not None and abs(v) < 0.005 else v


def gauge(value: float, lower: float | None = None, upper: float | None = None, title: str = "", reference: float | None = None) -> go.Figure:
    steps = [dict(range=[0, 10], color="rgba(57,135,229,0.08)")]
    if lower is not None and upper is not None:
        steps.append(dict(range=[lower, upper], color="rgba(57,135,229,0.30)"))
    fig = go.Figure(go.Indicator(
        mode="gauge+number" + ("+delta" if reference is not None else ""), value=value,
        number=dict(valueformat=".2f", font=dict(size=54, color=TEXT)),
        delta=dict(reference=reference, valueformat="+.2f", increasing=dict(color=SERIES_COLORS[2]), decreasing=dict(color=SERIES_COLORS[7])) if reference is not None else None,
        gauge=dict(axis=dict(range=[0, 10], tickcolor=MUTED, tickfont=dict(color=MUTED), dtick=1), bar=dict(color=SERIES_COLORS[0], thickness=0.28),
                   bgcolor="rgba(0,0,0,0)", borderwidth=0, steps=steps,
                   threshold=dict(line=dict(color=TEXT, width=2), thickness=0.8, value=reference) if reference is not None else None),
        title=dict(text=title, font=dict(color=MUTED, size=13)),
    ))
    fig.update_layout(margin=dict(l=30, r=30, t=50, b=10))
    return _finish(fig, None, height=300, legend=False)


def scenario_compare(names: list[str], values: list[float], baseline: float, title: str = "") -> go.Figure:
    cols = [SERIES_COLORS[2] if v >= baseline else SERIES_COLORS[7] for v in values]
    fig = go.Figure(go.Bar(x=names, y=values, marker=dict(color=cols, cornerradius=4), text=[f"{v:.2f}<br>({v - baseline:+.2f})" for v in values],
                           textposition="outside", textfont=dict(color=MUTED), cliponaxis=False, hovertemplate="%{x}<br>%{y:.3f}<extra></extra>"))
    fig.add_hline(y=baseline, line=dict(color=TEXT, dash="dash", width=1.5), annotation_text=f"baseline {baseline:.2f}", annotation_font_color=MUTED)
    lo = min(values + [baseline])
    hi = max(values + [baseline])
    fig.update_yaxes(title="Predicted Life Ladder", range=[max(0, lo - 0.6), min(10, hi + 0.6)])
    return _finish(fig, title, height=400, legend=False)


def tornado_chart(t: pd.DataFrame, title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(y=t["indicator"], x=t["pred_low"], orientation="h", name="indicator − 1 SD", marker=dict(color=SERIES_COLORS[7], cornerradius=4),
                         customdata=t["low_value"], hovertemplate="%{y} at %{customdata:.3f}<br>Δ prediction %{x:+.3f}<extra></extra>"))
    fig.add_trace(go.Bar(y=t["indicator"], x=t["pred_high"], orientation="h", name="indicator + 1 SD", marker=dict(color=SERIES_COLORS[0], cornerradius=4),
                         customdata=t["high_value"], hovertemplate="%{y} at %{customdata:.3f}<br>Δ prediction %{x:+.3f}<extra></extra>"))
    fig.update_layout(barmode="overlay")
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    fig.update_xaxes(title="Change in predicted Life Ladder vs baseline")
    return _finish(fig, title, height=360)


def sweep_plot(curve: pd.DataFrame, feature: str, current: float, modified: float | None = None, title: str = "") -> go.Figure:
    fig = go.Figure(go.Scatter(x=curve["value"], y=curve["prediction"], mode="lines", line=dict(color=SERIES_COLORS[0], width=2.5),
                               hovertemplate=short(feature) + " %{x:.3f}<br>prediction %{y:.3f}<extra></extra>", name="model response"))
    fig.add_vline(x=current, line=dict(color=MUTED, dash="dot"), annotation_text="baseline", annotation_font_color=MUTED)
    if modified is not None:
        fig.add_vline(x=modified, line=dict(color=SERIES_COLORS[1], dash="dash"), annotation_text="scenario", annotation_font_color=MUTED)
    fig.update_xaxes(title=lab(feature))
    fig.update_yaxes(title="Predicted Life Ladder")
    return _finish(fig, title, height=320, legend=False)


def sparkline(values: list[float], color: str = SERIES_COLORS[0], height: int = 60) -> go.Figure:
    fig = go.Figure(go.Scatter(y=values, mode="lines", line=dict(color=color, width=2), fill="tozeroy", fillcolor=_rgba(color, 0.12), hoverinfo="skip"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=height, showlegend=False)
    return fig
