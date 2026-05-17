"""
Plotly chart builders for the Streamlit dashboard.
All functions return plotly Figure objects ready for st.plotly_chart().
"""

from __future__ import annotations

from typing import Any

import plotly.express as px
import plotly.graph_objects as go

# ── Shared theme ───────────────────────────────────────────────────────────────
_BG = "#0a0e1a"
_PAPER = "#0d1117"
_GRID = "#1e293b"
_TEXT = "#94A3B8"
_ACCENT = "#00C896"

_BASE_LAYOUT = dict(
    paper_bgcolor=_PAPER,
    plot_bgcolor=_BG,
    font=dict(color=_TEXT, family="IBM Plex Mono, monospace", size=11),
    margin=dict(l=16, r=16, t=32, b=16),
    showlegend=True,
)

CAT_COLORS: dict[str, str] = {
    "groceries": "#00C896",
    "dining": "#FF6B6B",
    "transport": "#4ECDC4",
    "subscriptions": "#A78BFA",
    "utilities": "#F59E0B",
    "healthcare": "#06B6D4",
    "entertainment": "#F97316",
    "shopping": "#EC4899",
    "transfers": "#6B7280",
    "income": "#10B981",
    "other": "#94A3B8",
}


def spending_donut(by_category: dict[str, float]) -> go.Figure:
    """Donut chart of spending by category."""
    labels = list(by_category.keys())
    values = list(by_category.values())
    colors = [CAT_COLORS.get(l, "#888") for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color=_BG, width=2)),
        textinfo="label+percent",
        textfont=dict(size=10),
        hovertemplate="<b>%{label}</b><br>$%{value:,.2f}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(**_BASE_LAYOUT, title="Spending by Category", showlegend=False)
    return fig


def category_bar(by_category: dict[str, float]) -> go.Figure:
    """Horizontal bar chart of category totals."""
    sorted_cats = sorted(by_category.items(), key=lambda x: x[1])
    cats, vals = zip(*sorted_cats) if sorted_cats else ([], [])
    colors = [CAT_COLORS.get(c, "#888") for c in cats]

    fig = go.Figure(go.Bar(
        x=vals,
        y=cats,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>$%{x:,.2f}<extra></extra>",
        text=[f"${v:,.0f}" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(
        **_BASE_LAYOUT,
        title="Expenses by Category",
        xaxis=dict(gridcolor=_GRID, showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    return fig


def weekly_trend(by_week: dict[str, float]) -> go.Figure:
    """Bar + line chart of weekly spending."""
    weeks = list(by_week.keys())
    values = list(by_week.values())

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=weeks, y=values,
        name="Weekly Spend",
        marker=dict(color=_ACCENT, opacity=0.6),
        hovertemplate="Week %{x}<br>$%{y:,.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=weeks, y=values,
        name="Trend",
        mode="lines+markers",
        line=dict(color="#A78BFA", width=2),
        marker=dict(size=6),
    ))
    fig.update_layout(
        **_BASE_LAYOUT,
        title="Week-over-Week Spending",
        xaxis=dict(gridcolor=_GRID),
        yaxis=dict(gridcolor=_GRID),
        barmode="overlay",
    )
    return fig


def anomaly_scatter(
    transactions: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
) -> go.Figure:
    """
    Scatter plot of all expense transactions.
    Anomalies are highlighted with a different marker and colour.
    """
    import pandas as pd

    anomaly_ids = {a["id"] for a in anomalies}
    severity_colors = {"HIGH": "#FF4444", "MEDIUM": "#FF9500", "LOW": "#FFD700"}

    normal = [t for t in transactions if t["amount"] < 0 and t["id"] not in anomaly_ids]
    flagged = [a for a in anomalies]

    fig = go.Figure()

    # Normal transactions
    if normal:
        fig.add_trace(go.Scatter(
            x=[t["date"] for t in normal],
            y=[abs(t["amount"]) for t in normal],
            mode="markers",
            name="Normal",
            marker=dict(color=_ACCENT, size=6, opacity=0.5),
            hovertemplate="<b>%{customdata}</b><br>$%{y:,.2f}<extra></extra>",
            customdata=[t["description"] for t in normal],
        ))

    # Flagged anomalies (by severity)
    for sev in ("HIGH", "MEDIUM", "LOW"):
        grp = [a for a in flagged if a["severity"] == sev]
        if not grp:
            continue
        fig.add_trace(go.Scatter(
            x=[a["date"] for a in grp],
            y=[abs(a["amount"]) for a in grp],
            mode="markers",
            name=f"{sev} anomaly",
            marker=dict(
                color=severity_colors[sev],
                size=12 if sev == "HIGH" else 9,
                symbol="diamond",
                line=dict(width=1, color=_BG),
            ),
            hovertemplate="<b>%{customdata[0]}</b><br>$%{y:,.2f}<br>Z=%{customdata[1]}<extra></extra>",
            customdata=[[a["description"], a.get("z_score", "?")] for a in grp],
        ))

    fig.update_layout(
        **_BASE_LAYOUT,
        title="Transaction Anomaly Timeline",
        xaxis=dict(gridcolor=_GRID, title="Date"),
        yaxis=dict(gridcolor=_GRID, title="Amount ($)"),
    )
    return fig


def merchant_treemap(by_merchant: dict[str, float]) -> go.Figure:
    """Treemap of top merchant spend."""
    merchants = list(by_merchant.keys())
    values = list(by_merchant.values())

    fig = go.Figure(go.Treemap(
        labels=merchants,
        parents=[""] * len(merchants),
        values=values,
        marker=dict(
            colors=values,
            colorscale=[[0, "#0d1117"], [1, _ACCENT]],
            showscale=False,
        ),
        hovertemplate="<b>%{label}</b><br>$%{value:,.2f}<extra></extra>",
        textinfo="label+value",
    ))
    fig.update_layout(**_BASE_LAYOUT, title="Top Merchant Spend")
    return fig
