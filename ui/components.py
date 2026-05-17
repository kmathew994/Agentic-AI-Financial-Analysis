"""
Reusable Streamlit UI components for the Expense Analyser dashboard.
"""

from __future__ import annotations

from typing import Any

import streamlit as st


def kpi_card(label: str, value: str, delta: str | None = None, color: str = "#00C896") -> None:
    """Render a single KPI metric card."""
    st.markdown(
        f"""
        <div style="
            background:#0d1117;
            border:1px solid #1e293b;
            border-radius:10px;
            padding:16px 20px;
            text-align:center;
        ">
            <div style="font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px">{label}</div>
            <div style="font-size:28px;font-weight:700;color:{color};letter-spacing:-1px;font-family:'IBM Plex Mono',monospace">{value}</div>
            {"" if delta is None else f'<div style="font-size:11px;color:#64748b;margin-top:4px">{delta}</div>'}
        </div>
        """,
        unsafe_allow_html=True,
    )


def severity_badge(severity: str) -> str:
    """Return HTML for a severity badge."""
    colors = {"HIGH": "#FF4444", "MEDIUM": "#FF9500", "LOW": "#FFD700"}
    c = colors.get(severity, "#94A3B8")
    return (
        f'<span style="'
        f"background:{c}20;color:{c};border:1px solid {c}40;"
        f"padding:2px 8px;border-radius:3px;font-size:9px;font-weight:700;"
        f'letter-spacing:0.1em">{severity}</span>'
    )


def anomaly_card(anomaly: dict[str, Any], on_explain: Any | None = None) -> None:
    """Render an anomaly detail card with optional explain button."""
    sev = anomaly.get("severity", "LOW")
    sev_colors = {"HIGH": "#FF4444", "MEDIUM": "#FF9500", "LOW": "#FFD700"}
    border_color = sev_colors.get(sev, "#94A3B8")

    with st.container():
        st.markdown(
            f"""
            <div style="
                background:#0d1117;
                border:1px solid {border_color}30;
                border-left:3px solid {border_color};
                border-radius:10px;
                padding:14px 16px;
                margin-bottom:10px;
            ">
                <div style="display:flex;justify-content:space-between;align-items:flex-start">
                    <div>
                        {severity_badge(sev)}
                        <span style="font-size:11px;color:#64748b;margin-left:8px">{anomaly.get('date','')}</span>
                        <div style="font-size:14px;font-weight:600;margin-top:6px;color:#e2e8f0">{anomaly.get('description','')}</div>
                        <div style="font-size:11px;color:#64748b;margin-top:3px">
                            Z-score: {anomaly.get('z_score','?')}σ &nbsp;·&nbsp;
                            Category: {anomaly.get('category','other')} &nbsp;·&nbsp;
                            Flags: {', '.join(anomaly.get('flags', []))}
                        </div>
                    </div>
                    <div style="font-size:22px;font-weight:700;color:{border_color};font-family:'IBM Plex Mono',monospace;flex-shrink:0;margin-left:16px">
                        ${abs(anomaly.get('amount', 0)):.2f}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if on_explain:
            if st.button(f"🤖 Explain this anomaly", key=f"explain_{anomaly.get('id')}"):
                on_explain(anomaly)


def pipeline_progress(current_step: int) -> None:
    """Render the 5-step pipeline progress bar."""
    steps = [
        ("⬆", "INGEST"),
        ("🏷", "CATEGORISE"),
        ("🔍", "DETECT"),
        ("📊", "ANALYSE"),
        ("📝", "REPORT"),
    ]
    cols = st.columns(len(steps))
    for i, (icon, label) in enumerate(steps):
        done = current_step > i
        active = current_step == i
        border = "#00C896" if done else ("#A78BFA" if active else "#1e293b")
        text_color = "#00C896" if done else ("#A78BFA" if active else "#334155")
        with cols[i]:
            st.markdown(
                f"""
                <div style="
                    padding:8px 4px;
                    background:#0d1117;
                    border:1px solid {border};
                    border-radius:6px;
                    text-align:center;
                    transition:all 0.3s;
                ">
                    <div style="font-size:16px">{icon}</div>
                    <div style="font-size:8px;color:{text_color};margin-top:2px;letter-spacing:0.06em">
                        {"✓ " if done else ""}{label}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def reasoning_log(logs: list[str]) -> None:
    """Render the agent reasoning log in a scrollable box."""
    type_colors = {
        "→": "#4ECDC4",   # tool calls
        "✓": "#00C896",   # success
        "✗": "#FF6B6B",   # error
        "⚠": "#F59E0B",   # warning
        "🚀": "#A78BFA",   # system
        "🏁": "#A78BFA",
    }
    lines_html = ""
    for log in logs:
        first_char = log.strip()[:1] if log.strip() else ""
        color = type_colors.get(first_char, "#64748b")
        lines_html += (
            f'<div style="color:{color};font-size:11px;margin-bottom:3px;'
            f'font-family:\'IBM Plex Mono\',monospace">{log}</div>'
        )

    st.markdown(
        f"""
        <div style="
            background:#0d1117;
            border:1px solid #1e293b;
            border-radius:8px;
            padding:12px;
            height:280px;
            overflow-y:auto;
            font-family:'IBM Plex Mono',monospace;
        ">
            {lines_html if lines_html else '<div style="color:#334155;font-size:11px">Awaiting agent run...</div>'}
        </div>
        """,
        unsafe_allow_html=True,
    )
