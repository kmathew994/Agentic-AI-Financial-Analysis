"""
Agentic AI Expense Analyser — Streamlit Dashboard
Run with: streamlit run app.py
"""

import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic Expense Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'IBM Plex Mono', monospace !important;
        background-color: #0a0e1a;
        color: #e2e8f0;
    }
    .stApp { background-color: #0a0e1a; }
    .stTabs [data-baseweb="tab-list"] { background-color: #0d1117; border-bottom: 1px solid #1e293b; gap: 0; }
    .stTabs [data-baseweb="tab"] { background-color: transparent; color: #475569; font-family: 'IBM Plex Mono', monospace; font-size: 12px; }
    .stTabs [aria-selected="true"] { background-color: transparent !important; color: #00C896 !important; border-bottom: 2px solid #00C896 !important; }
    .stButton > button { font-family: 'IBM Plex Mono', monospace; background: linear-gradient(135deg, #00C896, #00A878); color: #000; font-weight: 700; border: none; border-radius: 6px; }
    .stButton > button:hover { opacity: 0.85; }
    div[data-testid="metric-container"] { background: #0d1117; border: 1px solid #1e293b; border-radius: 10px; padding: 12px; }
    .stTextInput > div > div > input { background: #0d1117; border: 1px solid #1e293b; color: #e2e8f0; font-family: 'IBM Plex Mono', monospace; }
    .stMarkdown, .stText { font-family: 'IBM Plex Mono', monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Imports (after env loaded) ─────────────────────────────────────────────────
from agent.graph import expense_agent
from agent.state import AgentState
from ui.charts import anomaly_scatter, category_bar, merchant_treemap, spending_donut, weekly_trend
from ui.components import anomaly_card, kpi_card, pipeline_progress, reasoning_log

SAMPLE_CSV_PATH = Path("data/sample_transactions.csv")

# ── Session state defaults ─────────────────────────────────────────────────────
for key, default in {
    "agent_state": None,
    "running": False,
    "step": -1,
    "logs": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Header ─────────────────────────────────────────────────────────────────────
col_logo, col_title, col_status = st.columns([1, 6, 3])
with col_logo:
    st.markdown(
        '<div style="width:42px;height:42px;border-radius:8px;background:linear-gradient(135deg,#00C896,#00A878);'
        'display:flex;align-items:center;justify-content:center;font-size:22px;box-shadow:0 0 24px #00C89650">⚡</div>',
        unsafe_allow_html=True,
    )
with col_title:
    st.markdown(
        '<div style="font-size:18px;font-weight:700;letter-spacing:-0.5px">Agentic Expense Intelligence</div>'
        '<div style="font-size:10px;color:#475569;letter-spacing:0.1em;text-transform:uppercase">'
        "LangGraph · Claude claude-sonnet-4-20250514 · Multi-Step Reasoning Pipeline</div>",
        unsafe_allow_html=True,
    )
with col_status:
    if st.session_state.agent_state:
        st.success("✓ Analysis complete", icon="⚡")
    elif st.session_state.running:
        st.warning("⟳ Agent running...", icon="🔄")
    else:
        st.info("Ready — upload a CSV to begin", icon="📂")

st.markdown('<hr style="border-color:#1e293b;margin:8px 0 16px">', unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_upload, tab_overview, tab_anomalies, tab_report = st.tabs([
    "📤 Upload & Process",
    "📊 Spending Overview",
    "🚨 Anomaly Report",
    "📝 AI Report",
])


# ────────────────────────────────────────────────────────────────────────────────
# TAB 1 — Upload & Process
# ────────────────────────────────────────────────────────────────────────────────
with tab_upload:
    col_upload, col_log = st.columns([1, 1], gap="medium")

    with col_upload:
        st.markdown("#### 📂 Upload Transaction CSV")

        uploaded = st.file_uploader(
            "Drag & drop or click to browse",
            type=["csv"],
            help="Supports CBA, ANZ, Westpac, NAB formats",
        )

        csv_text = None
        if uploaded:
            csv_text = uploaded.read().decode("utf-8")
            st.success(f"✓ Loaded: {uploaded.name} ({len(csv_text.splitlines())-1} rows)")

        if st.button("📥 Load Sample CSV (Australian transactions)"):
            if SAMPLE_CSV_PATH.exists():
                csv_text = SAMPLE_CSV_PATH.read_text()
                st.session_state["loaded_csv"] = csv_text
                st.success(f"✓ Sample loaded — {len(csv_text.splitlines())-1} transactions")
            else:
                st.error("Sample CSV not found at data/sample_transactions.csv")

        if "loaded_csv" in st.session_state and csv_text is None:
            csv_text = st.session_state["loaded_csv"]

        st.markdown("---")

        run_disabled = csv_text is None or st.session_state.running
        if st.button("▶ Run Agentic Pipeline", disabled=run_disabled, use_container_width=True):
            st.session_state.running = True
            st.session_state.step = 0
            st.session_state.logs = []

            progress_placeholder = st.empty()
            log_placeholder = st.empty()

            def _run_pipeline(csv: str):
                initial_state: AgentState = {
                    "raw_csv_content": csv,
                    "user_query": None,
                    "messages": [],
                    "parsed_transactions": [],
                    "categorised_transactions": [],
                    "anomalies": [],
                    "spending_summary": {},
                    "final_report": "",
                    "agent_reasoning_log": [],
                    "current_step": "ingest_csv",
                    "error": None,
                    "iteration_count": 0,
                }
                result = expense_agent.invoke(initial_state)
                return result

            with st.spinner("Running multi-step agent pipeline..."):
                result = _run_pipeline(csv_text)

            st.session_state.agent_state = result
            st.session_state.running = False
            st.session_state.step = 5
            st.session_state.logs = result.get("agent_reasoning_log", [])
            st.rerun()

        st.markdown("---")
        st.markdown(
            '<div style="font-size:10px;color:#475569">Pipeline nodes: '
            'ingest_csv → categorise → detect_anomalies → analyse → generate_report</div>',
            unsafe_allow_html=True,
        )

    with col_log:
        st.markdown("#### 🧠 Agent Reasoning Log")
        reasoning_log(st.session_state.logs)

        st.markdown("#### Pipeline Steps")
        pipeline_progress(st.session_state.step)


# ────────────────────────────────────────────────────────────────────────────────
# TAB 2 — Spending Overview
# ────────────────────────────────────────────────────────────────────────────────
with tab_overview:
    state = st.session_state.agent_state

    if not state:
        st.info("Run the agent pipeline first to see spending analysis.", icon="📊")
    else:
        summary = state.get("spending_summary", {})
        anomalies = state.get("anomalies", [])

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi_card("Total Income", f"${summary['total_income']:,.2f}", color="#00C896")
        with k2:
            kpi_card("Total Expenses", f"${summary['total_expenses']:,.2f}", color="#FF6B6B")
        with k3:
            kpi_card("Savings Rate", f"{summary['savings_rate']}%", color="#4ECDC4")
        with k4:
            anomaly_color = "#F59E0B" if len(anomalies) > 2 else "#00C896"
            kpi_card("Anomalies Found", str(len(anomalies)), color=anomaly_color)

        st.markdown("")

        col_donut, col_bar = st.columns(2)
        with col_donut:
            st.plotly_chart(spending_donut(summary["by_category"]), use_container_width=True)
        with col_bar:
            st.plotly_chart(category_bar(summary["by_category"]), use_container_width=True)

        st.plotly_chart(weekly_trend(summary["by_week"]), use_container_width=True)

        col_tree, col_merchants = st.columns([3, 2])
        with col_tree:
            st.plotly_chart(merchant_treemap(summary["by_merchant"]), use_container_width=True)
        with col_merchants:
            st.markdown("#### Top Merchants")
            for merchant, amount in summary["by_merchant"].items():
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:6px 10px;background:#0d1117;border-radius:6px;'
                    f'margin-bottom:4px;font-size:11px">'
                    f'<span style="color:#64748b;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:200px">{merchant}</span>'
                    f'<span style="color:#e2e8f0;flex-shrink:0;margin-left:8px">${amount:,.2f}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ────────────────────────────────────────────────────────────────────────────────
# TAB 3 — Anomaly Report
# ────────────────────────────────────────────────────────────────────────────────
with tab_anomalies:
    state = st.session_state.agent_state

    if not state:
        st.info("Run the agent pipeline first to see anomaly detection results.", icon="🚨")
    else:
        anomalies = state.get("anomalies", [])
        transactions = state.get("categorised_transactions", [])

        if not anomalies:
            st.success("No anomalies detected in your transaction history.", icon="✅")
        else:
            st.markdown(
                f'<div style="font-size:11px;color:#475569;margin-bottom:12px">'
                f"{len(anomalies)} transactions flagged via Z-score + IQR statistical analysis</div>",
                unsafe_allow_html=True,
            )

            st.plotly_chart(anomaly_scatter(transactions, anomalies), use_container_width=True)

            st.markdown("#### Flagged Transactions")

            # Store explain results in session state
            if "explain_results" not in st.session_state:
                st.session_state.explain_results = {}

            for anomaly in anomalies:
                col_card, col_btn = st.columns([5, 1])
                with col_card:
                    anomaly_card(anomaly)
                with col_btn:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("🤖 Explain", key=f"explain_{anomaly['id']}"):
                        with st.spinner("Analysing anomaly..."):
                            llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
                            prompt = (
                                f"Explain in 2–3 sentences why this bank transaction is flagged as anomalous:\n"
                                f"Description: {anomaly['description']}\n"
                                f"Amount: ${abs(anomaly['amount']):.2f}\n"
                                f"Date: {anomaly['date']}\n"
                                f"Z-score: {anomaly.get('z_score', '?')}\n"
                                f"Severity: {anomaly['severity']}\n"
                                f"Flags: {anomaly.get('flags', [])}\n\n"
                                "Be specific, practical, and reference the statistical reason."
                            )
                            resp = llm.invoke([HumanMessage(content=prompt)])
                            st.session_state.explain_results[anomaly["id"]] = resp.content

                if anomaly["id"] in st.session_state.explain_results:
                    st.markdown(
                        f'<div style="background:#A78BFA10;border:1px solid #A78BFA30;'
                        f'border-radius:6px;padding:10px 14px;font-size:12px;color:#94A3B8;'
                        f'margin-bottom:8px;line-height:1.6">'
                        f'<span style="color:#A78BFA">🤖 </span>'
                        f"{st.session_state.explain_results[anomaly['id']]}</div>",
                        unsafe_allow_html=True,
                    )


# ────────────────────────────────────────────────────────────────────────────────
# TAB 4 — AI Report
# ────────────────────────────────────────────────────────────────────────────────
with tab_report:
    state = st.session_state.agent_state

    if not state:
        st.info("Run the agent pipeline first to generate the AI report.", icon="📝")
    else:
        report = state.get("final_report", "")

        if report:
            st.markdown(
                '<div style="background:#0d1117;border:1px solid #1e293b;'
                'border-radius:10px;padding:24px;margin-bottom:20px">',
                unsafe_allow_html=True,
            )
            st.markdown(report)
            st.markdown("</div>", unsafe_allow_html=True)

            # Download button
            st.download_button(
                label="⬇ Download Report (.md)",
                data=report,
                file_name="expense_report.md",
                mime="text/markdown",
            )

        st.markdown("---")
        st.markdown("#### 💬 Ask a Follow-up Question (ReAct Loop)")
        st.markdown(
            '<div style="font-size:11px;color:#475569;margin-bottom:8px">'
            "The agent has full context of your transactions and can answer specific questions.</div>",
            unsafe_allow_html=True,
        )

        follow_up = st.text_input(
            label="Question",
            placeholder="e.g. How can I reduce my grocery spending? What was my biggest week?",
            label_visibility="collapsed",
        )

        if st.button("Ask Agent", disabled=not follow_up):
            with st.spinner("Agent thinking..."):
                summary = state.get("spending_summary", {})
                anomalies = state.get("anomalies", [])
                llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.3)

                context = (
                    f"Income: ${summary.get('total_income', 0):,.2f}, "
                    f"Expenses: ${summary.get('total_expenses', 0):,.2f}, "
                    f"Savings rate: {summary.get('savings_rate', 0)}%.\n"
                    f"Categories: {', '.join(f'{k} ${v:.2f}' for k,v in summary.get('by_category', {}).items())}.\n"
                    f"Anomalies: {len(anomalies)} flagged."
                )
                system = SystemMessage(content=(
                    "You are a financial AI assistant with access to the user's transaction analysis. "
                    "Answer concisely in 2–4 sentences with specific dollar amounts where possible. "
                    "Australian English."
                ))
                resp = llm.invoke([
                    system,
                    HumanMessage(content=f"Context:\n{context}\n\nQuestion: {follow_up}"),
                ])

                st.markdown(
                    f'<div style="background:#00C89610;border:1px solid #00C89630;'
                    f'border-radius:8px;padding:14px;font-size:13px;color:#94A3B8;'
                    f'line-height:1.7;margin-top:12px">'
                    f'<span style="color:#00C896">Agent: </span>{resp.content}</div>',
                    unsafe_allow_html=True,
                )
