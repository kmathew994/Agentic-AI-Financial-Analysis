"""
LangGraph node functions for the Agentic Expense Analyser pipeline.

Node execution order:
  ingest_csv → categorise → detect_anomalies → analyse → generate_report
                  ↑                                  |
                  └── (conditional re-analysis loop) ┘
"""

import json
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import AgentState
from agent.tools import (
    calculate_spending_summary,
    categorise_transaction,
    detect_spending_anomalies,
    generate_budget_recommendations,
    parse_csv_transactions,
)

# ── Shared LLM instance ────────────────────────────────────────────────────────
_llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
_llm_with_tools = _llm.bind_tools([categorise_transaction])


def _log(state: AgentState, msg: str) -> list[str]:
    return state.get("agent_reasoning_log", []) + [msg]


# ── Node 1: CSV Ingestion ──────────────────────────────────────────────────────

def csv_ingestion_node(state: AgentState) -> dict[str, Any]:
    """Parse raw CSV content into structured transaction records."""
    log = _log(state, "→ Node: ingest_csv | Tool: parse_csv_transactions")

    result = parse_csv_transactions.invoke({"csv_content": state["raw_csv_content"]})

    if "error" in result:
        log.append(f"⚠ Parse error: {result['error']} — attempting fallback format detection")
        # Attempt with stripped whitespace / BOM
        cleaned = state["raw_csv_content"].lstrip("\ufeff").strip()
        result = parse_csv_transactions.invoke({"csv_content": cleaned})
        if "error" in result:
            return {
                "error": result["error"],
                "agent_reasoning_log": log + [f"✗ Could not parse CSV: {result['error']}"],
                "current_step": "ingest_csv",
            }

    transactions = result["transactions"]
    dr = result.get("date_range", {})
    log.append(
        f"✓ Ingested {result['row_count']} transactions "
        f"({dr.get('start', '?')} → {dr.get('end', '?')}) | "
        f"Columns: {result['columns_detected']}"
    )

    return {
        "parsed_transactions": transactions,
        "agent_reasoning_log": log,
        "current_step": "categorise",
        "iteration_count": 0,
        "error": None,
    }


# ── Node 2: Categorisation ─────────────────────────────────────────────────────

def categorisation_node(state: AgentState) -> dict[str, Any]:
    """
    Categorise every transaction.
    Rule-based for confidence >= 0.7; LLM fallback for ambiguous entries.
    """
    log = _log(state, "→ Node: categorise | Rule-based + LLM hybrid")

    transactions = state["parsed_transactions"]
    categorised: list[dict] = []
    llm_needed: list[dict] = []

    # First pass: rule-based
    for t in transactions:
        result = categorise_transaction.invoke({
            "description": t["description"],
            "amount": t["amount"],
        })
        enriched = {**t, **result}
        categorised.append(enriched)
        if result["confidence"] < 0.7:
            llm_needed.append(enriched)

    log.append(
        f"  Rule-based: {len(transactions) - len(llm_needed)} categorised (confidence ≥ 0.7)"
    )

    # Second pass: LLM fallback for low-confidence entries
    if llm_needed:
        log.append(f"  LLM fallback invoked for {len(llm_needed)} ambiguous transactions")
        system = SystemMessage(content=(
            "You are a financial transaction categoriser for Australian bank data. "
            "Respond ONLY in valid JSON with keys: category, subcategory, confidence, reasoning. "
            "Valid categories: groceries, dining, transport, utilities, entertainment, "
            "healthcare, shopping, subscriptions, income, transfers, other."
        ))
        for t in llm_needed:
            prompt = (
                f"Categorise this transaction:\n"
                f"Description: {t['description']}\n"
                f"Amount: {'credit' if t['amount'] > 0 else 'debit'} ${abs(t['amount']):.2f}"
            )
            try:
                resp = _llm.invoke([system, HumanMessage(content=prompt)])
                raw = resp.content.strip().lstrip("```json").rstrip("```").strip()
                llm_result = json.loads(raw)
                # Update in categorised list
                for i, c in enumerate(categorised):
                    if c["id"] == t["id"]:
                        categorised[i] = {**c, **llm_result}
                        break
            except Exception as e:
                log.append(f"  ⚠ LLM categorisation failed for '{t['description']}': {e}")

    cats_used = len({c.get("category", "other") for c in categorised})
    log.append(f"✓ All {len(categorised)} transactions categorised — {cats_used} distinct categories")

    return {
        "categorised_transactions": categorised,
        "agent_reasoning_log": log,
        "current_step": "detect_anomalies",
    }


# ── Node 3: Anomaly Detection ──────────────────────────────────────────────────

def anomaly_detection_node(state: AgentState) -> dict[str, Any]:
    """Flag statistically unusual transactions using Z-score + IQR methods."""
    log = _log(state, "→ Node: detect_anomalies | Tool: detect_spending_anomalies")
    log.append("  Method: Z-score (threshold: 2.0σ) + IQR (1.5× fence)")

    transactions_json = json.dumps(state["categorised_transactions"])
    result = detect_spending_anomalies.invoke({"transactions_json": transactions_json})

    anomalies = result.get("anomalies", [])
    st = result.get("stats", {})

    if st:
        log.append(
            f"  Stats: mean=${st.get('mean', 0):.2f}, std=${st.get('std', 0):.2f}, "
            f"IQR fence=${st.get('upper_fence', 0):.2f}"
        )

    high = sum(1 for a in anomalies if a["severity"] == "HIGH")
    med = sum(1 for a in anomalies if a["severity"] == "MEDIUM")
    low = sum(1 for a in anomalies if a["severity"] == "LOW")
    log.append(f"✓ Detected {len(anomalies)} anomalies — HIGH: {high}, MEDIUM: {med}, LOW: {low}")

    if len(anomalies) > 5:
        log.append(
            f"⚠ Anomaly count ({len(anomalies)}) > 5 — conditional edge: triggering re-analysis pass"
        )

    return {
        "anomalies": anomalies,
        "agent_reasoning_log": log,
        "current_step": "analyse",
    }


# ── Node 4: Analysis ───────────────────────────────────────────────────────────

def analysis_node(state: AgentState) -> dict[str, Any]:
    """Compute aggregate spending statistics and budget recommendations."""
    log = _log(state, "→ Node: analyse | Tools: calculate_spending_summary, generate_budget_recommendations")

    transactions_json = json.dumps(state["categorised_transactions"])

    summary_result = calculate_spending_summary.invoke({"transactions_json": transactions_json})
    recs_result = generate_budget_recommendations.invoke({"summary_json": json.dumps(summary_result)})

    top_cat = max(summary_result["by_category"].items(), key=lambda x: x[1], default=("n/a", 0))
    log.append(
        f"  Income: ${summary_result['total_income']:.2f} | "
        f"Expenses: ${summary_result['total_expenses']:.2f} | "
        f"Savings rate: {summary_result['savings_rate']}%"
    )
    log.append(
        f"  Top category: {top_cat[0]} (${top_cat[1]:.2f}) | "
        f"Recommendations priority: {recs_result.get('priority', 'N/A')} | "
        f"Potential savings: ${recs_result.get('potential_savings', 0):.2f}"
    )
    log.append("✓ Spending summary complete — weekly trends and merchant analysis ready")

    spending_summary = {**summary_result, "recommendations": recs_result}

    return {
        "spending_summary": spending_summary,
        "agent_reasoning_log": log,
        "current_step": "generate_report",
    }


# ── Node 5: Report Generation ──────────────────────────────────────────────────

_REPORT_SYSTEM = """You are a financial analyst AI for an Agentic Expense Intelligence platform built on CBA's infrastructure.
Your reports are professional, data-driven, and written in Australian English.
Always use specific dollar amounts. Structure your report clearly with markdown headings."""

_REPORT_TEMPLATE = """Analyse the following transaction data and write a comprehensive financial report.

=== PERIOD ===
{date_range}

=== TRANSACTION SUMMARY ===
Total transactions: {tx_count}
Total income: ${total_income}
Total expenses: ${total_expenses}
Savings rate: {savings_rate}%

=== SPENDING BY CATEGORY ===
{by_category}

=== TOP MERCHANTS ===
{by_merchant}

=== WEEKLY TREND ===
{by_week}

=== ANOMALIES DETECTED ({anomaly_count}) ===
{anomalies}

=== BUDGET RECOMMENDATIONS ===
{recommendations}

Write a financial report with EXACTLY these sections:

## 1. Executive Summary
Three sentences covering total income, total expenses, savings rate, and top concern.

## 2. Spending Breakdown
Discuss each significant category with insights about patterns and whether spend seems appropriate.

## 3. Anomaly Alerts
For each anomaly, explain: what it is, why it was flagged, and whether the user should be concerned.

## 4. Trends & Patterns
Identify weekly patterns, habitual merchants, recurring costs, and any seasonal factors.

## 5. Recommendations
Provide 3–5 specific, actionable recommendations. Each must include: the category, a specific dollar amount, and a concrete action step.

## 6. Month Outlook
Project the end-of-month balance trajectory if current patterns continue. Flag any risks.

Keep total length around 600 words. Be direct, specific, and practical."""


def report_generation_node(state: AgentState) -> dict[str, Any]:
    """Generate a comprehensive natural language financial report using Claude."""
    log = _log(state, f"→ Node: generate_report | Model: claude-sonnet-4-20250514")

    summary = state["spending_summary"]
    anomalies = state["anomalies"]
    transactions = state["categorised_transactions"]

    # Build date range from transactions
    dates = sorted(t["date"] for t in transactions if t.get("date"))
    date_range = f"{dates[0]} to {dates[-1]}" if dates else "Unknown"

    def fmt_dict(d: dict) -> str:
        return "\n".join(f"  {k}: ${v:.2f}" for k, v in d.items())

    anom_text = "\n".join(
        f"  [{a['severity']}] {a['date']} | {a['description']} | ${abs(a['amount']):.2f} | Z={a.get('z_score', '?')}"
        for a in anomalies
    ) or "  None detected"

    recs = summary.get("recommendations", {}).get("recommendations", [])
    rec_text = "\n".join(
        f"  [{r['category'].upper()}] {r['insight']}" for r in recs
    ) or "  No specific recommendations at this time"

    prompt = _REPORT_TEMPLATE.format(
        date_range=date_range,
        tx_count=len(transactions),
        total_income=f"{summary['total_income']:.2f}",
        total_expenses=f"{summary['total_expenses']:.2f}",
        savings_rate=summary["savings_rate"],
        by_category=fmt_dict(summary["by_category"]),
        by_merchant=fmt_dict(summary["by_merchant"]),
        by_week=fmt_dict(summary["by_week"]),
        anomaly_count=len(anomalies),
        anomalies=anom_text,
        recommendations=rec_text,
    )

    try:
        response = _llm.invoke([
            SystemMessage(content=_REPORT_SYSTEM),
            HumanMessage(content=prompt),
        ])
        report = response.content
        log.append(f"✓ Report generated — {len(report)} characters, 6 sections")
    except Exception as e:
        report = f"Report generation failed: {e}"
        log.append(f"✗ Report generation error: {e}")

    return {
        "final_report": report,
        "agent_reasoning_log": log,
        "current_step": "done",
    }


# ── Conditional Edge ───────────────────────────────────────────────────────────

def should_reanalyse(state: AgentState) -> str:
    """
    Conditional routing after the analysis node.
    If anomaly density is high and we haven't re-analysed yet, loop back to categorise.
    Otherwise, proceed to report generation.
    """
    anomaly_count = len(state.get("anomalies", []))
    iteration = state.get("iteration_count", 0)

    if anomaly_count > 5 and iteration < 1:
        # Increment iteration counter to prevent infinite loops
        state["iteration_count"] = iteration + 1  # type: ignore[index]
        return "reanalyse"
    return "generate_report"
