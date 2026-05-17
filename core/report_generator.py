"""
Natural language report generator.
Takes completed AgentState data and produces a structured markdown report via Claude.
"""

from __future__ import annotations

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

_SYSTEM = """You are a financial analyst AI for an Agentic Expense Intelligence platform.
Your role is to produce clear, data-driven financial reports in Australian English.
Always reference specific dollar amounts. Be direct, practical, and actionable."""

_TEMPLATE = """Produce a comprehensive financial report from the data below.

PERIOD: {date_range}
TRANSACTIONS: {tx_count} total

INCOME / EXPENSES:
  Total income:   ${total_income}
  Total expenses: ${total_expenses}
  Savings rate:   {savings_rate}%

SPENDING BY CATEGORY:
{by_category}

TOP MERCHANTS:
{by_merchant}

WEEKLY SPENDING:
{by_week}

ANOMALIES ({anomaly_count} flagged):
{anomalies}

BUDGET RECOMMENDATIONS:
{recommendations}

---
Write the report using EXACTLY these markdown sections:

## 1. Executive Summary
(3 sentences — income, expenses, savings rate, headline risk)

## 2. Spending Breakdown
(Analyse each category — is it reasonable? Any concerns? Specific amounts.)

## 3. Anomaly Alerts
(Each anomaly: what it is, why flagged, whether action is needed)

## 4. Trends & Patterns
(Weekly cadence, habitual merchants, recurring spend, any seasonal factors)

## 5. Recommendations
(3–5 items; each must include: category, specific dollar amount, concrete action step)

## 6. Month Outlook
(Project trajectory; flag risks; estimated end-of-month balance if patterns hold)

Australian English. ~600 words. Professional but conversational tone."""


def generate_report(
    transactions: list[dict[str, Any]],
    spending_summary: dict[str, Any],
    anomalies: list[dict[str, Any]],
    model: str = "claude-sonnet-4-20250514",
) -> str:
    """
    Generate a natural language financial report.

    Args:
        transactions: categorised transaction list
        spending_summary: output of calculate_spending_summary tool
        anomalies: output of detect_anomalies
        model: Anthropic model name

    Returns:
        Markdown-formatted report string
    """
    llm = ChatAnthropic(model=model, temperature=0.3)

    dates = sorted(t["date"] for t in transactions if t.get("date"))
    date_range = f"{dates[0]} to {dates[-1]}" if dates else "Unknown period"

    def fmt(d: dict) -> str:
        return "\n".join(f"  {k}: ${v:.2f}" for k, v in d.items()) if d else "  (none)"

    anom_lines = "\n".join(
        f"  [{a['severity']}] {a['date']} | {a['description']} | ${abs(a['amount']):.2f} | "
        f"Z={a.get('z_score', '?')} | flags={a.get('flags', [])}"
        for a in anomalies
    ) or "  None detected"

    recs = spending_summary.get("recommendations", {}).get("recommendations", [])
    rec_lines = "\n".join(
        f"  [{r['category'].upper()}] {r['insight']}" for r in recs
    ) or "  No specific recommendations generated"

    prompt = _TEMPLATE.format(
        date_range=date_range,
        tx_count=len(transactions),
        total_income=f"{spending_summary['total_income']:.2f}",
        total_expenses=f"{spending_summary['total_expenses']:.2f}",
        savings_rate=spending_summary["savings_rate"],
        by_category=fmt(spending_summary["by_category"]),
        by_merchant=fmt(spending_summary["by_merchant"]),
        by_week=fmt(spending_summary["by_week"]),
        anomaly_count=len(anomalies),
        anomalies=anom_lines,
        recommendations=rec_lines,
    )

    try:
        response = llm.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=prompt),
        ])
        return response.content
    except Exception as e:
        return f"# Report Generation Error\n\n{e}\n\nPlease check your ANTHROPIC_API_KEY."
