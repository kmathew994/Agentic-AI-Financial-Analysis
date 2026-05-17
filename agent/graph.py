"""
LangGraph StateGraph for the Agentic Expense Analyser.

Pipeline:
  ingest_csv
      ↓
  categorise
      ↓
  detect_anomalies
      ↓
  analyse ──(anomalies > 5 & iter=0)──→ categorise (re-analysis loop)
      ↓
  generate_report
      ↓
   END
"""

from langgraph.graph import END, StateGraph

from agent.nodes import (
    analysis_node,
    anomaly_detection_node,
    categorisation_node,
    csv_ingestion_node,
    report_generation_node,
    should_reanalyse,
)
from agent.state import AgentState


def build_expense_agent():
    """Compile and return the LangGraph expense analysis agent."""
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("ingest_csv", csv_ingestion_node)
    graph.add_node("categorise", categorisation_node)
    graph.add_node("detect_anomalies", anomaly_detection_node)
    graph.add_node("analyse", analysis_node)
    graph.add_node("generate_report", report_generation_node)

    # Entry point
    graph.set_entry_point("ingest_csv")

    # Linear edges
    graph.add_edge("ingest_csv", "categorise")
    graph.add_edge("categorise", "detect_anomalies")
    graph.add_edge("detect_anomalies", "analyse")

    # Conditional edge: re-analysis loop vs proceed to report
    graph.add_conditional_edges(
        "analyse",
        should_reanalyse,
        {
            "reanalyse": "categorise",
            "generate_report": "generate_report",
        },
    )

    graph.add_edge("generate_report", END)

    return graph.compile()


# Singleton instance (import and use directly)
expense_agent = build_expense_agent()
