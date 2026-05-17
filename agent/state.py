from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Input
    raw_csv_content: str
    user_query: Optional[str]

    # Processing stages
    messages: Annotated[list, add_messages]
    parsed_transactions: List[Dict[str, Any]]
    categorised_transactions: List[Dict[str, Any]]
    anomalies: List[Dict[str, Any]]
    spending_summary: Dict[str, Any]

    # Output
    final_report: str
    agent_reasoning_log: List[str]

    # Control flow
    current_step: str
    error: Optional[str]
    iteration_count: int
