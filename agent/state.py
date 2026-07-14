"""LangGraph agent state definition."""

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Shared state carried through every node in the graph.

    Notes
    -----
    * ``messages`` uses LangGraph's ``add_messages`` reducer so new messages
      are appended rather than replacing the list.
    * All other keys are overwritten on each write.
    """

    # ── conversation ──
    messages: Annotated[list[Any], add_messages]

    # ── identity ──
    user_id: str
    session_id: str

    # ── routing ──
    intent: str  # general_qa | return_request | human_support

    # ── RAG ──
    rag_context: str

    # ── return flow ──
    return_order_id: str
    return_reason: str
    return_step: str
    return_eligible: bool
    return_attempts: int
    return_collecting_info: str  # which piece of info we're asking for next

    # ── handoff ──
    handoff_reason: str

    # ── output ──
    final_response: str
