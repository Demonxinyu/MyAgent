"""LangGraph agent graph — main customer-service state machine.

Graph topology
--------------

::

                        START
                          │
                   ┌──────▼──────┐
                   │classify_intent│
                   └──────┬──────┘
                          │
               ┌──────────┼──────────┐
               │          │          │
        general_qa  return_request  human_support
               │          │          │
        ┌──────▼──────┐ ┌▼─────────┐ ┌▼──────────┐
        │rag_retrieve │ │return    │ │human_handoff│
        └──────┬──────┘ │_subgraph │ └───────────┘
               │        └┬────────┘
        ┌──────▼──────┐  │
        │generate     │  │
        │_response    │  │
        └──────┬──────┘  │
               │         │
               └────┬────┘
                    │
               ┌────▼────┐
               │   END   │
               └─────────┘

The return subgraph runs through:
start → validate_order → check_policy → collect_reason → initiate → confirm
"""

import asyncio
import logging
from pathlib import Path

import aiosqlite

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph

from config import settings

from agent.nodes.classify import classify_intent
from agent.nodes.generate import generate_response
from agent.nodes.handoff import human_handoff
from agent.nodes.rag import rag_retrieve
from agent.nodes.return_flow import (
    return_check_policy,
    return_collect_reason,
    return_confirm,
    return_initiate,
    return_start,
    return_validate_order,
)
from agent.state import AgentState

logger = logging.getLogger(__name__)

# ── routing helpers ──────────────────────────────────────────────────────────


def route_by_intent(state: AgentState) -> str:
    """After classification, route to the correct branch."""
    intent = state.get("intent", "general_qa")
    logger.info("Routing intent: %s", intent)

    if intent == "return_request":
        return "return_start"
    if intent == "human_support":
        return "human_handoff"
    return "rag_retrieve"  # general_qa (default)


def route_return_step(state: AgentState) -> str:
    """Within the return subgraph, decide the next node.

    Terminal steps → END (graph stops, waits for next user message).
    Progression steps → the next node in the return pipeline.
    """
    step = state.get("return_step", "")

    # ── terminal: graph stops, response already in state ─────────────────
    if step in ("confirmed", "not_eligible"):
        return "generate_response"
    if step in ("waiting_order_id", "collecting_reason"):
        return END  # stop and wait for user input

    # ── error ───────────────────────────────────────────────────────────
    if step == "failed":
        return "human_handoff"

    # ── progression ─────────────────────────────────────────────────────
    step_order: dict[str, str] = {
        "order_extracted": "return_validate_order",
        "validated": "return_check_policy",
        "policy_checked": "return_collect_reason",
        "need_reason": "return_collect_reason",  # re-enter after user provides reason
        "reason_collected": "return_initiate",
        "initiated": "return_confirm",
    }
    next_node = step_order.get(step, "return_start")
    logger.info("Return step: %s → %s", step, next_node)
    return next_node


def should_handoff_after_rag(state: AgentState) -> str:
    """After RAG retrieval, decide: generate or handoff (RAG unavailable)."""
    if state.get("handoff_reason"):
        return "human_handoff"
    return "generate_response"


def after_return_or_response(state: AgentState) -> str:
    """After response generation, check if we need to handoff."""
    if state.get("handoff_reason") and state.get("return_step") == "failed":
        return "human_handoff"
    return END


# ── graph construction ───────────────────────────────────────────────────────


async def build_main_graph() -> StateGraph:
    """Construct and compile the top-level agent graph.

    The return flow is a separate compiled subgraph that is added as a single
    node in the main graph.
    """
    graph = StateGraph(AgentState)

    # ── nodes ────────────────────────────────────────────────────────────
    graph.add_node("classify_intent", classify_intent)

    # QA branch
    graph.add_node("rag_retrieve", rag_retrieve)
    graph.add_node("generate_response", generate_response)

    # Return branch (single node wrapping the subgraph)
    graph.add_node("return_start", return_start)
    graph.add_node("return_validate_order", return_validate_order)
    graph.add_node("return_check_policy", return_check_policy)
    graph.add_node("return_collect_reason", return_collect_reason)
    graph.add_node("return_initiate", return_initiate)
    graph.add_node("return_confirm", return_confirm)

    # Handoff
    graph.add_node("human_handoff", human_handoff)

    # ── edges ─────────────────────────────────────────────────────────────
    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges("classify_intent", route_by_intent, {
        "rag_retrieve": "rag_retrieve",
        "return_start": "return_start",
        "human_handoff": "human_handoff",
    })

    # QA pipeline
    graph.add_conditional_edges("rag_retrieve", should_handoff_after_rag, {
        "generate_response": "generate_response",
        "human_handoff": "human_handoff",
    })
    graph.add_edge("generate_response", END)

    # Return flow edges.
    # Every node can route to END (when waiting for user input) or to another node.
    _return_edge_map = {
        "return_validate_order": "return_validate_order",
        "return_check_policy": "return_check_policy",
        "return_collect_reason": "return_collect_reason",
        "return_initiate": "return_initiate",
        "return_confirm": "return_confirm",
        "return_start": "return_start",
        "human_handoff": "human_handoff",
        "generate_response": "generate_response",
        END: END,
    }
    for node in (
        "return_start",
        "return_validate_order",
        "return_check_policy",
        "return_collect_reason",
        "return_initiate",
        "return_confirm",
    ):
        graph.add_conditional_edges(node, route_return_step, _return_edge_map)

    graph.add_edge("human_handoff", END)

    # AsyncSqliteSaver for persistent checkpointing — survives restarts.
    # Uses aiosqlite (async wrapper around SQLite) to work with LangGraph's async API.
    db_dir = Path(settings.db_path).parent
    db_dir.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(settings.db_path)
    await conn.execute("PRAGMA journal_mode=WAL")
    checkpointer = AsyncSqliteSaver(conn)
    return graph.compile(checkpointer=checkpointer)
