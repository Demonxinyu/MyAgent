"""FastAPI routes for the customer-service agent."""

import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage

from agent.graph import build_main_graph
from agent.state import AgentState
from api.schemas import (
    ChatRequest,
    ChatResponse,
    HandoffPickupRequest,
    HandoffPickupResponse,
    SessionInfo,
)
from services.session_store import Session, session_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["customer-service"])

# Compiled graph — built once at import time
_agent_graph = build_main_graph()


# ── helpers ──────────────────────────────────────────────────────────────────


def _build_initial_state(user_id: str, session_id: str, message: str) -> AgentState:
    return AgentState(
        messages=[HumanMessage(content=message)],
        user_id=user_id,
        session_id=session_id,
        intent="",
        rag_context="",
        return_order_id="",
        return_reason="",
        return_step="",
        return_eligible=False,
        return_attempts=0,
        return_collecting_info="",
        handoff_reason="",
        final_response="",
    )


def _session_to_info(s: Session) -> SessionInfo:
    return SessionInfo(
        session_id=s.session_id,
        user_id=s.user_id,
        messages=s.messages,
        need_handoff=s.need_handoff,
        handoff_reason=s.handoff_reason,
        return_order_id=s.return_order_id,
        return_reason=s.return_reason,
        return_step=s.return_step,
    )


# ── routes ───────────────────────────────────────────────────────────────────


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Main conversation endpoint.

    Send a user message and receive the agent's response.  Multi-turn
    conversations are supported via ``session_id``.
    """
    # --- resolve session ---
    session = session_store.get(req.session_id) if req.session_id else None
    if session is None:
        session = session_store.create(req.user_id)

    # If this session is already pending handoff, don't process further
    if session.need_handoff:
        return ChatResponse(
            session_id=session.session_id,
            response="您已进入人工客服排队队列，请耐心等待。如需取消排队，请联系在线客服。",
            need_handoff=True,
            handoff_reason=session.handoff_reason,
        )

    # --- build state ---
    # For multi-turn we seed the return-flow fields from the session so
    # the graph continues where it left off.
    initial_state = _build_initial_state(
        user_id=req.user_id,
        session_id=session.session_id,
        message=req.message,
    )
    initial_state["return_order_id"] = session.return_order_id
    initial_state["return_reason"] = session.return_reason
    initial_state["return_step"] = session.return_step
    initial_state["return_attempts"] = 0  # reset per-turn

    # --- invoke the graph ---
    config: dict[str, Any] = {
        "configurable": {"thread_id": session.session_id}
    }

    try:
        result = await _agent_graph.ainvoke(initial_state, config)
    except Exception as exc:
        logger.exception("Graph invocation failed")
        return ChatResponse(
            session_id=session.session_id,
            response="系统繁忙，正在为您转接人工客服...",
            intent="human_support",
            need_handoff=True,
            handoff_reason=f"Agent异常: {exc}",
        )

    # --- persist session state ---
    final_response = result.get("final_response", "")
    intent = result.get("intent", "")
    handoff_reason = result.get("handoff_reason", "")

    session.messages.append({"role": "user", "content": req.message})
    session.messages.append({"role": "assistant", "content": final_response})
    session.return_order_id = result.get("return_order_id", "")
    session.return_reason = result.get("return_reason", "")
    session.return_step = result.get("return_step", "")

    if handoff_reason:
        session.need_handoff = True
        session.handoff_reason = handoff_reason

    session_store.save(session)

    return ChatResponse(
        session_id=session.session_id,
        response=final_response,
        intent=intent,
        need_handoff=session.need_handoff,
        handoff_reason=handoff_reason or None,
    )


@router.get("/history/{session_id}", response_model=SessionInfo)
async def get_history(session_id: str):
    """Return the full conversation history (and metadata) for a session."""
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return _session_to_info(session)


@router.get("/handoff/queue", response_model=list[SessionInfo])
async def list_handoff_queue():
    """List all sessions currently waiting for a human agent."""
    return [_session_to_info(s) for s in session_store.pending_handoffs]


@router.post("/handoff/{session_id}/pickup", response_model=HandoffPickupResponse)
async def pickup_handoff(session_id: str, req: HandoffPickupRequest):
    """A human agent picks up a pending handoff session."""
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    if not session.need_handoff:
        raise HTTPException(status_code=400, detail="Session is not pending handoff")

    session.need_handoff = False
    session_store.save(session)

    return HandoffPickupResponse(
        session_id=session.session_id,
        user_id=session.user_id,
        messages=session.messages,
        handoff_reason=session.handoff_reason,
        assigned_agent=req.agent_name,
    )


@router.delete("/session/{session_id}")
async def close_session(session_id: str):
    """Close (delete) a session."""
    if session_store.delete(session_id):
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")
