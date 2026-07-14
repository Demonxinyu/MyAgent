"""Human-handoff node.

Marks the session as needing a human agent and returns a polite message
to the user while they wait.
"""

import logging

from langchain_core.messages import AIMessage

from agent.state import AgentState
from services.session_store import session_store

logger = logging.getLogger(__name__)

HANDOFF_MESSAGE = """已为您转接人工客服，请稍候。

⏳ 当前排队人数：{queue_position} 人
⏱ 预计等待时间：约 {wait_minutes} 分钟

您也可以留言，客服上线后会第一时间回复您。感谢您的耐心等待！"""


async def human_handoff(state: AgentState) -> dict:
    """Save the session to the handoff queue and notify the user."""
    session_id = state.get("session_id", "")
    handoff_reason = state.get("handoff_reason", "用户请求人工服务")

    # Mark the session for handoff
    session = session_store.get(session_id)
    if session:
        session.need_handoff = True
        session.handoff_reason = handoff_reason
        session_store.save(session)
    else:
        logger.warning("Session %s not found for handoff", session_id)

    # Estimate queue position (all currently pending sessions)
    queue_pos = len(session_store.pending_handoffs)
    wait_minutes = max(1, queue_pos * 2)  # rough estimate

    response_text = HANDOFF_MESSAGE.format(
        queue_position=queue_pos,
        wait_minutes=wait_minutes,
    )

    logger.info(
        "Session %s handed off — reason: %s, queue_pos: %s",
        session_id, handoff_reason, queue_pos,
    )

    return {
        "final_response": response_text,
        "handoff_reason": handoff_reason,
        "messages": [AIMessage(content=response_text)],
    }
