"""Intent-classification node.

Classification is done via LangChain's ``with_structured_output()`` (native
function calling), so the return value is a validated Pydantic model —
never a malformed JSON string.
"""

import logging

from agent.state import AgentState
from services.llm import Intent, IntentClassification, llm_classify

logger = logging.getLogger(__name__)

# Keywords that strongly suggest human handoff
HANDOFF_KEYWORDS = [
    "转人工", "人工客服", "找真人", "找活人", "叫你们经理",
    "投诉你", "投诉你们", "我要投诉", "12315",
]


def _check_handoff_keywords(text: str) -> bool:
    return any(kw in text for kw in HANDOFF_KEYWORDS)


async def classify_intent(state: AgentState) -> dict:
    """Analyse the latest user message, set ``intent`` in state."""
    messages = state["messages"]
    if not messages:
        return {"intent": "general_qa"}

    last_msg = messages[-1]
    user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    # ── fast-path: handoff keywords → bypass LLM ──────────────────────────
    if _check_handoff_keywords(user_text):
        return {
            "intent": "human_support",
            "handoff_reason": "用户明确要求转人工",
        }

    # ── fast-path: active return flow + short input → follow-up reply ─────
    active_return_step = state.get("return_step", "")
    if active_return_step in ("waiting_order_id", "collecting_reason"):
        if len(user_text) <= 50:
            logger.info("Active return flow (%s) + short msg — forcing return_request", active_return_step)
            return {"intent": "return_request"}
        # Long message during active flow → probably a new topic, fall through to LLM

    # ── structured LLM classification ────────────────────────────────────
    result: IntentClassification = await llm_classify(user_text)
    intent = result.intent.value  # Enum → string
    reason = result.reason

    logger.info("Intent: %s | reason: %s", intent, reason)

    updates: dict = {"intent": intent}
    if intent == "human_support":
        updates["handoff_reason"] = reason or "系统判定需转人工"
    return updates
