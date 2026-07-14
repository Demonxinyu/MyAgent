"""Intent-classification node."""

import logging

from langchain_core.messages import AIMessage

from agent.state import AgentState
from services.llm import llm_classify

logger = logging.getLogger(__name__)

CLASSIFY_PROMPT = """你是电商客服意图分类器。分析用户消息，只输出一个JSON对象，不要输出任何其他内容。

{
  "intent": "general_qa|return_request|human_support",
  "reason": "简短分类理由"
}

意图规则：
- general_qa: 咨询商品信息、价格、规格、库存、使用说明、售后政策等知识类问题
- return_request: 用户想退货、退款、换货、取消订单；或询问退货流程
- human_support: 用户明确要求"转人工""人工客服""找真人""投诉"；或消息中包含辱骂、威胁、强烈不满情绪
"""

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

    # Fast-path: explicit handoff keywords → bypass LLM
    if _check_handoff_keywords(user_text):
        return {
            "intent": "human_support",
            "handoff_reason": "用户明确要求转人工",
        }

    result = await llm_classify(CLASSIFY_PROMPT, user_text)
    intent = result.get("intent", "human_support")
    reason = result.get("reason", "")

    logger.info("Intent: %s | reason: %s", intent, reason)

    updates: dict = {"intent": intent}
    if intent == "human_support":
        updates["handoff_reason"] = reason or "系统判定需转人工"
    return updates
