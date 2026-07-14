"""RAG-retrieval node — calls the external knowledge-base HTTP API."""

import logging

from langchain_core.messages import AIMessage

from agent.state import AgentState
from services.rag_client import retrieve_knowledge, rag_health_check

logger = logging.getLogger(__name__)


async def rag_retrieve(state: AgentState) -> dict:
    """Fetch relevant documents from the external RAG service.

    If the RAG service is unavailable the node sets ``handoff_reason`` so the
    router can optionally escalate to a human.
    """
    messages = state["messages"]
    last_msg = messages[-1]
    user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    # Quick health check first
    healthy = await rag_health_check()
    if not healthy:
        logger.warning("RAG service unavailable — will handoff")
        return {
            "rag_context": "",
            "handoff_reason": "RAG知识库服务不可用",
        }

    context = await retrieve_knowledge(user_text, top_k=5)
    logger.info("RAG returned %d chars of context", len(context))

    return {"rag_context": context}
