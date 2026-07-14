"""LLM service — thin wrapper around an OpenAI-compatible chat API."""

import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton LLM instance (lazy init)
# ---------------------------------------------------------------------------

_llm: ChatOpenAI | None = None


def get_llm() -> ChatOpenAI:
    """Return a cached ChatOpenAI instance configured from settings."""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            temperature=0.3,
            max_tokens=1024,
        )
    return _llm


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------

async def llm_classify(prompt: str, user_message: str) -> dict[str, Any]:
    """Classify user intent.  Returns a dict with ``intent`` and ``reason``."""
    llm = get_llm()
    full_prompt = f"{prompt}\n\n用户消息: {user_message}"

    try:
        response = await llm.ainvoke(full_prompt)
        content = response.content.strip()
        # Strip markdown fences if present
        if content.startswith("```"):
            content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(content)
        return result
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("Intent classification failed: %s", exc)
        return {"intent": "human_support", "reason": f"分类异常: {exc}"}


async def llm_generate(system_prompt: str, context: str, user_message: str) -> str:
    """Generate a customer-service reply given RAG context."""
    llm = get_llm()
    messages = [
        ("system", system_prompt),
        ("system", f"参考知识: {context}"),
        ("user", user_message),
    ]

    try:
        response = await llm.ainvoke(messages)
        return response.content.strip()
    except Exception as exc:
        logger.error("LLM generation failed: %s", exc)
        return "抱歉，我暂时无法处理您的问题，正在为您转接人工客服..."
