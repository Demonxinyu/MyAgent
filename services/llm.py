"""LLM service — wraps an OpenAI-compatible chat API.

Classification strategy (auto-selects on first call):

1. **Native** — ``with_structured_output()`` via function calling.  Zero-parse,
   always well-typed.  Used when the provider supports it (OpenAI, Claude, etc.).

2. **Fallback** — ``PydanticOutputParser`` injects the JSON schema into the
   prompt and parses the raw text output.  Used for providers that don't
   support native structured output (e.g. DeepSeek).
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

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
            timeout=30,
            max_retries=1,
        )
    return _llm


# ---------------------------------------------------------------------------
# Structured output models
# ---------------------------------------------------------------------------


class Intent(str, Enum):
    GENERAL_QA = "general_qa"
    RETURN_REQUEST = "return_request"
    HUMAN_SUPPORT = "human_support"


class IntentClassification(BaseModel):
    """Result of classifying a customer-service message."""

    intent: Intent = Field(
        description=(
            "general_qa: 商品/价格/规格/库存/使用说明/售后政策等知识类问题; "
            "return_request: 退货/退款/换货/取消订单; "
            "human_support: 要求转人工/找真人/投诉, 或包含辱骂/威胁/强烈不满情绪"
        ),
    )
    reason: str = Field(description="一句话说明为什么是这个意图")


# ---------------------------------------------------------------------------
# Auto-detection: native structured output vs prompt-fallback
# ---------------------------------------------------------------------------

_parser: PydanticOutputParser | None = None


def _get_parser() -> PydanticOutputParser:
    global _parser
    if _parser is None:
        _parser = PydanticOutputParser(pydantic_object=IntentClassification)
    return _parser


# After the first failure we remember to skip native mode for this provider
_use_native: bool | None = None  # None = not tested yet, True/False = decided


async def _llm_classify_native(user_text: str) -> IntentClassification:
    """Via function-calling (requires provider support)."""
    llm = get_llm()
    structured_llm = llm.with_structured_output(IntentClassification)
    result = await structured_llm.ainvoke(user_text)

    if isinstance(result, IntentClassification):
        return result
    if isinstance(result, dict):
        return IntentClassification(**result)
    raise TypeError(f"Unexpected type from structured output: {type(result).__name__}")


async def _llm_classify_fallback(user_text: str) -> IntentClassification:
    """Via prompt-injected JSON schema + manual parse (works with any model)."""
    parser = _get_parser()
    llm = get_llm()

    prompt = (
        "你是电商客服意图分类器。分析用户消息并输出一个JSON对象。\n\n"
        f"{parser.get_format_instructions()}\n\n"
        "只输出JSON，不要输出任何其他内容。\n\n"
        f"用户消息: {user_text}"
    )

    response = await llm.ainvoke(prompt)
    raw = response.content.strip()

    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        parsed = json.loads(raw)
        return IntentClassification(**parsed)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        # Last resort: try to extract just the JSON object from the response
        m = re.search(r"\{[^{}]*\}", raw)
        if m:
            try:
                parsed = json.loads(m.group(0))
                return IntentClassification(**parsed)
            except Exception:
                pass
        # Truly unparseable — safe fallback
        logger.warning("Fallback JSON parse failed: %s. Raw: %s", exc, raw[:200])
        return IntentClassification(
            intent=Intent.HUMAN_SUPPORT,
            reason=f"LLM返回格式异常, 原始内容: {raw[:80]}",
        )


async def llm_classify(user_text: str) -> IntentClassification:
    """Classify user intent.  Auto-detects native vs fallback mode.

    Returns an ``IntentClassification`` — **always** valid, even on error.
    """
    global _use_native

    # ── try native (first call) ───────────────────────────────────────
    if _use_native is None or _use_native is True:
        try:
            result = await _llm_classify_native(user_text)
            if _use_native is None:
                _use_native = True
                logger.info("Using native structured output for classification")
            return result
        except Exception as exc:
            if _use_native is None:
                logger.info(
                    "Native structured output unavailable (%s: %s) — switching to prompt fallback",
                    type(exc).__name__, exc,
                )
                _use_native = False
            else:
                logger.warning("Native classification failed: %s", exc)

    # ── fallback ─────────────────────────────────────────────────────
    try:
        return await _llm_classify_fallback(user_text)
    except Exception as exc:
        logger.warning("Fallback classification also failed (%s): %s", type(exc).__name__, exc)
        return IntentClassification(
            intent=Intent.HUMAN_SUPPORT,
            reason=f"分类失败: {type(exc).__name__}",
        )


# ── response generation ─────────────────────────────────────────────────────


async def llm_generate(system_prompt: str, context: str, user_message: str) -> str:
    """Generate a customer-service reply, with RAG context when available."""
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
        logger.error("LLM generation failed (%s): %s", type(exc).__name__, exc)
        return "抱歉，我暂时无法处理您的问题，正在为您转接人工客服..."


async def llm_health_check() -> bool:
    """Quick check that the LLM API is reachable."""
    llm = get_llm()
    try:
        response = await llm.ainvoke("回复: ok")
        return bool(response.content)
    except Exception:
        return False
