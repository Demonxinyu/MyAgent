"""Async HTTP client for the external RAG knowledge-base service."""

import logging
from typing import Any

import httpx

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expected RAG API contract
#
#   POST {base_url}/api/retrieve
#   Body: {"query": "...", "top_k": N}
#   Response: {"documents": [{"content": "...", "score": 0.95}, ...]}
# ---------------------------------------------------------------------------

RAG_TIMEOUT = 15.0  # seconds


async def retrieve_knowledge(query: str, top_k: int = 5) -> str:
    """Call the external RAG service and return concatenated document texts.

    Returns an empty string when the RAG service is unreachable.
    """
    url = f"{settings.rag_base_url.rstrip('/')}/api/retrieve"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if settings.rag_api_key:
        headers["Authorization"] = f"Bearer {settings.rag_api_key}"

    payload: dict[str, Any] = {"query": query, "top_k": top_k}

    try:
        async with httpx.AsyncClient(timeout=RAG_TIMEOUT) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        logger.error("RAG HTTP error: %s", exc)
        return ""
    except Exception as exc:
        logger.error("RAG unexpected error: %s", exc)
        return ""

    documents: list[dict[str, Any]] = data.get("documents", [])
    if not documents:
        logger.info("RAG returned no documents for query: %s", query)
        return ""

    # Concatenate documents with separators
    parts = [doc.get("content", "") for doc in documents if doc.get("content")]
    return "\n---\n".join(parts)


async def rag_health_check() -> bool:
    """Return True when the RAG service responds OK."""
    url = f"{settings.rag_base_url.rstrip('/')}/health"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            return resp.status_code == 200
    except Exception:
        return False
