"""E-commerce Customer Service Agent — FastAPI entry point.

Start the server::

    uvicorn main:app --reload
"""

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router as api_router
from config import settings

# ── logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── app ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="电商客服 Agent",
    description="基于 LangGraph 的智能电商客服系统，支持 RAG 问答、退货流程、转人工",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(api_router)

# Global exception handler — guarantee JSON even on unhandled errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "session_id": "",
            "response": "系统繁忙，请稍后重试。如问题持续，请联系人工客服。",
            "intent": "human_support",
            "need_handoff": True,
            "handoff_reason": f"系统异常: {type(exc).__name__}",
        },
    )

# Static files (front-end test console)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    """Redirect to the test console."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")


@app.get("/health")
async def health():
    """Health check — includes LLM and RAG connectivity status."""
    from services.llm import llm_health_check
    from services.rag_client import rag_health_check

    llm_ok = await llm_health_check()
    rag_ok = await rag_health_check()

    overall = "ok" if llm_ok and rag_ok else "degraded"
    return {
        "status": overall,
        "service": "customer-service-agent",
        "checks": {
            "llm": {"ok": llm_ok, "url": settings.llm_base_url, "model": settings.llm_model},
            "rag": {"ok": rag_ok, "url": settings.rag_base_url},
        },
    }


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting server on %s:%s", settings.host, settings.port)
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)
