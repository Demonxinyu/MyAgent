"""E-commerce Customer Service Agent — FastAPI entry point.

Start the server::

    uvicorn main:app --reload
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


@app.get("/health")
async def health():
    return {"status": "ok", "service": "customer-service-agent"}


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting server on %s:%s", settings.host, settings.port)
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)
