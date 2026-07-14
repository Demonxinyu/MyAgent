"""Pydantic request / response schemas for the customer-service API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier", examples=["user_001"])
    message: str = Field(..., description="User's message text", examples=["我要退货"])
    session_id: str | None = Field(
        default=None,
        description="Existing session ID for multi-turn; omit to create a new session",
    )


class ChatResponse(BaseModel):
    session_id: str
    response: str
    intent: str | None = None
    need_handoff: bool = False
    handoff_reason: str | None = None


class SessionInfo(BaseModel):
    session_id: str
    user_id: str
    messages: list[dict[str, Any]] = []
    need_handoff: bool = False
    handoff_reason: str | None = None
    return_order_id: str | None = None
    return_reason: str | None = None
    return_step: str | None = None


class HandoffPickupRequest(BaseModel):
    agent_name: str = Field(..., description="Name of the human agent picking up")


class HandoffPickupResponse(BaseModel):
    session_id: str
    user_id: str
    messages: list[dict[str, Any]]
    handoff_reason: str | None = None
    assigned_agent: str
