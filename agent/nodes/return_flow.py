"""Return-flow nodes.

The return process is a **multi-turn conversation**.  Each node that needs user
input (order ID, return reason) sets a *terminal* step that routes to ``END`` so
the graph stops and waits for the next user message.  On the next turn the graph
re-enters through ``return_start``, which acts as a dispatcher: it reads the
current ``return_step`` and routes accordingly.

Step progression
----------------
order_extracted → validated → policy_checked → reason_collected → initiated → confirmed

Terminal steps (graph stops, waits for user)
--------------------------------------------
waiting_order_id   — waiting for user to provide order ID
collecting_reason  — waiting for user to explain return reason
confirmed          — return complete
not_eligible       — can't return (politely rejected)
"""

from __future__ import annotations

import logging
import re

from langchain_core.messages import AIMessage

from agent.state import AgentState
from config import settings
from tools.order import lookup_order
from tools.return_policy import check_return_eligibility, create_return

logger = logging.getLogger(__name__)

# ── helpers ──────────────────────────────────────────────────────────────────

# Common order-id patterns: pure digits (12-20 chars) or ORD-xxx
_ORDER_ID_PATTERNS = [
    r"\b\d{12,20}\b",
    r"\bORD\d{8,}\b",
    r"\b[A-Z]{2,4}\d{10,}\b",
]
_ORDER_ID_RE = re.compile("|".join(f"({p})" for p in _ORDER_ID_PATTERNS), re.IGNORECASE)


def _extract_order_id(user_text: str) -> str:
    m = _ORDER_ID_RE.search(user_text)
    return m.group(0) if m else ""


# ── step nodes ───────────────────────────────────────────────────────────────


async def return_start(state: AgentState) -> dict:
    """Dispatcher for the return subgraph — handles every turn's entry.

    Logic
    -----
    1. If we already have an ``order_id`` from a previous turn, skip extraction
       and dispatch according to the current ``return_step``.
    2. Otherwise, try to extract an order ID from the latest message.
    3. If still missing, ask the user and set a terminal step (→ END).
    """
    messages = state["messages"]
    last_msg = messages[-1]
    user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    existing_oid = state.get("return_order_id", "")
    current_step = state.get("return_step", "")

    # ── already have order ID — dispatch based on where we left off ────────
    if existing_oid:
        if current_step == "waiting_order_id":
            return {"return_step": "order_extracted"}
        if current_step == "collecting_reason":
            # User just replied with the reason — re-enter the collect node
            return {"return_step": "need_reason"}
        # Other steps: pass through, routing handles the rest
        return {}

    # ── try to extract order ID from this message ───────────────────────────
    order_id = _extract_order_id(user_text)
    if order_id:
        logger.info("Extracted order ID from message: %s", order_id)
        return {"return_order_id": order_id, "return_step": "order_extracted"}

    # ── still no order ID → ask user ────────────────────────────────────────
    return {
        "return_step": "waiting_order_id",
        "final_response": "好的，我来帮您处理退货。请提供您的订单号（可以在「我的订单」中找到）。",
        "messages": [
            AIMessage(content="好的，我来帮您处理退货。请提供您的订单号（可以在「我的订单」中找到）。")
        ],
    }


async def return_validate_order(state: AgentState) -> dict:
    """Check that the order exists and belongs to the current user."""
    order_id = state.get("return_order_id", "")
    user_id = state.get("user_id", "")
    attempts = state.get("return_attempts", 0)

    order = await lookup_order(order_id, user_id)

    if order:
        logger.info("Order %s validated for user %s", order_id, user_id)
        return {"return_step": "validated", "return_attempts": 0}

    # ── invalid ─────────────────────────────────────────────────────────────
    attempts += 1
    max_attempts = settings.max_return_attempts

    if attempts >= max_attempts:
        logger.warning("Order validation failed %d times — handing off", attempts)
        return {
            "return_step": "failed",
            "return_attempts": attempts,
            "handoff_reason": f"订单验证失败{attempts}次，订单号: {order_id}",
            "final_response": "抱歉，我们多次验证您的订单信息均未成功。正在为您转接人工客服，请稍候...",
            "messages": [
                AIMessage(content="抱歉，我们多次验证您的订单信息均未成功。正在为您转接人工客服，请稍候...")
            ],
        }

    # Allow retry — ask user to re-enter and stop the graph
    return {
        "return_step": "waiting_order_id",
        "return_attempts": attempts,
        "return_order_id": "",  # clear the bad one so return_start re-extracts
        "final_response": (
            f"未查询到您的订单号 {order_id}，请确认订单号是否正确并重新输入"
            f"（剩余尝试次数 {max_attempts - attempts} 次）。"
        ),
        "messages": [
            AIMessage(
                content=f"未查询到您的订单号 {order_id}，请确认订单号是否正确并重新输入"
                f"（剩余尝试次数 {max_attempts - attempts} 次）。"
            )
        ],
    }


async def return_check_policy(state: AgentState) -> dict:
    """Check whether the order is eligible for return."""
    order_id = state["return_order_id"]
    eligible, reason = await check_return_eligibility(order_id)

    if eligible:
        return {"return_eligible": True, "return_step": "policy_checked"}
    else:
        return {
            "return_eligible": False,
            "return_step": "not_eligible",
            "final_response": (
                f"抱歉，订单 {order_id} 暂不支持退货。原因：{reason}。"
                f"如有疑问，可联系人工客服。"
            ),
            "messages": [
                AIMessage(
                    content=f"抱歉，订单 {order_id} 暂不支持退货。原因：{reason}。"
                    f"如有疑问，可联系人工客服。"
                )
            ],
        }


async def return_collect_reason(state: AgentState) -> dict:
    """Collect the return reason from the user, or ask for it."""
    messages = state["messages"]
    last_msg = messages[-1]
    user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    # If we already collected the reason, pass through
    if state.get("return_reason", ""):
        return {"return_step": "reason_collected"}

    # Try to extract a meaningful reason from user message
    has_order_id = _extract_order_id(user_text)
    # Use the message as reason if it's substantive (not just the order ID)
    if len(user_text) >= 3 and "退货" not in user_text and not has_order_id:
        return {"return_reason": user_text, "return_step": "reason_collected"}

    # Ask for reason — terminal step
    return {
        "return_step": "collecting_reason",
        "final_response": (
            "为了更好地为您处理，请告诉我退货原因"
            "（例如：商品与描述不符、质量问题、不想要了等）。"
        ),
        "messages": [
            AIMessage(
                content="为了更好地为您处理，请告诉我退货原因"
                "（例如：商品与描述不符、质量问题、不想要了等）。"
            )
        ],
    }


async def return_initiate(state: AgentState) -> dict:
    """Create the return record."""
    order_id = state["return_order_id"]
    reason = state.get("return_reason", "用户未说明")
    user_id = state["user_id"]

    result = await create_return(order_id, user_id, reason)

    if result.get("success"):
        return {"return_step": "initiated", "return_order_id": order_id}
    else:
        error_msg = result.get("message", "系统错误")
        return {
            "return_step": "failed",
            "handoff_reason": f"创建退货单失败: {error_msg}",
            "final_response": (
                f"抱歉，创建退货单时出现问题（{error_msg}）。"
                f"正在为您转接人工客服处理..."
            ),
            "messages": [
                AIMessage(
                    content=f"抱歉，创建退货单时出现问题（{error_msg}）。"
                    f"正在为您转接人工客服处理..."
                )
            ],
        }


async def return_confirm(state: AgentState) -> dict:
    """Confirm the return to the user."""
    order_id = state["return_order_id"]
    reason = state.get("return_reason", "")

    msg = (
        f"退货申请已提交成功！\n\n"
        f"📦 订单号：{order_id}\n"
        f"📝 退货原因：{reason}\n\n"
        f"接下来请按以下步骤操作：\n"
        f"1. 将商品及配件、赠品、包装完整装回\n"
        f"2. 我们会在1-2个工作日内安排快递员上门取件\n"
        f"3. 仓库收到退货后将在3-5个工作日内完成退款\n\n"
        f"您可以在「我的订单」中随时查看退货进度。如有问题请联系人工客服。"
    )

    return {
        "return_step": "confirmed",
        "final_response": msg,
        "messages": [AIMessage(content=msg)],
    }
