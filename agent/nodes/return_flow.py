"""Return-flow nodes.

The return process has these steps:

1. **return_start**        — greet the user, ask for order ID if missing
2. **return_validate_order** — call order service to verify the order
3. **return_check_policy**   — check return-eligibility against policy
4. **return_collect_reason** — ask why the user wants to return
5. **return_initiate**       — create the return record
6. **return_confirm**        — give the user next steps

Any step that fails fatally routes to ``human_handoff``.
"""

import logging

from langchain_core.messages import AIMessage

from agent.state import AgentState
from config import settings
from tools.order import lookup_order
from tools.return_policy import check_return_eligibility, create_return

logger = logging.getLogger(__name__)

# ── helpers ──────────────────────────────────────────────────────────────────

EXTRACT_ORDER_PROMPT = """从用户消息中提取订单号。只回复订单号，不要任何其他文字。
如果找不到订单号，回复 "NOT_FOUND"。

用户消息: {message}"""


async def _extract_order_id(user_text: str) -> str:
    """Minimal extraction: look for numbers with 12+ digits or common patterns.

    In production you'd use the LLM for this, but a regex fast-path is fine here.
    """
    import re

    # Match common order-id patterns: pure digits (12-20 chars) or ORD-xxx
    patterns = [
        r"\b\d{12,20}\b",
        r"\bORD\d{8,}\b",
        r"\b[A-Z]{2,4}\d{10,}\b",
    ]
    for pat in patterns:
        m = re.search(pat, user_text, re.IGNORECASE)
        if m:
            return m.group(0)
    return ""


# ── step nodes ───────────────────────────────────────────────────────────────


async def return_start(state: AgentState) -> dict:
    """Entry-point for the return subgraph.

    If the user already provided an order ID we seed it; otherwise we ask.
    """
    messages = state["messages"]
    last_msg = messages[-1]
    user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    order_id = _extract_order_id(user_text)
    if order_id:
        return {
            "return_order_id": order_id,
            "return_step": "start",
            "return_attempts": state.get("return_attempts", 0),
        }

    # Ask for order ID
    return {
        "return_step": "start",
        "return_attempts": state.get("return_attempts", 0),
        "final_response": "好的，我来帮您处理退货。请提供您的订单号（可以在「我的订单」中找到）。",
        "messages": [AIMessage(content="好的，我来帮您处理退货。请提供您的订单号（可以在「我的订单」中找到）。")],
    }


async def return_validate_order(state: AgentState) -> dict:
    """Check whether the order exists and belongs to the current user.

    Routes
    ------
    * Order valid → continue to policy check.
    * Order invalid → retry or handoff.
    * No order ID yet → keep asking (the user may reply with it next turn).
    """
    order_id = state.get("return_order_id", "")
    user_id = state.get("user_id", "")
    attempts = state.get("return_attempts", 0)

    if not order_id:
        # User hasn't provided an order ID yet; stay in start phase
        return {"return_step": "start"}

    order = await lookup_order(order_id, user_id)

    if order:
        logger.info("Order %s validated for user %s", order_id, user_id)
        return {"return_step": "validated"}
    else:
        attempts += 1
        max_attempts = settings.max_return_attempts
        if attempts >= max_attempts:
            logger.warning("Order validation failed %d times — handing off", attempts)
            return {
                "return_step": "failed",
                "return_attempts": attempts,
                "handoff_reason": f"订单验证失败{attempts}次，订单号: {order_id}",
                "final_response": "抱歉，我们多次验证您的订单信息均未成功。正在为您转接人工客服，请稍候...",
                "messages": [AIMessage(content="抱歉，我们多次验证您的订单信息均未成功。正在为您转接人工客服，请稍候...")],
            }
        else:
            return {
                "return_step": "start",
                "return_attempts": attempts,
                "final_response": f"没有找到订单 {order_id}，请确认订单号是否正确并重新输入（剩余尝试次数 {max_attempts - attempts} 次）。",
                "messages": [AIMessage(content=f"没有找到订单 {order_id}，请确认订单号是否正确并重新输入（剩余尝试次数 {max_attempts - attempts} 次）。")],
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
            "final_response": f"抱歉，订单 {order_id} 暂不支持退货。原因：{reason}。如有疑问，可联系人工客服。",
            "messages": [AIMessage(content=f"抱歉，订单 {order_id} 暂不支持退货。原因：{reason}。如有疑问，可联系人工客服。")],
        }


async def return_collect_reason(state: AgentState) -> dict:
    """Ask the user for the return reason, or extract it from their message."""
    messages = state["messages"]
    last_msg = messages[-1]
    user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    existing_reason = state.get("return_reason", "")
    if existing_reason:
        return {"return_step": "reason_collected"}

    # Simple heuristic: if there's a substantive message, use it as reason
    if len(user_text) >= 3 and "退货" not in user_text:
        return {"return_reason": user_text, "return_step": "reason_collected"}

    # Otherwise ask
    return {
        "return_step": "collecting_reason",
        "final_response": "为了更好地为您处理，请告诉我退货原因（例如：商品与描述不符、质量问题、不想要了等）。",
        "messages": [AIMessage(content="为了更好地为您处理，请告诉我退货原因（例如：商品与描述不符、质量问题、不想要了等）。")],
    }


async def return_initiate(state: AgentState) -> dict:
    """Create the return record via the order service."""
    order_id = state["return_order_id"]
    reason = state.get("return_reason", "用户未说明")
    user_id = state["user_id"]

    result = await create_return(order_id, user_id, reason)

    if result.get("success"):
        return {
            "return_step": "initiated",
            "return_order_id": order_id,
        }
    else:
        error_msg = result.get("message", "系统错误")
        return {
            "return_step": "failed",
            "handoff_reason": f"创建退货单失败: {error_msg}",
            "final_response": f"抱歉，创建退货单时出现问题（{error_msg}）。正在为您转接人工客服处理...",
            "messages": [AIMessage(content=f"抱歉，创建退货单时出现问题（{error_msg}）。正在为您转接人工客服处理...")],
        }


async def return_confirm(state: AgentState) -> dict:
    """Confirm the return to the user and provide next steps."""
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
