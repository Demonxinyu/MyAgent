"""Return-policy utilities.

Production version would query an order-service API and / or a policy engine.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── policy rules ─────────────────────────────────────────────────────────────

RETURN_WINDOW_DAYS = 7  # days after delivery
NON_RETURNABLE_CATEGORIES = {"digital_goods", "gift_cards", "personal_care_opened"}


async def check_return_eligibility(order_id: str) -> tuple[bool, str]:
    """Check whether *order_id* is eligible for return.

    Returns
    -------
    (eligible: bool, reason: str)
    """
    # Stub: in production, look up order details and apply policy rules.
    # Here we just mark everything as eligible.
    logger.info("Checking return eligibility for %s → eligible", order_id)
    return True, ""


async def create_return(
    order_id: str, user_id: str, reason: str
) -> dict[str, Any]:
    """Create a return record in the back-end system.

    Returns
    -------
    {"success": bool, "return_id"?: str, "message"?: str}
    """
    logger.info("Creating return for order %s, user %s, reason: %s", order_id, user_id, reason)
    # Stub: always succeeds
    return {
        "success": True,
        "return_id": f"RET_{order_id}",
    }
