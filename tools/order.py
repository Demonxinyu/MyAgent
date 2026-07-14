"""Order-related utilities.

In production these would call an internal order-management HTTP API or
database.  They're stubbed here for demonstration.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── stub data (replace with real API calls) ──────────────────────────────────

_MOCK_ORDERS: dict[str, dict[str, Any]] = {
    "ORD20240701001": {
        "order_id": "ORD20240701001",
        "user_id": "user_001",
        "status": "delivered",
        "items": [{"sku": "PHONE_X1", "name": "XPhone X1 256GB", "price": 4999}],
        "delivered_at": "2024-07-05T10:30:00Z",
    },
    "20240701000001": {
        "order_id": "20240701000001",
        "user_id": "user_001",
        "status": "delivered",
        "items": [{"sku": "HEADPHONE_A1", "name": "无线蓝牙耳机", "price": 399}],
        "delivered_at": "2024-07-03T14:00:00Z",
    },
}


async def lookup_order(order_id: str, user_id: str) -> dict[str, Any] | None:
    """Look up an order by ID and verify it belongs to ``user_id``.

    Returns the order dict on success, ``None`` when not found or mismatched.
    """
    logger.info("Looking up order %s for user %s", order_id, user_id)
    order = _MOCK_ORDERS.get(order_id.upper())
    if order is None:
        logger.info("Order %s not found", order_id)
        return None
    if order["user_id"] != user_id:
        logger.info("Order %s does not belong to user %s", order_id, user_id)
        return None
    return order
