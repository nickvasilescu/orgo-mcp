"""Account tools — user profile, credits, and transactions."""

import json

from orgo_mcp.server import mcp
from orgo_mcp.auth import get_current_api_key
from orgo_mcp.client import api_request
from orgo_mcp.errors import handle_orgo_error


@mcp.tool(
    name="orgo_get_profile",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def orgo_get_profile() -> str:
    """Get the current user's profile: ID, email, name, avatar, and subscription tier."""
    try:
        api_key = get_current_api_key(mcp)
        profile = await api_request("GET", "user/profile", api_key)
        subscription = await api_request("GET", "user/subscription", api_key)
        profile["tier"] = subscription.get("tier", "free")
        return json.dumps(profile, indent=2)
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_get_credits",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def orgo_get_credits() -> str:
    """Get the current credit balance and subscription tier. Balance is in cents (e.g. 5000 = $50.00)."""
    try:
        api_key = get_current_api_key(mcp)
        data = await api_request("GET", "credits", api_key)
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_get_transactions",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def orgo_get_transactions() -> str:
    """Get credit transaction history: purchases, usage, and balance changes."""
    try:
        api_key = get_current_api_key(mcp)
        data = await api_request("GET", "credits/transactions", api_key)
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)
