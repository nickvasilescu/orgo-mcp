"""Thread management tools for multi-turn agent conversations.

Threads persist context across multiple instructions to the same computer.
The AI remembers previous actions so you can build on prior steps.
"""

import json

from orgo_mcp.server import mcp
from orgo_mcp.auth import get_current_api_key
from orgo_mcp.client import api_request, ORGO_V1_BASE
from orgo_mcp.errors import handle_orgo_error
from orgo_mcp.models import ListThreadsInput, ThreadIdInput


@mcp.tool(
    name="orgo_list_threads",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def orgo_list_threads(params: ListThreadsInput) -> str:
    """List conversation threads for a computer. Returns thread IDs, titles, and message counts."""
    try:
        api_key = get_current_api_key(mcp)
        data = await api_request(
            "GET", "threads", api_key,
            params={"computer_id": params.computer_id},
            base_url=ORGO_V1_BASE,
        )
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_get_thread",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def orgo_get_thread(params: ThreadIdInput) -> str:
    """Get a thread with its full message history."""
    try:
        api_key = get_current_api_key(mcp)
        data = await api_request(
            "GET", f"threads/{params.thread_id}", api_key,
            base_url=ORGO_V1_BASE,
        )
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_delete_thread",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def orgo_delete_thread(params: ThreadIdInput) -> str:
    """Delete a conversation thread."""
    try:
        api_key = get_current_api_key(mcp)
        data = await api_request(
            "DELETE", f"threads/{params.thread_id}", api_key,
            base_url=ORGO_V1_BASE,
        )
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)
