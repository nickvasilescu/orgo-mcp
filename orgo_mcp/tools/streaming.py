"""RTMP streaming tools — stream computer display to Twitch, YouTube, etc.

Requires pre-configured RTMP connections in your Orgo account settings.
"""

import json

from orgo_mcp.server import mcp
from orgo_mcp.auth import get_current_api_key
from orgo_mcp.client import computer_action
from orgo_mcp.errors import handle_orgo_error
from orgo_mcp.models import StartStreamInput, ComputerIdInput


@mcp.tool(
    name="orgo_start_stream",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def orgo_start_stream(params: StartStreamInput) -> str:
    """Start RTMP streaming from a computer to a pre-configured connection. One stream per computer."""
    try:
        api_key = get_current_api_key(mcp)
        data = await computer_action(
            "POST", params.computer_id, "stream/start", api_key,
            json={"connection_name": params.connection_name},
        )
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_stream_status",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def orgo_stream_status(params: ComputerIdInput) -> str:
    """Get current streaming status: idle, streaming, or terminated."""
    try:
        api_key = get_current_api_key(mcp)
        data = await computer_action("GET", params.computer_id, "stream/status", api_key)
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_stop_stream",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def orgo_stop_stream(params: ComputerIdInput) -> str:
    """Stop an active RTMP stream."""
    try:
        api_key = get_current_api_key(mcp)
        data = await computer_action("POST", params.computer_id, "stream/stop", api_key)
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)
