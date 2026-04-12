"""Access tools — VNC password for direct connection."""

import json

from orgo_mcp.server import mcp
from orgo_mcp.auth import get_current_api_key, resolve_computer_id
from orgo_mcp.client import api_request
from orgo_mcp.errors import handle_orgo_error
from orgo_mcp.models import ComputerIdInput


@mcp.tool(
    name="orgo_vnc_password",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def orgo_vnc_password(params: ComputerIdInput) -> str:
    """Get the VNC password for direct connection to a computer's display.
    Used for VNC clients, terminal WebSocket, and the orgo-vnc React component."""
    try:
        api_key = get_current_api_key(mcp)
        computer_id = resolve_computer_id(params.computer_id)
        data = await api_request("GET", f"computers/{computer_id}/vnc-password", api_key)
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)
