"""Shell command execution tools — bash and Python."""

import json

from orgo_mcp.server import mcp
from orgo_mcp.auth import get_current_api_key
from orgo_mcp.client import computer_action
from orgo_mcp.errors import handle_orgo_error
from orgo_mcp.models import BashInput, ExecInput


@mcp.tool(
    name="orgo_bash",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": True},
)
async def orgo_bash(params: BashInput) -> str:
    """Execute a bash command on the computer. Returns stdout+stderr output."""
    try:
        api_key = get_current_api_key(mcp)
        data = await computer_action("POST", params.computer_id, "bash", api_key,
                                     json={"command": params.command})
        output = data.get("output", "")
        return f"$ {params.command}\n\n{output}"
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_exec",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": True},
)
async def orgo_exec(params: ExecInput) -> str:
    """Execute Python code on the computer. Returns output or error details."""
    try:
        api_key = get_current_api_key(mcp)
        data = await computer_action("POST", params.computer_id, "exec", api_key,
                                     json={"code": params.code, "timeout": params.timeout})
        return json.dumps(data, indent=2) if isinstance(data, dict) else str(data)
    except Exception as e:
        return handle_orgo_error(e)
