"""Shell command execution tools — bash and Python."""

import json
import asyncio

from orgo_mcp.server import mcp
from orgo_mcp.auth import get_current_api_key
from orgo_mcp.client import get_computer
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

        def run():
            computer = get_computer(params.computer_id, api_key)
            output = computer.bash(params.command)
            return f"$ {params.command}\n\n{output}"

        return await asyncio.to_thread(run)
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

        def run():
            computer = get_computer(params.computer_id, api_key)
            result = computer.exec(params.code, timeout=params.timeout)
            return json.dumps(result, indent=2) if isinstance(result, dict) else str(result)

        return await asyncio.to_thread(run)
    except Exception as e:
        return handle_orgo_error(e)
