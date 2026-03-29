"""Computer management tools.

Computers are virtual machines within workspaces. They boot in under 500ms
and can be controlled via actions, shell commands, or AI agents.
"""

import json
import asyncio

from orgo_mcp.server import mcp
from orgo_mcp.auth import get_current_api_key
from orgo_mcp.client import api_request, get_sdk_client
from orgo_mcp.errors import handle_orgo_error
from orgo_mcp.models import ListComputersInput, CreateComputerInput, ComputerIdInput


@mcp.tool(
    name="orgo_list_computers",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def orgo_list_computers(params: ListComputersInput) -> str:
    """List all computers in a workspace. Returns IDs, names, status, specs."""
    try:
        api_key = get_current_api_key(mcp)
        data = await api_request("GET", f"workspaces/{params.workspace_id}", api_key)
        computers = data.get("computers", [])
        return json.dumps({"computers": computers, "count": len(computers)}, indent=2)
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_create_computer",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def orgo_create_computer(params: CreateComputerInput) -> str:
    """Create a virtual computer in a workspace. Boots in <500ms. Returns computer ID and details.

    Supports GPU (a10, l40s, a100-40gb, a100-80gb), custom resolution, auto-stop config, and template images.
    """
    try:
        api_key = get_current_api_key(mcp)

        # Use the SDK to create the computer (handles workspace creation if needed)
        def create():
            from orgo import Computer
            kwargs = {
                "project": params.workspace,
                "api_key": api_key,
                "os": params.os,
                "ram": params.ram,
                "cpu": params.cpu,
                "gpu": params.gpu,
                "verbose": False,
            }
            if params.name:
                kwargs["name"] = params.name
            if params.image:
                kwargs["image"] = params.image
            computer = Computer(**kwargs)
            return {
                "id": computer.computer_id,
                "name": computer.name,
                "workspace": params.workspace,
                "os": params.os,
                "ram": params.ram,
                "cpu": params.cpu,
                "gpu": params.gpu,
                "status": "running",
                "url": computer.url,
            }

        result = await asyncio.to_thread(create)

        # If auto_stop_minutes or resolution specified, we'd need an additional API call
        # The SDK doesn't support these yet, so note them in the response
        if params.auto_stop_minutes is not None or params.resolution != "1280x720x24":
            result["note"] = "auto_stop_minutes and custom resolution require direct API calls (not yet supported by SDK v0.0.38)"

        return json.dumps(result, indent=2)
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_get_computer",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def orgo_get_computer(params: ComputerIdInput) -> str:
    """Get computer details including status, specs, and dashboard URL."""
    try:
        api_key = get_current_api_key(mcp)
        data = await api_request("GET", f"computers/{params.computer_id}", api_key)
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_delete_computer",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def orgo_delete_computer(params: ComputerIdInput) -> str:
    """Permanently delete a computer and all its data. Cannot be undone."""
    try:
        api_key = get_current_api_key(mcp)
        data = await api_request("DELETE", f"computers/{params.computer_id}", api_key)
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_start_computer",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def orgo_start_computer(params: ComputerIdInput) -> str:
    """Start a stopped computer. State is preserved from when it was stopped. Idempotent."""
    try:
        api_key = get_current_api_key(mcp)
        data = await api_request("POST", f"computers/{params.computer_id}/start", api_key)
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_stop_computer",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def orgo_stop_computer(params: ComputerIdInput) -> str:
    """Stop a running computer. State is preserved. Stopped computers don't incur charges."""
    try:
        api_key = get_current_api_key(mcp)
        data = await api_request("POST", f"computers/{params.computer_id}/stop", api_key)
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_restart_computer",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def orgo_restart_computer(params: ComputerIdInput) -> str:
    """Restart a computer. Useful for recovering from unresponsive states."""
    try:
        api_key = get_current_api_key(mcp)
        data = await api_request("POST", f"computers/{params.computer_id}/restart", api_key)
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)
