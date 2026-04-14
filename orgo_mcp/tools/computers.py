"""Computer management tools.

Computers are virtual machines within workspaces. They boot in under 500ms
and can be controlled via actions, shell commands, or AI agents.
"""

import json
import asyncio

from orgo_mcp.server import mcp
from orgo_mcp.auth import get_current_api_key, resolve_computer_id
from orgo_mcp.client import api_request, resolve_fly_instance_id
from orgo_mcp.errors import handle_orgo_error
from orgo_mcp.models import (
    ListComputersInput, CreateComputerInput, ComputerIdInput,
    CloneComputerInput, ResizeComputerInput,
)


@mcp.tool(
    name="orgo_list_computers",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def orgo_list_computers(params: ListComputersInput) -> str:
    """List all computers in a workspace. Returns IDs, names, status, specs."""
    try:
        api_key = get_current_api_key(mcp)
        data = await api_request("GET", f"projects/{params.workspace_id}", api_key)
        computers = data.get("desktops", [])
        return json.dumps({"computers": computers, "count": len(computers)}, indent=2)
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_create_computer",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def orgo_create_computer(params: CreateComputerInput) -> str:
    """Create a virtual computer in a workspace. Boots in <500ms. Returns computer ID and details."""
    try:
        api_key = get_current_api_key(mcp)

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
        computer_id = resolve_computer_id(params.computer_id)
        data = await api_request("GET", f"computers/{computer_id}", api_key)
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
        computer_id = resolve_computer_id(params.computer_id)
        data = await api_request("DELETE", f"computers/{computer_id}", api_key)
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
        computer_id = resolve_computer_id(params.computer_id)
        fly_id = await resolve_fly_instance_id(computer_id, api_key)
        data = await api_request("POST", f"computers/{fly_id}/start", api_key)
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
        computer_id = resolve_computer_id(params.computer_id)
        fly_id = await resolve_fly_instance_id(computer_id, api_key)
        data = await api_request("POST", f"computers/{fly_id}/stop", api_key)
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
        computer_id = resolve_computer_id(params.computer_id)
        fly_id = await resolve_fly_instance_id(computer_id, api_key)
        data = await api_request("POST", f"computers/{fly_id}/restart", api_key)
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_clone_computer",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def orgo_clone_computer(params: CloneComputerInput) -> str:
    """Clone/duplicate a computer including its full disk state. Creates an identical copy in the same or different workspace."""
    try:
        api_key = get_current_api_key(mcp)
        computer_id = resolve_computer_id(params.computer_id)
        fly_id = await resolve_fly_instance_id(computer_id, api_key)
        body = {}
        if params.name:
            body["name"] = params.name
        if params.workspace_id:
            body["targetProjectId"] = params.workspace_id
        data = await api_request("POST", f"computers/{fly_id}/clone", api_key, json=body, timeout=120.0)
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_ensure_running",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def orgo_ensure_running(params: ComputerIdInput) -> str:
    """Ensure a computer is running. Resumes suspended VMs automatically. Idempotent — safe to call on already-running computers."""
    try:
        api_key = get_current_api_key(mcp)
        computer_id = resolve_computer_id(params.computer_id)
        data = await api_request("POST", f"computers/{computer_id}/ensure-running", api_key)
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_resize_computer",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def orgo_resize_computer(params: ResizeComputerInput) -> str:
    """Resize a computer's CPU, RAM, disk, or bandwidth. Some changes may require a restart."""
    try:
        api_key = get_current_api_key(mcp)
        computer_id = resolve_computer_id(params.computer_id)
        body = {}
        if params.cpu is not None:
            body["cpu"] = params.cpu
        if params.ram is not None:
            body["ram"] = params.ram
        if params.disk_size_gb is not None:
            body["disk_size_gb"] = params.disk_size_gb
        if params.bandwidth_limit_mbps is not None:
            body["bandwidth_limit_mbps"] = params.bandwidth_limit_mbps
        data = await api_request("PATCH", f"computers/{computer_id}/resize", api_key, json=body)
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)
