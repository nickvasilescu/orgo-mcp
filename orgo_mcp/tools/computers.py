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
    CloneComputerInput, ResizeComputerInput, AutoStopInput,
    SkillInstallInput, StarComputerInput, MoveComputerInput,
    WaitComputerInput,
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
    name="orgo_auto_stop_get",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def orgo_auto_stop_get(params: ComputerIdInput) -> str:
    """Get the current auto-stop configuration for a computer."""
    try:
        api_key = get_current_api_key(mcp)
        computer_id = resolve_computer_id(params.computer_id)
        data = await api_request("GET", f"computers/{computer_id}/auto-stop", api_key)
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_auto_stop_set",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def orgo_auto_stop_set(params: AutoStopInput) -> str:
    """Set auto-stop timeout for a computer. 0 disables auto-stop (paid plans only). Free tier always enforces 15min."""
    try:
        api_key = get_current_api_key(mcp)
        computer_id = resolve_computer_id(params.computer_id)
        data = await api_request("PATCH", f"computers/{computer_id}/auto-stop", api_key, json={"minutes": params.minutes})
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_install_skill",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def orgo_install_skill(params: SkillInstallInput) -> str:
    """Install skill files onto a running computer via the Orgo API. Uploads files and executes the install command on the VM."""
    try:
        import base64
        import httpx
        from orgo_mcp.client import ORGO_API_BASE

        api_key = get_current_api_key(mcp)

        computer_id = resolve_computer_id(params.computer_id)

        async with httpx.AsyncClient() as client:
            files_list = []
            for filename, b64_content in params.files_base64.items():
                file_bytes = base64.b64decode(b64_content)
                files_list.append(("files", (filename, file_bytes, "application/octet-stream")))

            response = await client.post(
                f"{ORGO_API_BASE}/computers/{computer_id}/skills/install",
                headers={"Authorization": f"Bearer {api_key}"},
                files=files_list,
                data={"skillName": params.skill_name},
                timeout=120.0,
            )
            response.raise_for_status()
            return json.dumps(response.json(), indent=2)
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_star_computer",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def orgo_star_computer(params: StarComputerInput) -> str:
    """Star or unstar a computer for quick access."""
    try:
        api_key = get_current_api_key(mcp)
        computer_id = resolve_computer_id(params.computer_id)
        data = await api_request("POST", f"computers/{computer_id}/star", api_key, json={"starred": params.starred})
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_starred_computers",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def orgo_starred_computers() -> str:
    """List all starred computer IDs for quick access."""
    try:
        api_key = get_current_api_key(mcp)
        data = await api_request("GET", "computers/starred", api_key)
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_move_computer",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def orgo_move_computer(params: MoveComputerInput) -> str:
    """Move a computer to a different workspace. The computer keeps all its data and state."""
    try:
        api_key = get_current_api_key(mcp)
        computer_id = resolve_computer_id(params.computer_id)
        data = await api_request(
            "PATCH", f"computers/{computer_id}/move", api_key,
            json={"project_id": params.workspace_id},
        )
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_wait_computer",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def orgo_wait_computer(params: WaitComputerInput) -> str:
    """Wait for a computer to reach a target state (e.g. 'running' after start). Blocks until the state is reached or timeout."""
    try:
        api_key = get_current_api_key(mcp)
        computer_id = resolve_computer_id(params.computer_id)
        data = await api_request(
            "POST", f"computers/{computer_id}/wait", api_key,
            json={"state": params.state, "timeout": params.timeout},
            timeout=float(params.timeout + 10),
        )
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)
