"""Workspace management tools.

Workspaces are containers for computers. Use them to organize different projects,
environments, or agent fleets.
"""

import json

from orgo_mcp.server import mcp
from orgo_mcp.auth import get_current_api_key
from orgo_mcp.client import api_request
from orgo_mcp.errors import handle_orgo_error
from orgo_mcp.models import CreateWorkspaceInput, WorkspaceIdInput, WorkspaceByNameInput


@mcp.tool(
    name="orgo_list_workspaces",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def orgo_list_workspaces() -> str:
    """List all workspaces in your Orgo account. Returns workspace IDs, names, and computer counts."""
    try:
        api_key = get_current_api_key(mcp)
        data = await api_request("GET", "projects", api_key)
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_create_workspace",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def orgo_create_workspace(params: CreateWorkspaceInput) -> str:
    """Create a new workspace. Workspace names must be unique. Returns the workspace ID."""
    try:
        api_key = get_current_api_key(mcp)
        data = await api_request("POST", "projects", api_key, json={"name": params.name})
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_get_workspace",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def orgo_get_workspace(params: WorkspaceIdInput) -> str:
    """Get workspace details including its computers. Returns workspace info and computer list."""
    try:
        api_key = get_current_api_key(mcp)
        data = await api_request("GET", f"projects/{params.workspace_id}", api_key)
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_workspace_by_name",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def orgo_workspace_by_name(params: WorkspaceByNameInput) -> str:
    """Look up a workspace by name instead of ID. Returns workspace details if found."""
    try:
        api_key = get_current_api_key(mcp)
        data = await api_request("GET", f"projects/by-name/{params.name}", api_key)
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)
