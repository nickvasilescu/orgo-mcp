"""File management tools — upload, export, list, download, delete."""

import json
import base64

import httpx

from orgo_mcp.server import mcp
from orgo_mcp.auth import get_current_api_key
from orgo_mcp.client import api_request, ORGO_API_BASE
from orgo_mcp.errors import handle_orgo_error
from orgo_mcp.models import ListFilesInput, ExportFileInput, UploadFileInput, FileIdInput


@mcp.tool(
    name="orgo_list_files",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def orgo_list_files(params: ListFilesInput) -> str:
    """List files in a workspace, optionally filtered by computer ID."""
    try:
        api_key = get_current_api_key(mcp)
        query = {"projectId": params.workspace_id}
        if params.computer_id:
            query["desktopId"] = params.computer_id
        data = await api_request("GET", "files", api_key, params=query)
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_export_file",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def orgo_export_file(params: ExportFileInput) -> str:
    """Export a file from the computer's filesystem. Returns file info and a download URL (expires in 1 hour).
    Files must be under /home/user."""
    try:
        api_key = get_current_api_key(mcp)
        data = await api_request("POST", "files/export", api_key, json={
            "desktopId": params.computer_id,
            "path": params.path,
        })
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_upload_file",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
async def orgo_upload_file(params: UploadFileInput) -> str:
    """Upload a file to a workspace (max 10MB). Pass content as base64. File is accessible by all computers in the workspace."""
    try:
        api_key = get_current_api_key(mcp)
        file_bytes = base64.b64decode(params.content_base64)
        content_type = params.content_type or "application/octet-stream"

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ORGO_API_BASE}/files/upload",
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (params.filename, file_bytes, content_type)},
                data={
                    "projectId": params.workspace_id,
                    **({"desktopId": params.computer_id} if params.computer_id else {}),
                },
                timeout=60.0,
            )
            response.raise_for_status()
            return json.dumps(response.json(), indent=2)
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_download_file",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def orgo_download_file(params: FileIdInput) -> str:
    """Get a signed download URL for a file. URL expires in 1 hour."""
    try:
        api_key = get_current_api_key(mcp)
        data = await api_request("GET", "files/download", api_key, params={"id": params.file_id})
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_delete_file",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": False},
)
async def orgo_delete_file(params: FileIdInput) -> str:
    """Permanently delete a file from storage. Cannot be undone."""
    try:
        api_key = get_current_api_key(mcp)
        data = await api_request("DELETE", "files/delete", api_key, params={"id": params.file_id})
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)
