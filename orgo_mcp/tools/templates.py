"""Template management tools — list, star/unstar VM templates."""

import json

from orgo_mcp.server import mcp
from orgo_mcp.auth import get_current_api_key
from orgo_mcp.client import api_request
from orgo_mcp.errors import handle_orgo_error
from orgo_mcp.models import TemplateStarInput


@mcp.tool(
    name="orgo_list_templates",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def orgo_list_templates() -> str:
    """List all available VM templates. Templates are pre-configured images that can be used when creating computers."""
    try:
        api_key = get_current_api_key(mcp)
        data = await api_request("GET", "templates", api_key)
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_starred_templates",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def orgo_starred_templates() -> str:
    """List starred VM templates for quick access."""
    try:
        api_key = get_current_api_key(mcp)
        data = await api_request("GET", "templates/starred", api_key)
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_star_template",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def orgo_star_template(params: TemplateStarInput) -> str:
    """Star or unstar a VM template for quick access."""
    try:
        api_key = get_current_api_key(mcp)
        data = await api_request("POST", f"templates/{params.template_id}/star", api_key, json={"starred": params.starred})
        return json.dumps(data, indent=2)
    except Exception as e:
        return handle_orgo_error(e)
