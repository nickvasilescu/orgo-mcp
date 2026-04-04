"""Screen action tools — mouse, keyboard, and screenshot control."""

import base64

from mcp.server.fastmcp import Image

from orgo_mcp.server import mcp
from orgo_mcp.auth import get_current_api_key
from orgo_mcp.client import computer_action
from orgo_mcp.errors import handle_orgo_error
from orgo_mcp.models import ComputerIdInput, ClickInput, TypeInput, KeyInput, ScrollInput, DragInput


@mcp.tool(
    name="orgo_screenshot",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def orgo_screenshot(params: ComputerIdInput) -> Image:
    """Take a screenshot of the computer's display. Returns a JPEG image."""
    try:
        api_key = get_current_api_key(mcp)
        data = await computer_action("GET", params.computer_id, "screenshot", api_key)
        image_b64 = data.get("image", "")
        return Image(data=base64.b64decode(image_b64), format="png")
    except Exception as e:
        raise RuntimeError(handle_orgo_error(e))


@mcp.tool(
    name="orgo_click",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
async def orgo_click(params: ClickInput) -> str:
    """Click at (x, y) coordinates. Supports left/right click and double-click."""
    try:
        api_key = get_current_api_key(mcp)
        payload = {"x": params.x, "y": params.y, "button": params.button}
        if params.double:
            payload["double"] = True
        await computer_action("POST", params.computer_id, "click", api_key, json=payload)
        action = "Double-clicked" if params.double else f"{'Right' if params.button == 'right' else ''}Clicked"
        return f"{action} at ({params.x}, {params.y})"
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_type",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
async def orgo_type(params: TypeInput) -> str:
    """Type text at the current cursor position. Click a field first."""
    try:
        api_key = get_current_api_key(mcp)
        await computer_action("POST", params.computer_id, "type", api_key, json={"text": params.text})
        preview = params.text[:50] + "..." if len(params.text) > 50 else params.text
        return f"Typed: {preview}"
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_key",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
async def orgo_key(params: KeyInput) -> str:
    """Press a key or combo: Enter, Tab, Escape, ctrl+c, alt+Tab, ctrl+shift+s, F1-F12."""
    try:
        api_key = get_current_api_key(mcp)
        await computer_action("POST", params.computer_id, "key", api_key, json={"key": params.key})
        return f"Pressed: {params.key}"
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_scroll",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
async def orgo_scroll(params: ScrollInput) -> str:
    """Scroll up or down by a specified amount."""
    try:
        api_key = get_current_api_key(mcp)
        await computer_action("POST", params.computer_id, "scroll", api_key,
                              json={"direction": params.direction, "amount": params.amount})
        return f"Scrolled {params.direction} by {params.amount}"
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_drag",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
async def orgo_drag(params: DragInput) -> str:
    """Drag from (start_x, start_y) to (end_x, end_y). For drag-and-drop, text selection, window resizing."""
    try:
        api_key = get_current_api_key(mcp)
        await computer_action("POST", params.computer_id, "drag", api_key, json={
            "start_x": params.start_x, "start_y": params.start_y,
            "end_x": params.end_x, "end_y": params.end_y,
            "button": "left", "duration": params.duration,
        })
        return f"Dragged ({params.start_x},{params.start_y}) -> ({params.end_x},{params.end_y})"
    except Exception as e:
        return handle_orgo_error(e)
