"""Screen action tools — mouse, keyboard, and screenshot control."""

import base64

from mcp.server.fastmcp import Image

from orgo_mcp.server import mcp
from orgo_mcp.auth import get_current_api_key, resolve_computer_id
from orgo_mcp.client import computer_action
from orgo_mcp.errors import handle_orgo_error
from orgo_mcp.models import ComputerIdInput, ClickInput, TypeInput, KeyInput, ScrollInput, DragInput


@mcp.tool(
    name="orgo_screenshot",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def orgo_screenshot(params: ComputerIdInput) -> Image:
    """Take a screenshot of the full VM display (all windows, desktop). For Chrome browser tab only, prefer orgo_chrome_screenshot."""
    try:
        api_key = get_current_api_key(mcp)
        computer_id = resolve_computer_id(params.computer_id)
        data = await computer_action("GET", computer_id, "screenshot", api_key, direct=True)
        image_b64 = data.get("image", "")
        image_b64_clean = image_b64.strip().replace("\n", "").replace("\r", "").replace(" ", "")
        pad = len(image_b64_clean) % 4
        if pad:
            image_b64_clean += "=" * (4 - pad)
        return Image(data=base64.b64decode(image_b64_clean), format="png")
    except Exception as e:
        raise RuntimeError(handle_orgo_error(e))


@mcp.tool(
    name="orgo_click",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
async def orgo_click(params: ClickInput) -> str:
    """Click at pixel (x, y) coordinates on the VM display. For Chrome elements, prefer orgo_chrome_click with element refs."""
    try:
        api_key = get_current_api_key(mcp)
        computer_id = resolve_computer_id(params.computer_id)
        payload = {"x": params.x, "y": params.y, "button": params.button}
        if params.double:
            payload["double"] = True
        await computer_action("POST", computer_id, "click", api_key, json=payload)
        action = "Double-clicked" if params.double else f"{'Right' if params.button == 'right' else ''}Clicked"
        return f"{action} at ({params.x}, {params.y})"
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_type",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
async def orgo_type(params: TypeInput) -> str:
    """Type text at the current cursor position on the VM. For Chrome form fields, prefer orgo_chrome_form_input with element refs."""
    try:
        api_key = get_current_api_key(mcp)
        computer_id = resolve_computer_id(params.computer_id)
        await computer_action("POST", computer_id, "type", api_key, json={"text": params.text})
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
        computer_id = resolve_computer_id(params.computer_id)
        await computer_action("POST", computer_id, "key", api_key, json={"key": params.key})
        return f"Pressed: {params.key}"
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_scroll",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
async def orgo_scroll(params: ScrollInput) -> str:
    """Scroll the VM display up or down. For Chrome page scrolling, prefer orgo_chrome_scroll."""
    try:
        api_key = get_current_api_key(mcp)
        computer_id = resolve_computer_id(params.computer_id)
        await computer_action("POST", computer_id, "scroll", api_key,
                              json={"direction": params.direction, "amount": params.amount})
        return f"Scrolled {params.direction} by {params.amount}"
    except Exception as e:
        return handle_orgo_error(e)


@mcp.tool(
    name="orgo_drag",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
)
async def orgo_drag(params: DragInput) -> str:
    """Drag from (start_x, start_y) to (end_x, end_y) on the VM display. For drag-and-drop, text selection, window resizing in native apps."""
    try:
        api_key = get_current_api_key(mcp)
        computer_id = resolve_computer_id(params.computer_id)
        await computer_action("POST", computer_id, "drag", api_key, json={
            "start_x": params.start_x, "start_y": params.start_y,
            "end_x": params.end_x, "end_y": params.end_y,
            "button": "left", "duration": params.duration,
        })
        return f"Dragged ({params.start_x},{params.start_y}) -> ({params.end_x},{params.end_y})"
    except Exception as e:
        return handle_orgo_error(e)
