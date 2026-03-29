"""Screen action tools — mouse, keyboard, and screenshot control."""

import json
import asyncio
import base64

from mcp.server.fastmcp import Image

from orgo_mcp.server import mcp
from orgo_mcp.auth import get_current_api_key
from orgo_mcp.client import get_computer
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

        def take():
            computer = get_computer(params.computer_id, api_key)
            return computer.screenshot_base64()

        b64 = await asyncio.to_thread(take)
        return Image(data=base64.b64decode(b64), format="jpeg")
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

        def click():
            computer = get_computer(params.computer_id, api_key)
            if params.double:
                computer.double_click(params.x, params.y)
                return f"Double-clicked at ({params.x}, {params.y})"
            elif params.button == "right":
                computer.right_click(params.x, params.y)
                return f"Right-clicked at ({params.x}, {params.y})"
            else:
                computer.left_click(params.x, params.y)
                return f"Clicked at ({params.x}, {params.y})"

        return await asyncio.to_thread(click)
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

        def do_type():
            computer = get_computer(params.computer_id, api_key)
            computer.type(params.text)
            preview = params.text[:50] + "..." if len(params.text) > 50 else params.text
            return f"Typed: {preview}"

        return await asyncio.to_thread(do_type)
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

        def press():
            computer = get_computer(params.computer_id, api_key)
            computer.key(params.key)
            return f"Pressed: {params.key}"

        return await asyncio.to_thread(press)
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

        def scroll():
            computer = get_computer(params.computer_id, api_key)
            computer.scroll(params.direction, params.amount)
            return f"Scrolled {params.direction} by {params.amount}"

        return await asyncio.to_thread(scroll)
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

        def drag():
            computer = get_computer(params.computer_id, api_key)
            computer.drag(params.start_x, params.start_y, params.end_x, params.end_y, duration=params.duration)
            return f"Dragged ({params.start_x},{params.start_y}) -> ({params.end_x},{params.end_y})"

        return await asyncio.to_thread(drag)
    except Exception as e:
        return handle_orgo_error(e)
