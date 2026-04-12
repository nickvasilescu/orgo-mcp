"""Orgo MCP Server — FastMCP initialization and transport configuration."""

import os
import logging

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse, JSONResponse

from orgo_mcp.auth import _request_api_key

# Configure logging (stderr for stdio transport)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orgo-mcp")

# Transport configuration
TRANSPORT_MODE = os.environ.get("MCP_TRANSPORT", "stdio").lower()
IS_HTTP_MODE = TRANSPORT_MODE in ("http", "streamable-http")
HTTP_HOST = os.environ.get("MCP_HOST", "0.0.0.0")
HTTP_PORT = int(os.environ.get("MCP_PORT", os.environ.get("PORT", "8000")))

# =============================================================================
# MCP Instructions — guides the model on tool selection and usage
# =============================================================================

ORGO_MCP_INSTRUCTIONS = """
# Orgo MCP — Tool Selection Guide

Orgo provides cloud virtual machines for AI agents. This MCP gives you full control:
VM lifecycle, shell commands, screen interaction, file management, and AI agent orchestration.

## Default Computer

If ORGO_DEFAULT_COMPUTER_ID is set, you can omit computer_id on any tool call.
The default is used automatically. Pass computer_id explicitly to target a different VM.

## When orgo-chrome-mcp is also active

If both `orgo_*` and `orgo_chrome_*` tools are available, use this hierarchy:

### For Chrome/browser tasks → prefer orgo_chrome_* tools
- **Reading pages**: `orgo_chrome_read_page` (DOM/accessibility tree) over `orgo_screenshot` (pixels)
- **Clicking elements**: `orgo_chrome_click` (element refs, reliable) over `orgo_click` (pixel coordinates, fragile)
- **Typing in forms**: `orgo_chrome_form_input` (sets value by ref) over `orgo_type` (simulates keystrokes)
- **Scrolling pages**: `orgo_chrome_scroll` over `orgo_scroll`
- **Running JS**: `orgo_chrome_evaluate` (in-page context) over `orgo_bash` with a curl
- **Screenshots**: `orgo_chrome_screenshot` (browser tab only) for web tasks

### For non-browser / native desktop tasks → use orgo_* tools
- **Full VM screenshots**: `orgo_screenshot` — shows entire desktop, all windows
- **Native app interaction**: `orgo_click`, `orgo_type`, `orgo_key`, `orgo_drag` — pixel-based, works everywhere
- **Shell commands**: `orgo_bash` — run any command on the VM
- **Python execution**: `orgo_exec` — run Python code directly
- **File operations**: `orgo_export_file`, `orgo_upload_file` — move files to/from the VM

### General rule
DOM-aware tools (orgo_chrome_*) are faster and more reliable for web content.
Pixel-based tools (orgo_*) are the right choice for native apps, terminal windows, or full desktop.

## Tool Categories

- **VM Lifecycle**: create, start, stop, restart, delete, clone, resize, ensure_running, wait
- **Screen Actions**: screenshot, click, type, key, scroll, drag
- **Shell**: bash, exec (Python)
- **Files**: list, upload, export, download, delete
- **AI Agent**: completions (autonomous agent), threads (conversation history)
- **Streaming**: start/stop RTMP streams
- **Account**: profile, credits, transactions
""".strip()

# Initialize FastMCP
if IS_HTTP_MODE:
    mcp = FastMCP(
        "orgo_mcp",
        instructions=ORGO_MCP_INSTRUCTIONS,
        stateless_http=True,
        json_response=True,
        host=HTTP_HOST,
        port=HTTP_PORT,
    )
else:
    mcp = FastMCP("orgo_mcp", instructions=ORGO_MCP_INSTRUCTIONS)


# =============================================================================
# HTTP Middleware — Extract API key from request headers
# =============================================================================

class OrgoAPIKeyMiddleware:
    """ASGI middleware to extract X-Orgo-API-Key header into context."""

    EXEMPT_PATHS = {"/health", "/health/"}

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            if path in self.EXEMPT_PATHS:
                await self.app(scope, receive, send)
                return

            headers = dict(scope.get("headers", []))
            api_key = headers.get(b"x-orgo-api-key", b"").decode("utf-8")

            if not api_key:
                response = JSONResponse(
                    {"error": "X-Orgo-API-Key header required. Get your key at https://orgo.ai"},
                    status_code=401,
                )
                await response(scope, receive, send)
                return

            token = _request_api_key.set(api_key)
            try:
                await self.app(scope, receive, send)
            finally:
                _request_api_key.reset(token)
        else:
            await self.app(scope, receive, send)


# =============================================================================
# Health Check
# =============================================================================

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    """Health check for load balancers."""
    return PlainTextResponse("OK", status_code=200)


# =============================================================================
# Register all tools
# =============================================================================

# Import tool modules — registration happens at import time via @mcp.tool decorators
import orgo_mcp.tools  # noqa: E402, F401


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Start the Orgo MCP server."""
    if TRANSPORT_MODE == "stdio":
        if not os.environ.get("ORGO_API_KEY"):
            logger.error("ORGO_API_KEY not set. Get your key at https://orgo.ai")
            exit(1)
        logger.info("Starting Orgo MCP server (stdio transport)")
        mcp.run()

    elif TRANSPORT_MODE in ("http", "streamable-http"):
        logger.info(f"Starting Orgo MCP server (HTTP) on {HTTP_HOST}:{HTTP_PORT}")
        mcp.run(transport="streamable-http")

    else:
        logger.error(f"Unknown transport: {TRANSPORT_MODE}. Use 'stdio' or 'http'")
        exit(1)


if __name__ == "__main__":
    main()
