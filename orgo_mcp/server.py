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

# Initialize FastMCP
if IS_HTTP_MODE:
    mcp = FastMCP(
        "orgo_mcp",
        stateless_http=True,
        json_response=True,
        host=HTTP_HOST,
        port=HTTP_PORT,
    )
else:
    mcp = FastMCP("orgo_mcp")


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
