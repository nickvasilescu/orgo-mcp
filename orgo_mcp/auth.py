"""API key and computer ID resolution for Orgo MCP.

Supports two modes:
- HTTP transport: key from X-Orgo-API-Key request header (via MCP context)
- stdio transport: key from ORGO_API_KEY environment variable

Computer ID resolution:
- Explicit computer_id parameter (always wins)
- ORGO_DEFAULT_COMPUTER_ID environment variable (fallback)
"""

import os
from contextvars import ContextVar
from typing import Optional

# Per-request API key for HTTP transport (set by middleware)
_request_api_key: ContextVar[Optional[str]] = ContextVar("request_api_key", default=None)


def get_current_api_key(mcp_server=None) -> str:
    """Get the API key for the current request.

    Priority:
    1. MCP request context header (HTTP transport)
    2. ContextVar from middleware (HTTP fallback)
    3. ORGO_API_KEY environment variable (stdio transport)
    """
    # Try MCP request context (HTTP transport)
    if mcp_server is not None:
        try:
            ctx = mcp_server.get_context()
            if ctx and ctx.request_context and ctx.request_context.request:
                api_key = ctx.request_context.request.headers.get("x-orgo-api-key")
                if api_key:
                    return api_key
        except Exception:
            pass

    # Try per-request ContextVar (middleware fallback)
    request_key = _request_api_key.get()
    if request_key:
        return request_key

    # Fall back to environment variable (stdio transport)
    env_key = os.environ.get("ORGO_API_KEY")
    if env_key:
        return env_key

    raise ValueError(
        "API key required. HTTP: include X-Orgo-API-Key header. "
        "stdio: set ORGO_API_KEY env var. Get your key at https://orgo.ai"
    )


def resolve_computer_id(computer_id: Optional[str] = None) -> str:
    """Resolve computer_id from explicit param or ORGO_DEFAULT_COMPUTER_ID env var.

    Priority:
    1. Explicit computer_id parameter
    2. ORGO_DEFAULT_COMPUTER_ID environment variable
    """
    if computer_id:
        return computer_id

    default_id = os.environ.get("ORGO_DEFAULT_COMPUTER_ID")
    if default_id:
        return default_id

    raise ValueError(
        "computer_id required. Either pass it explicitly or set "
        "ORGO_DEFAULT_COMPUTER_ID env var."
    )
