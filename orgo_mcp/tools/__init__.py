"""Register all Orgo MCP tools.

Importing this module registers tools with the FastMCP server via @mcp.tool decorators.
"""

from orgo_mcp.tools import workspaces  # noqa: F401
from orgo_mcp.tools import computers  # noqa: F401
from orgo_mcp.tools import actions  # noqa: F401
from orgo_mcp.tools import shell  # noqa: F401
from orgo_mcp.tools import files  # noqa: F401
from orgo_mcp.tools import agent  # noqa: F401
from orgo_mcp.tools import threads  # noqa: F401
from orgo_mcp.tools import streaming  # noqa: F401
from orgo_mcp.tools import access  # noqa: F401
from orgo_mcp.tools import templates  # noqa: F401
