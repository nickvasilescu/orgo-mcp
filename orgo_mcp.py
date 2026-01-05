#!/usr/bin/env python3
"""
Orgo MCP Server - Virtual Computer Control

An MCP server that gives AI agents the ability to control virtual computers via Orgo.
Exposes 35 tools for VM management, screen actions, shell commands, file operations,
streaming, AI agents, and AI completion.

Requires: ORGO_API_KEY environment variable
Get your key at: https://orgo.ai
"""

import os
import json
import asyncio
import logging
import base64
from typing import Optional, Literal, List, Dict, Any
from enum import Enum
from contextvars import ContextVar

from mcp.server.fastmcp import FastMCP, Image
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv
import httpx

# HTTP transport imports (only used when MCP_TRANSPORT=http)
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import Response, PlainTextResponse, JSONResponse

# Per-request API key storage (async-safe for HTTP transport)
_request_api_key: ContextVar[Optional[str]] = ContextVar("request_api_key", default=None)

# Load environment variables
load_dotenv()

# Configure logging (stderr for stdio transport)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orgo-mcp")


# ============================================================================
# Enums
# ============================================================================

class ResponseFormat(str, Enum):
    """Output format for tool responses."""
    MARKDOWN = "markdown"
    JSON = "json"


# ============================================================================
# Error Handling
# ============================================================================

def _handle_orgo_error(e: Exception) -> str:
    """
    Consistent error formatting for Orgo operations.

    Provides actionable error messages that guide agents toward solutions.
    """
    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        if status == 401:
            return "Error: Invalid API key. Check ORGO_API_KEY environment variable. Get your key at https://orgo.ai"
        elif status == 404:
            return "Error: Resource not found. Use orgo_list_projects or orgo_list_computers to find valid IDs."
        elif status == 402:
            return "Error: Insufficient credits. Check your Orgo account balance at https://orgo.ai"
        elif status == 409:
            return "Error: Conflict. Resource already exists or operation in progress."
        elif status == 429:
            return "Error: Rate limit exceeded. Wait a moment before retrying."
        elif status == 503:
            return "Error: Service unavailable. The service may be temporarily down."
        return f"Error: Orgo API returned status {status}. Check your request parameters."
    elif isinstance(e, httpx.TimeoutException):
        return "Error: Request timed out. The computer may be starting up - try again in 10 seconds."
    return f"Error: {type(e).__name__}: {str(e)}"


# ============================================================================
# HTTP Transport Middleware
# ============================================================================

class OrgoAPIKeyMiddleware:
    """
    ASGI middleware to extract X-Orgo-API-Key header and store in context.

    For HTTP transport, each request must include the API key in the header.
    The health check endpoint is exempt from authentication.
    """

    EXEMPT_PATHS = {"/health", "/health/"}

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path", "")

            # Skip auth for health check endpoint
            if path in self.EXEMPT_PATHS:
                await self.app(scope, receive, send)
                return

            # Extract API key from headers
            headers = dict(scope.get("headers", []))
            api_key = headers.get(b"x-orgo-api-key", b"").decode("utf-8")

            if not api_key:
                response = JSONResponse(
                    {"error": "X-Orgo-API-Key header required. Get your key at https://orgo.ai"},
                    status_code=401
                )
                await response(scope, receive, send)
                return

            # Store API key in context for this request
            token = _request_api_key.set(api_key)
            try:
                await self.app(scope, receive, send)
            finally:
                _request_api_key.reset(token)
        else:
            await self.app(scope, receive, send)


# ============================================================================
# Initialize FastMCP Server
# ============================================================================

# Determine transport mode from environment
TRANSPORT_MODE = os.environ.get("MCP_TRANSPORT", "stdio").lower()
IS_HTTP_MODE = TRANSPORT_MODE in ("http", "streamable-http")

# Get host/port for HTTP mode
HTTP_HOST = os.environ.get("MCP_HOST", "0.0.0.0")
HTTP_PORT = int(os.environ.get("MCP_PORT", os.environ.get("PORT", "8000")))

# Initialize FastMCP with conditional settings for HTTP vs stdio
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


# ============================================================================
# Pydantic Output Models
# ============================================================================

class ProjectInfo(BaseModel):
    """Project information returned by list operations."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    id: str = Field(..., description="Unique project identifier")
    name: str = Field(..., description="Project name")
    created_at: Optional[str] = Field(default=None, description="ISO timestamp of creation")


class ComputerInfo(BaseModel):
    """Computer information returned by list/create operations."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    id: str = Field(..., description="Unique computer identifier (use with action tools)")
    name: str = Field(..., description="Computer display name")
    status: str = Field(..., description="Current status: running, stopped, starting")
    os: str = Field(..., description="Operating system: linux or windows")
    ram: Optional[int] = Field(default=None, description="RAM in GB")
    cpu: Optional[int] = Field(default=None, description="CPU cores")


class FileInfo(BaseModel):
    """File information returned by list operations."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    id: str = Field(..., description="Unique file identifier")
    filename: str = Field(..., description="File name")
    size_bytes: int = Field(..., description="File size in bytes")
    content_type: Optional[str] = Field(default=None, description="MIME type")
    created_at: Optional[str] = Field(default=None, description="ISO timestamp")


class ProjectDetails(BaseModel):
    """Full project details with computers."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    id: str = Field(..., description="Unique project identifier")
    name: str = Field(..., description="Project name")
    status: str = Field(..., description="Project status")
    created_at: Optional[str] = Field(default=None, description="ISO timestamp of creation")
    updated_at: Optional[str] = Field(default=None, description="ISO timestamp of last update")
    icon_url: Optional[str] = Field(default=None, description="Project icon URL")
    desktops: List[ComputerInfo] = Field(default_factory=list, description="Computers in this project")


class ComputerDetails(BaseModel):
    """Full computer details including access URL."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    id: str = Field(..., description="Unique computer identifier")
    name: str = Field(..., description="Computer display name")
    project_name: str = Field(..., description="Parent project name")
    os: str = Field(..., description="Operating system: linux or windows")
    ram: int = Field(..., description="RAM in GB")
    cpu: int = Field(..., description="CPU cores")
    status: str = Field(..., description="Current status: running, stopped, starting")
    url: str = Field(..., description="Access URL for the computer")
    created_at: Optional[str] = Field(default=None, description="ISO timestamp of creation")


class StreamStatus(BaseModel):
    """RTMP stream status."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    status: Literal["idle", "streaming", "terminated"] = Field(..., description="Current stream status")
    pid: Optional[int] = Field(default=None, description="Process ID of stream")
    start_time: Optional[str] = Field(default=None, description="When streaming started")
    message: Optional[str] = Field(default=None, description="Status message")


class AIModel(BaseModel):
    """Available AI model info."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    id: str = Field(..., description="Model identifier (e.g., 'openai/gpt-4')")
    name: str = Field(..., description="Display name")
    description: Optional[str] = Field(default=None, description="Model description")
    context_length: Optional[int] = Field(default=None, description="Max context window")
    pricing: Optional[Dict[str, Any]] = Field(default=None, description="Pricing info")


# ============================================================================
# Pydantic Input Models
# ============================================================================

class ListProjectsInput(BaseModel):
    """Input for listing projects."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    limit: int = Field(default=20, ge=1, le=100, description="Maximum results to return (1-100)")
    offset: int = Field(default=0, ge=0, description="Number of results to skip for pagination")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format: 'markdown' or 'json'")


class ListComputersInput(BaseModel):
    """Input for listing computers in a project."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    project_name: str = Field(..., description="Name of the project to list computers from (e.g., 'my-project')", min_length=1)
    limit: int = Field(default=20, ge=1, le=100, description="Maximum results to return (1-100)")
    offset: int = Field(default=0, ge=0, description="Number of results to skip for pagination")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format: 'markdown' or 'json'")


class CreateProjectInput(BaseModel):
    """Input for creating a project."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    name: str = Field(..., description="Unique name for the new project (e.g., 'qa-automation')", min_length=1, max_length=100)
    icon_url: Optional[str] = Field(default=None, description="Optional URL for project icon")


class GetProjectInput(BaseModel):
    """Input for getting project details."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    name: str = Field(..., description="Project name to look up", min_length=1)


class ProjectIdInput(BaseModel):
    """Input for project operations requiring ID."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    project_id: str = Field(..., description="Project ID (from orgo_list_projects)", min_length=1)


class CreateComputerInput(BaseModel):
    """Input for creating a computer."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    project_name: str = Field(..., description="Project name to create computer in", min_length=1)
    name: str = Field(..., description="Name for the new computer (e.g., 'dev-box', 'test-env')", min_length=1, max_length=100)
    os: Literal["linux", "windows"] = Field(default="linux", description="Operating system: 'linux' or 'windows'")
    ram: Literal[1, 2, 4, 8, 16, 32, 64] = Field(default=2, description="RAM in GB (1, 2, 4, 8, 16, 32, or 64)")
    cpu: Literal[1, 2, 4, 8, 16] = Field(default=2, description="CPU cores (1, 2, 4, 8, or 16)")


class ComputerIdInput(BaseModel):
    """Input for computer operations requiring ID."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    computer_id: str = Field(..., description="Computer ID (from orgo_list_computers)", min_length=1)


class ClickInput(BaseModel):
    """Input for click operations."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    computer_id: str = Field(..., description="Computer ID", min_length=1)
    x: int = Field(..., ge=0, description="X coordinate in pixels from left edge")
    y: int = Field(..., ge=0, description="Y coordinate in pixels from top edge")
    button: Literal["left", "right", "middle"] = Field(default="left", description="Mouse button: 'left', 'right', or 'middle'")


class DoubleClickInput(BaseModel):
    """Input for double-click operations."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    computer_id: str = Field(..., description="Computer ID", min_length=1)
    x: int = Field(..., ge=0, description="X coordinate in pixels")
    y: int = Field(..., ge=0, description="Y coordinate in pixels")


class TypeInput(BaseModel):
    """Input for typing text."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    computer_id: str = Field(..., description="Computer ID", min_length=1)
    text: str = Field(..., description="Text to type at the current cursor position", min_length=1)


class KeyInput(BaseModel):
    """Input for pressing keyboard keys."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    computer_id: str = Field(..., description="Computer ID", min_length=1)
    key: str = Field(..., description="Key or combo: Enter, Tab, Escape, ctrl+c, alt+Tab, ctrl+shift+s", min_length=1)


class ScrollInput(BaseModel):
    """Input for scrolling."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    computer_id: str = Field(..., description="Computer ID", min_length=1)
    direction: Literal["up", "down", "left", "right"] = Field(..., description="Scroll direction")
    amount: int = Field(default=3, ge=1, le=10, description="Scroll amount (1-10)")


class DragInput(BaseModel):
    """Input for drag operations."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    computer_id: str = Field(..., description="Computer ID", min_length=1)
    start_x: int = Field(..., ge=0, description="Starting X coordinate")
    start_y: int = Field(..., ge=0, description="Starting Y coordinate")
    end_x: int = Field(..., ge=0, description="Ending X coordinate")
    end_y: int = Field(..., ge=0, description="Ending Y coordinate")
    duration: float = Field(default=0.5, ge=0.1, le=5.0, description="Duration in seconds (0.1-5.0)")


class WaitInput(BaseModel):
    """Input for wait operations."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    computer_id: str = Field(..., description="Computer ID", min_length=1)
    seconds: float = Field(default=2.0, ge=0.1, le=60.0, description="Seconds to wait (0.1-60)")


class BashInput(BaseModel):
    """Input for bash command execution."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    computer_id: str = Field(..., description="Computer ID", min_length=1)
    command: str = Field(..., description="Bash command to execute (e.g., 'ls -la', 'pip install requests')", min_length=1)


class ExecInput(BaseModel):
    """Input for Python code execution."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    computer_id: str = Field(..., description="Computer ID", min_length=1)
    code: str = Field(..., description="Python code to execute", min_length=1)
    timeout: int = Field(default=30, ge=1, le=300, description="Timeout in seconds (1-300)")


class ListFilesInput(BaseModel):
    """Input for listing files."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    computer_id: str = Field(..., description="Computer ID to list files for", min_length=1)
    limit: int = Field(default=20, ge=1, le=100, description="Maximum results to return")
    offset: int = Field(default=0, ge=0, description="Number of results to skip")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format")


class ExportFileInput(BaseModel):
    """Input for exporting a file."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    computer_id: str = Field(..., description="Computer ID to export from", min_length=1)
    path: str = Field(..., description="Path to file on computer (e.g., 'Desktop/results.txt', '~/Documents/report.pdf')", min_length=1)


class UploadFileInput(BaseModel):
    """Input for uploading a file."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    computer_id: str = Field(..., description="Computer ID to upload to", min_length=1)
    filename: str = Field(..., description="Name for the uploaded file", min_length=1, max_length=255)
    content_base64: str = Field(..., description="Base64-encoded file content", min_length=1)
    content_type: Optional[str] = Field(default=None, description="MIME type (e.g., 'text/plain', 'image/png')")


class FileIdInput(BaseModel):
    """Input for file operations requiring ID."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    file_id: str = Field(..., description="File ID (from orgo_list_files or orgo_export_file)", min_length=1)


class StartStreamInput(BaseModel):
    """Input for starting RTMP stream."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    computer_id: str = Field(..., description="Computer ID to stream from", min_length=1)
    rtmp_url: str = Field(..., description="RTMP URL (e.g., 'rtmp://live.twitch.tv/app/your_stream_key')", min_length=1)
    resolution: Literal["1920x1080", "1280x720", "854x480"] = Field(default="1280x720", description="Stream resolution")
    fps: Literal[15, 30, 60] = Field(default=30, description="Frames per second")
    bitrate: str = Field(default="2500k", description="Video bitrate (e.g., '2500k', '4000k')")


class ListAIModelsInput(BaseModel):
    """Input for listing AI models."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    limit: int = Field(default=50, ge=1, le=200, description="Maximum results to return")
    offset: int = Field(default=0, ge=0, description="Number of results to skip")
    response_format: ResponseFormat = Field(default=ResponseFormat.MARKDOWN, description="Output format")


class AICompletionInput(BaseModel):
    """Input for AI completion."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    model: str = Field(..., description="Model ID (e.g., 'openai/gpt-4', 'anthropic/claude-3-opus')", min_length=1)
    prompt: str = Field(..., description="The prompt to send to the model", min_length=1)
    system: Optional[str] = Field(default=None, description="Optional system message")
    max_tokens: int = Field(default=1024, ge=1, le=100000, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature (0-2)")


class PromptInput(BaseModel):
    """Input for launching an AI agent task on Orgo.

    This model validates parameters for asynchronous agent execution.
    The agent runs autonomously on Orgo's infrastructure.
    """
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra='forbid'
    )

    prompt: str = Field(
        ...,
        description="Natural language instruction for the AI agent (e.g., 'Open Firefox and search for AI news')",
        min_length=1,
        max_length=10000
    )
    computer_id: Optional[str] = Field(
        default=None,
        description="Existing computer ID to run task on. If omitted, creates a new computer.",
        min_length=1
    )
    project_name: Optional[str] = Field(
        default=None,
        description="Project name for new computer (only used when computer_id is omitted, defaults to 'MCP Agents')",
        max_length=100
    )
    computer_name: Optional[str] = Field(
        default=None,
        description="Display name for new computer (only used when computer_id is omitted)",
        max_length=100
    )
    max_iterations: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum agent iterations before stopping (1-200, default: 50)"
    )


# ============================================================================
# Helper Functions
# ============================================================================

def get_current_api_key() -> str:
    """
    Get the API key for the current request.

    Priority:
    1. Per-request key from X-Orgo-API-Key header (HTTP transport via context)
    2. Per-request key from middleware ContextVar (fallback)
    3. Environment variable ORGO_API_KEY (stdio transport)

    Raises:
        ValueError: If no API key is available
    """
    # Try to get from MCP context (HTTP transport - request headers)
    if IS_HTTP_MODE:
        try:
            ctx = mcp.get_context()
            if ctx and ctx.request_context and ctx.request_context.request:
                request = ctx.request_context.request
                # Starlette Request has headers as a case-insensitive mapping
                api_key = request.headers.get("x-orgo-api-key")
                if api_key:
                    return api_key
        except Exception:
            # Context may not be available outside of tool calls
            pass

    # Try per-request key from middleware ContextVar (fallback)
    request_key = _request_api_key.get()
    if request_key:
        return request_key

    # Fall back to environment variable (stdio transport)
    env_key = os.environ.get("ORGO_API_KEY")
    if env_key:
        return env_key

    raise ValueError(
        "API key required. For HTTP: include X-Orgo-API-Key header. "
        "For stdio: set ORGO_API_KEY environment variable. "
        "Get your key at https://orgo.ai"
    )


def get_computer(computer_id: str):
    """Get a Computer instance by ID with per-request API key."""
    from orgo import Computer
    api_key = get_current_api_key()
    return Computer(computer_id=computer_id, api_key=api_key)


def get_api_client():
    """Get the Orgo API client with per-request API key."""
    from orgo.api.client import ApiClient
    api_key = get_current_api_key()
    return ApiClient(api_key=api_key)


def _format_pagination_info(total: int, count: int, offset: int) -> Dict[str, Any]:
    """Generate pagination metadata."""
    return {
        "total": total,
        "count": count,
        "offset": offset,
        "has_more": total > offset + count,
        "next_offset": offset + count if total > offset + count else None
    }


# ============================================================================
# Project Management Tools
# ============================================================================

@mcp.tool(
    name="orgo_list_projects",
    annotations={
        "title": "List Orgo Projects",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def orgo_list_projects(params: ListProjectsInput) -> str:
    """
    List all Orgo projects in your account.

    Returns project names and IDs with pagination support. Use project names
    with orgo_list_computers or orgo_create_computer to manage computers.

    Args:
        params (ListProjectsInput): Input parameters containing:
            - limit (int): Maximum results to return, 1-100 (default: 20)
            - offset (int): Number of results to skip for pagination (default: 0)
            - response_format (ResponseFormat): 'markdown' or 'json' (default: markdown)

    Returns:
        str: Formatted response containing:

        Markdown format:
            # Orgo Projects
            Found X projects (showing Y)

            ## project-name (proj_123)
            - **Created**: 2024-01-15 10:30:00

        JSON format:
            {
                "total": int,
                "count": int,
                "offset": int,
                "has_more": bool,
                "next_offset": int | null,
                "projects": [{"id": str, "name": str, "created_at": str}]
            }

    Examples:
        - "List my Orgo projects" -> orgo_list_projects with defaults
        - "Show next page of projects" -> orgo_list_projects with offset=20

    Error Handling:
        - Returns "Error: Invalid API key..." if ORGO_API_KEY is invalid
        - Returns "No projects found" if account has no projects
    """
    def get_projects():
        client = get_api_client()
        all_projects = client.list_projects()
        total = len(all_projects)
        paginated = all_projects[params.offset:params.offset + params.limit]
        return total, [
            ProjectInfo(
                id=p.get("id", ""),
                name=p.get("name", ""),
                created_at=p.get("created_at")
            )
            for p in paginated
        ]

    try:
        total, projects = await asyncio.to_thread(get_projects)

        if not projects:
            return "No projects found. Create one with orgo_create_project."

        if params.response_format == ResponseFormat.JSON:
            response = _format_pagination_info(total, len(projects), params.offset)
            response["projects"] = [p.model_dump() for p in projects]
            return json.dumps(response, indent=2)

        # Markdown format
        lines = ["# Orgo Projects", ""]
        lines.append(f"Found {total} projects (showing {len(projects)})")
        lines.append("")

        for p in projects:
            lines.append(f"## {p.name} (`{p.id}`)")
            if p.created_at:
                lines.append(f"- **Created**: {p.created_at}")
            lines.append("")

        if total > params.offset + len(projects):
            lines.append(f"*More projects available. Use offset={params.offset + len(projects)} to see next page.*")

        return "\n".join(lines)

    except Exception as e:
        return _handle_orgo_error(e)


@mcp.tool(
    name="orgo_list_computers",
    annotations={
        "title": "List Computers in Project",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def orgo_list_computers(params: ListComputersInput) -> str:
    """
    List all computers in a project.

    Returns computer IDs, names, and status with pagination. Use computer_id
    with action tools like orgo_screenshot, orgo_click, orgo_bash, etc.

    Args:
        params (ListComputersInput): Input parameters containing:
            - project_name (str): Project name from orgo_list_projects
            - limit (int): Maximum results, 1-100 (default: 20)
            - offset (int): Skip for pagination (default: 0)
            - response_format (ResponseFormat): 'markdown' or 'json'

    Returns:
        str: Formatted response containing:

        Markdown format:
            # Computers in project-name
            Found X computers (showing Y)

            ## dev-box (`comp_abc`)
            - **Status**: running
            - **OS**: linux
            - **RAM**: 4 GB | **CPU**: 2 cores

        JSON format:
            {
                "total": int,
                "count": int,
                "offset": int,
                "has_more": bool,
                "next_offset": int | null,
                "computers": [{"id": str, "name": str, "status": str, "os": str, "ram": int, "cpu": int}]
            }

    Examples:
        - "List computers in my-project" -> params with project_name="my-project"
        - "Show running VMs" -> list then filter by status

    Error Handling:
        - Returns "Project 'name' not found" if project doesn't exist
        - Returns "No computers found in project" if empty
    """
    def get_computers():
        client = get_api_client()
        all_projects = client.list_projects()

        project = next((p for p in all_projects if p.get("name") == params.project_name), None)
        if not project:
            return None, []

        desktops = project.get("desktops", [])
        total = len(desktops)
        paginated = desktops[params.offset:params.offset + params.limit]

        return total, [
            ComputerInfo(
                id=c.get("id", ""),
                name=c.get("name", ""),
                status=c.get("status", "unknown"),
                os=c.get("os", "linux"),
                ram=c.get("ram"),
                cpu=c.get("cpu")
            )
            for c in paginated
        ]

    try:
        total, computers = await asyncio.to_thread(get_computers)

        if total is None:
            return f"Error: Project '{params.project_name}' not found. Use orgo_list_projects to see available projects."

        if not computers:
            return f"No computers found in project '{params.project_name}'. Create one with orgo_create_computer."

        if params.response_format == ResponseFormat.JSON:
            response = _format_pagination_info(total, len(computers), params.offset)
            response["computers"] = [c.model_dump() for c in computers]
            return json.dumps(response, indent=2)

        # Markdown format
        lines = [f"# Computers in {params.project_name}", ""]
        lines.append(f"Found {total} computers (showing {len(computers)})")
        lines.append("")

        for c in computers:
            status_emoji = {"running": "🟢", "stopped": "🔴", "starting": "🟡"}.get(c.status, "⚪")
            lines.append(f"## {c.name} (`{c.id}`)")
            lines.append(f"- **Status**: {status_emoji} {c.status}")
            lines.append(f"- **OS**: {c.os}")
            if c.ram and c.cpu:
                lines.append(f"- **RAM**: {c.ram} GB | **CPU**: {c.cpu} cores")
            lines.append("")

        if total > params.offset + len(computers):
            lines.append(f"*More computers available. Use offset={params.offset + len(computers)}*")

        return "\n".join(lines)

    except Exception as e:
        return _handle_orgo_error(e)


@mcp.tool(
    name="orgo_create_project",
    annotations={
        "title": "Create Orgo Project",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def orgo_create_project(params: CreateProjectInput) -> str:
    """
    Create a new Orgo project.

    Projects are containers for computers. Create a project first,
    then add computers to it with orgo_create_computer.

    Args:
        params (CreateProjectInput): Input parameters containing:
            - name (str): Unique name for the project (e.g., 'qa-automation')
            - icon_url (Optional[str]): URL for project icon

    Returns:
        str: JSON with created project details:
            {
                "id": str,
                "name": str,
                "status": str,
                "created_at": str,
                "desktops": []
            }

    Examples:
        - "Create project for QA testing" -> params with name="qa-automation"
        - "Make new project called dev-env" -> params with name="dev-env"

    Error Handling:
        - Returns "Error: Conflict..." if project name already exists
    """
    def create():
        client = get_api_client()
        payload = {"name": params.name}
        if params.icon_url:
            payload["icon_url"] = params.icon_url
        data = client._request("POST", "projects", payload)
        project = data.get("project", data)
        return ProjectDetails(
            id=project.get("id", ""),
            name=project.get("name", params.name),
            status=project.get("status", "active"),
            created_at=project.get("created_at"),
            updated_at=project.get("updated_at"),
            icon_url=project.get("icon_url"),
            desktops=[]
        )

    try:
        result = await asyncio.to_thread(create)
        return json.dumps(result.model_dump(), indent=2)
    except Exception as e:
        return _handle_orgo_error(e)


@mcp.tool(
    name="orgo_get_project",
    annotations={
        "title": "Get Project Details",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def orgo_get_project(params: GetProjectInput) -> str:
    """
    Get project details by name.

    Returns full project information including all computers.

    Args:
        params (GetProjectInput): Input containing:
            - name (str): Project name to look up

    Returns:
        str: JSON with project details:
            {
                "id": str,
                "name": str,
                "status": str,
                "created_at": str,
                "desktops": [{"id": str, "name": str, "status": str, ...}]
            }

    Examples:
        - "Get details for my-project" -> params with name="my-project"

    Error Handling:
        - Returns "Error: Resource not found..." if project doesn't exist
    """
    def get_project():
        client = get_api_client()
        data = client._request("GET", f"projects/by-name/{params.name}")
        project = data.get("project", data)
        desktops = [
            ComputerInfo(
                id=c.get("id", ""),
                name=c.get("name", ""),
                status=c.get("status", "unknown"),
                os=c.get("os", "linux"),
                ram=c.get("ram"),
                cpu=c.get("cpu")
            )
            for c in project.get("desktops", [])
        ]
        return ProjectDetails(
            id=project.get("id", ""),
            name=project.get("name", params.name),
            status=project.get("status", "active"),
            created_at=project.get("created_at"),
            updated_at=project.get("updated_at"),
            icon_url=project.get("icon_url"),
            desktops=desktops
        )

    try:
        result = await asyncio.to_thread(get_project)
        return json.dumps(result.model_dump(), indent=2)
    except Exception as e:
        return _handle_orgo_error(e)


@mcp.tool(
    name="orgo_delete_project",
    annotations={
        "title": "Delete Project",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def orgo_delete_project(params: ProjectIdInput) -> str:
    """
    Permanently delete a project and ALL its computers.

    WARNING: This is destructive and cannot be undone.
    All computers and their data in the project will be lost.

    Args:
        params (ProjectIdInput): Input containing:
            - project_id (str): Project ID from orgo_list_projects

    Returns:
        str: Confirmation message

    Examples:
        - "Delete project proj_123" -> params with project_id="proj_123"

    Error Handling:
        - Returns "Error: Resource not found..." if project doesn't exist
    """
    def delete():
        client = get_api_client()
        client._request("POST", f"projects/{params.project_id}/delete")
        return f"Project {params.project_id} and all its computers deleted."

    try:
        return await asyncio.to_thread(delete)
    except Exception as e:
        return _handle_orgo_error(e)


@mcp.tool(
    name="orgo_start_project",
    annotations={
        "title": "Start All Computers in Project",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def orgo_start_project(params: ProjectIdInput) -> str:
    """
    Start all computers in a project.

    Batch operation to boot all computers simultaneously.
    Computers boot in under 500ms each.

    Args:
        params (ProjectIdInput): Input containing:
            - project_id (str): Project ID

    Returns:
        str: Confirmation message

    Examples:
        - "Start all computers in project proj_123" -> params with project_id="proj_123"
    """
    def start():
        client = get_api_client()
        client._request("POST", f"projects/{params.project_id}/start")
        return f"All computers in project {params.project_id} starting."

    try:
        return await asyncio.to_thread(start)
    except Exception as e:
        return _handle_orgo_error(e)


@mcp.tool(
    name="orgo_stop_project",
    annotations={
        "title": "Stop All Computers in Project",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def orgo_stop_project(params: ProjectIdInput) -> str:
    """
    Stop all computers in a project.

    Batch operation to stop all computers and save costs.
    Computers can be restarted later with orgo_start_project.

    Args:
        params (ProjectIdInput): Input containing:
            - project_id (str): Project ID

    Returns:
        str: Confirmation message

    Examples:
        - "Stop all computers in project proj_123" -> params with project_id="proj_123"
    """
    def stop():
        client = get_api_client()
        client._request("POST", f"projects/{params.project_id}/stop")
        return f"All computers in project {params.project_id} stopping."

    try:
        return await asyncio.to_thread(stop)
    except Exception as e:
        return _handle_orgo_error(e)


@mcp.tool(
    name="orgo_restart_project",
    annotations={
        "title": "Restart All Computers in Project",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def orgo_restart_project(params: ProjectIdInput) -> str:
    """
    Restart all computers in a project.

    Batch operation to restart all computers simultaneously.

    Args:
        params (ProjectIdInput): Input containing:
            - project_id (str): Project ID

    Returns:
        str: Confirmation message

    Examples:
        - "Restart all computers in project proj_123"
    """
    def restart():
        client = get_api_client()
        client._request("POST", f"projects/{params.project_id}/restart")
        return f"All computers in project {params.project_id} restarting."

    try:
        return await asyncio.to_thread(restart)
    except Exception as e:
        return _handle_orgo_error(e)


@mcp.tool(
    name="orgo_create_computer",
    annotations={
        "title": "Create Virtual Computer",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def orgo_create_computer(params: CreateComputerInput) -> str:
    """
    Create a new virtual computer in a project.

    The computer boots in under 500ms and starts in 'running' status.
    Returns the computer ID for use with action tools.

    Args:
        params (CreateComputerInput): Input containing:
            - project_name (str): Project to create the computer in
            - name (str): Display name for the computer
            - os (Literal): 'linux' or 'windows' (default: linux)
            - ram (Literal): RAM in GB - 1, 2, 4, 8, 16, 32, 64 (default: 2)
            - cpu (Literal): CPU cores - 1, 2, 4, 8, 16 (default: 2)

    Returns:
        str: JSON with created computer details:
            {
                "id": str,
                "name": str,
                "status": "running",
                "os": str,
                "ram": int,
                "cpu": int
            }

    Examples:
        - "Create Linux computer with 4GB RAM" -> params with ram=4
        - "Create Windows dev-box" -> params with os="windows", name="dev-box"

    Error Handling:
        - Returns "Error: Insufficient credits..." if account balance is low
    """
    def create():
        from orgo import Computer
        computer = Computer(
            project=params.project_name,
            name=params.name,
            os=params.os,
            ram=params.ram,
            cpu=params.cpu,
            api_key=get_current_api_key(),
        )
        return ComputerInfo(
            id=computer.computer_id,
            name=params.name,
            status="running",
            os=params.os,
            ram=params.ram,
            cpu=params.cpu
        )

    try:
        result = await asyncio.to_thread(create)
        return json.dumps(result.model_dump(), indent=2)
    except Exception as e:
        return _handle_orgo_error(e)


@mcp.tool(
    name="orgo_get_computer",
    annotations={
        "title": "Get Computer Details",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def orgo_get_computer(params: ComputerIdInput) -> str:
    """
    Get full details for a computer including access URL.

    Returns comprehensive information about a computer including its
    access URL for direct browser viewing.

    Args:
        params (ComputerIdInput): Input containing:
            - computer_id (str): Computer ID

    Returns:
        str: JSON with computer details:
            {
                "id": str,
                "name": str,
                "project_name": str,
                "os": str,
                "ram": int,
                "cpu": int,
                "status": str,
                "url": str,
                "created_at": str
            }

    Examples:
        - "Get details for computer abc123" -> params with computer_id="abc123"
        - "What's the URL for my computer?" -> get_computer then access url field
    """
    def get_details():
        client = get_api_client()
        data = client._request("GET", f"computers/{params.computer_id}")
        computer = data.get("computer", data)
        return ComputerDetails(
            id=computer.get("id", params.computer_id),
            name=computer.get("name", ""),
            project_name=computer.get("project_name", ""),
            os=computer.get("os", "linux"),
            ram=computer.get("ram", 2),
            cpu=computer.get("cpu", 2),
            status=computer.get("status", "unknown"),
            url=computer.get("url", ""),
            created_at=computer.get("created_at")
        )

    try:
        result = await asyncio.to_thread(get_details)
        return json.dumps(result.model_dump(), indent=2)
    except Exception as e:
        return _handle_orgo_error(e)


@mcp.tool(
    name="orgo_start_computer",
    annotations={
        "title": "Start Computer",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def orgo_start_computer(params: ComputerIdInput) -> str:
    """
    Start a stopped computer.

    Boots in under 500ms. Use orgo_list_computers to find computer IDs.

    Args:
        params (ComputerIdInput): Input containing:
            - computer_id (str): Computer ID to start

    Returns:
        str: Confirmation message

    Examples:
        - "Start computer abc123" -> params with computer_id="abc123"
    """
    def start():
        client = get_api_client()
        client._request("POST", f"computers/{params.computer_id}/start")
        return f"Computer {params.computer_id} starting"

    try:
        return await asyncio.to_thread(start)
    except Exception as e:
        return _handle_orgo_error(e)


@mcp.tool(
    name="orgo_stop_computer",
    annotations={
        "title": "Stop Computer",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def orgo_stop_computer(params: ComputerIdInput) -> str:
    """
    Stop a running computer to save costs.

    The computer can be restarted later with orgo_start_computer.
    Data on the computer is preserved.

    Args:
        params (ComputerIdInput): Input containing:
            - computer_id (str): Computer ID to stop

    Returns:
        str: Confirmation message
    """
    def stop():
        client = get_api_client()
        client._request("POST", f"computers/{params.computer_id}/stop")
        return f"Computer {params.computer_id} stopping"

    try:
        return await asyncio.to_thread(stop)
    except Exception as e:
        return _handle_orgo_error(e)


@mcp.tool(
    name="orgo_restart_computer",
    annotations={
        "title": "Restart Computer",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def orgo_restart_computer(params: ComputerIdInput) -> str:
    """
    Restart a computer.

    Useful for recovering from unresponsive states or resetting to a clean environment.

    Args:
        params (ComputerIdInput): Input containing:
            - computer_id (str): Computer ID to restart

    Returns:
        str: Confirmation message
    """
    def restart():
        client = get_api_client()
        client._request("POST", f"computers/{params.computer_id}/restart")
        return f"Computer {params.computer_id} restarting"

    try:
        return await asyncio.to_thread(restart)
    except Exception as e:
        return _handle_orgo_error(e)


@mcp.tool(
    name="orgo_delete_computer",
    annotations={
        "title": "Delete Computer",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def orgo_delete_computer(params: ComputerIdInput) -> str:
    """
    Permanently delete a computer.

    WARNING: This is destructive and cannot be undone.
    All data on the computer will be lost.

    Args:
        params (ComputerIdInput): Input containing:
            - computer_id (str): Computer ID to delete

    Returns:
        str: Confirmation message
    """
    def delete():
        client = get_api_client()
        client._request("DELETE", f"computers/{params.computer_id}")
        return f"Computer {params.computer_id} deleted"

    try:
        return await asyncio.to_thread(delete)
    except Exception as e:
        return _handle_orgo_error(e)


# ============================================================================
# Screen Action Tools
# ============================================================================

@mcp.tool(
    name="orgo_screenshot",
    annotations={
        "title": "Take Screenshot",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def orgo_screenshot(params: ComputerIdInput) -> Image:
    """
    Take a screenshot of the computer's display.

    Returns a JPEG image of the current screen. Use this to see what's
    on screen before clicking or typing.

    Args:
        params (ComputerIdInput): Input containing:
            - computer_id (str): Computer ID to screenshot

    Returns:
        Image: JPEG screenshot of the current display

    Examples:
        - "Take a screenshot of computer abc123"
        - "Show me what's on the screen"
    """
    def take_screenshot():
        computer = get_computer(params.computer_id)
        return computer.screenshot_base64()

    try:
        screenshot_base64 = await asyncio.to_thread(take_screenshot)
        screenshot_bytes = base64.b64decode(screenshot_base64)
        return Image(data=screenshot_bytes, format="jpeg")
    except Exception as e:
        raise RuntimeError(_handle_orgo_error(e))


@mcp.tool(
    name="orgo_click",
    annotations={
        "title": "Click at Coordinates",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def orgo_click(params: ClickInput) -> str:
    """
    Click at (x, y) coordinates on the screen.

    Use orgo_screenshot first to see the screen and identify click targets.
    Coordinates are in pixels from top-left corner.

    Args:
        params (ClickInput): Input containing:
            - computer_id (str): Computer ID
            - x (int): Horizontal position in pixels
            - y (int): Vertical position in pixels
            - button (Literal): 'left', 'right', or 'middle' (default: left)

    Returns:
        str: Confirmation of click action

    Examples:
        - "Click at (500, 300)" -> params with x=500, y=300
        - "Right-click at 100, 200" -> params with x=100, y=200, button="right"
    """
    def click():
        computer = get_computer(params.computer_id)
        if params.button == "left":
            computer.left_click(params.x, params.y)
        elif params.button == "right":
            computer.right_click(params.x, params.y)
        return f"Clicked {params.button} at ({params.x}, {params.y})"

    try:
        return await asyncio.to_thread(click)
    except Exception as e:
        return _handle_orgo_error(e)


@mcp.tool(
    name="orgo_double_click",
    annotations={
        "title": "Double-Click at Coordinates",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def orgo_double_click(params: DoubleClickInput) -> str:
    """
    Double-click at (x, y) coordinates.

    Use for opening files/apps on desktop or selecting text.

    Args:
        params (DoubleClickInput): Input containing:
            - computer_id (str): Computer ID
            - x (int): Horizontal position in pixels
            - y (int): Vertical position in pixels

    Returns:
        str: Confirmation of double-click action
    """
    def double_click():
        computer = get_computer(params.computer_id)
        computer.double_click(params.x, params.y)
        return f"Double-clicked at ({params.x}, {params.y})"

    try:
        return await asyncio.to_thread(double_click)
    except Exception as e:
        return _handle_orgo_error(e)


@mcp.tool(
    name="orgo_type",
    annotations={
        "title": "Type Text",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def orgo_type(params: TypeInput) -> str:
    """
    Type text at the current cursor position.

    Click on an input field first, then use this to type text.

    Args:
        params (TypeInput): Input containing:
            - computer_id (str): Computer ID
            - text (str): Text to type

    Returns:
        str: Confirmation showing what was typed (truncated if long)

    Examples:
        - "Type 'hello world'" -> params with text="hello world"
    """
    def type_text():
        computer = get_computer(params.computer_id)
        computer.type(params.text)
        preview = params.text[:50] + '...' if len(params.text) > 50 else params.text
        return f"Typed: {preview}"

    try:
        return await asyncio.to_thread(type_text)
    except Exception as e:
        return _handle_orgo_error(e)


@mcp.tool(
    name="orgo_key",
    annotations={
        "title": "Press Keyboard Key",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def orgo_key(params: KeyInput) -> str:
    """
    Press a keyboard key or combination.

    Supports single keys and combinations with modifiers (ctrl, alt, shift, cmd).

    Args:
        params (KeyInput): Input containing:
            - computer_id (str): Computer ID
            - key (str): Key name or combination
                - Single keys: Enter, Tab, Escape, Backspace, Delete, Space
                - Arrow keys: Up, Down, Left, Right
                - Function keys: F1-F12
                - Combos: ctrl+c, ctrl+v, alt+Tab, ctrl+shift+s

    Returns:
        str: Confirmation of key press

    Examples:
        - "Press Enter" -> params with key="Enter"
        - "Press Ctrl+C" -> params with key="ctrl+c"
    """
    def press_key():
        computer = get_computer(params.computer_id)
        computer.key(params.key)
        return f"Pressed: {params.key}"

    try:
        return await asyncio.to_thread(press_key)
    except Exception as e:
        return _handle_orgo_error(e)


@mcp.tool(
    name="orgo_scroll",
    annotations={
        "title": "Scroll Page",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def orgo_scroll(params: ScrollInput) -> str:
    """
    Scroll the page in the specified direction.

    Args:
        params (ScrollInput): Input containing:
            - computer_id (str): Computer ID
            - direction (Literal): 'up', 'down', 'left', or 'right'
            - amount (int): How much to scroll, 1-10 (default: 3)

    Returns:
        str: Confirmation of scroll action
    """
    def scroll():
        computer = get_computer(params.computer_id)
        computer.scroll(params.direction, params.amount)
        return f"Scrolled {params.direction} by {params.amount}"

    try:
        return await asyncio.to_thread(scroll)
    except Exception as e:
        return _handle_orgo_error(e)


@mcp.tool(
    name="orgo_drag",
    annotations={
        "title": "Drag Mouse",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def orgo_drag(params: DragInput) -> str:
    """
    Drag from one position to another.

    Useful for drag-and-drop operations, selecting text, or resizing windows.

    Args:
        params (DragInput): Input containing:
            - computer_id (str): Computer ID
            - start_x (int): Starting horizontal position
            - start_y (int): Starting vertical position
            - end_x (int): Ending horizontal position
            - end_y (int): Ending vertical position
            - duration (float): How long the drag takes, 0.1-5.0 (default: 0.5)

    Returns:
        str: Confirmation of drag action
    """
    def drag():
        computer = get_computer(params.computer_id)
        computer.drag(params.start_x, params.start_y, params.end_x, params.end_y, duration=params.duration)
        return f"Dragged from ({params.start_x}, {params.start_y}) to ({params.end_x}, {params.end_y})"

    try:
        return await asyncio.to_thread(drag)
    except Exception as e:
        return _handle_orgo_error(e)


@mcp.tool(
    name="orgo_wait",
    annotations={
        "title": "Wait",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def orgo_wait(params: WaitInput) -> str:
    """
    Wait for a specified duration.

    Useful for waiting for pages to load, animations to complete,
    or applications to start.

    Args:
        params (WaitInput): Input containing:
            - computer_id (str): Computer ID
            - seconds (float): How long to wait, 0.1-60 (default: 2)

    Returns:
        str: Confirmation of wait completion
    """
    def wait():
        computer = get_computer(params.computer_id)
        computer.wait(params.seconds)
        return f"Waited {params.seconds} seconds"

    try:
        return await asyncio.to_thread(wait)
    except Exception as e:
        return _handle_orgo_error(e)


# ============================================================================
# Shell Command Tools
# ============================================================================

@mcp.tool(
    name="orgo_bash",
    annotations={
        "title": "Execute Bash Command",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def orgo_bash(params: BashInput) -> str:
    """
    Execute a bash command on the computer.

    Useful for file operations, installing packages, running scripts,
    checking system state, etc.

    Args:
        params (BashInput): Input containing:
            - computer_id (str): Computer ID
            - command (str): Bash command to run

    Returns:
        str: Command output (stdout and stderr combined)

    Examples:
        - "Run ls -la" -> params with command="ls -la"
        - "Install requests" -> params with command="pip install requests"
    """
    def run_bash():
        computer = get_computer(params.computer_id)
        output = computer.bash(params.command)
        return f"$ {params.command}\n\n{output}"

    try:
        return await asyncio.to_thread(run_bash)
    except Exception as e:
        return _handle_orgo_error(e)


@mcp.tool(
    name="orgo_exec",
    annotations={
        "title": "Execute Python Code",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def orgo_exec(params: ExecInput) -> str:
    """
    Execute Python code on the computer.

    Returns the output of the execution. Useful for data processing,
    file manipulation, and quick scripts.

    Args:
        params (ExecInput): Input containing:
            - computer_id (str): Computer ID
            - code (str): Python code to execute
            - timeout (int): Max execution time, 1-300 seconds (default: 30)

    Returns:
        str: Python execution output
    """
    def run_exec():
        computer = get_computer(params.computer_id)
        output = computer.exec(params.code, timeout=params.timeout)
        return f">>> Python execution:\n{output}"

    try:
        return await asyncio.to_thread(run_exec)
    except Exception as e:
        return _handle_orgo_error(e)


# ============================================================================
# AI Agent Tools
# ============================================================================

# Storage for fire-and-forget background tasks (prevents garbage collection)
_background_prompt_tasks: Dict[str, asyncio.Task] = {}


async def _execute_prompt_in_background(
    computer_id: str,
    prompt: str,
    max_iterations: int,
    api_key: str
) -> None:
    """
    Execute computer.prompt() in background thread.

    This function runs asynchronously without blocking the MCP response.
    Errors are logged but not raised (fire-and-forget pattern).
    """
    def run_prompt_sync():
        from orgo import Computer
        computer = Computer(computer_id=computer_id, api_key=api_key)
        computer.prompt(
            prompt,
            max_iterations=max_iterations,
            verbose=False
        )

    try:
        await asyncio.to_thread(run_prompt_sync)
        logger.info(f"Background prompt completed for computer {computer_id}")
    except Exception as e:
        # Log but don't raise - task is fire-and-forget
        logger.error(f"Background prompt failed for computer {computer_id}: {e}")
    finally:
        # Clean up task reference
        _background_prompt_tasks.pop(computer_id, None)


@mcp.tool(
    name="orgo_prompt",
    annotations={
        "title": "Run AI Agent Task",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def orgo_prompt(params: PromptInput) -> str:
    """
    Send an AI agent to complete a task on an Orgo computer (fire-and-forget).

    Launches a task asynchronously and returns immediately with a URL to monitor
    progress. The agent runs on Orgo's hosted AI infrastructure, controlling
    the computer with mouse, keyboard, and bash commands to complete the task.

    This is ideal for long-running tasks where you don't want to wait for
    completion. Check progress at the returned URL or use orgo_screenshot.

    Args:
        params (PromptInput): Input containing:
            - prompt (str): Natural language instruction for the AI agent
            - computer_id (Optional[str]): Existing computer ID. Creates new if omitted
            - project_name (Optional[str]): Project for new computer (default: 'MCP Agents')
            - computer_name (Optional[str]): Display name for new computer
            - max_iterations (int): Max agent loops, 1-200 (default: 50)

    Returns:
        str: Markdown-formatted response containing:
            - Computer ID for reference
            - Task summary (truncated if long)
            - URL to monitor progress at orgo.ai
            - Instructions for checking status and cleanup

    Examples:
        - "Search for AI news and create a summary document"
          -> Creates new computer, starts agent, returns URL immediately
        - "Fill out the contact form with test data" (with computer_id)
          -> Uses existing computer, starts agent on current screen state

    Error Handling:
        - Returns "Error: Resource not found..." if computer_id is invalid
        - Returns "Error: Insufficient credits..." if account balance is low
        - Returns "Error: Invalid API key..." if authentication fails
    """
    api_key = get_current_api_key()

    def setup_computer() -> tuple:
        """Create or connect to computer (synchronous, runs in thread)."""
        from orgo import Computer

        if params.computer_id:
            # Connect to existing computer
            computer = Computer(computer_id=params.computer_id, api_key=api_key)
            return computer.computer_id, computer.url
        else:
            # Create new computer in specified project
            computer = Computer(
                project=params.project_name or "MCP Agents",
                name=params.computer_name,
                api_key=api_key
            )
            return computer.computer_id, computer.url

    try:
        # Step 1: Create or connect to computer (fast, ~500ms)
        computer_id, computer_url = await asyncio.to_thread(setup_computer)

        # Step 2: Launch prompt in background (non-blocking)
        task = asyncio.create_task(
            _execute_prompt_in_background(
                computer_id=computer_id,
                prompt=params.prompt,
                max_iterations=params.max_iterations,
                api_key=api_key
            )
        )
        _background_prompt_tasks[computer_id] = task

        # Step 3: Return immediately with monitoring info
        task_preview = params.prompt[:100] + ('...' if len(params.prompt) > 100 else '')

        return (
            f"# AI Agent Dispatched\n\n"
            f"**Computer ID:** `{computer_id}`\n\n"
            f"**Task:** {task_preview}\n\n"
            f"**Monitor progress:** {computer_url}\n\n"
            f"---\n\n"
            f"The agent is now running autonomously on Orgo's infrastructure.\n\n"
            f"**To check status:**\n"
            f"- Visit the URL above to watch the agent in real-time\n"
            f"- Use `orgo_screenshot` with computer_id `{computer_id}`\n\n"
            f"**When complete:**\n"
            f"- Use `orgo_delete_computer` to clean up (or leave running for future tasks)"
        )

    except Exception as e:
        return _handle_orgo_error(e)


# ============================================================================
# File Management Tools
# ============================================================================

@mcp.tool(
    name="orgo_list_files",
    annotations={
        "title": "List Files",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def orgo_list_files(params: ListFilesInput) -> str:
    """
    List all files associated with a computer.

    Shows both uploaded files and files exported from the computer.
    Use file IDs with orgo_download_file or orgo_delete_file.

    Args:
        params (ListFilesInput): Input containing:
            - computer_id (str): Computer ID
            - limit (int): Maximum results, 1-100 (default: 20)
            - offset (int): Skip for pagination (default: 0)
            - response_format (ResponseFormat): 'markdown' or 'json'

    Returns:
        str: Formatted list of files with pagination info
    """
    def list_files():
        client = get_api_client()
        data = client._request("GET", f"computers/{params.computer_id}/files")
        all_files = data.get("files", [])
        total = len(all_files)
        paginated = all_files[params.offset:params.offset + params.limit]
        return total, [
            FileInfo(
                id=f.get("id", ""),
                filename=f.get("filename", ""),
                size_bytes=f.get("size_bytes", 0),
                content_type=f.get("content_type"),
                created_at=f.get("created_at")
            )
            for f in paginated
        ]

    try:
        total, files = await asyncio.to_thread(list_files)

        if not files:
            return "No files found. Use orgo_upload_file or orgo_export_file to add files."

        if params.response_format == ResponseFormat.JSON:
            response = _format_pagination_info(total, len(files), params.offset)
            response["files"] = [f.model_dump() for f in files]
            return json.dumps(response, indent=2)

        # Markdown format
        lines = ["# Files", ""]
        lines.append(f"Found {total} files (showing {len(files)})")
        lines.append("")

        for f in files:
            size_kb = f.size_bytes / 1024
            lines.append(f"## {f.filename} (`{f.id}`)")
            lines.append(f"- **Size**: {size_kb:.1f} KB")
            if f.content_type:
                lines.append(f"- **Type**: {f.content_type}")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        return _handle_orgo_error(e)


@mcp.tool(
    name="orgo_export_file",
    annotations={
        "title": "Export File from Computer",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def orgo_export_file(params: ExportFileInput) -> str:
    """
    Export a file from the computer's filesystem.

    The computer must be running. Files can only be exported from /home/user.
    Returns a download URL that expires in 1 hour.

    Args:
        params (ExportFileInput): Input containing:
            - computer_id (str): Computer ID
            - path (str): Path to file (e.g., 'Desktop/results.txt')

    Returns:
        str: File info and download URL
    """
    def export_file():
        client = get_api_client()
        data = client._request("POST", "files/export", {
            "desktopId": params.computer_id,
            "path": params.path
        })
        file_info = data.get("file", {})
        url = data.get("url", "")
        return f"""# File Exported

**Filename:** {file_info.get('filename', 'unknown')}
**Size:** {file_info.get('size_bytes', 0)} bytes
**File ID:** `{file_info.get('id', 'unknown')}`

**Download URL:** {url}

*URL expires in 1 hour. Use orgo_download_file with the file ID to get a fresh URL later.*"""

    try:
        return await asyncio.to_thread(export_file)
    except Exception as e:
        return _handle_orgo_error(e)


@mcp.tool(
    name="orgo_upload_file",
    annotations={
        "title": "Upload File to Computer",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def orgo_upload_file(params: UploadFileInput) -> str:
    """
    Upload a file to the computer's Desktop.

    The file will sync to all running computers in the project.
    Maximum file size: 10MB.

    Args:
        params (UploadFileInput): Input containing:
            - computer_id (str): Computer ID
            - filename (str): Name for the file
            - content_base64 (str): Base64-encoded content
            - content_type (Optional[str]): MIME type

    Returns:
        str: Confirmation message
    """
    def upload_file():
        client = get_api_client()
        payload = {
            "filename": params.filename,
            "content": params.content_base64,
        }
        if params.content_type:
            payload["content_type"] = params.content_type

        client._request("POST", f"computers/{params.computer_id}/files/upload", payload)
        return f"File '{params.filename}' uploaded successfully to Desktop."

    try:
        return await asyncio.to_thread(upload_file)
    except Exception as e:
        return _handle_orgo_error(e)


@mcp.tool(
    name="orgo_download_file",
    annotations={
        "title": "Get File Download URL",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def orgo_download_file(params: FileIdInput) -> str:
    """
    Get a signed download URL for a file.

    The URL expires after 1 hour. Use orgo_list_files to find file IDs.

    Args:
        params (FileIdInput): Input containing:
            - file_id (str): File ID

    Returns:
        str: Download URL (expires in 1 hour)
    """
    def download_file():
        client = get_api_client()
        data = client._request("GET", f"files/{params.file_id}/download")
        return f"**Download URL:** {data.get('url', '')}\n\n*URL expires in 1 hour.*"

    try:
        return await asyncio.to_thread(download_file)
    except Exception as e:
        return _handle_orgo_error(e)


@mcp.tool(
    name="orgo_delete_file",
    annotations={
        "title": "Delete File",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def orgo_delete_file(params: FileIdInput) -> str:
    """
    Delete a file from storage.

    WARNING: This permanently removes the file from cloud storage.

    Args:
        params (FileIdInput): Input containing:
            - file_id (str): File ID to delete

    Returns:
        str: Confirmation message
    """
    def delete_file():
        client = get_api_client()
        client._request("DELETE", f"files/{params.file_id}")
        return f"File {params.file_id} deleted."

    try:
        return await asyncio.to_thread(delete_file)
    except Exception as e:
        return _handle_orgo_error(e)


# ============================================================================
# Streaming Tools
# ============================================================================

@mcp.tool(
    name="orgo_start_stream",
    annotations={
        "title": "Start RTMP Stream",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def orgo_start_stream(params: StartStreamInput) -> str:
    """
    Start RTMP streaming from a computer.

    Stream the computer's display to Twitch, YouTube, or any RTMP endpoint.
    Computer must be running. Only one stream per computer at a time.

    Args:
        params (StartStreamInput): Input containing:
            - computer_id (str): Computer ID
            - rtmp_url (str): Full RTMP URL with stream key
            - resolution (Literal): '1920x1080', '1280x720', '854x480' (default: 720p)
            - fps (Literal): 15, 30, or 60 (default: 30)
            - bitrate (str): Video bitrate (default: '2500k')

    Returns:
        str: JSON with stream status

    Examples:
        - "Stream to Twitch" -> params with rtmp_url="rtmp://live.twitch.tv/app/key"
    """
    def start_stream():
        client = get_api_client()
        data = client._request("POST", f"computers/{params.computer_id}/stream/start", {
            "rtmp_url": params.rtmp_url,
            "resolution": params.resolution,
            "fps": params.fps,
            "bitrate": params.bitrate
        })
        return StreamStatus(
            status="streaming",
            pid=data.get("pid"),
            start_time=data.get("start_time"),
            message=data.get("message", "Stream started successfully")
        )

    try:
        result = await asyncio.to_thread(start_stream)
        return json.dumps(result.model_dump(), indent=2)
    except Exception as e:
        return _handle_orgo_error(e)


@mcp.tool(
    name="orgo_stream_status",
    annotations={
        "title": "Get Stream Status",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def orgo_stream_status(params: ComputerIdInput) -> str:
    """
    Get the current streaming status of a computer.

    Check if a stream is active, idle, or terminated.

    Args:
        params (ComputerIdInput): Input containing:
            - computer_id (str): Computer ID

    Returns:
        str: JSON with stream status
    """
    def get_status():
        client = get_api_client()
        data = client._request("GET", f"computers/{params.computer_id}/stream/status")
        return StreamStatus(
            status=data.get("status", "idle"),
            pid=data.get("pid"),
            start_time=data.get("start_time"),
            message=data.get("message")
        )

    try:
        result = await asyncio.to_thread(get_status)
        return json.dumps(result.model_dump(), indent=2)
    except Exception as e:
        return _handle_orgo_error(e)


@mcp.tool(
    name="orgo_stop_stream",
    annotations={
        "title": "Stop RTMP Stream",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def orgo_stop_stream(params: ComputerIdInput) -> str:
    """
    Stop RTMP streaming from a computer.

    Args:
        params (ComputerIdInput): Input containing:
            - computer_id (str): Computer ID

    Returns:
        str: JSON with final stream status
    """
    def stop_stream():
        client = get_api_client()
        data = client._request("POST", f"computers/{params.computer_id}/stream/stop")
        return StreamStatus(
            status="terminated",
            pid=None,
            start_time=None,
            message=data.get("message", "Stream stopped successfully")
        )

    try:
        result = await asyncio.to_thread(stop_stream)
        return json.dumps(result.model_dump(), indent=2)
    except Exception as e:
        return _handle_orgo_error(e)


# ============================================================================
# AI / OpenRouter Integration Tools
# ============================================================================

@mcp.tool(
    name="orgo_list_ai_models",
    annotations={
        "title": "List AI Models",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True
    }
)
async def orgo_list_ai_models(params: ListAIModelsInput) -> str:
    """
    List available AI models through OpenRouter.

    Returns a list of 400+ AI models available for use with orgo_ai_completion.
    Requires OpenRouter API key configured in your Orgo account.

    Args:
        params (ListAIModelsInput): Input containing:
            - limit (int): Maximum results, 1-200 (default: 50)
            - offset (int): Skip for pagination (default: 0)
            - response_format (ResponseFormat): 'markdown' or 'json'

    Returns:
        str: Formatted list of AI models with pagination

    Examples:
        - "List available AI models" -> orgo_list_ai_models with defaults
    """
    def list_models():
        client = get_api_client()
        data = client._request("GET", "ai")
        all_models = data.get("models", data.get("data", []))
        total = len(all_models)
        paginated = all_models[params.offset:params.offset + params.limit]
        return total, [
            AIModel(
                id=m.get("id", ""),
                name=m.get("name", m.get("id", "")),
                description=m.get("description"),
                context_length=m.get("context_length"),
                pricing=m.get("pricing")
            )
            for m in paginated
        ]

    try:
        total, models = await asyncio.to_thread(list_models)

        if not models:
            return "No AI models found. Configure OpenRouter API key in your Orgo account."

        if params.response_format == ResponseFormat.JSON:
            response = _format_pagination_info(total, len(models), params.offset)
            response["models"] = [m.model_dump() for m in models]
            return json.dumps(response, indent=2)

        # Markdown format
        lines = ["# AI Models", ""]
        lines.append(f"Found {total} models (showing {len(models)})")
        lines.append("")

        for m in models:
            lines.append(f"## {m.name}")
            lines.append(f"- **ID**: `{m.id}`")
            if m.context_length:
                lines.append(f"- **Context**: {m.context_length:,} tokens")
            if m.description:
                lines.append(f"- {m.description[:100]}...")
            lines.append("")

        if total > params.offset + len(models):
            lines.append(f"*More models available. Use offset={params.offset + len(models)}*")

        return "\n".join(lines)

    except Exception as e:
        return _handle_orgo_error(e)


@mcp.tool(
    name="orgo_ai_completion",
    annotations={
        "title": "Run AI Completion",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def orgo_ai_completion(params: AICompletionInput) -> str:
    """
    Run an AI completion using OpenRouter's 400+ models.

    Access models from OpenAI, Anthropic, Google, Meta, and more through
    a unified API. Requires OpenRouter key in your Orgo account settings.

    Args:
        params (AICompletionInput): Input containing:
            - model (str): Model ID (e.g., 'openai/gpt-4')
            - prompt (str): The prompt to send
            - system (Optional[str]): Optional system message
            - max_tokens (int): Max response length, 1-100000 (default: 1024)
            - temperature (float): Randomness, 0-2 (default: 0.7)

    Returns:
        str: The model's response text

    Examples:
        - "Ask GPT-4 to explain quantum computing" ->
          params with model="openai/gpt-4", prompt="Explain quantum computing"
    """
    def run_completion():
        client = get_api_client()
        payload = {
            "model": params.model,
            "messages": [{"role": "user", "content": params.prompt}],
            "max_tokens": params.max_tokens,
            "temperature": params.temperature
        }
        if params.system:
            payload["messages"].insert(0, {"role": "system", "content": params.system})

        data = client._request("POST", "ai", payload)
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return data.get("content", str(data))

    try:
        return await asyncio.to_thread(run_completion)
    except Exception as e:
        return _handle_orgo_error(e)


# ============================================================================
# Health Check Endpoint (for HTTP transport / cloud deployments)
# ============================================================================

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    """
    Health check endpoint for load balancers and container orchestrators.

    Returns 200 OK if the server is running. Does not require authentication.
    """
    return PlainTextResponse("OK", status_code=200)


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """
    Entry point for the Orgo MCP server with dual transport support.

    Transport modes:
    - stdio (default): For local CLI usage, requires ORGO_API_KEY env var
    - http: For cloud deployment, accepts X-Orgo-API-Key header per request
    """
    if TRANSPORT_MODE == "stdio":
        # Stdio transport requires ORGO_API_KEY environment variable
        if not os.environ.get("ORGO_API_KEY"):
            logger.error("ORGO_API_KEY not set. Get your key at https://orgo.ai")
            exit(1)

        logger.info("Starting Orgo MCP server (stdio transport, 34 tools)")
        mcp.run()

    elif TRANSPORT_MODE in ("http", "streamable-http"):
        # HTTP transport - API key extracted from request headers in tools (BYOK)
        # Host/port configured during FastMCP initialization at module level
        logger.info(f"Starting Orgo MCP server (HTTP transport) on {HTTP_HOST}:{HTTP_PORT}")
        logger.info("Users must include X-Orgo-API-Key header with each request")

        # Use FastMCP's built-in streamable HTTP transport
        # Health check is available via @mcp.custom_route("/health")
        # MCP endpoint is at /mcp
        mcp.run(transport="streamable-http")

    else:
        logger.error(f"Unknown transport: {TRANSPORT_MODE}. Use 'stdio' or 'http'")
        exit(1)


if __name__ == "__main__":
    main()
