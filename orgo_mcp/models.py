"""Pydantic input models for Orgo MCP tools."""

from typing import Optional, Literal
from pydantic import BaseModel, Field


# =============================================================================
# Workspace Models
# =============================================================================

class CreateWorkspaceInput(BaseModel):
    name: str = Field(..., description="Unique workspace name (letters, numbers, hyphens, underscores)", min_length=1, max_length=64)


class WorkspaceIdInput(BaseModel):
    workspace_id: str = Field(..., description="Workspace ID (from orgo_list_workspaces)", min_length=1)


# =============================================================================
# Computer Models
# =============================================================================

class ListComputersInput(BaseModel):
    workspace_id: str = Field(..., description="Workspace ID to list computers from", min_length=1)


class CreateComputerInput(BaseModel):
    workspace: str = Field(..., description="Workspace name (created automatically if doesn't exist)", min_length=1)
    name: Optional[str] = Field(default=None, description="Computer name (auto-generated if omitted)", max_length=100)
    os: Literal["linux"] = Field(default="linux", description="Operating system (only linux supported)")
    ram: Literal[4, 8, 16, 32, 64] = Field(default=4, description="RAM in GB")
    cpu: Literal[1, 2, 4, 8, 16] = Field(default=2, description="CPU cores")
    gpu: Literal["none", "a10", "l40s", "a100-40gb", "a100-80gb"] = Field(default="none", description="GPU type")
    resolution: str = Field(default="1280x720x24", description="Display resolution (WIDTHxHEIGHTxDEPTH)")
    auto_stop_minutes: Optional[int] = Field(default=None, description="Auto-stop after N minutes idle. 0 to disable. Free tier: always 15min.")
    image: Optional[str] = Field(default=None, description="Custom template image reference (from orgo Forge/Templates)")


class ComputerIdInput(BaseModel):
    computer_id: str = Field(..., description="Computer ID", min_length=1)


# =============================================================================
# Action Models
# =============================================================================

class ClickInput(BaseModel):
    computer_id: str = Field(..., description="Computer ID", min_length=1)
    x: int = Field(..., ge=0, description="X coordinate (pixels from left)")
    y: int = Field(..., ge=0, description="Y coordinate (pixels from top)")
    button: Literal["left", "right"] = Field(default="left", description="Mouse button")
    double: bool = Field(default=False, description="Double-click if true")


class TypeInput(BaseModel):
    computer_id: str = Field(..., description="Computer ID", min_length=1)
    text: str = Field(..., description="Text to type at cursor position", min_length=1)


class KeyInput(BaseModel):
    computer_id: str = Field(..., description="Computer ID", min_length=1)
    key: str = Field(..., description="Key or combo: Enter, Tab, Escape, ctrl+c, alt+Tab, ctrl+shift+s", min_length=1)


class ScrollInput(BaseModel):
    computer_id: str = Field(..., description="Computer ID", min_length=1)
    direction: Literal["up", "down"] = Field(..., description="Scroll direction")
    amount: int = Field(default=3, ge=1, le=20, description="Scroll clicks (1-20)")


class DragInput(BaseModel):
    computer_id: str = Field(..., description="Computer ID", min_length=1)
    start_x: int = Field(..., ge=0, description="Start X coordinate")
    start_y: int = Field(..., ge=0, description="Start Y coordinate")
    end_x: int = Field(..., ge=0, description="End X coordinate")
    end_y: int = Field(..., ge=0, description="End Y coordinate")
    duration: float = Field(default=0.5, ge=0.1, le=5.0, description="Drag duration in seconds")


# =============================================================================
# Shell Models
# =============================================================================

class BashInput(BaseModel):
    computer_id: str = Field(..., description="Computer ID", min_length=1)
    command: str = Field(..., description="Bash command to execute", min_length=1)


class ExecInput(BaseModel):
    computer_id: str = Field(..., description="Computer ID", min_length=1)
    code: str = Field(..., description="Python code to execute", min_length=1)
    timeout: int = Field(default=10, ge=1, le=300, description="Timeout in seconds")


# =============================================================================
# File Models
# =============================================================================

class ListFilesInput(BaseModel):
    workspace_id: str = Field(..., description="Workspace ID", min_length=1)
    computer_id: Optional[str] = Field(default=None, description="Optional computer ID to filter by")


class ExportFileInput(BaseModel):
    computer_id: str = Field(..., description="Computer ID (must be running)", min_length=1)
    path: str = Field(..., description="Path on computer (e.g. 'Desktop/report.pdf', '~/Documents/data.csv')", min_length=1)


class UploadFileInput(BaseModel):
    workspace_id: str = Field(..., description="Workspace ID to upload to", min_length=1)
    filename: str = Field(..., description="Filename for the uploaded file", min_length=1, max_length=255)
    content_base64: str = Field(..., description="Base64-encoded file content", min_length=1)
    computer_id: Optional[str] = Field(default=None, description="Optional computer ID to associate with")
    content_type: Optional[str] = Field(default=None, description="MIME type (e.g. 'text/plain', 'image/png')")


class FileIdInput(BaseModel):
    file_id: str = Field(..., description="File ID (from orgo_list_files or orgo_export_file)", min_length=1)


# =============================================================================
# Agent / Completions Models
# =============================================================================

class CompletionsInput(BaseModel):
    computer_id: str = Field(..., description="Computer ID (must be running)", min_length=1)
    instruction: str = Field(..., description="What the AI agent should do (natural language)", min_length=1)
    model: str = Field(default="claude-sonnet-4.6", description="Model: claude-sonnet-4.6 or claude-opus-4.6")
    thread_id: Optional[str] = Field(default=None, description="Thread ID to continue a previous conversation")
    max_steps: Optional[int] = Field(default=None, ge=1, le=500, description="Max agent steps (default: 100)")
    anthropic_key: Optional[str] = Field(default=None, description="Your Anthropic API key for BYOK mode (no Orgo credits used)")


# =============================================================================
# Thread Models
# =============================================================================

class ListThreadsInput(BaseModel):
    computer_id: str = Field(..., description="Computer ID to list threads for", min_length=1)


class ThreadIdInput(BaseModel):
    thread_id: str = Field(..., description="Thread ID", min_length=1)


# =============================================================================
# Streaming Models
# =============================================================================

class StartStreamInput(BaseModel):
    computer_id: str = Field(..., description="Computer ID (must be running)", min_length=1)
    connection_name: str = Field(..., description="Name of pre-configured RTMP connection (set up at orgo.ai)", min_length=1)


# =============================================================================
# Clone / Resize / Auto-Stop Models
# =============================================================================

class CloneComputerInput(BaseModel):
    computer_id: str = Field(..., description="Computer ID to clone (uses fly_instance_id internally)", min_length=1)
    name: Optional[str] = Field(default=None, description="Name for the cloned computer (defaults to '{original}-clone')")
    workspace_id: Optional[str] = Field(default=None, description="Target workspace ID (defaults to same workspace as source)")


class ResizeComputerInput(BaseModel):
    computer_id: str = Field(..., description="Computer ID", min_length=1)
    cpu: Optional[int] = Field(default=None, description="New CPU cores (1, 2, 4, 8, 16)")
    ram: Optional[int] = Field(default=None, description="New RAM in GB (4, 8, 16, 32, 64)")
    disk_size_gb: Optional[int] = Field(default=None, description="New disk size in GB")
    bandwidth_limit_mbps: Optional[int] = Field(default=None, description="Bandwidth limit in Mbps (null for unlimited)")


class AutoStopInput(BaseModel):
    computer_id: str = Field(..., description="Computer ID", min_length=1)
    minutes: Optional[int] = Field(default=None, ge=0, description="Auto-stop after N minutes idle. 0 to disable. Free tier always enforces 15min.")


class SkillInstallInput(BaseModel):
    computer_id: str = Field(..., description="Computer ID (must be running)", min_length=1)
    skill_name: str = Field(..., description="Name for the skill being installed", min_length=1)
    files_base64: dict[str, str] = Field(..., description="Map of filename -> base64-encoded content for skill files (e.g. {'SKILL.md': '...', 'script.py': '...'})")


class StarComputerInput(BaseModel):
    computer_id: str = Field(..., description="Computer ID", min_length=1)
    starred: bool = Field(default=True, description="True to star, false to unstar")


# =============================================================================
# Workspace Member Models
# =============================================================================

class WorkspaceMembersInput(BaseModel):
    workspace_id: str = Field(..., description="Workspace ID", min_length=1)


class WorkspaceInviteInput(BaseModel):
    workspace_id: str = Field(..., description="Workspace ID", min_length=1)
    email: str = Field(..., description="Email address to invite", min_length=1)
    permission: Literal["viewer", "admin"] = Field(default="viewer", description="Permission level for the invited user")


class WorkspaceByNameInput(BaseModel):
    name: str = Field(..., description="Workspace name to look up", min_length=1)


# =============================================================================
# Template Models
# =============================================================================

class TemplateStarInput(BaseModel):
    template_id: str = Field(..., description="Template ID", min_length=1)
    starred: bool = Field(default=True, description="True to star, false to unstar")


# =============================================================================
# Access Models
# =============================================================================

# VNC password uses ComputerIdInput
