"""AI agent completions tool — the core agent API.

Uses the OpenAI-compatible POST /api/v1/chat/completions endpoint.
The AI agent handles the full loop: screenshot -> decide -> act -> repeat.
"""

import json

import httpx

from orgo_mcp.server import mcp
from orgo_mcp.auth import get_current_api_key, resolve_computer_id
from orgo_mcp.client import ORGO_V1_BASE
from orgo_mcp.errors import handle_orgo_error
from orgo_mcp.models import CompletionsInput


@mcp.tool(
    name="orgo_completions",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def orgo_completions(params: CompletionsInput) -> str:
    """Send an instruction to an AI agent controlling an Orgo computer.

    The agent sees the screen, clicks, types, and runs commands autonomously until
    the task is done. Returns the agent's final response and a thread_id for follow-up.
    Use thread_id to continue multi-turn conversations.

    Models: claude-sonnet-4.6 (fast, default), claude-opus-4.6 (most capable).
    Set anthropic_key for BYOK mode (your key, no Orgo credits consumed).
    """
    try:
        api_key = get_current_api_key(mcp)
        computer_id = resolve_computer_id(params.computer_id)

        body = {
            "model": params.model,
            "messages": [{"role": "user", "content": params.instruction}],
            "computer_id": computer_id,
        }
        if params.thread_id:
            body["thread_id"] = params.thread_id
        if params.max_steps is not None:
            body["max_steps"] = params.max_steps

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if params.anthropic_key:
            headers["X-Anthropic-Key"] = params.anthropic_key

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ORGO_V1_BASE}/chat/completions",
                json=body,
                headers=headers,
                timeout=300.0,
            )
            response.raise_for_status()
            return json.dumps(response.json(), indent=2)
    except Exception as e:
        return handle_orgo_error(e)
