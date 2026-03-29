"""Unified error handling for Orgo MCP tools."""

import httpx


def handle_orgo_error(e: Exception) -> str:
    """Format errors from both the orgo SDK (requests) and direct httpx calls."""

    # Direct httpx calls (completions, threads, files, etc.)
    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        detail = ""
        try:
            detail = e.response.json().get("error", "")
        except Exception:
            pass
        if status == 401:
            return f"Error: Invalid API key. {detail or 'Get your key at https://orgo.ai'}"
        if status == 402:
            return f"Error: Insufficient credits. {detail or 'Add credits at https://orgo.ai/settings/billing'}"
        if status == 404:
            return f"Error: Not found. {detail or 'Use orgo_list_workspaces / orgo_list_computers to find valid IDs.'}"
        if status == 400:
            return f"Error: Bad request. {detail}"
        if status == 429:
            return "Error: Rate limited. Wait before retrying."
        if status == 503:
            return "Error: Orgo service temporarily unavailable."
        return f"Error: API returned {status}. {detail}"

    if isinstance(e, httpx.TimeoutException):
        return "Error: Request timed out. The computer may be starting — retry in a few seconds."

    # orgo SDK errors (generic Exception with message strings)
    msg = str(e)
    if "status 401" in msg or "invalid api key" in msg.lower():
        return "Error: Invalid API key. Get your key at https://orgo.ai"
    if "status 402" in msg:
        return "Error: Insufficient credits. Add credits at https://orgo.ai/settings/billing"
    if "status 404" in msg or "not found" in msg.lower():
        return "Error: Resource not found. Use orgo_list_workspaces or orgo_list_computers to find valid IDs."
    if "status 429" in msg:
        return "Error: Rate limited. Wait before retrying."
    if "failed to connect" in msg.lower():
        return "Error: Cannot reach Orgo API. Check your network connection."
    return f"Error: {msg}"
