"""Client factories for Orgo API access.

Two client types:
- Direct HTTP to computer VMs for actions (click, type, screenshot, bash, etc.)
- httpx.AsyncClient for platform endpoints (completions, threads, files, workspaces)
"""

import httpx

# Base URLs
ORGO_API_BASE = "https://www.orgo.ai/api"
ORGO_V1_BASE = "https://api.orgo.ai/api/v1"

# Cache computer connection info: computer_id -> (direct_url, vnc_password)
_connection_cache: dict[str, tuple[str, str]] = {}


def get_auth_headers(api_key: str) -> dict[str, str]:
    """Get standard auth headers for platform API calls."""
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


async def api_request(
    method: str,
    path: str,
    api_key: str,
    json: dict = None,
    params: dict = None,
    timeout: float = 30.0,
    base_url: str = ORGO_API_BASE,
) -> dict:
    """Make an authenticated async request to the Orgo platform API."""
    url = f"{base_url}/{path}"
    headers = get_auth_headers(api_key)
    async with httpx.AsyncClient() as client:
        response = await client.request(
            method, url, headers=headers, json=json, params=params, timeout=timeout
        )
        response.raise_for_status()
        return response.json()


async def _get_computer_connection(computer_id: str, api_key: str) -> tuple[str, str]:
    """Resolve computer_id to (direct_url, vnc_password), with caching."""
    if computer_id in _connection_cache:
        return _connection_cache[computer_id]

    # Fetch computer info and VNC password from platform API in parallel
    async with httpx.AsyncClient() as client:
        headers = get_auth_headers(api_key)
        info_resp, vnc_resp = await asyncio.gather(
            client.get(f"{ORGO_API_BASE}/computers/{computer_id}", headers=headers, timeout=15.0),
            client.get(f"{ORGO_API_BASE}/computers/{computer_id}/vnc-password", headers=headers, timeout=15.0),
        )
        info_resp.raise_for_status()
        vnc_resp.raise_for_status()

    direct_url = info_resp.json().get("url", "").rstrip("/")
    vnc_password = vnc_resp.json().get("password", "")

    if not direct_url or not vnc_password:
        raise RuntimeError(f"Could not resolve connection for computer {computer_id}")

    _connection_cache[computer_id] = (direct_url, vnc_password)
    return direct_url, vnc_password


async def computer_action(
    method: str,
    computer_id: str,
    endpoint: str,
    api_key: str,
    json: dict = None,
    timeout: float = 30.0,
) -> dict:
    """Make an authenticated request directly to a computer's VM API.

    Uses the computer's direct URL + VNC password (not the platform proxy).
    """
    direct_url, vnc_password = await _get_computer_connection(computer_id, api_key)
    url = f"{direct_url}/{endpoint}"
    headers = {
        "Authorization": f"Bearer {vnc_password}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient() as client:
        response = await client.request(
            method, url, headers=headers, json=json, timeout=timeout
        )
        response.raise_for_status()
        return response.json()


import asyncio  # noqa: E402 — used by _get_computer_connection
