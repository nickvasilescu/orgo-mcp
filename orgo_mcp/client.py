"""Client factories for Orgo API access.

Two client types:
- Platform API proxy for computer actions (click, type, screenshot, bash, etc.)
- httpx.AsyncClient for platform endpoints (completions, threads, files, workspaces)

Computer actions route through the platform API at /api/computers/{id}/{action},
which handles VM port resolution and auth internally. This avoids the need to
resolve direct VM URLs and ports, which differ between Metal and Fly providers.
"""

import asyncio

import httpx

# Base URLs
ORGO_API_BASE = "https://www.orgo.ai/api"
ORGO_V1_BASE = "https://api.orgo.ai/api/v1"

# Cache VNC passwords: computer_id -> vnc_password
_vnc_password_cache: dict[str, str] = {}


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


async def _get_vnc_password(computer_id: str, api_key: str) -> str:
    """Fetch and cache VNC password for a computer."""
    if computer_id in _vnc_password_cache:
        return _vnc_password_cache[computer_id]

    data = await api_request("GET", f"computers/{computer_id}/vnc-password", api_key, timeout=15.0)
    password = data.get("password", "")
    if not password:
        raise RuntimeError(f"Could not get VNC password for computer {computer_id}")

    _vnc_password_cache[computer_id] = password
    return password


async def computer_action(
    method: str,
    computer_id: str,
    endpoint: str,
    api_key: str,
    json: dict = None,
    timeout: float = 30.0,
) -> dict:
    """Make an authenticated request to a computer via the platform API proxy.

    Routes through /api/computers/{id}/{endpoint} which handles VM port
    resolution internally. Falls back to direct VM connection if the proxy
    returns auth errors (e.g. workspace API keys not matching platform keys).
    """
    # Try platform API proxy first — handles Metal/Fly port resolution
    try:
        return await api_request(
            method,
            f"computers/{computer_id}/{endpoint}",
            api_key,
            json=json,
            timeout=timeout,
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code not in (401, 403):
            raise
        # Auth failed on platform proxy — fall back to direct VM connection

    # Fallback: direct connection using VNC password + instance_details
    vnc_password = await _get_vnc_password(computer_id, api_key)
    info = await api_request("GET", f"computers/{computer_id}", api_key, timeout=15.0)

    # Resolve the correct API URL from instance_details when available.
    # The top-level `url` field is the noVNC web proxy, NOT the VM API endpoint.
    # Metal VMs expose the API on a separate port stored in instance_details.apiPort.
    details = info.get("instance_details") or {}
    api_port = details.get("apiPort")
    host = details.get("publicHost") or details.get("vncHost")
    if api_port and host:
        direct_url = f"http://{host}:{api_port}"
    else:
        direct_url = info.get("url", "").rstrip("/")

    if not direct_url:
        raise RuntimeError(f"Could not resolve VM URL for computer {computer_id}")

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
