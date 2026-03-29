"""Client factories for Orgo API access.

Two client types:
- orgo SDK's ApiClient (sync, wrapped with asyncio.to_thread) for actions, screenshots, bash
- httpx.AsyncClient for new endpoints (completions, threads, files, workspaces)
"""

import httpx
from orgo import Computer
from orgo.api.client import ApiClient

# Base URLs
ORGO_API_BASE = "https://www.orgo.ai/api"
ORGO_V1_BASE = "https://api.orgo.ai/api/v1"

# Cache SDK clients per API key
_sdk_client_cache: dict[str, ApiClient] = {}


def get_sdk_client(api_key: str) -> ApiClient:
    """Get a cached orgo SDK ApiClient for the given API key."""
    if api_key not in _sdk_client_cache:
        _sdk_client_cache[api_key] = ApiClient(api_key=api_key)
    return _sdk_client_cache[api_key]


def get_computer(computer_id: str, api_key: str) -> Computer:
    """Get a Computer instance for direct actions (click, type, screenshot, etc.)."""
    return Computer(computer_id=computer_id, api_key=api_key, verbose=False)


def get_auth_headers(api_key: str) -> dict[str, str]:
    """Get standard auth headers for direct httpx calls."""
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
    """Make an authenticated async request to the Orgo API."""
    url = f"{base_url}/{path}"
    headers = get_auth_headers(api_key)
    async with httpx.AsyncClient() as client:
        response = await client.request(
            method, url, headers=headers, json=json, params=params, timeout=timeout
        )
        response.raise_for_status()
        return response.json()
