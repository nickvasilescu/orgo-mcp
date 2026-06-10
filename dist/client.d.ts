/**
 * HTTP client for Orgo API access.
 *
 * Two routing strategies:
 * - Platform API proxy for computer actions (click, type, screenshot, bash, etc.)
 * - Direct VM connection as fallback (uses the VM desktop-API token + instance_details)
 *
 * Computer actions route through /api/computers/{id}/{action}, which handles
 * VM port resolution internally. Falls back to direct VM connection on errors.
 *
 * VM auth: new metal-image VMs issue a `desktop_api_token` distinct from the
 * VNC password and reject `Bearer <vncPassword>`; older cold-boot images set
 * desktop_api_token == password. GET /computers/{id}/vnc-password returns both,
 * so `desktop_api_token ?? password` is correct for every image generation.
 */
declare const ORGO_API_BASE = "https://www.orgo.ai/api";
declare function tenantKey(apiKey: string, computerId: string): string;
/**
 * Make an authenticated request to the Orgo platform API.
 */
declare function apiRequest(method: string, path: string, apiKey: string, options?: {
    json?: Record<string, unknown>;
    params?: Record<string, string>;
    timeout?: number;
    baseUrl?: string;
}): Promise<Record<string, unknown>>;
/**
 * Fetch and cache the VM desktop-API auth token for a computer.
 *
 * Prefers `desktop_api_token` (required by new metal-image VMs) and falls back
 * to the VNC password (== the token on older images).
 */
declare function getVmAuthToken(computerId: string, apiKey: string): Promise<string>;
/**
 * Resolve the direct VM endpoint (host + apiPort) for WebSocket/HTTP access.
 */
declare function getVmEndpoint(computerId: string, apiKey: string): Promise<{
    host: string;
    apiPort: number;
} | null>;
/**
 * Make a request to a computer via the platform API proxy.
 *
 * Falls back to direct VM connection if the proxy returns server errors
 * or is unreachable (stale ports after restart).
 *
 * Set direct=true to bypass the proxy (required for screenshot).
 */
declare function computerAction(method: string, computerId: string, endpoint: string, apiKey: string, options?: {
    json?: Record<string, unknown>;
    timeout?: number;
    direct?: boolean;
}): Promise<Record<string, unknown>>;
/**
 * Resolve a computer UUID to its fly_instance_id.
 *
 * Short IDs (<=12 chars, no dashes) are already fly_instance_ids.
 */
declare function resolveFlyInstanceId(computerId: string, apiKey: string): Promise<string>;
/**
 * Upload a file via multipart/form-data (cannot use JSON for file uploads).
 */
declare function uploadFile(apiKey: string, workspaceId: string, filename: string, fileBytes: Buffer, options?: {
    computerId?: string;
    contentType?: string;
    timeout?: number;
}): Promise<Record<string, unknown>>;
/**
 * Clear cached VM credentials/endpoints for a computer (e.g., after restart),
 * across all tenants that cached it.
 */
declare function clearVncCache(computerId?: string): void;
export { ORGO_API_BASE, apiRequest, computerAction, resolveFlyInstanceId, getVmAuthToken, getVmEndpoint, uploadFile, clearVncCache, tenantKey, };
