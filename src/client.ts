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

import { createHash } from "node:crypto";
import { HttpError } from "./errors.js";
import type { ComputerInfo, InstanceDetails, VncPasswordResponse } from "./types.js";

const ORGO_API_BASE = "https://www.orgo.ai/api";

// Caches are keyed by (api key, computer) — never by computer alone. The HTTP
// transport hosts many tenants in one process; a computer-only key would serve
// tenant A's credentials/connections to tenant B on a cache hit.
function tenantKey(apiKey: string, computerId: string): string {
  const keyHash = createHash("sha256").update(apiKey).digest("hex").slice(0, 16);
  return `${keyHash}:${computerId}`;
}

// VM desktop-API tokens, keyed by tenantKey.
const vmTokenCache = new Map<string, string>();

// instance_details (host/port) per computer, keyed by tenantKey, with a short
// TTL — ports only change on restart, and refetching via ensure-running on
// every direct call added 1-2 round trips per screenshot.
const INSTANCE_DETAILS_TTL_MS = 60_000;
const instanceDetailsCache = new Map<string, { details: ComputerInfo; expires: number }>();

function authHeaders(apiKey: string): Record<string, string> {
  return {
    Authorization: `Bearer ${apiKey}`,
    "Content-Type": "application/json",
  };
}

/**
 * Make an authenticated request to the Orgo platform API.
 */
async function apiRequest(
  method: string,
  path: string,
  apiKey: string,
  options: {
    json?: Record<string, unknown>;
    params?: Record<string, string>;
    timeout?: number;
    baseUrl?: string;
  } = {}
): Promise<Record<string, unknown>> {
  const { json: body, params, timeout = 30000, baseUrl = ORGO_API_BASE } = options;

  let url = `${baseUrl}/${path}`;
  if (params) {
    const search = new URLSearchParams(params);
    url += `?${search.toString()}`;
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(url, {
      method,
      headers: authHeaders(apiKey),
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });

    if (!response.ok) {
      let detail = "";
      try {
        const errJson = (await response.json()) as Record<string, unknown>;
        detail = (errJson.error as string) || "";
      } catch {
        // ignore parse errors
      }
      throw new HttpError(response.status, detail, response.statusText);
    }

    return (await response.json()) as Record<string, unknown>;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Fetch and cache the VM desktop-API auth token for a computer.
 *
 * Prefers `desktop_api_token` (required by new metal-image VMs) and falls back
 * to the VNC password (== the token on older images).
 */
async function getVmAuthToken(computerId: string, apiKey: string): Promise<string> {
  const cacheKey = tenantKey(apiKey, computerId);
  const cached = vmTokenCache.get(cacheKey);
  if (cached) return cached;

  const data = (await apiRequest(
    "GET",
    `computers/${computerId}/vnc-password`,
    apiKey,
    { timeout: 15000 }
  )) as unknown as VncPasswordResponse;

  const token = data.desktop_api_token || data.password;
  if (!token) {
    throw new Error(`Could not get VM auth token for computer ${computerId}`);
  }

  vmTokenCache.set(cacheKey, token);
  return token;
}

/**
 * Resolve the VM's direct connection info (host + apiPort), with a short TTL
 * cache. Uses ensure-running so suspended VMs come back with fresh ports.
 */
async function getInstanceInfo(computerId: string, apiKey: string): Promise<ComputerInfo> {
  const cacheKey = tenantKey(apiKey, computerId);
  const cached = instanceDetailsCache.get(cacheKey);
  if (cached && cached.expires > Date.now()) return cached.details;

  const info = (await apiRequest(
    "POST",
    `computers/${computerId}/ensure-running`,
    apiKey,
    { timeout: 15000 }
  )) as unknown as ComputerInfo;

  instanceDetailsCache.set(cacheKey, { details: info, expires: Date.now() + INSTANCE_DETAILS_TTL_MS });
  return info;
}

/**
 * Resolve the direct VM endpoint (host + apiPort) for WebSocket/HTTP access.
 */
async function getVmEndpoint(
  computerId: string,
  apiKey: string
): Promise<{ host: string; apiPort: number } | null> {
  const info = await getInstanceInfo(computerId, apiKey);
  const details: InstanceDetails = info.instance_details || {};
  const host = details.publicHost || details.vncHost;
  if (details.apiPort && host) {
    return { host, apiPort: details.apiPort };
  }
  return null;
}

/**
 * Make a request directly to the VM, bypassing the platform proxy.
 */
async function directVmRequest(
  method: string,
  computerId: string,
  endpoint: string,
  apiKey: string,
  options: { json?: Record<string, unknown>; timeout?: number } = {}
): Promise<Record<string, unknown>> {
  const { json: body, timeout = 30000 } = options;

  const vmToken = await getVmAuthToken(computerId, apiKey);
  const info = await getInstanceInfo(computerId, apiKey);

  const details = info.instance_details || {};
  const apiPort = details.apiPort;
  const host = details.publicHost || details.vncHost;

  let directUrl: string;
  if (apiPort && host) {
    directUrl = `http://${host}:${apiPort}`;
  } else {
    directUrl = (info.url || "").replace(/\/+$/, "");
  }

  if (!directUrl) {
    throw new Error(`Could not resolve VM URL for computer ${computerId}`);
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(`${directUrl}/${endpoint}`, {
      method,
      headers: {
        Authorization: `Bearer ${vmToken}`,
        "Content-Type": "application/json",
      },
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });

    if (!response.ok) {
      let detail = "";
      try {
        const errJson = (await response.json()) as Record<string, unknown>;
        detail = (errJson.error as string) || "";
      } catch {
        // ignore
      }
      throw new HttpError(response.status, detail, response.statusText);
    }

    return (await response.json()) as Record<string, unknown>;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Make a request to a computer via the platform API proxy.
 *
 * Falls back to direct VM connection if the proxy returns server errors
 * or is unreachable (stale ports after restart).
 *
 * Set direct=true to bypass the proxy (required for screenshot).
 */
async function computerAction(
  method: string,
  computerId: string,
  endpoint: string,
  apiKey: string,
  options: {
    json?: Record<string, unknown>;
    timeout?: number;
    direct?: boolean;
  } = {}
): Promise<Record<string, unknown>> {
  const { json: body, timeout = 30000, direct = false } = options;

  if (direct) {
    return directVmRequest(method, computerId, endpoint, apiKey, { json: body, timeout });
  }

  // Try platform API proxy first
  try {
    return await apiRequest(method, `computers/${computerId}/${endpoint}`, apiKey, {
      json: body,
      timeout,
    });
  } catch (e) {
    if (e instanceof HttpError) {
      // Client errors (including auth) don't warrant fallback — only 5xx does
      if (e.status < 500) {
        throw e;
      }
      // 5xx -> fall back to direct VM connection
    } else if (e instanceof Error) {
      if (
        !e.message.includes("ECONNREFUSED") &&
        !e.message.includes("fetch failed") &&
        e.name !== "AbortError"
      ) {
        throw e;
      }
      // Connection errors and timeouts -> fall back to direct
    } else {
      throw e;
    }
  }

  // Fallback: direct connection
  return directVmRequest(method, computerId, endpoint, apiKey, { json: body, timeout });
}

/**
 * Resolve a computer UUID to its fly_instance_id.
 *
 * Short IDs (<=12 chars, no dashes) are already fly_instance_ids.
 */
async function resolveFlyInstanceId(computerId: string, apiKey: string): Promise<string> {
  if (computerId.length <= 12 && !computerId.includes("-")) {
    return computerId;
  }
  const data = (await apiRequest("GET", `computers/${computerId}`, apiKey, {
    timeout: 15000,
  })) as unknown as ComputerInfo;
  const flyId = data.fly_instance_id;
  if (!flyId) {
    throw new Error(`Computer ${computerId} has no fly_instance_id`);
  }
  return flyId;
}

/**
 * Upload a file via multipart/form-data (cannot use JSON for file uploads).
 */
async function uploadFile(
  apiKey: string,
  workspaceId: string,
  filename: string,
  fileBytes: Buffer,
  options: { computerId?: string; contentType?: string; timeout?: number } = {}
): Promise<Record<string, unknown>> {
  const { computerId, contentType = "application/octet-stream", timeout = 60000 } = options;

  const formData = new FormData();
  const blob = new Blob([new Uint8Array(fileBytes)], { type: contentType });
  formData.append("file", blob, filename);
  formData.append("projectId", workspaceId);
  if (computerId) {
    formData.append("desktopId", computerId);
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);

  try {
    const response = await fetch(`${ORGO_API_BASE}/files/upload`, {
      method: "POST",
      headers: { Authorization: `Bearer ${apiKey}` },
      body: formData,
      signal: controller.signal,
    });

    if (!response.ok) {
      let detail = "";
      try {
        const errJson = (await response.json()) as Record<string, unknown>;
        detail = (errJson.error as string) || "";
      } catch {
        // ignore
      }
      throw new HttpError(response.status, detail, response.statusText);
    }

    return (await response.json()) as Record<string, unknown>;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Clear cached VM credentials/endpoints for a computer (e.g., after restart),
 * across all tenants that cached it.
 */
function clearVncCache(computerId?: string): void {
  if (computerId) {
    for (const cache of [vmTokenCache, instanceDetailsCache]) {
      for (const key of cache.keys()) {
        if (key.endsWith(`:${computerId}`)) cache.delete(key);
      }
    }
  } else {
    vmTokenCache.clear();
    instanceDetailsCache.clear();
  }
}

export {
  ORGO_API_BASE,
  apiRequest,
  computerAction,
  resolveFlyInstanceId,
  getVmAuthToken,
  getVmEndpoint,
  uploadFile,
  clearVncCache,
  tenantKey,
};
