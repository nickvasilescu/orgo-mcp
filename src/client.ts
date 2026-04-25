/**
 * HTTP client for Orgo API access.
 *
 * Two routing strategies:
 * - Platform API proxy for computer actions (click, type, screenshot, bash, etc.)
 * - Direct VM connection as fallback (uses VNC password + instance_details)
 *
 * Computer actions route through /api/computers/{id}/{action}, which handles
 * VM port resolution internally. Falls back to direct VM connection on errors.
 */

import { HttpError } from "./errors.js";
import type { ComputerInfo, VncPasswordResponse } from "./types.js";

const ORGO_API_BASE = "https://www.orgo.ai/api";
const ORGO_V1_BASE = "https://api.orgo.ai/api/v1";

// Cache VNC passwords per computer
const vncPasswordCache = new Map<string, string>();

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
 * Fetch and cache VNC password for a computer.
 */
async function getVncPassword(computerId: string, apiKey: string): Promise<string> {
  const cached = vncPasswordCache.get(computerId);
  if (cached) return cached;

  const data = (await apiRequest(
    "GET",
    `computers/${computerId}/vnc-password`,
    apiKey,
    { timeout: 15000 }
  )) as unknown as VncPasswordResponse;

  const password = data.password;
  if (!password) {
    throw new Error(`Could not get VNC password for computer ${computerId}`);
  }

  vncPasswordCache.set(computerId, password);
  return password;
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

  const vncPassword = await getVncPassword(computerId, apiKey);

  // Use ensure-running to get fresh instance_details (ports change on restart)
  const info = (await apiRequest(
    "POST",
    `computers/${computerId}/ensure-running`,
    apiKey,
    { timeout: 15000 }
  )) as unknown as ComputerInfo;

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
        Authorization: `Bearer ${vncPassword}`,
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
 * Clear cached VNC password for a computer (e.g., after restart).
 */
function clearVncCache(computerId?: string): void {
  if (computerId) {
    vncPasswordCache.delete(computerId);
  } else {
    vncPasswordCache.clear();
  }
}

export {
  ORGO_API_BASE,
  ORGO_V1_BASE,
  apiRequest,
  computerAction,
  resolveFlyInstanceId,
  getVncPassword,
  uploadFile,
  clearVncCache,
};
