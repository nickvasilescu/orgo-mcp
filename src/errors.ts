/**
 * Unified error handling for Orgo MCP tools.
 */

export interface FetchErrorLike {
  status?: number;
  statusText?: string;
  body?: string;
}

/**
 * Format errors into user-friendly messages with actionable guidance.
 */
export function handleError(e: unknown): string {
  if (e instanceof HttpError) {
    const detail = e.detail || "";
    switch (e.status) {
      case 401:
        return `Error: Invalid API key. ${detail || "Get your key at https://orgo.ai"}`;
      case 402:
        return `Error: Insufficient credits. ${detail || "Add credits at https://orgo.ai/settings/billing"}`;
      case 404:
        return `Error: Not found. ${detail || "Use orgo_list_workspaces / orgo_list_computers to find valid IDs."}`;
      case 400:
        return `Error: Bad request. ${detail}`;
      case 429:
        return "Error: Rate limited. Wait before retrying.";
      case 503:
        return "Error: Orgo service temporarily unavailable.";
      default:
        return `Error: API returned ${e.status}. ${detail}`;
    }
  }

  if (e instanceof Error) {
    if (e.name === "AbortError" || e.message.includes("timeout")) {
      return "Error: Request timed out. The computer may be starting -- retry in a few seconds.";
    }
    if (e.message.includes("fetch failed") || e.message.includes("ECONNREFUSED")) {
      return "Error: Cannot reach Orgo API. Check your network connection.";
    }

    const msg = e.message;
    if (msg.includes("status 401") || msg.toLowerCase().includes("invalid api key")) {
      return "Error: Invalid API key. Get your key at https://orgo.ai";
    }
    if (msg.includes("status 402")) {
      return "Error: Insufficient credits. Add credits at https://orgo.ai/settings/billing";
    }
    if (msg.includes("status 404") || msg.toLowerCase().includes("not found")) {
      return "Error: Resource not found. Use orgo_list_workspaces or orgo_list_computers to find valid IDs.";
    }
    return `Error: ${msg}`;
  }

  return `Error: ${String(e)}`;
}

/**
 * Custom HTTP error class for API responses.
 */
export class HttpError extends Error {
  constructor(
    public status: number,
    public detail: string,
    public statusText: string = ""
  ) {
    super(`HTTP ${status}: ${detail || statusText}`);
    this.name = "HttpError";
  }
}
