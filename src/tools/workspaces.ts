/**
 * Workspace management tools.
 *
 * Workspaces are containers for computers. Use them to organize different
 * projects, environments, or agent fleets.
 */

import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { getApiKey } from "../auth.js";
import { apiRequest } from "../client.js";
import { handleError } from "../errors.js";
import { applyLimit, jsonText, jsonTextCompact } from "./format.js";
import { registerOrgoTool } from "./registry.js";

const COMPACT_DESC = "Return only essential fields (id, name, status, timestamps) instead of the full Orgo API response. Defaults to true to minimize token usage; pass false when you need the complete payload.";
const LIMIT_DESC = "Optional cap on the number of items returned. The Orgo API doesn't paginate server-side, so this trims client-side. When set and truncation occurs, the response includes `total` and `truncated: true`. Omit to return everything (current default).";

export function registerWorkspaceTools(server: McpServer): void {
  registerOrgoTool(server, {
    name: "orgo_list_workspaces",
    title: "List Workspaces",
    description: "List all workspaces in your Orgo account. Returns workspace IDs, names, and computer counts. Use as a session-start discovery call, or when an agent needs to pick a workspace by name/status.",
    inputSchema: {
      compact: z.boolean().optional().default(true).describe(COMPACT_DESC),
      limit: z.number().int().min(1).max(500).optional().describe(LIMIT_DESC),
    },
    toolsets: ["core"],
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: true,
    },
    handler: async ({ compact, limit }) => {
      try {
        const apiKey = getApiKey();
        const data = (await apiRequest("GET", "projects", apiKey)) as Record<string, unknown>;
        const payload = applyLimit(data, "projects", limit);
        return { content: [{ type: "text" as const, text: compact ? jsonTextCompact(payload) : jsonText(payload) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });

  registerOrgoTool(server, {
    name: "orgo_create_workspace",
    title: "Create Workspace",
    description: "Create a new workspace. Workspace names must be unique. Returns the workspace ID. Use when organizing a new project, environment, or agent fleet — workspaces are the container for computers.",
    inputSchema: {
      name: z.string().min(1).max(64).describe("Unique workspace name (letters, numbers, hyphens, underscores)"),
    },
    toolsets: ["admin"],
    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
    handler: async ({ name }) => {
      try {
        const apiKey = getApiKey();
        const data = await apiRequest("POST", "projects", apiKey, { json: { name } });
        return { content: [{ type: "text" as const, text: jsonText(data) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });

  registerOrgoTool(server, {
    name: "orgo_get_workspace",
    title: "Get Workspace",
    description: "Get workspace details including its computers. Returns workspace info and computer list. Use when you already have a workspace ID and want to enumerate its computers in one call (cheaper than list_workspaces + filter).",
    inputSchema: {
      workspace_id: z.string().min(1).describe("Workspace ID (from orgo_list_workspaces)"),
      compact: z.boolean().optional().default(true).describe(COMPACT_DESC),
    },
    toolsets: ["core"],
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: true,
    },
    handler: async ({ workspace_id, compact }) => {
      try {
        const apiKey = getApiKey();
        const data = await apiRequest("GET", `projects/${workspace_id}`, apiKey);
        return { content: [{ type: "text" as const, text: compact ? jsonTextCompact(data) : jsonText(data) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });

  registerOrgoTool(server, {
    name: "orgo_workspace_by_name",
    title: "Get Workspace By Name",
    description: "Look up a workspace by name instead of ID. Returns workspace details if found. Use when the workspace name comes from configuration (env var, user input) but no ID is on hand — saves a list_workspaces + filter step.",
    inputSchema: {
      name: z.string().min(1).describe("Workspace name to look up"),
      compact: z.boolean().optional().default(true).describe(COMPACT_DESC),
    },
    toolsets: ["core"],
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: true,
    },
    handler: async ({ name, compact }) => {
      try {
        const apiKey = getApiKey();
        const data = await apiRequest("GET", `projects/by-name/${encodeURIComponent(name)}`, apiKey);
        return { content: [{ type: "text" as const, text: compact ? jsonTextCompact(data) : jsonText(data) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });

  registerOrgoTool(server, {
    name: "orgo_delete_workspace",
    title: "Delete Workspace",
    description: "Permanently delete a workspace and all of its computers. Cannot be undone. Use only when explicitly instructed to clean up — this removes every computer inside; prefer deleting individual computers if scope is narrower.",
    inputSchema: {
      workspace_id: z.string().min(1).describe("Workspace ID to delete"),
    },
    toolsets: ["admin"],
    annotations: {
      readOnlyHint: false,
      destructiveHint: true,
      idempotentHint: false,
      openWorldHint: true,
    },
    handler: async ({ workspace_id }) => {
      try {
        const apiKey = getApiKey();
        // Workspace deletion is POST /projects/{id}/delete on the platform
        // (DELETE /projects/{id} is not allowed → 405).
        const data = await apiRequest("POST", `projects/${workspace_id}/delete`, apiKey);
        return { content: [{ type: "text" as const, text: jsonText(data) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });
}
