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

const COMPACT_DESC = "Return only essential fields (id, name, status, timestamps) instead of the full Orgo API response. Recommended for agent contexts to minimize token usage.";
const LIMIT_DESC = "Optional cap on the number of items returned. The Orgo API doesn't paginate server-side, so this trims client-side. When set and truncation occurs, the response includes `total` and `truncated: true`. Omit to return everything (current default).";

export function registerWorkspaceTools(server: McpServer): void {
  registerOrgoTool(server, {
    name: "orgo_list_workspaces",
    title: "List Workspaces",
    description: "List all workspaces in your Orgo account. Returns workspace IDs, names, and computer counts.",
    inputSchema: {
      compact: z.boolean().optional().default(false).describe(COMPACT_DESC),
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
    description: "Create a new workspace. Workspace names must be unique. Returns the workspace ID.",
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
    description: "Get workspace details including its computers. Returns workspace info and computer list.",
    inputSchema: {
      workspace_id: z.string().min(1).describe("Workspace ID (from orgo_list_workspaces)"),
      compact: z.boolean().optional().default(false).describe(COMPACT_DESC),
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
    description: "Look up a workspace by name instead of ID. Returns workspace details if found.",
    inputSchema: {
      name: z.string().min(1).describe("Workspace name to look up"),
      compact: z.boolean().optional().default(false).describe(COMPACT_DESC),
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
        const data = await apiRequest("GET", `projects/by-name/${name}`, apiKey);
        return { content: [{ type: "text" as const, text: compact ? jsonTextCompact(data) : jsonText(data) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });

  registerOrgoTool(server, {
    name: "orgo_delete_workspace",
    title: "Delete Workspace",
    description: "Permanently delete a workspace and all of its computers. Cannot be undone.",
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
        const data = await apiRequest("DELETE", `projects/${workspace_id}`, apiKey);
        return { content: [{ type: "text" as const, text: jsonText(data) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });
}
