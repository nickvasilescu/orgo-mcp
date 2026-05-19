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
import { jsonText } from "./format.js";
import { registerOrgoTool } from "./registry.js";

export function registerWorkspaceTools(server: McpServer): void {
  registerOrgoTool(server, {
    name: "orgo_list_workspaces",
    title: "List Workspaces",
    description: "List all workspaces in your Orgo account. Returns workspace IDs, names, and computer counts.",
    inputSchema: {},
    toolsets: ["core"],
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: true,
    },
    handler: async () => {
      try {
        const apiKey = getApiKey();
        const data = await apiRequest("GET", "projects", apiKey);
        return { content: [{ type: "text" as const, text: jsonText(data) }] };
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
    },
    toolsets: ["core"],
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: true,
    },
    handler: async ({ workspace_id }) => {
      try {
        const apiKey = getApiKey();
        const data = await apiRequest("GET", `projects/${workspace_id}`, apiKey);
        return { content: [{ type: "text" as const, text: jsonText(data) }] };
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
    },
    toolsets: ["core"],
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: true,
    },
    handler: async ({ name }) => {
      try {
        const apiKey = getApiKey();
        const data = await apiRequest("GET", `projects/by-name/${name}`, apiKey);
        return { content: [{ type: "text" as const, text: jsonText(data) }] };
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
