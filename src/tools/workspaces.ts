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

export function registerWorkspaceTools(server: McpServer): void {
  server.tool(
    "orgo_list_workspaces",
    "List all workspaces in your Orgo account. Returns workspace IDs, names, and computer counts.",
    {},
    async () => {
      try {
        const apiKey = getApiKey();
        const data = await apiRequest("GET", "projects", apiKey);
        return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    }
  );

  server.tool(
    "orgo_create_workspace",
    "Create a new workspace. Workspace names must be unique. Returns the workspace ID.",
    {
      name: z.string().min(1).max(64).describe("Unique workspace name (letters, numbers, hyphens, underscores)"),
    },
    async ({ name }) => {
      try {
        const apiKey = getApiKey();
        const data = await apiRequest("POST", "projects", apiKey, { json: { name } });
        return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    }
  );

  server.tool(
    "orgo_get_workspace",
    "Get workspace details including its computers. Returns workspace info and computer list.",
    {
      workspace_id: z.string().min(1).describe("Workspace ID (from orgo_list_workspaces)"),
    },
    async ({ workspace_id }) => {
      try {
        const apiKey = getApiKey();
        const data = await apiRequest("GET", `projects/${workspace_id}`, apiKey);
        return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    }
  );

  server.tool(
    "orgo_workspace_by_name",
    "Look up a workspace by name instead of ID. Returns workspace details if found.",
    {
      name: z.string().min(1).describe("Workspace name to look up"),
    },
    async ({ name }) => {
      try {
        const apiKey = getApiKey();
        const data = await apiRequest("GET", `projects/by-name/${name}`, apiKey);
        return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    }
  );
}
