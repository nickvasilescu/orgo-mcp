/**
 * Template management tools -- list, star/unstar VM templates.
 */

import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { getApiKey } from "../auth.js";
import { apiRequest } from "../client.js";
import { handleError } from "../errors.js";

export function registerTemplateTools(server: McpServer): void {
  server.tool(
    "orgo_list_templates",
    "List all available VM templates. Templates are pre-configured images for creating computers.",
    {},
    async () => {
      try {
        const apiKey = getApiKey();
        const data = await apiRequest("GET", "templates", apiKey);
        return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    }
  );

  server.tool(
    "orgo_starred_templates",
    "List starred VM templates for quick access.",
    {},
    async () => {
      try {
        const apiKey = getApiKey();
        const data = await apiRequest("GET", "templates/starred", apiKey);
        return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    }
  );

  server.tool(
    "orgo_star_template",
    "Star or unstar a VM template for quick access.",
    {
      template_id: z.string().min(1).describe("Template ID"),
      starred: z.boolean().default(true).describe("True to star, false to unstar"),
    },
    async ({ template_id, starred }) => {
      try {
        const apiKey = getApiKey();
        const data = await apiRequest("POST", `templates/${template_id}/star`, apiKey, {
          json: { starred },
        });
        return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    }
  );
}
