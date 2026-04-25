/**
 * Thread management tools for multi-turn agent conversations.
 *
 * Threads persist context across multiple instructions to the same computer.
 */

import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { getApiKey, resolveComputerId } from "../auth.js";
import { apiRequest, ORGO_V1_BASE } from "../client.js";
import { handleError } from "../errors.js";

export function registerThreadTools(server: McpServer): void {
  server.tool(
    "orgo_list_threads",
    "List conversation threads for a computer. Returns thread IDs, titles, and message counts.",
    {
      computer_id: z.string().optional().describe("Computer ID (uses ORGO_DEFAULT_COMPUTER_ID if omitted)"),
    },
    async ({ computer_id }) => {
      try {
        const apiKey = getApiKey();
        const id = resolveComputerId(computer_id);
        const data = await apiRequest("GET", "threads", apiKey, {
          params: { computer_id: id },
          baseUrl: ORGO_V1_BASE,
        });
        return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    }
  );

  server.tool(
    "orgo_get_thread",
    "Get a thread with its full message history.",
    {
      thread_id: z.string().min(1).describe("Thread ID"),
    },
    async ({ thread_id }) => {
      try {
        const apiKey = getApiKey();
        const data = await apiRequest("GET", `threads/${thread_id}`, apiKey, {
          baseUrl: ORGO_V1_BASE,
        });
        return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    }
  );

  server.tool(
    "orgo_delete_thread",
    "Delete a conversation thread.",
    {
      thread_id: z.string().min(1).describe("Thread ID"),
    },
    async ({ thread_id }) => {
      try {
        const apiKey = getApiKey();
        const data = await apiRequest("DELETE", `threads/${thread_id}`, apiKey, {
          baseUrl: ORGO_V1_BASE,
        });
        return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    }
  );
}
