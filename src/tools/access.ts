/**
 * Access tools -- VNC password for direct connection.
 */

import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { getApiKey, resolveComputerId } from "../auth.js";
import { apiRequest } from "../client.js";
import { handleError } from "../errors.js";

export function registerAccessTools(server: McpServer): void {
  server.tool(
    "orgo_vnc_password",
    "Get the VNC password for direct connection to a computer's display. Used for VNC clients, terminal WebSocket, and the orgo-vnc React component.",
    {
      computer_id: z.string().optional().describe("Computer ID (uses ORGO_DEFAULT_COMPUTER_ID if omitted)"),
    },
    async ({ computer_id }) => {
      try {
        const apiKey = getApiKey();
        const id = resolveComputerId(computer_id);
        const data = await apiRequest("GET", `computers/${id}/vnc-password`, apiKey);
        return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    }
  );
}
