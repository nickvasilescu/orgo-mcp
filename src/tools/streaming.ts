/**
 * RTMP streaming tools -- stream computer display to Twitch, YouTube, etc.
 *
 * Requires pre-configured RTMP connections in your Orgo account settings.
 */

import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { getApiKey, resolveComputerId } from "../auth.js";
import { computerAction } from "../client.js";
import { handleError } from "../errors.js";

export function registerStreamingTools(server: McpServer): void {
  server.tool(
    "orgo_start_stream",
    "Start RTMP streaming from a computer to a pre-configured connection. One stream per computer.",
    {
      computer_id: z.string().optional().describe("Computer ID (uses ORGO_DEFAULT_COMPUTER_ID if omitted)"),
      connection_name: z.string().min(1).describe("Name of pre-configured RTMP connection (set up at orgo.ai)"),
    },
    async ({ computer_id, connection_name }) => {
      try {
        const apiKey = getApiKey();
        const id = resolveComputerId(computer_id);
        const data = await computerAction("POST", id, "stream/start", apiKey, {
          json: { connection_name },
        });
        return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    }
  );

  server.tool(
    "orgo_stream_status",
    "Get current streaming status: idle, streaming, or terminated.",
    {
      computer_id: z.string().optional().describe("Computer ID (uses ORGO_DEFAULT_COMPUTER_ID if omitted)"),
    },
    async ({ computer_id }) => {
      try {
        const apiKey = getApiKey();
        const id = resolveComputerId(computer_id);
        const data = await computerAction("GET", id, "stream/status", apiKey);
        return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    }
  );

  server.tool(
    "orgo_stop_stream",
    "Stop an active RTMP stream.",
    {
      computer_id: z.string().optional().describe("Computer ID (uses ORGO_DEFAULT_COMPUTER_ID if omitted)"),
    },
    async ({ computer_id }) => {
      try {
        const apiKey = getApiKey();
        const id = resolveComputerId(computer_id);
        const data = await computerAction("POST", id, "stream/stop", apiKey);
        return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    }
  );
}
