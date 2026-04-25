/**
 * Shell command execution tools -- bash and Python.
 *
 * orgo_bash uses Terminal WSS as primary transport (more reliable),
 * falling back to REST API if WSS fails.
 */

import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { getApiKey, resolveComputerId } from "../auth.js";
import { computerAction } from "../client.js";
import { executeViaTerminal } from "../terminal.js";
import { handleError } from "../errors.js";

export function registerShellTools(server: McpServer): void {
  server.tool(
    "orgo_bash",
    "Execute a bash command on the VM. Uses WebSocket terminal (preferred) with REST API fallback.",
    {
      computer_id: z.string().optional().describe("Computer ID (uses ORGO_DEFAULT_COMPUTER_ID if omitted)"),
      command: z.string().min(1).describe("Bash command to execute"),
    },
    async ({ computer_id, command }) => {
      try {
        const apiKey = getApiKey();
        const id = resolveComputerId(computer_id);

        // Try Terminal WSS first (preferred -- persistent session, no stale ports)
        try {
          const output = await executeViaTerminal(id, apiKey, command, 30000);
          return { content: [{ type: "text" as const, text: `$ ${command}\n\n${output}` }] };
        } catch {
          // WSS failed -- fall back to REST API
        }

        // Fallback: REST bash API
        const data = await computerAction("POST", id, "bash", apiKey, {
          json: { command },
        });
        const output = (data as Record<string, unknown>).output || "";
        return { content: [{ type: "text" as const, text: `$ ${command}\n\n${output}` }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    }
  );

  server.tool(
    "orgo_exec",
    "Execute Python code on the computer. Returns output or error details.",
    {
      computer_id: z.string().optional().describe("Computer ID (uses ORGO_DEFAULT_COMPUTER_ID if omitted)"),
      code: z.string().min(1).describe("Python code to execute"),
      timeout: z.number().int().min(1).max(300).default(10).describe("Timeout in seconds"),
    },
    async ({ computer_id, code, timeout }) => {
      try {
        const apiKey = getApiKey();
        const id = resolveComputerId(computer_id);
        const data = await computerAction("POST", id, "exec", apiKey, {
          json: { code, timeout },
        });
        return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    }
  );
}
