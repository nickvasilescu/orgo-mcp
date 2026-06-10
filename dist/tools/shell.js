/**
 * Shell command execution tools -- bash and Python.
 *
 * orgo_bash uses Terminal WSS as primary transport (more reliable),
 * falling back to REST API if WSS fails.
 */
import { z } from "zod";
import { getApiKey, resolveComputerId } from "../auth.js";
import { computerAction } from "../client.js";
import { executeViaTerminal } from "../terminal.js";
import { handleError } from "../errors.js";
import { jsonText } from "./format.js";
import { registerOrgoTool } from "./registry.js";
export function registerShellTools(server) {
    registerOrgoTool(server, {
        name: "orgo_bash",
        title: "Run Bash",
        description: "Execute a bash command on the VM. Uses WebSocket terminal (preferred) with REST API fallback. Returns output plus the exit code when non-zero. Prefer this over GUI clicks for anything scriptable — file ops, installs, git, curl, process management — and over `orgo_exec` when the task is a shell pipeline. For long-running commands (installs, builds), raise `timeout`.",
        inputSchema: {
            computer_id: z.string().optional().describe("Computer ID (uses ORGO_DEFAULT_COMPUTER_ID if omitted)"),
            command: z.string().min(1).describe("Bash command to execute"),
            timeout: z.number().int().min(1).max(300).default(30).describe("Timeout in seconds. On timeout, partial output is returned with an explicit truncation marker."),
        },
        toolsets: ["shell"],
        annotations: {
            readOnlyHint: false,
            destructiveHint: true,
            idempotentHint: false,
            openWorldHint: true,
        },
        handler: async ({ computer_id, command, timeout }) => {
            try {
                const apiKey = getApiKey();
                const id = resolveComputerId(computer_id);
                const timeoutMs = timeout * 1000;
                // Try Terminal WSS first (preferred -- persistent session, exit codes)
                try {
                    const output = await executeViaTerminal(id, apiKey, command, timeoutMs);
                    return { content: [{ type: "text", text: `$ ${command}\n\n${output}` }] };
                }
                catch {
                    // WSS failed -- fall back to REST API
                }
                // Fallback: REST bash API (HTTP timeout padded so the transport
                // doesn't abort before the command's own budget elapses)
                const data = await computerAction("POST", id, "bash", apiKey, {
                    json: { command },
                    timeout: timeoutMs + 5000,
                });
                const output = data.output || "";
                return { content: [{ type: "text", text: `$ ${command}\n\n${output}` }] };
            }
            catch (e) {
                return { content: [{ type: "text", text: handleError(e) }], isError: true };
            }
        },
    });
    registerOrgoTool(server, {
        name: "orgo_exec",
        title: "Run Python",
        description: "Execute Python code on the computer. Returns output or error details. Use for computation, JSON manipulation, HTTP calls, or any task naturally expressed in Python — `orgo_bash` is the right pick for shell-native operations.",
        inputSchema: {
            computer_id: z.string().optional().describe("Computer ID (uses ORGO_DEFAULT_COMPUTER_ID if omitted)"),
            code: z.string().min(1).describe("Python code to execute"),
            timeout: z.number().int().min(1).max(300).default(10).describe("Timeout in seconds"),
        },
        toolsets: ["shell"],
        annotations: {
            readOnlyHint: false,
            destructiveHint: true,
            idempotentHint: false,
            openWorldHint: true,
        },
        handler: async ({ computer_id, code, timeout }) => {
            try {
                const apiKey = getApiKey();
                const id = resolveComputerId(computer_id);
                const data = await computerAction("POST", id, "exec", apiKey, {
                    json: { code, timeout },
                    // HTTP timeout must outlast the code's own budget
                    timeout: (timeout + 5) * 1000,
                });
                return { content: [{ type: "text", text: jsonText(data) }] };
            }
            catch (e) {
                return { content: [{ type: "text", text: handleError(e) }], isError: true };
            }
        },
    });
}
//# sourceMappingURL=shell.js.map