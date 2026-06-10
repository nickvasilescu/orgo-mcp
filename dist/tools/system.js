/**
 * System / health tools — orgo_doctor.
 */
import { getApiKey, getApiKeySource } from "../auth.js";
import { apiRequest } from "../client.js";
import { HttpError } from "../errors.js";
import { jsonText } from "./format.js";
import { registerOrgoTool } from "./registry.js";
export function registerSystemTools(server) {
    registerOrgoTool(server, {
        name: "orgo_doctor",
        title: "Health Check",
        description: "Probe MCP server health. Returns auth source, API reachability, and latency. " +
            "Use as the first call to verify setup before invoking other tools, or after errors to " +
            "distinguish auth failures from network/API issues.",
        inputSchema: {},
        toolsets: ["core"],
        annotations: {
            readOnlyHint: true,
            destructiveHint: false,
            idempotentHint: true,
            openWorldHint: true,
        },
        handler: async () => {
            const authSource = getApiKeySource();
            const auth = {
                configured: authSource !== null,
                source: authSource,
            };
            const api = {
                reachable: false,
                status_code: null,
                latency_ms: 0,
                error: null,
            };
            if (auth.configured) {
                const start = Date.now();
                try {
                    const apiKey = getApiKey();
                    await apiRequest("GET", "projects", apiKey, { timeout: 10000 });
                    api.reachable = true;
                    api.status_code = 200;
                    api.latency_ms = Date.now() - start;
                }
                catch (e) {
                    api.latency_ms = Date.now() - start;
                    if (e instanceof HttpError) {
                        api.status_code = e.status;
                        api.error = `HTTP ${e.status}: ${e.detail || e.statusText || "request failed"}`;
                    }
                    else if (e instanceof Error) {
                        api.error = e.message;
                    }
                    else {
                        api.error = String(e);
                    }
                }
            }
            else {
                api.error = "No API key configured (no ORGO_API_KEY env var, no X-Orgo-API-Key header)";
            }
            const result = {
                ok: auth.configured && api.reachable,
                auth,
                api,
            };
            return { content: [{ type: "text", text: jsonText(result) }] };
        },
    });
}
//# sourceMappingURL=system.js.map