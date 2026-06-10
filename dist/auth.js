/**
 * API key and computer ID resolution for Orgo MCP.
 *
 * Supports two modes:
 * - HTTP transport: per-request context (X-Orgo-API-Key header; optional
 *   X-Orgo-Default-Computer-Id header / ?computer_id=) via AsyncLocalStorage
 * - stdio transport: ORGO_API_KEY / ORGO_DEFAULT_COMPUTER_ID environment vars
 */
import { AsyncLocalStorage } from "node:async_hooks";
const requestStore = new AsyncLocalStorage();
/**
 * Get the API key for the current request.
 *
 * Priority:
 * 1. AsyncLocalStorage request context (HTTP transport, set by middleware)
 * 2. ORGO_API_KEY environment variable (stdio transport)
 */
export function getApiKey() {
    const ctx = requestStore.getStore();
    if (ctx?.apiKey)
        return ctx.apiKey;
    const envKey = process.env.ORGO_API_KEY;
    if (envKey)
        return envKey;
    throw new Error("API key required. HTTP: include X-Orgo-API-Key header. " +
        "stdio: set ORGO_API_KEY env var. Get your key at https://orgo.ai");
}
/**
 * Resolve computer_id, in order: an explicit param, the per-request default
 * (HTTP: X-Orgo-Default-Computer-Id header / ?computer_id=), then the
 * ORGO_DEFAULT_COMPUTER_ID env var (stdio).
 */
export function resolveComputerId(computerId) {
    if (computerId)
        return computerId;
    const requestDefault = requestStore.getStore()?.defaultComputerId;
    if (requestDefault)
        return requestDefault;
    const defaultId = process.env.ORGO_DEFAULT_COMPUTER_ID;
    if (defaultId)
        return defaultId;
    throw new Error("computer_id required. Pass it explicitly, or set a default " +
        "(HTTP: X-Orgo-Default-Computer-Id header or ?computer_id=; " +
        "stdio: ORGO_DEFAULT_COMPUTER_ID env var).");
}
/**
 * Run a function with a per-request context (HTTP transport middleware).
 */
export function runWithRequestContext(ctx, fn) {
    return requestStore.run(ctx, fn);
}
/**
 * Back-compat: run a function with only an API key in context. Prefer
 * runWithRequestContext when a default computer is also available.
 */
export function runWithApiKey(key, fn) {
    return requestStore.run({ apiKey: key }, fn);
}
/**
 * Describe where the API key came from without leaking the value.
 * Returns "http_header" (HTTP transport middleware), "env:ORGO_API_KEY"
 * (stdio transport), or null when no key is configured.
 */
export function getApiKeySource() {
    if (requestStore.getStore()?.apiKey)
        return "http_header";
    if (process.env.ORGO_API_KEY)
        return "env:ORGO_API_KEY";
    return null;
}
//# sourceMappingURL=auth.js.map