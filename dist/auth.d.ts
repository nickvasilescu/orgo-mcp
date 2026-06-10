/**
 * API key and computer ID resolution for Orgo MCP.
 *
 * Supports two modes:
 * - HTTP transport: per-request context (X-Orgo-API-Key header; optional
 *   X-Orgo-Default-Computer-Id header / ?computer_id=) via AsyncLocalStorage
 * - stdio transport: ORGO_API_KEY / ORGO_DEFAULT_COMPUTER_ID environment vars
 */
/** Per-request context, carried across one MCP request on the HTTP transport. */
interface RequestContext {
    apiKey: string;
    /**
     * Default computer for this request when a tool omits computer_id. Lets a
     * SHARED hosted server pin a computer per connection — the env var
     * ORGO_DEFAULT_COMPUTER_ID can't be per-user on a multi-tenant host.
     */
    defaultComputerId?: string;
}
/**
 * Get the API key for the current request.
 *
 * Priority:
 * 1. AsyncLocalStorage request context (HTTP transport, set by middleware)
 * 2. ORGO_API_KEY environment variable (stdio transport)
 */
export declare function getApiKey(): string;
/**
 * Resolve computer_id, in order: an explicit param, the per-request default
 * (HTTP: X-Orgo-Default-Computer-Id header / ?computer_id=), then the
 * ORGO_DEFAULT_COMPUTER_ID env var (stdio).
 */
export declare function resolveComputerId(computerId?: string): string;
/**
 * Run a function with a per-request context (HTTP transport middleware).
 */
export declare function runWithRequestContext<T>(ctx: RequestContext, fn: () => T): T;
/**
 * Back-compat: run a function with only an API key in context. Prefer
 * runWithRequestContext when a default computer is also available.
 */
export declare function runWithApiKey<T>(key: string, fn: () => T): T;
/**
 * Describe where the API key came from without leaking the value.
 * Returns "http_header" (HTTP transport middleware), "env:ORGO_API_KEY"
 * (stdio transport), or null when no key is configured.
 */
export declare function getApiKeySource(): "http_header" | "env:ORGO_API_KEY" | null;
export {};
