/**
 * API key and computer ID resolution for Orgo MCP.
 *
 * Supports two modes:
 * - HTTP transport: key from X-Orgo-API-Key request header (via AsyncLocalStorage)
 * - stdio transport: key from ORGO_API_KEY environment variable
 */

import { AsyncLocalStorage } from "node:async_hooks";

const apiKeyStore = new AsyncLocalStorage<string>();

/**
 * Get the API key for the current request.
 *
 * Priority:
 * 1. AsyncLocalStorage (HTTP transport, set by middleware)
 * 2. ORGO_API_KEY environment variable (stdio transport)
 */
export function getApiKey(): string {
  const requestKey = apiKeyStore.getStore();
  if (requestKey) return requestKey;

  const envKey = process.env.ORGO_API_KEY;
  if (envKey) return envKey;

  throw new Error(
    "API key required. HTTP: include X-Orgo-API-Key header. " +
      "stdio: set ORGO_API_KEY env var. Get your key at https://orgo.ai"
  );
}

/**
 * Resolve computer_id from explicit param or ORGO_DEFAULT_COMPUTER_ID env var.
 */
export function resolveComputerId(computerId?: string): string {
  if (computerId) return computerId;

  const defaultId = process.env.ORGO_DEFAULT_COMPUTER_ID;
  if (defaultId) return defaultId;

  throw new Error(
    "computer_id required. Either pass it explicitly or set ORGO_DEFAULT_COMPUTER_ID env var."
  );
}

/**
 * Run a function with a specific API key in context (for HTTP transport middleware).
 */
export function runWithApiKey<T>(key: string, fn: () => T): T {
  return apiKeyStore.run(key, fn);
}
