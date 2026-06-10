#!/usr/bin/env node
/**
 * Orgo MCP Server -- Entry point with transport selection.
 *
 * Supports two transport modes:
 * - stdio (default): Direct process communication with Claude Code / Claude Desktop
 * - http: REST API with X-Orgo-API-Key header authentication
 *
 * Usage:
 *   ORGO_API_KEY=sk_live_... npx @orgo-ai/mcp
 *   ORGO_API_KEY=sk_live_... MCP_TRANSPORT=http npx @orgo-ai/mcp
 */
export {};
