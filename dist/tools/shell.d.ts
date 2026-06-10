/**
 * Shell command execution tools -- bash and Python.
 *
 * orgo_bash uses Terminal WSS as primary transport (more reliable),
 * falling back to REST API if WSS fails.
 */
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
export declare function registerShellTools(server: McpServer): void;
