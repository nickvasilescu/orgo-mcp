/**
 * Register all Orgo MCP tools with the server.
 */

import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { registerWorkspaceTools } from "./workspaces.js";
import { registerComputerTools } from "./computers.js";
import { registerActionTools } from "./actions.js";
import { registerShellTools } from "./shell.js";
import { registerFileTools } from "./files.js";
import { registerAccessTools } from "./access.js";

export function registerAllTools(server: McpServer): void {
  registerWorkspaceTools(server);
  registerComputerTools(server);
  registerActionTools(server);
  registerShellTools(server);
  registerFileTools(server);
  registerAccessTools(server);
}
