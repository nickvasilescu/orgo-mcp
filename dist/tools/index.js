/**
 * Register all Orgo MCP tools with the server.
 */
import { registerWorkspaceTools } from "./workspaces.js";
import { registerComputerTools } from "./computers.js";
import { registerActionTools } from "./actions.js";
import { registerShellTools } from "./shell.js";
import { registerFileTools } from "./files.js";
import { registerSystemTools } from "./system.js";
export function registerAllTools(server) {
    registerWorkspaceTools(server);
    registerComputerTools(server);
    registerActionTools(server);
    registerShellTools(server);
    registerFileTools(server);
    registerSystemTools(server);
}
//# sourceMappingURL=index.js.map