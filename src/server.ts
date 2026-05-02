/**
 * Orgo MCP Server -- McpServer instantiation and tool registration.
 */

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { registerAllTools } from "./tools/index.js";
import { getToolPolicySummary } from "./tools/policy.js";

const VERSION = "4.0.0";

const ORGO_MCP_INSTRUCTIONS = `
# Orgo MCP -- Tool Selection Guide

Orgo provides cloud virtual machines for AI agents. This MCP gives you focused control:
workspace/computer management, shell commands, screen interaction, and file management.

## Default Computer

If ORGO_DEFAULT_COMPUTER_ID is set, you can omit computer_id on any tool call.
The default is used automatically. Pass computer_id explicitly to target a different VM.

## When orgo-chrome-mcp is also active

If both orgo_* and orgo_chrome_* tools are available, use this hierarchy:

### For Chrome/browser tasks -> prefer orgo_chrome_* tools
- **Reading pages**: orgo_chrome_read_page (DOM/accessibility tree) over orgo_screenshot (pixels)
- **Clicking elements**: orgo_chrome_click (element refs, reliable) over orgo_click (pixel coordinates)
- **Typing in forms**: orgo_chrome_form_input (sets value by ref) over orgo_type (simulates keystrokes)
- **Scrolling pages**: orgo_chrome_scroll over orgo_scroll
- **Running JS**: orgo_chrome_evaluate (in-page context) over orgo_bash

### For non-browser / native desktop tasks -> use orgo_* tools
- **Full VM screenshots**: orgo_screenshot -- shows entire desktop, all windows
- **Native app interaction**: orgo_click, orgo_type, orgo_key, orgo_drag -- pixel-based, works everywhere
- **Shell commands**: orgo_bash -- run any command on the VM (uses WebSocket terminal for reliability)
- **Python execution**: orgo_exec -- run Python code directly
- **File operations**: orgo_export_file, orgo_upload_file -- move files to/from the VM

### General rule
DOM-aware tools (orgo_chrome_*) are faster and more reliable for web content.
Pixel-based tools (orgo_*) are the right choice for native apps, terminal windows, or full desktop.

## Tool Categories

- **VM Management**: list, create, get, delete, restart, clone, resize, ensure_running
- **Screen Actions**: screenshot, click, type, key, scroll, drag
- **Shell**: bash (WebSocket terminal preferred), exec (Python)
- **Files**: list, upload, export, download

## Production Policy Controls

Operators can restrict the exposed tool surface without code changes:
- ORGO_READ_ONLY=true exposes only tools marked read-only.
- ORGO_TOOLSETS chooses tool groups: core, admin, screen, shell, files.
- ORGO_ENABLED_TOOLS allowlists exact tool names.
- ORGO_DISABLED_TOOLS removes exact tool names.

Current policy:
${getToolPolicySummary()}
`.trim();

/**
 * Create and configure the Orgo MCP server with all tools registered.
 */
export function createOrgoMcpServer(): McpServer {
  const server = new McpServer(
    {
      name: "orgo-mcp",
      version: VERSION,
    },
    {
      instructions: ORGO_MCP_INSTRUCTIONS,
    }
  );

  registerAllTools(server);

  return server;
}
