import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { CallToolResult, ToolAnnotations } from "@modelcontextprotocol/sdk/types.js";
import type { z } from "zod";
import { isToolEnabled, type Toolset } from "./policy.js";

export interface OrgoToolDefinition {
  name: string;
  title: string;
  description: string;
  inputSchema: Record<string, z.ZodTypeAny>;
  toolsets: readonly Toolset[];
  annotations: ToolAnnotations;
  handler: (args: any) => CallToolResult | Promise<CallToolResult>;
}

export function registerOrgoTool(server: McpServer, tool: OrgoToolDefinition): void {
  if (!isToolEnabled(tool)) return;

  server.registerTool(
    tool.name,
    {
      title: tool.title,
      description: tool.description,
      inputSchema: tool.inputSchema,
      annotations: {
        ...tool.annotations,
        title: tool.annotations.title || tool.title,
      },
      _meta: {
        "orgo/toolsets": tool.toolsets,
      },
    },
    tool.handler as any
  );
}
