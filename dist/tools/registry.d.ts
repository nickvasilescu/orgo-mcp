import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { CallToolResult, ToolAnnotations } from "@modelcontextprotocol/sdk/types.js";
import type { z } from "zod";
import { type Toolset } from "./policy.js";
export interface OrgoToolDefinition {
    name: string;
    title: string;
    description: string;
    inputSchema: Record<string, z.ZodTypeAny>;
    toolsets: readonly Toolset[];
    annotations: ToolAnnotations;
    handler: (args: any) => CallToolResult | Promise<CallToolResult>;
}
export declare function registerOrgoTool(server: McpServer, tool: OrgoToolDefinition): void;
