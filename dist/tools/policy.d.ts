import type { ToolAnnotations } from "@modelcontextprotocol/sdk/types.js";
export type Toolset = "core" | "admin" | "screen" | "shell" | "files";
export declare const ALL_TOOLSETS: readonly Toolset[];
export interface ToolPolicyMetadata {
    name: string;
    toolsets: readonly Toolset[];
    annotations: ToolAnnotations;
}
export declare function isReadOnlyMode(): boolean;
export declare function isToolEnabled(tool: ToolPolicyMetadata): boolean;
export declare function getToolPolicySummary(): string;
