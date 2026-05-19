/**
 * File management tools -- upload, export, list, download.
 */

import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { getApiKey, resolveComputerId } from "../auth.js";
import { apiRequest, uploadFile } from "../client.js";
import { handleError } from "../errors.js";
import { jsonText, jsonTextCompact } from "./format.js";
import { registerOrgoTool } from "./registry.js";

const COMPACT_DESC = "Return only essential fields (id, name, status, timestamps) instead of the full Orgo API response. Recommended for agent contexts to minimize token usage.";

export function registerFileTools(server: McpServer): void {
  registerOrgoTool(server, {
    name: "orgo_list_files",
    title: "List Files",
    description: "List files in a workspace, optionally filtered by computer.",
    inputSchema: {
      workspace_id: z.string().min(1).describe("Workspace ID"),
      computer_id: z.string().optional().describe("Optional computer ID to filter by"),
      compact: z.boolean().optional().default(false).describe(COMPACT_DESC),
    },
    toolsets: ["files"],
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: true,
    },
    handler: async ({ workspace_id, computer_id, compact }) => {
      try {
        const apiKey = getApiKey();
        const params: Record<string, string> = { projectId: workspace_id };
        if (computer_id) params.desktopId = computer_id;
        const data = await apiRequest("GET", "files", apiKey, { params });
        return { content: [{ type: "text" as const, text: compact ? jsonTextCompact(data) : jsonText(data) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });

  registerOrgoTool(server, {
    name: "orgo_export_file",
    title: "Export File",
    description: "Export a file from the computer's filesystem. Returns file info and a download URL (expires in 1 hour).",
    inputSchema: {
      computer_id: z.string().optional().describe("Computer ID (uses ORGO_DEFAULT_COMPUTER_ID if omitted)"),
      path: z.string().min(1).describe("Path on computer (e.g. 'Desktop/report.pdf', '~/Documents/data.csv')"),
    },
    toolsets: ["files"],
    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
    handler: async ({ computer_id, path }) => {
      try {
        const apiKey = getApiKey();
        const id = resolveComputerId(computer_id);
        const data = await apiRequest("POST", "files/export", apiKey, {
          json: { desktopId: id, path },
        });
        return { content: [{ type: "text" as const, text: jsonText(data) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });

  registerOrgoTool(server, {
    name: "orgo_upload_file",
    title: "Upload File",
    description: "Upload a file to a workspace (max 10MB). Pass content as base64.",
    inputSchema: {
      workspace_id: z.string().min(1).describe("Workspace ID to upload to"),
      filename: z.string().min(1).max(255).describe("Filename for the uploaded file"),
      content_base64: z.string().min(1).describe("Base64-encoded file content"),
      computer_id: z.string().optional().describe("Optional computer ID to associate with"),
      content_type: z.string().optional().describe("MIME type (e.g. 'text/plain', 'image/png')"),
    },
    toolsets: ["files"],
    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
    handler: async ({ workspace_id, filename, content_base64, computer_id, content_type }) => {
      try {
        const apiKey = getApiKey();
        const fileBytes = Buffer.from(content_base64, "base64");
        const data = await uploadFile(apiKey, workspace_id, filename, fileBytes, {
          computerId: computer_id,
          contentType: content_type,
        });
        return { content: [{ type: "text" as const, text: jsonText(data) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });

  registerOrgoTool(server, {
    name: "orgo_download_file",
    title: "Download File",
    description: "Get a download URL for a file. Returns a signed URL that expires in 1 hour.",
    inputSchema: {
      file_id: z.string().min(1).describe("File ID (from orgo_list_files or orgo_export_file)"),
    },
    toolsets: ["files"],
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
    handler: async ({ file_id }) => {
      try {
        const apiKey = getApiKey();
        const data = await apiRequest("GET", "files/download", apiKey, {
          params: { id: file_id },
        });
        return { content: [{ type: "text" as const, text: jsonText(data) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });
}
