/**
 * File management tools -- upload, export, list, download.
 */

import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { getApiKey, resolveComputerId } from "../auth.js";
import { apiRequest, uploadFile } from "../client.js";
import { handleError } from "../errors.js";

export function registerFileTools(server: McpServer): void {
  server.tool(
    "orgo_list_files",
    "List files in a workspace, optionally filtered by computer.",
    {
      workspace_id: z.string().min(1).describe("Workspace ID"),
      computer_id: z.string().optional().describe("Optional computer ID to filter by"),
    },
    async ({ workspace_id, computer_id }) => {
      try {
        const apiKey = getApiKey();
        const params: Record<string, string> = { projectId: workspace_id };
        if (computer_id) params.desktopId = computer_id;
        const data = await apiRequest("GET", "files", apiKey, { params });
        return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    }
  );

  server.tool(
    "orgo_export_file",
    "Export a file from the computer's filesystem. Returns file info and a download URL (expires in 1 hour).",
    {
      computer_id: z.string().optional().describe("Computer ID (uses ORGO_DEFAULT_COMPUTER_ID if omitted)"),
      path: z.string().min(1).describe("Path on computer (e.g. 'Desktop/report.pdf', '~/Documents/data.csv')"),
    },
    async ({ computer_id, path }) => {
      try {
        const apiKey = getApiKey();
        const id = resolveComputerId(computer_id);
        const data = await apiRequest("POST", "files/export", apiKey, {
          json: { desktopId: id, path },
        });
        return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    }
  );

  server.tool(
    "orgo_upload_file",
    "Upload a file to a workspace (max 10MB). Pass content as base64.",
    {
      workspace_id: z.string().min(1).describe("Workspace ID to upload to"),
      filename: z.string().min(1).max(255).describe("Filename for the uploaded file"),
      content_base64: z.string().min(1).describe("Base64-encoded file content"),
      computer_id: z.string().optional().describe("Optional computer ID to associate with"),
      content_type: z.string().optional().describe("MIME type (e.g. 'text/plain', 'image/png')"),
    },
    async ({ workspace_id, filename, content_base64, computer_id, content_type }) => {
      try {
        const apiKey = getApiKey();
        const fileBytes = Buffer.from(content_base64, "base64");
        const data = await uploadFile(apiKey, workspace_id, filename, fileBytes, {
          computerId: computer_id,
          contentType: content_type,
        });
        return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    }
  );

  server.tool(
    "orgo_download_file",
    "Get a download URL for a file. Returns a signed URL that expires in 1 hour.",
    {
      file_id: z.string().min(1).describe("File ID (from orgo_list_files or orgo_export_file)"),
    },
    async ({ file_id }) => {
      try {
        const apiKey = getApiKey();
        const data = await apiRequest("GET", "files/download", apiKey, {
          params: { id: file_id },
        });
        return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    }
  );
}
