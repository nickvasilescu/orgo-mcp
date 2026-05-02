/**
 * Computer management tools.
 *
 * Computers are virtual machines within workspaces. They boot in under 500ms
 * and can be controlled via actions, shell commands, or AI agents.
 */

import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { getApiKey, resolveComputerId } from "../auth.js";
import { apiRequest, resolveFlyInstanceId } from "../client.js";
import { handleError } from "../errors.js";
import { registerOrgoTool } from "./registry.js";

export function registerComputerTools(server: McpServer): void {
  registerOrgoTool(server, {
    name: "orgo_list_computers",
    title: "List Computers",
    description: "List all computers in a workspace. Returns IDs, names, status, specs.",
    inputSchema: {
      workspace_id: z.string().min(1).describe("Workspace ID to list computers from"),
    },
    toolsets: ["core"],
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: true,
    },
    handler: async ({ workspace_id }) => {
      try {
        const apiKey = getApiKey();
        const data = await apiRequest("GET", `projects/${workspace_id}`, apiKey);
        const computers = (data as Record<string, unknown>).desktops || [];
        return {
          content: [{ type: "text" as const, text: JSON.stringify({ computers, count: (computers as unknown[]).length }, null, 2) }],
        };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });

  registerOrgoTool(server, {
    name: "orgo_create_computer",
    title: "Create Computer",
    description: "Create a virtual computer in a workspace. Boots in under 500ms. Returns computer ID and details.",
    inputSchema: {
      workspace: z.string().min(1).describe("Workspace name (created automatically if doesn't exist)"),
      name: z.string().max(100).optional().describe("Computer name (auto-generated if omitted)"),
      os: z.enum(["linux"]).default("linux").describe("Operating system (only linux supported)"),
      ram: z.enum(["4", "8", "16", "32", "64"]).default("4").describe("RAM in GB"),
      cpu: z.enum(["1", "2", "4", "8", "16"]).default("2").describe("CPU cores"),
      gpu: z.enum(["none", "a10", "l40s", "a100-40gb", "a100-80gb"]).default("none").describe("GPU type"),
      resolution: z.string().default("1280x720x24").describe("Display resolution (WIDTHxHEIGHTxDEPTH)"),
      image: z.string().optional().describe("Custom template image reference (from orgo Forge/Templates)"),
    },
    toolsets: ["admin"],
    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
    handler: async ({ workspace, name, os, ram, cpu, gpu, resolution, image }) => {
      try {
        const apiKey = getApiKey();
        const body: Record<string, unknown> = {
          project: workspace,
          os,
          ram: parseInt(ram),
          cpu: parseInt(cpu),
          gpu,
          resolution,
        };
        if (name) body.name = name;
        if (image) body.image = image;

        const data = await apiRequest("POST", "computers", apiKey, { json: body, timeout: 60000 });
        return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });

  registerOrgoTool(server, {
    name: "orgo_get_computer",
    title: "Get Computer",
    description: "Get computer details including status, specs, and dashboard URL.",
    inputSchema: {
      computer_id: z.string().optional().describe("Computer ID (uses ORGO_DEFAULT_COMPUTER_ID if omitted)"),
    },
    toolsets: ["core"],
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: true,
    },
    handler: async ({ computer_id }) => {
      try {
        const apiKey = getApiKey();
        const id = resolveComputerId(computer_id);
        const data = await apiRequest("GET", `computers/${id}`, apiKey);
        return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });

  registerOrgoTool(server, {
    name: "orgo_delete_computer",
    title: "Delete Computer",
    description: "Permanently delete a computer and all its data. Cannot be undone.",
    inputSchema: {
      computer_id: z.string().optional().describe("Computer ID (uses ORGO_DEFAULT_COMPUTER_ID if omitted)"),
    },
    toolsets: ["admin"],
    annotations: {
      readOnlyHint: false,
      destructiveHint: true,
      idempotentHint: false,
      openWorldHint: true,
    },
    handler: async ({ computer_id }) => {
      try {
        const apiKey = getApiKey();
        const id = resolveComputerId(computer_id);
        const data = await apiRequest("DELETE", `computers/${id}`, apiKey);
        return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });

  registerOrgoTool(server, {
    name: "orgo_restart_computer",
    title: "Restart Computer",
    description: "Restart a computer. Useful for recovering from unresponsive states.",
    inputSchema: {
      computer_id: z.string().optional().describe("Computer ID (uses ORGO_DEFAULT_COMPUTER_ID if omitted)"),
    },
    toolsets: ["admin"],
    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
    handler: async ({ computer_id }) => {
      try {
        const apiKey = getApiKey();
        const id = resolveComputerId(computer_id);
        const flyId = await resolveFlyInstanceId(id, apiKey);
        const data = await apiRequest("POST", `computers/${flyId}/restart`, apiKey);
        return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });

  registerOrgoTool(server, {
    name: "orgo_clone_computer",
    title: "Clone Computer",
    description: "Clone/duplicate a computer including its full disk state. Creates an identical copy.",
    inputSchema: {
      computer_id: z.string().optional().describe("Computer ID to clone (uses ORGO_DEFAULT_COMPUTER_ID if omitted)"),
      name: z.string().optional().describe("Name for the cloned computer"),
      workspace_id: z.string().optional().describe("Target workspace ID (defaults to same workspace)"),
    },
    toolsets: ["admin"],
    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
    handler: async ({ computer_id, name, workspace_id }) => {
      try {
        const apiKey = getApiKey();
        const id = resolveComputerId(computer_id);
        const flyId = await resolveFlyInstanceId(id, apiKey);
        const body: Record<string, unknown> = {};
        if (name) body.name = name;
        if (workspace_id) body.targetProjectId = workspace_id;
        const data = await apiRequest("POST", `computers/${flyId}/clone`, apiKey, {
          json: body,
          timeout: 120000,
        });
        return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });

  registerOrgoTool(server, {
    name: "orgo_ensure_running",
    title: "Ensure Computer Running",
    description: "Ensure a computer is running. Resumes suspended VMs automatically. Idempotent.",
    inputSchema: {
      computer_id: z.string().optional().describe("Computer ID (uses ORGO_DEFAULT_COMPUTER_ID if omitted)"),
    },
    toolsets: ["admin"],
    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: true,
    },
    handler: async ({ computer_id }) => {
      try {
        const apiKey = getApiKey();
        const id = resolveComputerId(computer_id);
        const data = await apiRequest("POST", `computers/${id}/ensure-running`, apiKey);
        return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });

  registerOrgoTool(server, {
    name: "orgo_resize_computer",
    title: "Resize Computer",
    description: "Resize a computer's CPU, RAM, disk, or bandwidth. Some changes may require a restart.",
    inputSchema: {
      computer_id: z.string().optional().describe("Computer ID (uses ORGO_DEFAULT_COMPUTER_ID if omitted)"),
      cpu: z.number().optional().describe("New CPU cores (1, 2, 4, 8, 16)"),
      ram: z.number().optional().describe("New RAM in GB (4, 8, 16, 32, 64)"),
      disk_size_gb: z.number().optional().describe("New disk size in GB"),
      bandwidth_limit_mbps: z.number().optional().describe("Bandwidth limit in Mbps"),
    },
    toolsets: ["admin"],
    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
    handler: async ({ computer_id, cpu, ram, disk_size_gb, bandwidth_limit_mbps }) => {
      try {
        const apiKey = getApiKey();
        const id = resolveComputerId(computer_id);
        const body: Record<string, unknown> = {};
        if (cpu !== undefined) body.cpu = cpu;
        if (ram !== undefined) body.ram = ram;
        if (disk_size_gb !== undefined) body.disk_size_gb = disk_size_gb;
        if (bandwidth_limit_mbps !== undefined) body.bandwidth_limit_mbps = bandwidth_limit_mbps;
        const data = await apiRequest("PATCH", `computers/${id}/resize`, apiKey, { json: body });
        return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });
}
