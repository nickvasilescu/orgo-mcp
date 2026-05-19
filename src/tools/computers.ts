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
import { applyLimit, jsonText, jsonTextCompact } from "./format.js";
import { registerOrgoTool } from "./registry.js";

const COMPACT_DESC = "Return only essential fields (id, name, status, timestamps) instead of the full Orgo API response. Recommended for agent contexts to minimize token usage.";
const LIMIT_DESC = "Optional cap on the number of items returned. The Orgo API doesn't paginate server-side, so this trims client-side. When set and truncation occurs, the response includes `total` and `truncated: true`. Omit to return everything (current default).";

export function registerComputerTools(server: McpServer): void {
  registerOrgoTool(server, {
    name: "orgo_list_computers",
    title: "List Computers",
    description: "List all computers in a workspace. Returns IDs, names, status, specs. Use to pick a target computer when you have a workspace ID — combine with `compact: true` to keep the payload lean.",
    inputSchema: {
      workspace_id: z.string().min(1).describe("Workspace ID to list computers from"),
      compact: z.boolean().optional().default(false).describe(COMPACT_DESC),
      limit: z.number().int().min(1).max(500).optional().describe(LIMIT_DESC),
    },
    toolsets: ["core"],
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: true,
    },
    handler: async ({ workspace_id, compact, limit }) => {
      try {
        const apiKey = getApiKey();
        const data = await apiRequest("GET", `projects/${workspace_id}`, apiKey);
        const computers = (data as Record<string, unknown>).desktops || [];
        const payload = applyLimit({ computers }, "computers", limit);
        // applyLimit only adds count/total/truncated when limit is set;
        // preserve the existing always-on count for backward compat.
        if (limit === undefined) {
          (payload as Record<string, unknown>).count = (computers as unknown[]).length;
        }
        return {
          content: [{ type: "text" as const, text: compact ? jsonTextCompact(payload) : jsonText(payload) }],
        };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });

  registerOrgoTool(server, {
    name: "orgo_create_computer",
    title: "Create Computer",
    description: "Create a virtual computer in a workspace. Boots in under 500ms. Returns computer ID and details. Use when a new VM is needed for a task; prefer `orgo_clone_computer` when an identical starting state is already configured elsewhere.",
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
        return { content: [{ type: "text" as const, text: jsonText(data) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });

  registerOrgoTool(server, {
    name: "orgo_get_computer",
    title: "Get Computer",
    description: "Get computer details including status, specs, and dashboard URL. Use to check VM status before sending actions, or to retrieve the dashboard URL for human handoff.",
    inputSchema: {
      computer_id: z.string().optional().describe("Computer ID (uses ORGO_DEFAULT_COMPUTER_ID if omitted)"),
      compact: z.boolean().optional().default(false).describe(COMPACT_DESC),
    },
    toolsets: ["core"],
    annotations: {
      readOnlyHint: true,
      destructiveHint: false,
      idempotentHint: true,
      openWorldHint: true,
    },
    handler: async ({ computer_id, compact }) => {
      try {
        const apiKey = getApiKey();
        const id = resolveComputerId(computer_id);
        const data = await apiRequest("GET", `computers/${id}`, apiKey);
        return { content: [{ type: "text" as const, text: compact ? jsonTextCompact(data) : jsonText(data) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });

  registerOrgoTool(server, {
    name: "orgo_delete_computer",
    title: "Delete Computer",
    description: "Permanently delete a computer and all its data. Cannot be undone. Use only when explicitly instructed to remove a VM; for transient bad states, prefer `orgo_restart_computer` first.",
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
        return { content: [{ type: "text" as const, text: jsonText(data) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });

  registerOrgoTool(server, {
    name: "orgo_restart_computer",
    title: "Restart Computer",
    description: "Restart a computer. Useful for recovering from unresponsive states. Use when commands hang, the VM is in a bad state, or after kernel/system config changes that need a boot. Destructive: interrupts running processes and unsaved state.",
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
        const flyId = await resolveFlyInstanceId(id, apiKey);
        const data = await apiRequest("POST", `computers/${flyId}/restart`, apiKey);
        return { content: [{ type: "text" as const, text: jsonText(data) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });

  registerOrgoTool(server, {
    name: "orgo_clone_computer",
    title: "Clone Computer",
    description: "Clone/duplicate a computer including its full disk state. Creates an identical copy. Use when spinning up parallel agents on the same starting environment, or snapshotting a known-good state before risky changes.",
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
        return { content: [{ type: "text" as const, text: jsonText(data) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });

  registerOrgoTool(server, {
    name: "orgo_ensure_running",
    title: "Ensure Computer Running",
    description: "Ensure a computer is running. Resumes suspended VMs automatically. Idempotent. Use before sending screen/shell actions when the VM may be suspended (cheaper than orgo_get_computer + conditional restart).",
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
        return { content: [{ type: "text" as const, text: jsonText(data) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });

  registerOrgoTool(server, {
    name: "orgo_resize_computer",
    title: "Resize Computer",
    description: "Resize a computer's CPU, RAM, disk, or bandwidth. Some changes may require a restart. Use when workload exceeds current specs, or to scale down for cost when a task is complete.",
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
        return { content: [{ type: "text" as const, text: jsonText(data) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });

  registerOrgoTool(server, {
    name: "orgo_move_computer",
    title: "Move Computer",
    description: "Move a computer to a different workspace. The computer keeps its ID and disk state. Use to reorganize fleets without recreating VMs; faster than clone + delete for live machines.",
    inputSchema: {
      computer_id: z.string().optional().describe("Computer ID (uses ORGO_DEFAULT_COMPUTER_ID if omitted)"),
      workspace_id: z.string().min(1).describe("Target workspace ID to move the computer to"),
    },
    toolsets: ["admin"],
    annotations: {
      readOnlyHint: false,
      destructiveHint: false,
      idempotentHint: false,
      openWorldHint: true,
    },
    handler: async ({ computer_id, workspace_id }) => {
      try {
        const apiKey = getApiKey();
        const id = resolveComputerId(computer_id);
        const data = await apiRequest("PATCH", `computers/${id}/move`, apiKey, {
          json: { project_id: workspace_id },
        });
        return { content: [{ type: "text" as const, text: jsonText(data) }] };
      } catch (e) {
        return { content: [{ type: "text" as const, text: handleError(e) }], isError: true };
      }
    },
  });
}
