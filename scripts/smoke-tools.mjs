#!/usr/bin/env node
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

const EXPECTED_DEFAULT_TOOLS = [
  "orgo_list_workspaces",
  "orgo_create_workspace",
  "orgo_get_workspace",
  "orgo_workspace_by_name",
  "orgo_list_computers",
  "orgo_create_computer",
  "orgo_get_computer",
  "orgo_delete_computer",
  "orgo_restart_computer",
  "orgo_clone_computer",
  "orgo_ensure_running",
  "orgo_resize_computer",
  "orgo_screenshot",
  "orgo_click",
  "orgo_type",
  "orgo_key",
  "orgo_scroll",
  "orgo_drag",
  "orgo_bash",
  "orgo_exec",
  "orgo_list_files",
  "orgo_export_file",
  "orgo_upload_file",
  "orgo_download_file",
];

const EXPECTED_READ_ONLY_TOOLS = [
  "orgo_list_workspaces",
  "orgo_get_workspace",
  "orgo_workspace_by_name",
  "orgo_list_computers",
  "orgo_get_computer",
  "orgo_screenshot",
  "orgo_list_files",
  "orgo_download_file",
];

const BANNED_TOOLS = [
  "orgo_start_computer",
  "orgo_stop_computer",
  "orgo_vnc_password",
  "orgo_get_profile",
  "orgo_get_credits",
  "orgo_get_transactions",
  "orgo_agent",
  "orgo_list_threads",
  "orgo_get_thread",
  "orgo_delete_thread",
  "orgo_start_stream",
  "orgo_stop_stream",
  "orgo_stream_status",
  "orgo_list_templates",
  "orgo_starred_templates",
  "orgo_star_template",
];

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function assertSameNames(actual, expected, label) {
  const actualSorted = [...actual].sort();
  const expectedSorted = [...expected].sort();
  assert(
    JSON.stringify(actualSorted) === JSON.stringify(expectedSorted),
    `${label} mismatch.\nExpected: ${expectedSorted.join(", ")}\nActual:   ${actualSorted.join(", ")}`
  );
}

async function listToolNames(env = {}) {
  const transport = new StdioClientTransport({
    command: process.execPath,
    args: ["dist/index.js"],
    cwd: process.cwd(),
    env: {
      PATH: process.env.PATH || "",
      HOME: process.env.HOME || "",
      ORGO_API_KEY: "sk_live_TEST_ONLY",
      ...env,
    },
    stderr: "pipe",
  });

  const client = new Client({ name: "orgo-mcp-smoke", version: "0.0.0" });
  try {
    await client.connect(transport);
    const result = await client.listTools();
    return result.tools;
  } finally {
    await client.close();
  }
}

async function run() {
  const defaultTools = await listToolNames();
  const defaultNames = defaultTools.map((tool) => tool.name);
  assertSameNames(defaultNames, EXPECTED_DEFAULT_TOOLS, "default tools");

  for (const banned of BANNED_TOOLS) {
    assert(!defaultNames.includes(banned), `banned tool is exposed: ${banned}`);
  }

  for (const tool of defaultTools) {
    assert(tool.annotations, `${tool.name} is missing annotations`);
    assert(typeof tool.annotations.openWorldHint === "boolean", `${tool.name} missing openWorldHint`);
    assert(typeof tool.annotations.readOnlyHint === "boolean", `${tool.name} missing readOnlyHint`);
    assert(typeof tool.annotations.destructiveHint === "boolean", `${tool.name} missing destructiveHint`);
  }
  const restartTool = defaultTools.find((tool) => tool.name === "orgo_restart_computer");
  assert(restartTool?.annotations?.destructiveHint === true, "orgo_restart_computer must be marked destructive");

  const readOnlyTools = await listToolNames({ ORGO_READ_ONLY: "true" });
  assertSameNames(
    readOnlyTools.map((tool) => tool.name),
    EXPECTED_READ_ONLY_TOOLS,
    "read-only tools"
  );
  for (const tool of readOnlyTools) {
    assert(tool.annotations?.readOnlyHint === true, `${tool.name} exposed in read-only mode without readOnlyHint`);
  }

  const shellTools = await listToolNames({ ORGO_TOOLSETS: "shell" });
  assertSameNames(
    shellTools.map((tool) => tool.name),
    ["orgo_bash", "orgo_exec"],
    "shell toolset"
  );

  const disabledShellTools = await listToolNames({
    ORGO_TOOLSETS: "shell",
    ORGO_DISABLED_TOOLS: "orgo_bash",
  });
  assertSameNames(
    disabledShellTools.map((tool) => tool.name),
    ["orgo_exec"],
    "disabled tool filter"
  );

  const allowlistedTools = await listToolNames({
    ORGO_ENABLED_TOOLS: "orgo_screenshot,orgo_bash",
  });
  assertSameNames(
    allowlistedTools.map((tool) => tool.name),
    ["orgo_screenshot", "orgo_bash"],
    "enabled tool allowlist"
  );

  console.log("Smoke tool checks passed");
}

run().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
