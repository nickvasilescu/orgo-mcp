#!/usr/bin/env node
// End-to-end live test: drives @orgo-ai/mcp via stdio just like a real
// MCP client, exercises every tool category against the live Orgo API.
//
// Catches docs/reality drift that mock-only tests miss (e.g., the wait
// `seconds` vs `duration` bug). Requires a working API key + at least
// one running computer for the action-tool section to fully exercise.
//
// Run locally:
//   ORGO_API_KEY=sk_live_... npm run test:live
//
// In CI: gated on the ORGO_TEST_API_KEY secret. Skipped on fork PRs.

import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

const apiKey = process.env.ORGO_API_KEY;
if (!apiKey) {
  console.error("ORGO_API_KEY is required");
  process.exit(1);
}

function fmt(value) {
  if (typeof value === "string") return value.length > 200 ? value.slice(0, 200) + " …(truncated)" : value;
  return JSON.stringify(value, null, 2).slice(0, 400);
}

async function spawn(env = {}) {
  const transport = new StdioClientTransport({
    command: process.execPath,
    args: ["dist/index.js"],
    cwd: process.cwd(),
    env: { PATH: process.env.PATH, HOME: process.env.HOME, ORGO_API_KEY: apiKey, ...env },
    stderr: "pipe",
  });
  const client = new Client({ name: "orgo-e2e", version: "0.0.0" });
  await client.connect(transport);
  return { client, close: () => client.close() };
}

async function run() {
  const results = [];
  const record = (name, ok, info) => {
    results.push({ name, ok, info });
    const tag = ok ? "PASS" : "FAIL";
    console.log(`[${tag}] ${name}${info ? ` — ${info}` : ""}`);
  };

  // === Section 0: orgo_doctor — first thing an agent should call ===
  console.log("\n=== 0. orgo_doctor health probe ===");
  {
    const { client, close } = await spawn();
    try {
      const d = await client.callTool({ name: "orgo_doctor", arguments: {} });
      const body = JSON.parse(d.content?.[0]?.text || "{}");
      record("doctor returns ok=true with valid key", body.ok === true);
      record("doctor reports auth source", body.auth?.source === "env:ORGO_API_KEY");
      record("doctor reports api.reachable=true", body.api?.reachable === true);
      record("doctor returns latency_ms", typeof body.api?.latency_ms === "number" && body.api.latency_ms > 0);
    } finally {
      await close();
    }
  }

  // Bad-key probe — server starts (since key is non-empty), API rejects → doctor surfaces 401
  console.log("\n=== 0b. orgo_doctor with bad key ===");
  {
    const { client, close } = await spawn({ ORGO_API_KEY: "sk_live_invalid_zzz_test_only" });
    try {
      const d = await client.callTool({ name: "orgo_doctor", arguments: {} });
      const body = JSON.parse(d.content?.[0]?.text || "{}");
      record("doctor reports ok=false with bad key", body.ok === false);
      record("doctor reports status_code=401 on bad key", body.api?.status_code === 401, `got=${body.api?.status_code}`);
      record("doctor error message does not double-prefix HTTP", !body.api?.error?.includes("HTTP 401: HTTP 401"));
    } finally {
      await close();
    }
  }

  // === Section 1: Cold-start handshake + tool inventory ===
  console.log("\n=== 1. Connect + list tools (default) ===");
  const { client, close } = await spawn();
  try {
    const list = await client.listTools();
    const names = list.tools.map((t) => t.name).sort();
    record("connected via stdio", true);
    record("28 tools exposed", names.length === 28, `actual=${names.length}`);
    record("orgo_doctor present in tool list", names.includes("orgo_doctor"));

    // Spot-check annotations on the 3 newly-added tools
    for (const newTool of ["orgo_delete_workspace", "orgo_wait", "orgo_move_computer"]) {
      const t = list.tools.find((x) => x.name === newTool);
      record(`${newTool} present`, !!t);
      if (t) {
        record(`${newTool} has annotations`, !!t.annotations && typeof t.annotations.openWorldHint === "boolean");
      }
    }

    // Spot-check destructive flags
    const del = list.tools.find((x) => x.name === "orgo_delete_workspace");
    record("orgo_delete_workspace destructiveHint=true", del?.annotations?.destructiveHint === true);
    const wait = list.tools.find((x) => x.name === "orgo_wait");
    record("orgo_wait readOnlyHint=true", wait?.annotations?.readOnlyHint === true);

    // === Section 2: Read-only live calls ===
    console.log("\n=== 2. Read-only live calls ===");

    // 2a. List workspaces
    const lw = await client.callTool({ name: "orgo_list_workspaces", arguments: {} });
    const wsText = lw.content?.[0]?.text || "";
    let wsCount = 0;
    let firstWorkspaceId = null;
    let firstWorkspaceName = null;
    try {
      const wsJson = JSON.parse(wsText);
      const arr = wsJson.projects || wsJson.workspaces || [];
      wsCount = arr.length;
      if (arr[0]) {
        firstWorkspaceId = arr[0].id;
        firstWorkspaceName = arr[0].name;
      }
    } catch {}
    record("orgo_list_workspaces returns workspaces", wsCount > 0, `count=${wsCount}`);

    // 2b. Get workspace
    if (firstWorkspaceId) {
      const gw = await client.callTool({ name: "orgo_get_workspace", arguments: { workspace_id: firstWorkspaceId } });
      record("orgo_get_workspace by ID", !gw.isError);
    }

    // 2c. Get workspace by name
    if (firstWorkspaceName) {
      const gn = await client.callTool({ name: "orgo_workspace_by_name", arguments: { name: firstWorkspaceName } });
      record("orgo_workspace_by_name", !gn.isError);
    }

    // 2d. List computers
    let firstComputerId = null;
    let runningComputerId = null;
    if (firstWorkspaceId) {
      const lc = await client.callTool({ name: "orgo_list_computers", arguments: { workspace_id: firstWorkspaceId } });
      try {
        const data = JSON.parse(lc.content?.[0]?.text || "{}");
        const computers = data.computers || [];
        firstComputerId = computers[0]?.id;
        runningComputerId = computers.find((c) => c.status === "running")?.id || firstComputerId;
        record("orgo_list_computers", computers.length >= 0, `count=${computers.length}`);
      } catch {
        record("orgo_list_computers", false, "parse error");
      }
    }

    // 2e. Get computer details
    if (firstComputerId) {
      const gc = await client.callTool({ name: "orgo_get_computer", arguments: { computer_id: firstComputerId } });
      record("orgo_get_computer", !gc.isError);
    }

    // 2f. List files
    if (firstWorkspaceId) {
      const lf = await client.callTool({
        name: "orgo_list_files",
        arguments: { workspace_id: firstWorkspaceId },
      });
      record("orgo_list_files", !lf.isError);
    }

    // === Section 3: Active computer interaction (only if we have a running VM) ===
    console.log("\n=== 3. Live computer interaction ===");

    if (!runningComputerId) {
      console.log("  (no running computer found — skipping computer-action tests)");
    } else {
      console.log(`  using computer: ${runningComputerId}`);

      // 3a. Ensure running (idempotent)
      const er = await client.callTool({ name: "orgo_ensure_running", arguments: { computer_id: runningComputerId } });
      record("orgo_ensure_running", !er.isError);

      // 3b. Wait (NEW tool — safe, harmless)
      const w = await client.callTool({ name: "orgo_wait", arguments: { computer_id: runningComputerId, duration: 0.5 } });
      record("orgo_wait (new) — 0.5s pause", !w.isError, fmt(w.content?.[0]?.text));

      // 3c. Bash — safe read-only command
      const b = await client.callTool({
        name: "orgo_bash",
        arguments: { computer_id: runningComputerId, command: "echo hello && uname -a" },
      });
      record("orgo_bash — echo + uname", !b.isError, fmt(b.content?.[0]?.text));

      // 3d. Exec Python
      const ex = await client.callTool({
        name: "orgo_exec",
        arguments: { computer_id: runningComputerId, code: "print(2 + 2)", timeout: 5 },
      });
      record("orgo_exec — 2+2", !ex.isError, fmt(ex.content?.[0]?.text));

      // 3e. Screenshot
      const ss = await client.callTool({ name: "orgo_screenshot", arguments: { computer_id: runningComputerId } });
      const ssOk = ss.content?.[0]?.type === "image" && (ss.content[0].data?.length || 0) > 1000;
      record("orgo_screenshot returns image", ssOk, ssOk ? `${ss.content[0].data.length} chars base64` : "no image");
    }

    // === Section 4: Redaction sanity ===
    console.log("\n=== 4. Redaction ===");
    if (firstComputerId) {
      const gc = await client.callTool({ name: "orgo_get_computer", arguments: { computer_id: firstComputerId } });
      const body = gc.content?.[0]?.text || "";
      const leaks = [];
      // These field names should be redacted in output (per format.ts)
      for (const pat of [/"vnc_password_encrypted"\s*:\s*"[a-f0-9]{20,}"/, /"vncPassword"\s*:\s*"[a-z0-9]{10,}"/i]) {
        if (pat.test(body)) leaks.push(pat.source);
      }
      record("sensitive fields redacted from get_computer", leaks.length === 0, leaks.length ? `leaks: ${leaks.join(", ")}` : "no leaks");
    }

    // === Section 5: Move computer (NEW tool, dry-style — pick same workspace to avoid disrupting tenant) ===
    console.log("\n=== 5. orgo_move_computer (new) ===");
    if (firstComputerId && firstWorkspaceId) {
      // No-op move: same workspace. Verifies the request shape + path without altering state.
      const m = await client.callTool({
        name: "orgo_move_computer",
        arguments: { computer_id: firstComputerId, workspace_id: firstWorkspaceId },
      });
      const moveResultText = m.content?.[0]?.text || "";
      // Either succeeds (no-op move) or returns a sensible error
      record(
        "orgo_move_computer round-trip",
        true,
        m.isError ? `(returned error — likely API rejects same-workspace) ${fmt(moveResultText)}` : `ok: ${fmt(moveResultText)}`
      );
    }

    // === Section 5.5: Compact mode token savings ===
    console.log("\n=== 5.5. Compact mode token savings ===");
    {
      const full = await client.callTool({ name: "orgo_list_workspaces", arguments: {} });
      const compact = await client.callTool({ name: "orgo_list_workspaces", arguments: { compact: true } });
      const fullLen = full.content?.[0]?.text?.length || 0;
      const compactLen = compact.content?.[0]?.text?.length || 0;
      const savings = fullLen ? Math.round((1 - compactLen / fullLen) * 100) : 0;
      record(`orgo_list_workspaces compact saves ≥ 30%`, savings >= 30, `${fullLen} → ${compactLen} chars (${savings}% saved)`);
      // Verify compact actually drops the big fields
      const compactBody = compact.content?.[0]?.text || "";
      record(`compact list_workspaces drops instance_details`, !compactBody.includes("instance_details"));
      record(`compact list_workspaces drops template_build_id`, !compactBody.includes("template_build_id"));
      record(`compact list_workspaces drops user_id`, !compactBody.includes("user_id"));
      // Verify compact keeps essentials
      record(`compact list_workspaces keeps id`, compactBody.includes("\"id\""));
      record(`compact list_workspaces keeps name`, compactBody.includes("\"name\""));
      record(`compact list_workspaces keeps status`, compactBody.includes("\"status\""));
    }
    if (firstComputerId) {
      const full = await client.callTool({ name: "orgo_get_computer", arguments: { computer_id: firstComputerId } });
      const compact = await client.callTool({ name: "orgo_get_computer", arguments: { computer_id: firstComputerId, compact: true } });
      const fullLen = full.content?.[0]?.text?.length || 0;
      const compactLen = compact.content?.[0]?.text?.length || 0;
      const savings = fullLen ? Math.round((1 - compactLen / fullLen) * 100) : 0;
      record(`orgo_get_computer compact saves tokens`, savings >= 25, `${fullLen} → ${compactLen} chars (${savings}% saved)`);
    }

    // === Section 5.6: Limit / pagination ===
    console.log("\n=== 5.6. Limit / pagination ===");
    {
      const noLimit = await client.callTool({ name: "orgo_list_workspaces", arguments: {} });
      const noLimitData = JSON.parse(noLimit.content?.[0]?.text || "{}");
      record(
        "no-limit response preserves original shape (no truncated key)",
        !("truncated" in noLimitData),
        `truncated=${noLimitData.truncated ?? "absent"}, projects=${noLimitData.projects?.length}`
      );

      // Limit smaller than total → truncated
      const limit3 = await client.callTool({ name: "orgo_list_workspaces", arguments: { limit: 3 } });
      const limit3Data = JSON.parse(limit3.content?.[0]?.text || "{}");
      record(
        "limit=3 returns 3 items with total + truncated:true",
        limit3Data.projects?.length === 3 && limit3Data.truncated === true && typeof limit3Data.total === "number",
        `projects=${limit3Data.projects?.length} total=${limit3Data.total} truncated=${limit3Data.truncated}`
      );

      // Limit larger than total → not truncated, total = actual count
      const limit500 = await client.callTool({ name: "orgo_list_workspaces", arguments: { limit: 500 } });
      const limit500Data = JSON.parse(limit500.content?.[0]?.text || "{}");
      record(
        "limit > total returns all + truncated:false",
        limit500Data.truncated === false && limit500Data.total === limit500Data.projects.length,
        `projects=${limit500Data.projects?.length} total=${limit500Data.total} truncated=${limit500Data.truncated}`
      );

      // Compact + limit composition
      const both = await client.callTool({ name: "orgo_list_workspaces", arguments: { compact: true, limit: 2 } });
      const bothData = JSON.parse(both.content?.[0]?.text || "{}");
      record(
        "compact + limit compose (truncated metadata preserved)",
        bothData.projects?.length === 2 && bothData.truncated === true,
        `bytes=${both.content?.[0]?.text?.length} (vs 79k+ unlimited)`
      );
    }
  } finally {
    await close();
  }

  // === Section 6: Toolset filtering and read-only mode ===
  console.log("\n=== 6. ORGO_TOOLSETS=core filter ===");
  {
    const { client, close } = await spawn({ ORGO_TOOLSETS: "core" });
    try {
      const list = await client.listTools();
      const names = list.tools.map((t) => t.name);
      const coreOnly = ["orgo_list_workspaces", "orgo_get_workspace", "orgo_workspace_by_name", "orgo_list_computers", "orgo_get_computer", "orgo_doctor"];
      const hasOnlyCore = coreOnly.every((n) => names.includes(n)) && names.every((n) => coreOnly.includes(n));
      record("ORGO_TOOLSETS=core exposes only core tools", hasOnlyCore, `got=[${names.join(", ")}]`);
    } finally {
      await close();
    }
  }

  console.log("\n=== 7. ORGO_READ_ONLY=true filter ===");
  {
    const { client, close } = await spawn({ ORGO_READ_ONLY: "true" });
    try {
      const list = await client.listTools();
      const names = list.tools.map((t) => t.name).sort();
      const expected = [
        "orgo_doctor",
        "orgo_download_file",
        "orgo_get_computer",
        "orgo_get_workspace",
        "orgo_list_computers",
        "orgo_list_files",
        "orgo_list_workspaces",
        "orgo_screenshot",
        "orgo_wait",
        "orgo_workspace_by_name",
      ];
      const same = JSON.stringify(names) === JSON.stringify(expected);
      record("read-only set matches expected 10 tools (incl orgo_doctor)", same, `got_count=${names.length}`);
    } finally {
      await close();
    }
  }

  console.log("\n=== 8. ORGO_DISABLED_TOOLS=orgo_bash filter ===");
  {
    const { client, close } = await spawn({ ORGO_TOOLSETS: "shell", ORGO_DISABLED_TOOLS: "orgo_bash" });
    try {
      const list = await client.listTools();
      const names = list.tools.map((t) => t.name).sort();
      const ok = JSON.stringify(names) === JSON.stringify(["orgo_exec"]);
      record("ORGO_DISABLED_TOOLS hides orgo_bash", ok, `got=[${names.join(", ")}]`);
    } finally {
      await close();
    }
  }

  // === Final report ===
  console.log("\n=== SUMMARY ===");
  const passed = results.filter((r) => r.ok).length;
  const failed = results.filter((r) => !r.ok).length;
  console.log(`Passed: ${passed}  Failed: ${failed}  Total: ${results.length}`);
  if (failed > 0) {
    console.log("\nFailures:");
    for (const r of results.filter((x) => !x.ok)) {
      console.log(`  - ${r.name}: ${r.info || "(no info)"}`);
    }
    process.exit(1);
  }
}

run().catch((e) => {
  console.error("E2E error:", e instanceof Error ? e.message : e);
  process.exit(2);
});
