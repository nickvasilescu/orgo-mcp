#!/usr/bin/env node
import { spawn } from "node:child_process";
import { setTimeout as delay } from "node:timers/promises";

const port = 18000 + Math.floor(Math.random() * 1000);
const baseUrl = `http://127.0.0.1:${port}`;

const child = spawn(process.execPath, ["dist/index.js"], {
  cwd: process.cwd(),
  env: {
    ...process.env,
    MCP_TRANSPORT: "http",
    MCP_HOST: "127.0.0.1",
    PORT: String(port),
  },
  stdio: ["ignore", "pipe", "pipe"],
});

let stderr = "";
child.stderr.on("data", (chunk) => {
  stderr += chunk.toString();
});

async function waitForHealth() {
  const deadline = Date.now() + 10000;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`${baseUrl}/health`);
      if (response.status === 200 && (await response.text()) === "OK") return;
    } catch {
      // Server may still be binding.
    }
    await delay(200);
  }
  throw new Error(`HTTP health check did not pass.\n${stderr}`);
}

async function expectStatus(path, options, expectedStatus, label) {
  const response = await fetch(`${baseUrl}${path}`, options);
  if (response.status !== expectedStatus) {
    const body = await response.text();
    throw new Error(`${label} expected ${expectedStatus}, got ${response.status}: ${body}`);
  }
}

try {
  await waitForHealth();
  await expectStatus("/mcp", { method: "GET" }, 401, "missing API key");
  await expectStatus(
    "/mcp",
    { method: "GET", headers: { "X-Orgo-API-Key": "sk_live_TEST_ONLY" } },
    400,
    "missing MCP session"
  );
  console.log("HTTP smoke checks passed");
} finally {
  child.kill("SIGTERM");
}
