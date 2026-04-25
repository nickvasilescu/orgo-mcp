#!/usr/bin/env node
/**
 * Orgo MCP Server -- Entry point with transport selection.
 *
 * Supports two transport modes:
 * - stdio (default): Direct process communication with Claude Code / Claude Desktop
 * - http: REST API with X-Orgo-API-Key header authentication
 *
 * Usage:
 *   ORGO_API_KEY=sk_live_... npx @orgo-ai/mcp
 *   ORGO_API_KEY=sk_live_... MCP_TRANSPORT=http npx @orgo-ai/mcp
 */

import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { createOrgoMcpServer } from "./server.js";
import { disconnectAll } from "./terminal.js";

const TRANSPORT_MODE = (process.env.MCP_TRANSPORT || "stdio").toLowerCase();
const HTTP_HOST = process.env.MCP_HOST || "0.0.0.0";
const HTTP_PORT = parseInt(process.env.MCP_PORT || process.env.PORT || "8000", 10);

async function startStdio(): Promise<void> {
  if (!process.env.ORGO_API_KEY) {
    console.error("Error: ORGO_API_KEY not set. Get your key at https://orgo.ai");
    process.exit(1);
  }

  const server = createOrgoMcpServer();
  const transport = new StdioServerTransport();

  console.error("Orgo MCP server starting (stdio transport)");
  await server.connect(transport);
  console.error("Orgo MCP server running on stdio");
}

async function startHttp(): Promise<void> {
  // Dynamic imports for HTTP-only dependencies
  const { StreamableHTTPServerTransport } = await import(
    "@modelcontextprotocol/sdk/server/streamableHttp.js"
  );
  const { isInitializeRequest } = await import("@modelcontextprotocol/sdk/types.js");
  const { randomUUID } = await import("node:crypto");
  const { runWithApiKey } = await import("./auth.js");
  const http = await import("node:http");

  type TransportInstance = InstanceType<typeof StreamableHTTPServerTransport>;

  // Session store with idle timeout (30 min) to prevent memory leaks
  const SESSION_TIMEOUT_MS = 30 * 60 * 1000;
  const sessions = new Map<string, { transport: TransportInstance; lastAccess: number }>();

  // Periodic cleanup of stale sessions
  setInterval(() => {
    const now = Date.now();
    for (const [sid, entry] of sessions) {
      if (now - entry.lastAccess > SESSION_TIMEOUT_MS) {
        entry.transport.close?.();
        sessions.delete(sid);
      }
    }
  }, 60000).unref();

  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);
    const path = url.pathname;

    // Health check
    if (path === "/health" && req.method === "GET") {
      res.writeHead(200, { "Content-Type": "text/plain" });
      res.end("OK");
      return;
    }

    // Only handle /mcp
    if (path !== "/mcp") {
      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Not found" }));
      return;
    }

    // Extract API key from header
    const apiKey = req.headers["x-orgo-api-key"] as string | undefined;
    if (!apiKey) {
      res.writeHead(401, { "Content-Type": "application/json" });
      res.end(JSON.stringify({
        error: "X-Orgo-API-Key header required. Get your key at https://orgo.ai",
      }));
      return;
    }

    const sessionId = req.headers["mcp-session-id"] as string | undefined;

    // GET = SSE event stream, DELETE = session termination (no body needed)
    if (req.method === "GET" || req.method === "DELETE") {
      if (sessionId && sessions.has(sessionId)) {
        const entry = sessions.get(sessionId)!;
        entry.lastAccess = Date.now();
        await runWithApiKey(apiKey, async () => {
          await entry.transport.handleRequest(req, res);
        });
        return;
      }
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({
        jsonrpc: "2.0",
        error: { code: -32000, message: "Bad Request: No valid session" },
        id: null,
      }));
      return;
    }

    // POST — parse body
    const chunks: Buffer[] = [];
    for await (const chunk of req) {
      chunks.push(typeof chunk === "string" ? Buffer.from(chunk) : chunk);
    }
    const bodyStr = Buffer.concat(chunks).toString("utf-8");
    let body: unknown;
    try {
      body = JSON.parse(bodyStr);
    } catch {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Invalid JSON" }));
      return;
    }

    // Existing session
    if (sessionId && sessions.has(sessionId)) {
      const entry = sessions.get(sessionId)!;
      entry.lastAccess = Date.now();
      await runWithApiKey(apiKey, async () => {
        await entry.transport.handleRequest(req, res, body);
      });
      return;
    }

    // New session (initialize request)
    if (!sessionId && isInitializeRequest(body)) {
      const transport = new StreamableHTTPServerTransport({
        sessionIdGenerator: () => randomUUID(),
        onsessioninitialized: (sid: string) => {
          sessions.set(sid, { transport, lastAccess: Date.now() });
        },
      });

      transport.onclose = () => {
        if (transport.sessionId) {
          sessions.delete(transport.sessionId);
        }
      };

      const mcpServer = createOrgoMcpServer();
      await mcpServer.connect(transport);

      await runWithApiKey(apiKey, async () => {
        await transport.handleRequest(req, res, body);
      });
      return;
    }

    res.writeHead(400, { "Content-Type": "application/json" });
    res.end(JSON.stringify({
      jsonrpc: "2.0",
      error: { code: -32000, message: "Bad Request: No valid session or initialize request" },
      id: null,
    }));
  });

  server.listen(HTTP_PORT, HTTP_HOST, () => {
    console.error(`Orgo MCP server running on http://${HTTP_HOST}:${HTTP_PORT}/mcp`);
  });
}

async function main(): Promise<void> {
  // Cleanup on exit
  process.on("SIGINT", () => {
    disconnectAll();
    process.exit(0);
  });
  process.on("SIGTERM", () => {
    disconnectAll();
    process.exit(0);
  });

  if (TRANSPORT_MODE === "stdio") {
    await startStdio();
  } else if (TRANSPORT_MODE === "http" || TRANSPORT_MODE === "streamable-http") {
    await startHttp();
  } else {
    console.error(`Unknown transport: ${TRANSPORT_MODE}. Use 'stdio' or 'http'`);
    process.exit(1);
  }
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
