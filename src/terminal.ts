/**
 * Terminal WebSocket client for Orgo computers.
 *
 * Preferred over the HTTP bash API for shell execution:
 * - Persistent sessions (env vars and cwd preserved across commands)
 * - Real-time streaming output
 * - Exit codes captured via the end sentinel
 *
 * Transport: connects directly to the VM desktop API (host:apiPort/terminal).
 * The previous transport used the computer's `url` field, which points at the
 * noVNC port on old-image VMs — that port 101-accepts anything then speaks
 * RFB, hanging the session with no fallback.
 *
 * Auth diverges by VM image (both live-verified June 2026):
 * - new metal image: accepts `?token=<desktop_api_token>`, rejects Bearer
 * - old image: accepts `Authorization: Bearer <password>`, rejects `?token=`
 * so connect() tries the query form first (the growing fleet), then the
 * header form. Both reject instantly on mismatch, so the retry is cheap.
 */

import WebSocket from "ws";
import { getVmAuthToken, getVmEndpoint, tenantKey } from "./client.js";
import type { TerminalMessage } from "./types.js";

// Comprehensive ANSI escape sequence regex:
// CSI sequences, OSC sequences, single-char escapes, and mode changes
const ANSI_REGEX = /\x1b(?:\[[0-9;?]*[a-zA-Z]|\].*?(?:\x07|\x1b\\)|[()][AB012]|[>=])/g;

function stripAnsi(text: string): string {
  return text.replace(ANSI_REGEX, "");
}

let commandCounter = 0;

/**
 * A persistent WebSocket terminal connection to an Orgo computer.
 */
class TerminalConnection {
  private ws: WebSocket | null = null;
  private pingInterval: ReturnType<typeof setInterval> | null = null;
  private connected = false;

  constructor(
    private computerId: string,
    private apiKey: string
  ) {}

  /**
   * Open the WebSocket connection to the VM's terminal endpoint.
   */
  async connect(cols = 200, rows = 50): Promise<void> {
    if (this.connected && this.ws?.readyState === WebSocket.OPEN) return;

    const endpoint = await getVmEndpoint(this.computerId, this.apiKey);
    if (!endpoint) {
      throw new Error(
        `Computer ${this.computerId} has no direct API endpoint (instance_details missing)`
      );
    }

    const token = await getVmAuthToken(this.computerId, this.apiKey);
    const base = `ws://${endpoint.host}:${endpoint.apiPort}/terminal`;
    const attempts: Array<{ url: string; headers?: Record<string, string> }> = [
      // New metal image: query-token auth
      { url: `${base}?token=${encodeURIComponent(token)}&cols=${cols}&rows=${rows}` },
      // Old image: Bearer-header auth
      { url: `${base}?cols=${cols}&rows=${rows}`, headers: { Authorization: `Bearer ${token}` } },
    ];

    let lastError: Error | null = null;
    for (const attempt of attempts) {
      try {
        await this.open(attempt.url, attempt.headers);
        return;
      } catch (e) {
        lastError = e instanceof Error ? e : new Error(String(e));
      }
    }
    throw lastError ?? new Error(`Terminal connection failed for computer ${this.computerId}`);
  }

  private open(url: string, headers?: Record<string, string>): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      this.ws = new WebSocket(url, headers ? { headers } : undefined);

      const timeout = setTimeout(() => {
        this.ws?.terminate();
        reject(new Error(`Terminal connection timed out for computer ${this.computerId}`));
      }, 15000);

      this.ws.on("open", () => {
        clearTimeout(timeout);
        this.connected = true;

        // Keep-alive ping every 10 seconds
        this.pingInterval = setInterval(() => {
          if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: "ping" }));
          }
        }, 10000);

        resolve();
      });

      this.ws.on("error", (err) => {
        clearTimeout(timeout);
        this.connected = false;
        reject(err);
      });

      this.ws.on("close", () => {
        this.connected = false;
        if (this.pingInterval) {
          clearInterval(this.pingInterval);
          this.pingInterval = null;
        }
      });
    });
  }

  /**
   * Execute a command and collect output.
   *
   * Sentinel markers bracket the command: a BEGIN sentinel excludes stale
   * output from earlier (e.g. timed-out) commands, and the DONE sentinel
   * carries `$?` so callers get the exit code.
   *
   * Timeout semantics:
   * - No output at all -> rejects (the connection is likely dead or pointed at
   *   the wrong service), letting callers fall back to the REST bash API.
   * - Partial output -> resolves with an explicit truncation marker, and the
   *   connection is disposed so the still-running command can't bleed into the
   *   next call's output.
   */
  async execute(command: string, timeoutMs = 30000): Promise<string> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      await this.connect();
    }

    const ws = this.ws!;
    const id = `${Date.now()}_${++commandCounter}`;
    const beginSentinel = `__ORGO_BEGIN_${id}__`;
    const doneSentinel = `__ORGO_DONE_${id}__`;
    // The ; ensures the DONE echo runs regardless of command exit code; $?
    // captures that exit code.
    const wrappedCmd = `echo ${beginSentinel}; ${command}; echo ${doneSentinel}:$?`;

    // Match sentinels on their own line (actual echo output), NOT when they
    // appear inline in the echoed command (where DONE is followed by `:$?`,
    // not digits, and BEGIN is followed by `;`). Terminals separate lines
    // with \r\n or bare \r, so anchor on either.
    const beginLineRegex = new RegExp(`(?:^|[\\r\\n])${beginSentinel}\\s*[\\r\\n]`);
    const doneLineRegex = new RegExp(`[\\r\\n]${doneSentinel}:(\\d+)\\s*(?:[\\r\\n]|$)`);

    return new Promise<string>((resolve, reject) => {
      let output = "";
      let resolved = false;

      const finish = (fn: () => void) => {
        resolved = true;
        clearTimeout(timer);
        ws.removeListener("message", onMessage);
        fn();
      };

      const timer = setTimeout(() => {
        if (resolved) return;
        const cleaned = stripAnsi(output);
        const beginMatch = beginLineRegex.exec(cleaned);
        // A command may still be running on the session — dispose the
        // connection so its late output can't pollute the next command.
        finish(() => {
          this.disconnect();
          if (!beginMatch) {
            // The BEGIN echo never arrived: the terminal isn't functioning
            // (dead connection / wrong service) — reject so callers can fall
            // back to the REST bash API.
            reject(
              new Error(`Terminal produced no output within ${Math.round(timeoutMs / 1000)}s`)
            );
          } else {
            // The terminal works; the command itself exceeded the budget.
            const partial = cleaned.substring(beginMatch.index + beginMatch[0].length).trim();
            resolve(
              `${partial || "(no output yet)"}\n[orgo_bash: timed out after ${Math.round(timeoutMs / 1000)}s — command may still be running on the VM; output above is partial]`
            );
          }
        });
      }, timeoutMs);

      const onMessage = (data: WebSocket.Data) => {
        if (resolved) return;

        try {
          const msg = JSON.parse(data.toString()) as TerminalMessage;
          if (msg.type === "output" && msg.data) {
            output += msg.data;

            const cleaned = stripAnsi(output);
            const doneMatch = doneLineRegex.exec(cleaned);
            if (doneMatch) {
              const exitCode = doneMatch[1];
              const beginMatch = beginLineRegex.exec(cleaned);
              // Content runs from after the BEGIN line to the DONE line.
              const contentStart = beginMatch
                ? beginMatch.index + beginMatch[0].length
                : 0;
              const clean = cleaned.substring(contentStart, doneMatch.index).trim();
              finish(() =>
                resolve(exitCode === "0" ? clean : `${clean}\n[exit code: ${exitCode}]`.trim())
              );
            }
          } else if (msg.type === "error") {
            finish(() => reject(new Error(`Terminal error: ${msg.message || "unknown"}`)));
          } else if (msg.type === "exit") {
            finish(() => resolve(stripAnsi(output).trim()));
          }
        } catch {
          // Ignore non-JSON messages
        }
      };

      ws.on("message", onMessage);

      // Send the wrapped command
      ws.send(JSON.stringify({ type: "input", data: wrappedCmd + "\r" }));
    });
  }

  /**
   * Close the connection gracefully.
   */
  disconnect(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
    if (this.ws) {
      this.ws.removeAllListeners();
      if (this.ws.readyState === WebSocket.OPEN) {
        this.ws.close();
      }
      this.ws = null;
    }
    this.connected = false;
  }

  get isConnected(): boolean {
    return this.connected && this.ws?.readyState === WebSocket.OPEN;
  }
}

// Connection pool. Keyed by (api key, computer) — the HTTP transport hosts
// many tenants in one process, and a computer-only key would hand tenant A's
// authenticated terminal to tenant B.
const pool = new Map<string, TerminalConnection>();

/**
 * Get or create a terminal connection for a computer.
 */
async function getTerminalConnection(
  computerId: string,
  apiKey: string
): Promise<TerminalConnection> {
  const poolKey = tenantKey(apiKey, computerId);
  let conn = pool.get(poolKey);
  if (conn?.isConnected) return conn;

  // Clean up stale connection
  if (conn) {
    conn.disconnect();
    pool.delete(poolKey);
  }

  conn = new TerminalConnection(computerId, apiKey);
  await conn.connect();
  pool.set(poolKey, conn);
  return conn;
}

/**
 * Execute a bash command via Terminal WSS.
 *
 * Preferred over HTTP bash API for reliability.
 */
async function executeViaTerminal(
  computerId: string,
  apiKey: string,
  command: string,
  timeoutMs = 30000
): Promise<string> {
  const conn = await getTerminalConnection(computerId, apiKey);
  return conn.execute(command, timeoutMs);
}

/**
 * Drop pooled terminal connections for one computer (e.g., after restart),
 * across all tenants.
 */
function disposeTerminals(computerId: string): void {
  for (const [key, conn] of pool.entries()) {
    if (key.endsWith(`:${computerId}`)) {
      conn.disconnect();
      pool.delete(key);
    }
  }
}

/**
 * Disconnect all terminal connections (cleanup on shutdown).
 */
function disconnectAll(): void {
  for (const conn of pool.values()) {
    conn.disconnect();
  }
  pool.clear();
}

export { TerminalConnection, getTerminalConnection, executeViaTerminal, disposeTerminals, disconnectAll };
