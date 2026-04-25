/**
 * Terminal WebSocket client for Orgo computers.
 *
 * Preferred over the HTTP bash API for shell execution:
 * - Persistent sessions (env vars and cwd preserved across commands)
 * - Routes through stable Fly proxy (no stale port issues)
 * - Real-time streaming output
 *
 * Ported from utilities/terminal_wss.py.
 */

import WebSocket from "ws";
import { apiRequest, getVncPassword } from "./client.js";
import type { ComputerInfo, TerminalMessage } from "./types.js";

// Comprehensive ANSI escape sequence regex:
// CSI sequences, OSC sequences, single-char escapes, and mode changes
const ANSI_REGEX = /\x1b(?:\[[0-9;?]*[a-zA-Z]|\].*?(?:\x07|\x1b\\)|[()][AB012]|[>=])/g;

function stripAnsi(text: string): string {
  return text.replace(ANSI_REGEX, "");
}

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
   * Open the WebSocket connection.
   */
  async connect(cols = 200, rows = 50): Promise<void> {
    if (this.connected && this.ws?.readyState === WebSocket.OPEN) return;

    // Get computer URL
    const info = (await apiRequest(
      "GET",
      `computers/${this.computerId}`,
      this.apiKey,
      { timeout: 20000 }
    )) as unknown as ComputerInfo;

    const baseUrl = info.url;
    if (!baseUrl) {
      throw new Error(`Computer ${this.computerId} has no URL`);
    }

    // Get VNC password
    const password = await getVncPassword(this.computerId, this.apiKey);

    // Convert HTTP URL to WebSocket URL
    const wsBase = baseUrl.replace("http://", "ws://").replace("https://", "wss://");
    const url = `${wsBase}/terminal?token=${password}&cols=${cols}&rows=${rows}`;

    return new Promise<void>((resolve, reject) => {
      this.ws = new WebSocket(url);

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
   * Uses sentinel markers to detect command completion reliably.
   */
  async execute(command: string, timeoutMs = 30000): Promise<string> {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      await this.connect();
    }

    const ws = this.ws!;
    const sentinel = `__ORGO_DONE_${Date.now()}__`;
    // The ; ensures echo runs regardless of command exit code
    const wrappedCmd = `${command}; echo ${sentinel}`;

    // Regex to match sentinel on its own line (the actual echo output),
    // NOT when it appears embedded in the echoed command (e.g. `echo __ORGO_DONE_xxx__`)
    const sentinelLineRegex = new RegExp(`\\n${sentinel}\\s*(?:\\r?\\n|$)`);

    return new Promise<string>((resolve, reject) => {
      let output = "";
      let resolved = false;

      const timer = setTimeout(() => {
        if (!resolved) {
          resolved = true;
          ws.removeListener("message", onMessage);
          // Return whatever we collected so far
          resolve(stripAnsi(output).trim() || "(command timed out — no output captured)");
        }
      }, timeoutMs);

      const onMessage = (data: WebSocket.Data) => {
        if (resolved) return;

        try {
          const msg = JSON.parse(data.toString()) as TerminalMessage;
          if (msg.type === "output" && msg.data) {
            output += msg.data;

            // Check for sentinel on its own line (actual echo output).
            // The echoed command line contains `echo __SENTINEL__` (preceded
            // by a space), while the actual output has `\n__SENTINEL__\n`.
            const cleaned = stripAnsi(output);
            const match = sentinelLineRegex.exec(cleaned);
            if (match) {
              resolved = true;
              clearTimeout(timer);
              ws.removeListener("message", onMessage);

              // Extract output: everything between the echoed command line
              // and the sentinel line. Find the first newline after the
              // echoed command (which contains the sentinel text inline).
              const sentinelPos = match.index;
              // Skip past the echoed command line (first line containing wrappedCmd)
              const cmdEchoEnd = cleaned.indexOf("\n");
              const contentStart = cmdEchoEnd >= 0 && cmdEchoEnd < sentinelPos ? cmdEchoEnd + 1 : 0;
              const clean = cleaned.substring(contentStart, sentinelPos);
              resolve(clean.trim());
            }
          } else if (msg.type === "error") {
            resolved = true;
            clearTimeout(timer);
            ws.removeListener("message", onMessage);
            reject(new Error(`Terminal error: ${msg.message || "unknown"}`));
          } else if (msg.type === "exit") {
            resolved = true;
            clearTimeout(timer);
            ws.removeListener("message", onMessage);
            resolve(stripAnsi(output).trim());
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

// Connection pool: reuse connections per computer
const pool = new Map<string, TerminalConnection>();

/**
 * Get or create a terminal connection for a computer.
 */
async function getTerminalConnection(
  computerId: string,
  apiKey: string
): Promise<TerminalConnection> {
  let conn = pool.get(computerId);
  if (conn?.isConnected) return conn;

  // Clean up stale connection
  if (conn) {
    conn.disconnect();
    pool.delete(computerId);
  }

  conn = new TerminalConnection(computerId, apiKey);
  await conn.connect();
  pool.set(computerId, conn);
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
 * Disconnect all terminal connections (cleanup on shutdown).
 */
function disconnectAll(): void {
  for (const conn of pool.values()) {
    conn.disconnect();
  }
  pool.clear();
}

export { TerminalConnection, getTerminalConnection, executeViaTerminal, disconnectAll };
