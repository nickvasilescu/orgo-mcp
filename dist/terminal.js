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
import { clearVncCache, getVmAuthToken, getVmEndpoint, tenantKey } from "./client.js";
// Comprehensive ANSI escape sequence regex:
// CSI sequences, OSC sequences, single-char escapes, and mode changes
const ANSI_REGEX = /\x1b(?:\[[0-9;?]*[a-zA-Z]|\].*?(?:\x07|\x1b\\)|[()][AB012]|[>=])/g;
function stripAnsi(text) {
    return text.replace(ANSI_REGEX, "");
}
/** Last regex match in `text`, optionally only before `before`. */
function lastMatch(regex, text, before = Infinity) {
    const global = new RegExp(regex.source, "g");
    let last = null;
    let m;
    while ((m = global.exec(text)) !== null) {
        if (m.index >= before)
            break;
        last = m;
        // Zero-width safety
        if (m.index === global.lastIndex)
            global.lastIndex++;
    }
    return last;
}
let commandCounter = 0;
/**
 * A persistent WebSocket terminal connection to an Orgo computer.
 */
class TerminalConnection {
    computerId;
    apiKey;
    ws = null;
    pingInterval = null;
    connected = false;
    connecting = null;
    // Commands are strictly serialized per connection: the terminal is one
    // shared bash session, and interleaved sends would mix sentinel scopes,
    // consume each other's stdin, and double-run commands via the REST
    // fallback after a peer's timeout disposal.
    queue = Promise.resolve();
    constructor(computerId, apiKey) {
        this.computerId = computerId;
        this.apiKey = apiKey;
    }
    /**
     * Open the WebSocket connection to the VM's terminal endpoint.
     * Concurrent callers share one in-flight handshake.
     */
    connect(cols = 200, rows = 50) {
        if (this.connected && this.ws?.readyState === WebSocket.OPEN)
            return Promise.resolve();
        if (this.connecting)
            return this.connecting;
        this.connecting = this.doConnect(cols, rows).finally(() => {
            this.connecting = null;
        });
        return this.connecting;
    }
    async doConnect(cols, rows) {
        const endpoint = await getVmEndpoint(this.computerId, this.apiKey);
        if (!endpoint) {
            throw new Error(`Computer ${this.computerId} has no direct API endpoint (instance_details missing)`);
        }
        const token = await getVmAuthToken(this.computerId, this.apiKey);
        const base = `ws://${endpoint.host}:${endpoint.apiPort}/terminal`;
        const attempts = [
            // New metal image: query-token auth
            { url: `${base}?token=${encodeURIComponent(token)}&cols=${cols}&rows=${rows}` },
            // Old image: Bearer-header auth
            { url: `${base}?cols=${cols}&rows=${rows}`, headers: { Authorization: `Bearer ${token}` } },
        ];
        let lastError = null;
        for (const attempt of attempts) {
            try {
                await this.open(attempt.url, attempt.headers);
                return;
            }
            catch (e) {
                lastError = e instanceof Error ? e : new Error(String(e));
            }
        }
        // Both auth forms failed — the cached token may be stale (the VM restarts
        // and rotates tokens outside this process too). Evict so the next attempt
        // refetches fresh credentials instead of failing forever.
        clearVncCache(this.computerId);
        throw lastError ?? new Error(`Terminal connection failed for computer ${this.computerId}`);
    }
    open(url, headers) {
        return new Promise((resolve, reject) => {
            // A previous open attempt may have left a keep-alive behind.
            if (this.pingInterval) {
                clearInterval(this.pingInterval);
                this.pingInterval = null;
            }
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
     * Execute a command and collect output. Commands on one connection run
     * strictly one at a time; each command's timeout starts when it actually
     * dispatches, not when it queues behind a peer.
     */
    execute(command, timeoutMs = 30000) {
        const run = this.queue.then(() => this.executeNow(command, timeoutMs));
        // The chain must survive rejections, or one failure would poison every
        // later command on this connection.
        this.queue = run.catch(() => { });
        return run;
    }
    /**
     * The actual single-flight command execution.
     *
     * The command is shipped base64-encoded inside an `eval "$(… | base64 -d)"`
     * wrapper, so heredocs, comments, trailing `&`, quotes, and newlines in the
     * user's command can never interact with the sentinel syntax. Both sentinels
     * are emitted via `printf '\n…'` so they always start on a fresh line, even
     * when the command's output lacks a trailing newline or the prompt precedes
     * BEGIN without one.
     *
     * Timeout semantics:
     * - BEGIN never seen -> rejects (the terminal isn't functioning), letting
     *   callers fall back to the REST bash API — the command never dispatched,
     *   so the fallback cannot double-run it.
     * - BEGIN seen -> the terminal works and the command itself exceeded the
     *   budget: resolve with an explicit truncation marker and dispose the
     *   connection so the still-running command can't bleed into later calls.
     */
    async executeNow(command, timeoutMs) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            await this.connect();
        }
        const ws = this.ws;
        const id = `${Date.now()}_${++commandCounter}`;
        const beginSentinel = `__ORGO_BEGIN_${id}__`;
        const doneSentinel = `__ORGO_DONE_${id}__`;
        const encoded = Buffer.from(command, "utf8").toString("base64");
        const wrappedCmd = `printf '\\n%s\\n' ${beginSentinel}; ` +
            `eval "$(printf '%s' '${encoded}' | base64 -d)"; ` +
            `printf '\\n%s:%s\\n' ${doneSentinel} $?`;
        // Match sentinels on their own line (the printf output), NOT where they
        // appear inline in the echoed command (there BEGIN is followed by `;` and
        // DONE by ` $?`, never by `:digits`). Terminals separate lines with \r\n
        // or bare \r, so anchor on either. The real BEGIN is the LAST one before
        // DONE — the echoed command line (which can wrap at the terminal width)
        // always precedes it.
        const beginLineRegex = new RegExp(`(?:^|[\\r\\n])${beginSentinel}\\s*[\\r\\n]`);
        const doneLineRegex = new RegExp(`[\\r\\n]${doneSentinel}:(\\d+)\\s*(?:[\\r\\n]|$)`);
        return new Promise((resolve, reject) => {
            let output = "";
            let resolved = false;
            const finish = (fn) => {
                resolved = true;
                clearTimeout(timer);
                ws.removeListener("message", onMessage);
                fn();
            };
            const timer = setTimeout(() => {
                if (resolved)
                    return;
                const cleaned = stripAnsi(output);
                const beginMatch = lastMatch(beginLineRegex, cleaned);
                // A command may still be running on the session — dispose the
                // connection so its late output can't pollute the next command (the
                // next call in the queue reconnects fresh).
                finish(() => {
                    this.disconnect();
                    if (!beginMatch) {
                        // The BEGIN marker never arrived: the terminal isn't functioning
                        // (dead connection / wrong service) — reject so callers can fall
                        // back to the REST bash API.
                        reject(new Error(`Terminal produced no output within ${Math.round(timeoutMs / 1000)}s`));
                    }
                    else {
                        // The terminal works; the command itself exceeded the budget.
                        const partial = cleaned.substring(beginMatch.index + beginMatch[0].length).trim();
                        resolve(`${partial || "(no output yet)"}\n[orgo_bash: timed out after ${Math.round(timeoutMs / 1000)}s — command may still be running on the VM; output above is partial]`);
                    }
                });
            }, timeoutMs);
            const onMessage = (data) => {
                if (resolved)
                    return;
                try {
                    const msg = JSON.parse(data.toString());
                    if (msg.type === "output" && msg.data) {
                        output += msg.data;
                        const cleaned = stripAnsi(output);
                        const doneMatch = doneLineRegex.exec(cleaned);
                        if (doneMatch) {
                            const exitCode = doneMatch[1];
                            const beginMatch = lastMatch(beginLineRegex, cleaned, doneMatch.index);
                            // Content runs from after the BEGIN line to the DONE line.
                            const contentStart = beginMatch
                                ? beginMatch.index + beginMatch[0].length
                                : 0;
                            const clean = cleaned.substring(contentStart, doneMatch.index).trim();
                            finish(() => resolve(exitCode === "0" ? clean : `${clean}\n[exit code: ${exitCode}]`.trim()));
                        }
                    }
                    else if (msg.type === "error") {
                        finish(() => reject(new Error(`Terminal error: ${msg.message || "unknown"}`)));
                    }
                    else if (msg.type === "exit") {
                        finish(() => resolve(stripAnsi(output).trim()));
                    }
                }
                catch {
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
    disconnect() {
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
    get isConnected() {
        return this.connected && this.ws?.readyState === WebSocket.OPEN;
    }
}
// Connection pool. Keyed by (api key, computer) — the HTTP transport hosts
// many tenants in one process, and a computer-only key would hand tenant A's
// authenticated terminal to tenant B. The pool stores the connect PROMISE,
// set synchronously before the first await, so concurrent callers share one
// connection instead of minting and leaking N−1 of them.
const pool = new Map();
/**
 * Get or create a terminal connection for a computer.
 */
async function getTerminalConnection(computerId, apiKey) {
    const poolKey = tenantKey(apiKey, computerId);
    const existing = pool.get(poolKey);
    if (existing) {
        const conn = await existing.catch(() => null);
        if (conn?.isConnected)
            return conn;
        // Stale or failed — clean up and re-mint (only if no one re-minted while
        // we awaited).
        if (pool.get(poolKey) === existing) {
            conn?.disconnect();
            pool.delete(poolKey);
        }
        else {
            return getTerminalConnection(computerId, apiKey);
        }
    }
    const created = (async () => {
        const conn = new TerminalConnection(computerId, apiKey);
        await conn.connect();
        return conn;
    })();
    pool.set(poolKey, created);
    try {
        return await created;
    }
    catch (e) {
        if (pool.get(poolKey) === created)
            pool.delete(poolKey);
        throw e;
    }
}
/**
 * Execute a bash command via Terminal WSS.
 *
 * Preferred over HTTP bash API for reliability.
 */
async function executeViaTerminal(computerId, apiKey, command, timeoutMs = 30000) {
    const conn = await getTerminalConnection(computerId, apiKey);
    return conn.execute(command, timeoutMs);
}
/**
 * Drop pooled terminal connections for one computer (e.g., after restart),
 * across all tenants.
 */
function disposeTerminals(computerId) {
    for (const [key, promise] of pool.entries()) {
        if (key.endsWith(`:${computerId}`)) {
            pool.delete(key);
            promise.then((conn) => conn.disconnect()).catch(() => { });
        }
    }
}
/**
 * Disconnect all terminal connections (cleanup on shutdown).
 */
function disconnectAll() {
    for (const promise of pool.values()) {
        promise.then((conn) => conn.disconnect()).catch(() => { });
    }
    pool.clear();
}
export { TerminalConnection, getTerminalConnection, executeViaTerminal, disposeTerminals, disconnectAll };
//# sourceMappingURL=terminal.js.map