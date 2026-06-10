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
/**
 * A persistent WebSocket terminal connection to an Orgo computer.
 */
declare class TerminalConnection {
    private computerId;
    private apiKey;
    private ws;
    private pingInterval;
    private connected;
    private connecting;
    private queue;
    constructor(computerId: string, apiKey: string);
    /**
     * Open the WebSocket connection to the VM's terminal endpoint.
     * Concurrent callers share one in-flight handshake.
     */
    connect(cols?: number, rows?: number): Promise<void>;
    private doConnect;
    private open;
    /**
     * Execute a command and collect output. Commands on one connection run
     * strictly one at a time; each command's timeout starts when it actually
     * dispatches, not when it queues behind a peer.
     */
    execute(command: string, timeoutMs?: number): Promise<string>;
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
    private executeNow;
    /**
     * Close the connection gracefully.
     */
    disconnect(): void;
    get isConnected(): boolean;
}
/**
 * Get or create a terminal connection for a computer.
 */
declare function getTerminalConnection(computerId: string, apiKey: string): Promise<TerminalConnection>;
/**
 * Execute a bash command via Terminal WSS.
 *
 * Preferred over HTTP bash API for reliability.
 */
declare function executeViaTerminal(computerId: string, apiKey: string, command: string, timeoutMs?: number): Promise<string>;
/**
 * Drop pooled terminal connections for one computer (e.g., after restart),
 * across all tenants.
 */
declare function disposeTerminals(computerId: string): void;
/**
 * Disconnect all terminal connections (cleanup on shutdown).
 */
declare function disconnectAll(): void;
export { TerminalConnection, getTerminalConnection, executeViaTerminal, disposeTerminals, disconnectAll };
