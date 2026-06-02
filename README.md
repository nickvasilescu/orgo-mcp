# Orgo MCP Server

Official MCP server for controlling [Orgo](https://orgo.ai) cloud computers from Claude Code, Claude Desktop, and other Model Context Protocol clients.

This package is TypeScript-first. The current 4.x line is installable directly from GitHub today; the npm publish to `@orgo-ai/mcp` is pending — only 3.0.0 is on npm right now, so `npx @orgo-ai/mcp` will fetch the older release until 4.0.0 ships there.

## Quick Start

### 1. Get an Orgo API key

Sign up or log in at [orgo.ai](https://orgo.ai), then copy an API key from **Settings > API Keys**.

### 2. Add the MCP server

The recommended install path right now uses the GitHub URL so you get the current 4.x build (compact mode, limit, `orgo_doctor`, etc.). Pin to a commit SHA for production stability; omit the `#sha` to track the `main` branch.

Claude Code:

```bash
# Latest from main (auto-builds on install)
claude mcp add orgo -e ORGO_API_KEY=sk_live_YOUR_KEY -- npx -y github:nickvasilescu/orgo-mcp

# Or pin to a specific commit
claude mcp add orgo -e ORGO_API_KEY=sk_live_YOUR_KEY -- npx -y github:nickvasilescu/orgo-mcp#489170e
```

Claude Desktop:

```json
{
  "mcpServers": {
    "orgo": {
      "command": "npx",
      "args": ["-y", "github:nickvasilescu/orgo-mcp"],
      "env": {
        "ORGO_API_KEY": "sk_live_YOUR_KEY"
      }
    }
  }
}
```

Hosted Streamable HTTP server:

```json
{
  "mcpServers": {
    "orgo": {
      "type": "http",
      "url": "https://orgo-mcp.onrender.com/mcp",
      "headers": {
        "X-Orgo-API-Key": "${ORGO_API_KEY}",
        "X-Orgo-Default-Computer-Id": "your-computer-id"
      }
    }
  }
}
```

`X-Orgo-Default-Computer-Id` is optional: it pins a default computer for this
connection so tool calls can omit `computer_id` (the remote equivalent of the
`ORGO_DEFAULT_COMPUTER_ID` env var, which can't be per-user on a shared host).
Clients that only let you configure a URL can pass it as a query param instead:
`https://orgo-mcp.onrender.com/mcp?computer_id=your-computer-id`.

PowerShell users can wrap the Claude Code command because PowerShell may mangle `--` when it reaches npm shims:

```powershell
cmd /c "claude mcp add orgo -e ORGO_API_KEY=sk_live_YOUR_KEY -- npx -y github:nickvasilescu/orgo-mcp"
```

> **About the GitHub install.** `npx` clones the repo and runs the `prepare` script to build `dist/` on first use, so the install takes a few seconds longer than a registry install. Subsequent invocations are cached. Once 4.x is published to `@orgo-ai/mcp`, switch the `args` back to `["-y", "@orgo-ai/mcp"]` for the registry path.

## Tools

The server exposes 28 focused tools.

| Toolset | Tools |
| --- | --- |
| `core` | `orgo_list_workspaces`, `orgo_get_workspace`, `orgo_workspace_by_name`, `orgo_list_computers`, `orgo_get_computer`, `orgo_doctor` |
| `admin` | `orgo_create_workspace`, `orgo_delete_workspace`, `orgo_create_computer`, `orgo_delete_computer`, `orgo_restart_computer`, `orgo_clone_computer`, `orgo_ensure_running`, `orgo_resize_computer`, `orgo_move_computer` |
| `screen` | `orgo_screenshot`, `orgo_click`, `orgo_type`, `orgo_key`, `orgo_scroll`, `orgo_drag`, `orgo_wait` |
| `shell` | `orgo_bash`, `orgo_exec` |
| `files` | `orgo_list_files`, `orgo_export_file`, `orgo_upload_file`, `orgo_download_file` |

Each tool is registered with MCP annotations for `readOnlyHint`, `destructiveHint`, `idempotentHint`, and `openWorldHint`.
`orgo_restart_computer` is marked destructive because it can interrupt running processes and unsaved VM state.
Text responses are sanitized before they are returned to the MCP client, so password, token, secret, credential, and API-key fields are redacted even when Orgo API responses include them.

### Compact mode for read tools

Pass `compact: true` to any read tool (`orgo_list_workspaces`, `orgo_get_workspace`, `orgo_workspace_by_name`, `orgo_list_computers`, `orgo_get_computer`, `orgo_list_files`) to drop noisy fields like `instance_details`, `template_build_id`, and `user_id` from the response. Keeps `id`, `name`, `status`, timestamps, and key resource fields (`cpu`, `ram`, `os`, `disk_size_gb`, `fly_instance_id`, file `size`/`path`). Typical savings: 50%+ on list endpoints. Recommended for agent contexts where response size affects available tokens.

### Health check (`orgo_doctor`)

Call `orgo_doctor` (no arguments) to probe MCP server health: auth source detected (`env:ORGO_API_KEY` or `http_header`, never the key value), API reachability with status code, and round-trip latency in ms. Returns `{ ok, auth, api }` — `ok: true` means both auth is configured and the live Orgo API responded. Use as the first call from an agent harness to verify setup, or after a tool error to distinguish auth failures (`status_code: 401`) from network issues (no status_code, `error: "Cannot reach Orgo API"`).

### Client-side limit for list tools

Pass `limit: N` (1–500) to `orgo_list_workspaces`, `orgo_list_computers`, or `orgo_list_files` to cap the number of items returned. The Orgo API does not paginate server-side, so the MCP fetches everything and trims client-side. When truncation occurs, the response gains `total` (the real count) and `truncated: true`. Omit `limit` to return everything (current default — backward compatible). Combine with `compact: true` for the smallest payload: `{ limit: 5, compact: true }` typically returns under 3KB even on accounts with hundreds of computers.

Deliberately not exposed:

- computer start/stop tools
- VNC password access
- account/profile/credits/transactions
- autonomous agent/thread tools
- RTMP streaming tools
- template management tools

## Production Safety Controls

Use environment variables to restrict the exposed surface without changing client config.

| Variable | Default | Description |
| --- | --- | --- |
| `ORGO_READ_ONLY` | `false` | When `true`, only tools annotated as read-only are registered. |
| `ORGO_TOOLSETS` | `core,admin,screen,shell,files` | Comma-separated toolsets to expose. Example: `core,screen,files`. |
| `ORGO_ENABLED_TOOLS` | all registered tools | Exact comma-separated allowlist. Applied before `ORGO_DISABLED_TOOLS`. |
| `ORGO_DISABLED_TOOLS` | none | Exact comma-separated denylist. |

Examples:

```bash
# Observation-only mode
ORGO_READ_ONLY=true npx -y @orgo-ai/mcp

# Browserless VM control without shell access
ORGO_TOOLSETS=core,screen,files npx -y @orgo-ai/mcp

# Keep shell enabled, but remove bash
ORGO_TOOLSETS=shell ORGO_DISABLED_TOOLS=orgo_bash npx -y @orgo-ai/mcp
```

Read-only mode currently exposes:

```text
orgo_list_workspaces
orgo_get_workspace
orgo_workspace_by_name
orgo_list_computers
orgo_get_computer
orgo_screenshot
orgo_list_files
orgo_download_file
```

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `ORGO_API_KEY` | none | Required for stdio transport. HTTP deployments receive user keys via `X-Orgo-API-Key`. |
| `ORGO_DEFAULT_COMPUTER_ID` | none | Default computer ID so tool calls can omit `computer_id` (stdio). For HTTP, send a per-request `X-Orgo-Default-Computer-Id` header (or `?computer_id=`) instead. |
| `MCP_TRANSPORT` | `stdio` | `stdio`, `http`, or `streamable-http`. |
| `MCP_HOST` | `0.0.0.0` | HTTP bind address. |
| `MCP_PORT` / `PORT` | `8000` | HTTP port. |

## Self-Hosting

Local stdio:

```bash
git clone https://github.com/nickvasilescu/orgo-mcp.git
cd orgo-mcp
npm install
ORGO_API_KEY=sk_live_YOUR_KEY npm start
```

Local HTTP:

```bash
npm install
npm run build
MCP_TRANSPORT=http npm start
curl http://localhost:8000/health
```

Docker:

```bash
docker build -t orgo-mcp .
docker run -p 8000:8000 -e MCP_TRANSPORT=http orgo-mcp
```

Render:

1. Fork this repo.
2. Create a Render Blueprint from `render.yaml`.
3. Connect clients to `https://YOUR_RENDER_HOST/mcp` with an `X-Orgo-API-Key` header.

The hosted server does not store a shared Orgo key. Each request must include the user's Orgo API key header.

## Architecture

```text
npx / stdio:
  MCP client -> stdio -> @orgo-ai/mcp -> Orgo API
                           ORGO_API_KEY env var

Hosted HTTP:
  MCP client -> HTTPS -> orgo-mcp server -> Orgo API
                          X-Orgo-API-Key header

Shell commands:
  orgo_bash -> Terminal WebSocket primary -> VM
            -> REST /bash fallback        -> VM
```

## Development

```bash
npm install
npm run build
npm test
npm start
```

CI runs:

- TypeScript build from a clean `dist/`
- MCP tool-list smoke tests for default, read-only, toolset, allowlist, and denylist policies
- HTTP transport health and auth smoke tests
- npm package content checks
- Docker image build

Before publishing:

```bash
npm test
npm pack --dry-run
```

## Troubleshooting

| Error | Fix |
| --- | --- |
| `Invalid API key` | Check the Orgo API key and whether it is active. |
| `X-Orgo-API-Key header required` | HTTP clients must send this header on `/mcp` requests. |
| `computer_id required` | Pass `computer_id`, set `ORGO_DEFAULT_COMPUTER_ID` (stdio), or send `X-Orgo-Default-Computer-Id` / `?computer_id=` (HTTP). |
| `Connection refused` | Confirm the server is running and check `/health`. |
| Tools not appearing | Check `ORGO_READ_ONLY`, `ORGO_TOOLSETS`, `ORGO_ENABLED_TOOLS`, and `ORGO_DISABLED_TOOLS`. |

## Security

- Do not commit `.env` files or API keys.
- Prefer `ORGO_READ_ONLY=true` for observation-only clients.
- Disable the `shell` toolset for clients that should not execute VM commands.
- Treat `orgo_bash` and `orgo_exec` as high-power tools: they can modify anything reachable from the target VM.
- HTTP deployments should pass the user's key per request via `X-Orgo-API-Key`; do not bake a production Orgo key into the server.

## License

MIT. See [LICENSE](LICENSE).
