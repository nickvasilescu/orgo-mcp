# Orgo MCP Server

Official MCP server for controlling [Orgo](https://orgo.ai) cloud computers from Claude Code, Claude Desktop, and other Model Context Protocol clients.

This package is TypeScript-first and ships as `@orgo-ai/mcp`.

## Quick Start

### 1. Get an Orgo API key

Sign up or log in at [orgo.ai](https://orgo.ai), then copy an API key from **Settings > API Keys**.

### 2. Add the MCP server

Claude Code:

```bash
claude mcp add orgo -e ORGO_API_KEY=sk_live_YOUR_KEY -- npx -y @orgo-ai/mcp
```

Claude Desktop:

```json
{
  "mcpServers": {
    "orgo": {
      "command": "npx",
      "args": ["-y", "@orgo-ai/mcp"],
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
        "X-Orgo-API-Key": "${ORGO_API_KEY}"
      }
    }
  }
}
```

PowerShell users can wrap the Claude Code command because PowerShell may mangle `--` when it reaches npm shims:

```powershell
cmd /c "claude mcp add orgo -e ORGO_API_KEY=sk_live_YOUR_KEY -- npx -y @orgo-ai/mcp"
```

## Tools

The server exposes 24 focused tools.

| Toolset | Tools |
| --- | --- |
| `core` | `orgo_list_workspaces`, `orgo_get_workspace`, `orgo_workspace_by_name`, `orgo_list_computers`, `orgo_get_computer` |
| `admin` | `orgo_create_workspace`, `orgo_create_computer`, `orgo_delete_computer`, `orgo_restart_computer`, `orgo_clone_computer`, `orgo_ensure_running`, `orgo_resize_computer` |
| `screen` | `orgo_screenshot`, `orgo_click`, `orgo_type`, `orgo_key`, `orgo_scroll`, `orgo_drag` |
| `shell` | `orgo_bash`, `orgo_exec` |
| `files` | `orgo_list_files`, `orgo_export_file`, `orgo_upload_file`, `orgo_download_file` |

Each tool is registered with MCP annotations for `readOnlyHint`, `destructiveHint`, `idempotentHint`, and `openWorldHint`.

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
| `ORGO_DEFAULT_COMPUTER_ID` | none | Default computer ID so tool calls can omit `computer_id`. |
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
| `computer_id required` | Pass `computer_id` or set `ORGO_DEFAULT_COMPUTER_ID`. |
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
