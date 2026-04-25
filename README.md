# Orgo MCP Server

An MCP (Model Context Protocol) server that gives AI agents the ability to control virtual computers through [Orgo](https://orgo.ai). Works with Claude Code, Claude Desktop, and any MCP client.

## Quick Start

### 1. Get Your API Key

Sign up or log in at [orgo.ai](https://orgo.ai) and copy your API key from **Settings > API Keys**. It starts with `sk_live_`.

### 2. Connect to Claude

**Claude Code (recommended):**

```bash
# Option A: Use the hosted server (no install needed)
claude mcp add --transport http orgo https://orgo-mcp.onrender.com/mcp \
  --header "X-Orgo-API-Key: sk_live_YOUR_KEY_HERE"

# Option B: Run locally via npx (stdio transport)
claude mcp add orgo -- npx -y @orgo-ai/mcp
# Then set: ORGO_API_KEY=sk_live_YOUR_KEY_HERE
```

**Claude Desktop:**

```json
{
  "mcpServers": {
    "orgo": {
      "command": "npx",
      "args": ["-y", "@orgo-ai/mcp"],
      "env": {
        "ORGO_API_KEY": "sk_live_YOUR_KEY_HERE"
      }
    }
  }
}
```

**Team Project (`.mcp.json`):**

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

### 3. Start Using

```
"Create a Linux computer with 8GB RAM"
"Take a screenshot of my computer"
"Run 'ls -la' on the computer"
"Type 'hello world' and press Enter"
```

---

## Tools (40 total)

| Category | Tools |
|----------|-------|
| **Workspaces** | `orgo_list_workspaces`, `orgo_create_workspace`, `orgo_get_workspace`, `orgo_workspace_by_name` |
| **Computers** | `orgo_list_computers`, `orgo_create_computer`, `orgo_get_computer`, `orgo_delete_computer`, `orgo_start_computer`, `orgo_stop_computer`, `orgo_restart_computer`, `orgo_clone_computer`, `orgo_ensure_running`, `orgo_resize_computer` |
| **Actions** | `orgo_screenshot`, `orgo_click`, `orgo_type`, `orgo_key`, `orgo_scroll`, `orgo_drag` |
| **Shell** | `orgo_bash` (WebSocket terminal preferred, REST fallback), `orgo_exec` (Python) |
| **Files** | `orgo_list_files`, `orgo_upload_file`, `orgo_export_file`, `orgo_download_file` |
| **AI Agent** | `orgo_completions` (autonomous agent with screen vision) |
| **Threads** | `orgo_list_threads`, `orgo_get_thread`, `orgo_delete_thread` |
| **Streaming** | `orgo_start_stream`, `orgo_stream_status`, `orgo_stop_stream` |
| **Templates** | `orgo_list_templates`, `orgo_starred_templates`, `orgo_star_template` |
| **Access** | `orgo_vnc_password` |
| **Account** | `orgo_get_profile`, `orgo_get_credits`, `orgo_get_transactions` |

---

## Self-Hosting

### Local (stdio)

```bash
git clone https://github.com/nickvasilescu/orgo-mcp.git
cd orgo-mcp
npm install
export ORGO_API_KEY="sk_live_YOUR_KEY_HERE"
npm start
```

### Local (HTTP)

```bash
MCP_TRANSPORT=http npm start
# Server at http://localhost:8000/mcp
curl http://localhost:8000/health
```

### Docker

```bash
docker build -t orgo-mcp .
docker run -p 8000:8000 -e MCP_TRANSPORT=http orgo-mcp
```

### Render.com

1. Fork this repo
2. Go to [Render Dashboard](https://dashboard.render.com/blueprints) > New Blueprint Instance
3. Connect your GitHub repo and deploy

### Fly.io

```bash
fly auth login
fly launch --no-deploy
fly deploy
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ORGO_API_KEY` | -- | API key (required for stdio transport) |
| `ORGO_DEFAULT_COMPUTER_ID` | -- | Default computer ID (skip passing on every call) |
| `MCP_TRANSPORT` | `stdio` | Transport mode: `stdio` or `http` |
| `MCP_HOST` | `0.0.0.0` | HTTP bind address |
| `MCP_PORT` / `PORT` | `8000` | HTTP port |

---

## Architecture

```
npx / stdio:
  Claude  -->  stdio  -->  @orgo-ai/mcp  -->  Orgo API
                           ORGO_API_KEY env var

Cloud / HTTP:
  Claude  -->  HTTPS  -->  orgo-mcp server  -->  Orgo API
                           X-Orgo-API-Key header

Shell commands (enhanced):
  orgo_bash  -->  Terminal WebSocket (preferred)  -->  VM
             \->  REST /bash API (fallback)       -->  VM
```

### Project Structure

```
orgo-mcp/
├── src/
│   ├── index.ts          # Entry point (stdio/http transport selection)
│   ├── server.ts         # McpServer instantiation + tool registration
│   ├── auth.ts           # API key resolution (AsyncLocalStorage + env)
│   ├── client.ts         # HTTP client (proxy + direct VM fallback)
│   ├── terminal.ts       # WebSocket terminal (connection pool, keep-alive)
│   ├── errors.ts         # Unified error handling
│   ├── types.ts          # TypeScript interfaces
│   └── tools/            # 11 tool modules (38 tools total)
├── package.json
├── tsconfig.json
├── Dockerfile
├── render.yaml
├── fly.toml
└── README.md
```

---

## Development

```bash
npm install
npm run dev          # Watch mode (recompile on changes)
npm run build        # One-time build
npm start            # Run built server

# Test HTTP transport
MCP_TRANSPORT=http npm start
curl http://localhost:8000/health
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `Invalid API key` | Ensure key starts with `sk_live_` |
| `X-Orgo-API-Key header required` | Check header (no extra spaces) |
| `computer_id required` | Pass `computer_id` or set `ORGO_DEFAULT_COMPUTER_ID` |
| `Connection refused` | Check server is running; try `curl http://localhost:8000/health` |
| Tools not appearing | Wait 10-30s after connection; check Claude MCP settings |

---

## License

MIT -- see [LICENSE](LICENSE)

## Credits

- [Orgo](https://orgo.ai) -- Virtual computer infrastructure
- [Anthropic](https://anthropic.com) -- MCP protocol
