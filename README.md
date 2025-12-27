# Orgo MCP Server

An MCP (Model Context Protocol) server that gives AI agents the ability to control virtual computers through [Orgo](https://orgo.ai).

## Quick Start (Cloud-Hosted)

The fastest way to use Orgo MCP - no installation required!

### 1. Get Your API Key

Sign up at [orgo.ai](https://orgo.ai) and copy your API key.

### 2. Configure Your Client

<details>
<summary><strong>Claude Code (CLI)</strong></summary>

```bash
claude mcp add --transport http orgo https://orgo-mcp.onrender.com/mcp \
  --header "X-Orgo-API-Key: YOUR_API_KEY"
```

</details>

<details>
<summary><strong>Claude Desktop (via mcp-remote)</strong></summary>

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "orgo": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "https://orgo-mcp.onrender.com/mcp",
               "--header", "X-Orgo-API-Key:YOUR_API_KEY"]
    }
  }
}
```

**Note**: No space after the colon in `X-Orgo-API-Key:YOUR_API_KEY` (mcp-remote quirk).

</details>

<details>
<summary><strong>Project Configuration (.mcp.json)</strong></summary>

For team sharing, create `.mcp.json` at your project root:

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

Each team member sets their own key: `export ORGO_API_KEY="sk_live_xxx"`

</details>

### 3. Start Using

```
"Create a Linux computer with 4GB RAM"
"Take a screenshot of my computer"
"Run 'ls -la' on the computer"
"Type 'hello world' and press Enter"
```

---

## Features (34 tools)

| Category | Tools |
|----------|-------|
| **Projects** | `orgo_list_projects`, `orgo_create_project`, `orgo_get_project`, `orgo_delete_project`, `orgo_start_project`, `orgo_stop_project`, `orgo_restart_project` |
| **Computers** | `orgo_list_computers`, `orgo_create_computer`, `orgo_get_computer`, `orgo_start_computer`, `orgo_stop_computer`, `orgo_restart_computer`, `orgo_delete_computer` |
| **Actions** | `orgo_screenshot`, `orgo_click`, `orgo_double_click`, `orgo_type`, `orgo_key`, `orgo_scroll`, `orgo_drag`, `orgo_wait` |
| **Shell** | `orgo_bash`, `orgo_exec` |
| **Files** | `orgo_list_files`, `orgo_upload_file`, `orgo_export_file`, `orgo_download_file`, `orgo_delete_file` |
| **Streaming** | `orgo_start_stream`, `orgo_stream_status`, `orgo_stop_stream` |
| **AI** | `orgo_list_ai_models`, `orgo_ai_completion` |

---

## Self-Hosting Options

### Option 1: Local Development (stdio)

```bash
git clone https://github.com/nickvasilescu/orgo-mcp.git
cd orgo-mcp
pip install -e .
export ORGO_API_KEY="your_key"
python orgo_mcp.py
```

Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "orgo": {
      "command": "python3",
      "args": ["/path/to/orgo-mcp/orgo_mcp.py"],
      "env": {"ORGO_API_KEY": "your_key"}
    }
  }
}
```

### Option 2: Local HTTP Server

```bash
pip install -e .
MCP_TRANSPORT=http python orgo_mcp.py
```

Server available at `http://localhost:8000/mcp`

### Option 3: Docker

```bash
# Build
docker build -t orgo-mcp .

# Run
docker run -p 8000:8000 orgo-mcp

# Or use docker-compose
docker-compose up
```

### Option 4: Deploy to Render.com

1. Fork this repository
2. Go to [Render Dashboard](https://dashboard.render.com/blueprints)
3. Click "New Blueprint Instance"
4. Connect your GitHub repo
5. Deploy!

Your server will be at: `https://orgo-mcp-xxxx.onrender.com`

### Option 5: Deploy to Fly.io

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login and deploy
fly auth login
fly launch --no-deploy
fly deploy
```

Your server will be at: `https://orgo-mcp.fly.dev`

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_TRANSPORT` | `stdio` | Transport mode: `stdio` or `http` |
| `MCP_HOST` | `0.0.0.0` | HTTP bind address |
| `MCP_PORT` / `PORT` | `8000` | HTTP port |
| `ORGO_API_KEY` | - | API key (required for stdio transport) |
| `CORS_ORIGINS` | `*` | Allowed origins (comma-separated) |

---

## Usage Examples

### Project Management

```
"Create a new project called 'qa-automation'"
"List my Orgo projects"
"Start all computers in project proj_123"
"Delete project proj_123"
```

### Computer Management

```
"Create a new Linux computer called 'dev-box' with 4GB RAM"
"Get details for computer abc123"
"Start computer abc123"
"Stop computer abc123"
```

### Screen Actions

```
"Take a screenshot of computer abc123"
"Click at coordinates (500, 300)"
"Type 'hello world'"
"Press Enter"
"Scroll down"
```

### Shell Commands

```
"Run 'ls -la' on computer abc123"
"Execute Python code: print('hello')"
```

### File Operations

```
"List files on computer abc123"
"Export file ~/Documents/report.pdf from computer abc123"
"Upload file to computer abc123"
```

### Streaming

```
"Start streaming computer abc123 to Twitch"
"Check stream status for computer abc123"
"Stop streaming computer abc123"
```

### AI Completion

```
"List available AI models"
"Run GPT-4 completion: 'Explain quantum computing'"
```

---

## Troubleshooting

### "Invalid API key" Error

- Verify your key starts with `sk_live_`
- Check for extra spaces or quotes
- Regenerate key at [orgo.ai](https://orgo.ai)

### "X-Orgo-API-Key header required" Error

- Ensure the header is properly formatted
- For mcp-remote: use `Key:Value` format (no space after colon)
- Check environment variable is set

### "Connection refused" Error

- Cloud: Check `https://orgo-mcp.onrender.com/health`
- Local: Ensure server is running on correct port
- Check firewall/proxy settings

### Claude Desktop Not Connecting

- Restart Claude Desktop after config changes
- Check config file syntax (valid JSON)
- View logs: Help > Troubleshooting > Open Logs

### Tools Not Appearing

- Wait 10-30 seconds after connection
- Check server logs for errors
- Verify MCP server is listed in Claude settings

---

## Architecture

```
Cloud Deployment:
  Client (Claude) --> HTTPS --> Cloud Server --> Orgo API
                     X-Orgo-API-Key header

Local Deployment:
  Client (Claude) --> stdio --> orgo_mcp.py --> Orgo API
                     ORGO_API_KEY env var
```

### Project Structure

```
orgo-mcp/
├── orgo_mcp.py       # MCP server (34 tools, dual transport)
├── pyproject.toml    # Package configuration
├── Dockerfile        # Production container
├── docker-compose.yml # Local development
├── render.yaml       # Render.com deployment
├── fly.toml          # Fly.io deployment
├── .dockerignore     # Docker build exclusions
├── .env.example      # Environment template
├── README.md
└── LICENSE
```

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Format code
black orgo_mcp.py

# Lint
ruff check orgo_mcp.py

# Run tests
pytest

# Test HTTP transport locally
MCP_TRANSPORT=http python orgo_mcp.py

# Test health endpoint
curl http://localhost:8000/health

# Test MCP endpoint
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "X-Orgo-API-Key: your_key" \
  -d '{"jsonrpc":"2.0","method":"tools/list","id":1}'
```

---

## License

MIT License - see [LICENSE](LICENSE)

## Credits

- [Orgo](https://orgo.ai) - Virtual computer infrastructure
- [Anthropic](https://anthropic.com) - MCP protocol
