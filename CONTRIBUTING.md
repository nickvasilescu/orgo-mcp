# Contributing to Orgo MCP

This repository maintains the TypeScript Orgo MCP server published as `@orgo-ai/mcp`.

The supported implementation lives under `src/`. The legacy Python implementation has been removed and should not be reintroduced.

## Development Setup

Prerequisites:

- Node.js 22 for development and CI parity
- npm
- Git
- Optional Orgo API key for live tool testing

```bash
git clone https://github.com/YOUR_USERNAME/orgo-mcp.git
cd orgo-mcp
npm install
npm test
```

For local stdio testing:

```bash
ORGO_API_KEY=sk_live_YOUR_KEY npm start
```

For local HTTP testing:

```bash
npm run build
MCP_TRANSPORT=http npm start
curl http://localhost:8000/health
```

## Tool Surface Rules

The official tool surface is intentionally focused. Do not add broad tool families without a clear product reason and explicit review.

Current toolsets:

- `core`: read-only workspace and computer lookup
- `admin`: create/delete/restart/clone/resize/ensure computer state
- `screen`: screenshot, mouse, keyboard, scroll, drag
- `shell`: bash and Python execution
- `files`: list, export, upload, download

Do not re-add these removed tool families without review:

- computer start/stop
- VNC password access
- account/profile/credits/transactions
- autonomous agent/thread tools
- streaming tools
- template management tools

## Adding or Changing Tools

All tools must be registered through `registerOrgoTool` in `src/tools/registry.ts`.

Every tool needs:

- a stable `orgo_*` name
- a concise title and description
- a Zod input schema
- one or more toolsets
- MCP annotations for `readOnlyHint`, `destructiveHint`, `idempotentHint`, and `openWorldHint`
- a smoke-test update in `scripts/smoke-tools.mjs`
- README updates if the public surface changes

Be conservative with `readOnlyHint`. If a tool mutates Orgo control-plane state, VM state, files, UI, or exported artifacts, mark it non-read-only.

## Safety Controls

Keep these environment controls working:

- `ORGO_READ_ONLY`
- `ORGO_TOOLSETS`
- `ORGO_ENABLED_TOOLS`
- `ORGO_DISABLED_TOOLS`

Any change to tool registration should be tested against all four paths.

## Required Checks

Run these before opening a PR:

```bash
npm test
npm run test:http
npm pack --dry-run
```

If Docker is available:

```bash
docker build -t orgo-mcp-local .
docker run --rm -p 8000:8000 orgo-mcp-local
curl http://localhost:8000/health
```

CI also verifies npm package contents so stale source files or removed tool modules do not ship.

## Pull Requests

Keep PRs focused. Include:

- what changed
- why it changed
- which checks were run
- any compatibility or migration note

Use clear commit messages such as:

```text
feat: add read-only tool policy
fix: preserve API key header in HTTP sessions
docs: document hosted MCP setup
```

## Security

Do not include API keys, `.env` files, live MCP client configs, screenshots with secrets, or logs containing credentials.

Report security issues privately to the Orgo team instead of opening a public issue with exploit details.
