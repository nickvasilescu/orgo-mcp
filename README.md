# Orgo MCP

[![npm](https://img.shields.io/npm/v/orgo-mcp-server?color=black&label=npm)](https://www.npmjs.com/package/orgo-mcp-server)
[![MCP](https://img.shields.io/badge/MCP-server-black)](https://modelcontextprotocol.io)
[![license](https://img.shields.io/badge/license-MIT-black)](./LICENSE)

Cloud computers for AI agents, as [MCP](https://modelcontextprotocol.io) tools. Ask a model to
spin up a Linux desktop, drive its browser, run shell commands, and read the screen back.

**[docs.orgo.ai](https://docs.orgo.ai/guides/mcp)**

Every tool is generated from Orgo's published OpenAPI spec at startup, so the server cannot
drift from the API. When an endpoint ships in the docs, it is a tool on the next restart.

```
docs.orgo.ai/api-reference/openapi.json  ──►  43 MCP tools  ──►  www.orgo.ai/api
```

## Setup

**1. Get an API key** at [orgo.ai/workspaces](https://www.orgo.ai/workspaces) → Settings → API Keys.

**2. Add the server.**

```bash
claude mcp add orgo -e ORGO_API_KEY=sk_live_... -- npx -y orgo-mcp-server
```

**3. Ask for something.**

```text
spin up an orgo computer and screenshot the orgo.ai homepage
```

That is the whole setup. Node 20 or later, nothing to install or build.

Already use the [Orgo CLI](https://docs.orgo.ai/guides/cli)? Run `orgo login` once and drop the
`-e` flag: the server reads the same `~/.orgo/credentials.json`, so no key sits in a config file.

<details>
<summary><b>Other clients</b> - Codex, Claude Desktop, Cursor, VS Code, Windsurf</summary>

**Codex**

```bash
codex mcp add orgo --env ORGO_API_KEY=sk_live_... -- npx -y orgo-mcp-server
```

**Claude Desktop** - `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS,
`%APPDATA%\Claude\claude_desktop_config.json` on Windows. Quit and relaunch fully; a reload is
not enough.

```json
{
  "mcpServers": {
    "orgo": {
      "command": "npx",
      "args": ["-y", "orgo-mcp-server"],
      "env": { "ORGO_API_KEY": "sk_live_..." }
    }
  }
}
```

**Cursor** - `~/.cursor/mcp.json`, or `.cursor/mcp.json` per project. The project file gets
committed, so read the key from the environment instead of inlining it:

```json
{
  "mcpServers": {
    "orgo": {
      "command": "npx",
      "args": ["-y", "orgo-mcp-server"],
      "env": { "ORGO_API_KEY": "${env:ORGO_API_KEY}" }
    }
  }
}
```

**VS Code** - `.vscode/mcp.json`. The top-level key is `servers`, not `mcpServers`, and `type`
is required:

```json
{
  "inputs": [
    { "type": "promptString", "id": "orgo-key", "description": "Orgo API key", "password": true }
  ],
  "servers": {
    "orgo": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "orgo-mcp-server"],
      "env": { "ORGO_API_KEY": "${input:orgo-key}" }
    }
  }
}
```

**Windsurf** - `~/.codeium/windsurf/mcp_config.json` for the desktop editor,
`~/.codeium/mcp_config.json` for the JetBrains and VS Code plugins. Body matches Claude
Desktop's. Press refresh after editing.

**Anything else** takes the same four fields. See the MCP guide on
[connecting local servers](https://modelcontextprotocol.io/docs/develop/connect-local-servers)
if your client is not listed:

```
command:    npx
args:       -y orgo-mcp-server
env:        ORGO_API_KEY=sk_live_...
transport:  stdio
```

</details>

## More prompts

```text
open chrome on my orgo computer, search for the weather, screenshot the result
```

```text
list my orgo computers and stop the ones that are idle
```

```text
run the test suite on a clean linux desktop and show me what failed
```

Screenshots come back as images the model can actually see, not links.

## Tools

43 tools, one per API operation, named after its `operationId`. `createComputer` becomes
`orgo_create_computer`.

| Group | | Tools |
| --- | --- | --- |
| Workspaces | 4 | `list_workspaces` `create_workspace` `get_workspace` `delete_workspace` |
| Computers | 7 | `create_computer` `get_computer` `delete_computer` `clone_computer` `resize_computer` `move_computer` `get_vnc_password` |
| Computer Lifecycle | 3 | `start_computer` `stop_computer` `restart_computer` |
| Computer Actions | 9 | `get_screenshot` `mouse_click` `mouse_drag` `type_text` `press_key` `scroll` `wait` `execute_bash` `execute_python` |
| Files | 5 | `upload_file` `list_files` `export_file` `download_file` `delete_file` |
| Templates | 15 | `list_templates` `publish_template` `build_template` and 12 more |

<details>
<summary>All 15 Templates tools</summary>

`orgo_get_template_schema` `orgo_list_templates` `orgo_publish_template` `orgo_validate_template`
`orgo_list_curated_templates` `orgo_list_curated_template_versions` `orgo_list_template_versions`
`orgo_get_template` `orgo_delete_template` `orgo_build_template` `orgo_get_build_status`
`orgo_cancel_build` `orgo_stream_build_events` `orgo_test_run_template` `orgo_stop_template_run`

</details>

Group names are the spec's own tags, so `ORGO_TAGS="Workspaces,Computers"` narrows the surface
using the same words the [API reference](https://docs.orgo.ai/api-reference/introduction) uses.
Set it when a client's tool picker starts guessing.

## Configuration

Nothing below is required. `ORGO_API_KEY` is the only variable most setups ever set.

| Variable | Default | Description |
| --- | --- | --- |
| `ORGO_API_KEY` | (none) | Your API key. Falls back to `~/.orgo/credentials.json` from `orgo login`. |
| `ORGO_READ_ONLY` | `false` | Any value but `false`, `0`, `no` or `off` registers only the 15 `GET` tools. |
| `ORGO_TAGS` | all | Comma-separated spec tags to expose. Three contain a space, so quote it: `"Computers,Computer Actions"`. |
| `ORGO_DEFAULT_COMPUTER_ID` | (none) | Pin a computer so action tools can omit `id`. An explicit `id` always wins. |

<details>
<summary>Escape hatches</summary>

| Variable | Default | Description |
| --- | --- | --- |
| `ORGO_PROFILE` | `default` | Which credential profile to read. |
| `ORGO_API_BASE_URL` | `https://www.orgo.ai` | Point at a different host. |
| `ORGO_TIMEOUT_MS` | `120000` | Per-request timeout. |
| `ORGO_RETRIES` | `2` | Retries for idempotent requests on a 5xx or a dropped connection. |
| `ORGO_RAW` | unset | Return exactly what the API sent, without shaping. |
| `ORGO_SPEC_URL` | [the published spec](https://docs.orgo.ai/api-reference/openapi.json) | Where to load the spec from. |
| `ORGO_SPEC_MAX_AGE_MS` | `86400000` | How long a cached spec stays fresh. |
| `XDG_CACHE_HOME` | `~/.cache` | Where the spec cache lives. |

</details>

## Security

An `sk_live_` key is account-wide by default, and `orgo_execute_bash` and `orgo_execute_python`
run arbitrary commands on a live machine. Leave your client's tool confirmation on. For a
narrower blast radius, create a workspace-scoped key under Settings → API Keys, or set
`ORGO_READ_ONLY=true` to register only the 15 read tools.

## How it works

**Tools come from the spec.** Descriptions, types, enums, defaults and required fields are read
straight out of it, so `orgo_create_computer` already knows `ram` is one of `4, 8, 16, 32, 64`
without anyone writing that down twice. Path, query and body parameters flatten into one
argument list, so a call reads `orgo_resize_computer({ id, mem_gb })` rather than nesting a
body. Annotations follow the HTTP method: the 15 `GET` tools are read-only, and every write is
marked destructive, because `orgo_execute_bash` cannot promise otherwise.

**Responses are shaped to it too.** The API returns internal state the docs never promise, and
on a real account that is 80% of the bytes: encrypted VNC passwords, host and server ids, boot
details. Only the fields the spec declares survive. A schema matching none of the keys in hand
is treated as describing something else, so the payload passes through untouched instead of
being emptied. Spec drift costs verbosity, never data. `ORGO_RAW=1` turns this off.

**Failures retry only when that is provably safe.** A 5xx or a dropped connection is retried
with doubling backoff, but only for the methods the spec marks idempotent. `orgo_execute_bash`
and `orgo_mouse_click` get exactly one attempt, because running either twice is worse than
failing once.

**It speaks the current protocol.** The 2026-07-28 revision is stateless: no `initialize`
handshake, no session id, each request carrying its own protocol version and client identity.
That is served natively, alongside every 2025-era revision back to `2024-11-05`, so old and new
clients both work. `server/discover` answers with the tool capabilities, and `tools/list` is
advertised as publicly cacheable for as long as the spec stays fresh, since the list is the same
for every caller.

**The spec is loaded fresh.** In order, first one wins:

1. **Cache** - `$XDG_CACHE_HOME/orgo-mcp/openapi.json`, if fetched within `ORGO_SPEC_MAX_AGE_MS`.
2. **Network** - `ORGO_SPEC_URL`, 5 second timeout, written back to the cache.
3. **Stale cache** - whatever was last fetched.
4. **Bundled snapshot** - `openapi.snapshot.json`, shipped in the package.

A normal start is instant and offline. A start after the cache expires picks up API changes with
no release. A machine with no network still works. Refresh the committed snapshot with
`npm run sync` before cutting a release.

## Troubleshooting

Run the server directly to see what it thinks. It prints one line to stderr, then waits for
JSON-RPC on stdin:

```bash
ORGO_API_KEY=sk_live_... npx -y orgo-mcp-server
# [orgo-mcp] 43 tools from Orgo API 2.0.0 (https://docs.orgo.ai/api-reference/openapi.json) → https://www.orgo.ai/api
```

That line names the tool count, the API version and which spec source won, which answers most
setup questions on its own. Silence on stdout is correct; that channel carries the protocol.

<details>
<summary><code>spawn npx ENOENT</code> on Windows</summary>

npm ships `npx.cmd`, and Node's `spawn` will not resolve a `.cmd` without a shell:

```json
{
  "mcpServers": {
    "orgo": {
      "command": "cmd",
      "args": ["/c", "npx", "-y", "orgo-mcp-server"],
      "env": { "ORGO_API_KEY": "sk_live_..." }
    }
  }
}
```

</details>

<details>
<summary><code>HTTP 401: Invalid API key</code></summary>

Check the key on its own:

```bash
curl -H "Authorization: Bearer $ORGO_API_KEY" https://www.orgo.ai/api/workspaces
```

If that works but the server does not, your client is not forwarding `env`. Pass the key inline
with `claude mcp add -e` instead.

</details>

<details>
<summary>The tools don't appear in my client</summary>

Fully quit and relaunch. Most clients read MCP config only at startup. Then check
`ORGO_READ_ONLY` and `ORGO_TAGS`, which reduce the tool list by design. Claude Desktop logs to
`~/Library/Logs/Claude/mcp-server-orgo.log` on macOS.

</details>

## Development

```bash
npm install
npm test     # typecheck, then spec → tools against the bundled snapshot. No network, no key.
npm run dev  # run from source over stdio
```

764 lines of source, in seven files:

| File | Lines | Responsibility |
| --- | --- | --- |
| `src/tools.ts` | 184 | Operation → tool: name, description, flattened schema, annotations. |
| `src/client.ts` | 156 | Tool call → HTTP request (JSON, multipart, SSE, retries). |
| `src/spec.ts` | 126 | Load the OpenAPI document; inline `$ref`s. |
| `src/result.ts` | 87 | HTTP reply → MCP content, shaped to the spec's response. |
| `src/server.ts` | 73 | Public entry: build a server for any transport. |
| `src/http.ts` | 80 | HTTP transport, one server per request. |
| `src/index.ts` | 58 | The stdio bin: resolve auth, then hand off. |

The test suite pins the numbers in this README to the code, so a stale table fails `npm test`.

## Hosting it yourself

The package is a library as well as a binary, so the same 43 tools can be served over HTTP with
per-request credentials. In a Next.js route that is the whole file:

```ts
// app/api/mcp/route.ts
import { createHttpHandler } from 'orgo-mcp-server'

const handler = createHttpHandler()

export const POST = (request: Request) => handler.fetch(request)
```

`handler.fetch` is a web-standard `Request` to `Response`, so the same handler works in Hono,
Bun, or bare `node:http`. The spec loads once on the first request; every request after builds
its own server carrying that caller's key, so a shared deployment holds nothing between callers.

By default the key is read from the `Authorization` bearer token and requests go to
`https://www.orgo.ai`. Override any of it:

```ts
createHttpHandler({
  apiKey: (request) => request.headers.get('x-orgo-api-key') ?? undefined,
  baseUrl: process.env.ORGO_API_BASE_URL, // any origin, for another environment
  timeoutMs: 120_000,
})
```

`baseUrl` takes any origin, so the same server can point at a domain alias or a local
orgo-web. Leave it unset in production. `ORGO_API_BASE_URL` works for the stdio binary too.

A caller pins a computer with the `X-Orgo-Default-Computer-Id` header, or a `computer_id` query
param for clients that only let you configure a URL. It is read per request, so two callers on
one deployment never see each other's.

`baseUrl` also takes a function of the request, so a host serving this route from inside the API
itself can call its own origin and needs no configuration per environment:

```ts
createHttpHandler({ baseUrl: (request) => new URL(request.url).origin })
```

Clients then connect to the URL instead of spawning a process:

```json
{
  "mcpServers": {
    "orgo": {
      "type": "http",
      "url": "https://www.orgo.ai/api/mcp",
      "headers": { "Authorization": "Bearer sk_live_..." }
    }
  }
}
```

## License

MIT
