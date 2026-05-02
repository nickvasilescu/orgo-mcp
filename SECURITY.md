# Security Policy

## Supported Implementation

The supported Orgo MCP implementation is the TypeScript package `@orgo-ai/mcp`.

The legacy Python implementation has been removed from this repository and is not supported for production use.

## Key Handling

- Stdio deployments read `ORGO_API_KEY` from the local environment.
- HTTP deployments should receive the user's Orgo key on each `/mcp` request through the `X-Orgo-API-Key` header.
- Do not commit `.env` files, API keys, exported logs containing keys, or local MCP client configs with real credentials.

## Tool Surface Hardening

Production operators can restrict the server with:

- `ORGO_READ_ONLY=true`
- `ORGO_TOOLSETS=core,screen,files`
- `ORGO_ENABLED_TOOLS=orgo_screenshot,orgo_get_computer`
- `ORGO_DISABLED_TOOLS=orgo_bash,orgo_exec`

Disable the `shell` toolset for users or agents that should not execute commands inside Orgo computers.

## Reporting

Report security issues privately to the Orgo team instead of opening a public issue with exploit details.
