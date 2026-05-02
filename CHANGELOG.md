# Changelog

## 4.0.0

- Make the TypeScript MCP server the only supported implementation in the repository.
- Remove legacy Python source and unregistered TypeScript tool modules.
- Keep the official exposed surface to 24 tools.
- Remove start/stop computer, VNC password, account/credits, agent/thread, streaming, and template tools.
- Add MCP annotations to all exposed tools.
- Add production tool controls with `ORGO_READ_ONLY`, `ORGO_TOOLSETS`, `ORGO_ENABLED_TOOLS`, and `ORGO_DISABLED_TOOLS`.
- Add MCP tool-list smoke tests and GitHub Actions CI.
- Clean `dist/` before every build so removed tools cannot remain in the npm package.
- Mark `orgo_restart_computer` as destructive so clients can treat restart as an interrupting operation.
- Redact sensitive fields from text responses before returning Orgo API payloads to MCP clients.
