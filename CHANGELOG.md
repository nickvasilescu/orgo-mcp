# Changelog

## 4.0.0

- Make the TypeScript MCP server the only supported implementation in the repository.
- Remove legacy Python source and unregistered TypeScript tool modules.
- Expose 27 official tools covering workspaces, computers, computer actions, shell, and files.
- Remove start/stop computer, VNC password, account/credits, agent/thread, streaming, and template tools.
- Add MCP annotations to all exposed tools.
- Add production tool controls with `ORGO_READ_ONLY`, `ORGO_TOOLSETS`, `ORGO_ENABLED_TOOLS`, and `ORGO_DISABLED_TOOLS`.
- Add MCP tool-list smoke tests and GitHub Actions CI.
- Clean `dist/` before every build so removed tools cannot remain in the npm package.
- Mark `orgo_restart_computer` as destructive so clients can treat restart as an interrupting operation.
- Redact sensitive fields from text responses before returning Orgo API payloads to MCP clients.
- Add `orgo_delete_workspace` to permanently delete a workspace and all of its computers.
- Add `orgo_wait` to pause VM execution for a fixed duration, useful for action sequencing.
- Add `orgo_move_computer` to transfer a computer between workspaces while preserving disk state.
