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
- Fix `orgo_wait` body shape so the live API accepts it — sends `{ seconds }` not `{ duration }` (docs/reality mismatch).
- Add `orgo_doctor` health-check tool that reports auth source, API reachability, status code, and latency — bringing the exposed surface to 28 tools.
- Add `compact: true` flag to the 6 read tools (`orgo_list_workspaces`, `orgo_get_workspace`, `orgo_workspace_by_name`, `orgo_list_computers`, `orgo_get_computer`, `orgo_list_files`) to trim responses to identity/status/timestamp fields — typical 50%+ size reduction for agent contexts.
- Add optional `limit` parameter (1-500) to `orgo_list_workspaces`, `orgo_list_computers`, and `orgo_list_files`, with `total` and `truncated` metadata when the cap is hit. Omit `limit` for current (unbounded) behavior.
- Add `scripts/e2e-live.mjs` and a CI job gated on the `ORGO_TEST_API_KEY` secret that exercises every tool category against the live Orgo API end-to-end. Skips gracefully on fork PRs and when the secret is absent. Available locally via `npm run test:live`.
- Sharpen every tool description with an agent-facing "use when..." clause so tool selection is intent-driven; cross-link tools that overlap (e.g., `orgo_click` redirects to `orgo_bash` for scriptable work).
